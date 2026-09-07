"""Text-level baseline arms: HiAgent and ACON.

Both methods manage the agent's context at the TEXT level: the proxy
rewrites the request's history before it reaches the serving backend, and
the compression/summarization calls go to the SAME served ckpt-1088
endpoint (policy and compressor are the same model — the papers' own
protocol: HiAgent §4.1 uses gpt-4-turbo for both roles, ACON §4.2 uses
gpt-4.1 for both).  Cost columns must carry the compressor calls (one extra
LLM call per turn); the standing asymmetry (C2KV is trained for its
mechanism, these baselines are training-free here) is footnote material.

Sources:
* HiAgent  — arXiv 2408.09559.  The working-memory summarization prompt
  below is the PAPER's §3.3 prompt verbatim (subgoal-met judgment, one-line
  output constraint, {example} slot).  The repo's summarize.py
  (hiagent2024/hiagent @ cebdd8e) was AI-rebuilt in 2026-04 and post-dates
  the paper's numbers — audit ruling 1: not used.
  ``hiagent_summary`` contains the subgoal and observation-summarization
  components. ``hiagent_full`` additionally exposes Trajectory Retrieval as
  a proxy-owned meta-tool: the proxy expands selected completed trajectories
  and retries policy generation without forwarding retrieval to the task.
  ``full`` names the included method components, not a claim that this
  tool-native adaptation reproduces the paper's benchmark numbers; its
  task-specific in-context example slot remains empty below.
  - The repo's gripper/blocksworld summarization-off special case
    (cme_final.py:115-120) is NOT inherited (ruling 3: that is the paper's
    w/o-OS ablation, not the method).
  - User turns are NOT action-observation pairs: they survive verbatim in
    the output AND ride along in the summarizer input as "User:" lines
    (audit: a "$300 budget" constraint was silently dropped otherwise).
  - Compressor decode per paper §4.1: max_tokens~100, stop "\n\n",
    temperature 0, top_p 1; enable_thinking off (serving stack has no
    reasoning parser).
  - Degeneration is VISIBLE: when no assistant content ever declares a
    Subgoal (e.g. pure tool-call replies), the arm is a passthrough and
    stats["degenerate"] is True.
* ACON     — arXiv 2510.00615.  Thresholds per paper appendix B.3:
  T_obs = 1024 tok (~4096 chars), history 4096 tok, preserve the last ONE
  action/observation PAIR = k=2 messages. ``base`` uses the official repo's
  initial ``context_opt`` guidelines (microsoft/acon @ d63f9ae); ``ut`` and
  ``ut_co`` use the paper appendix's fixed guidelines after the utility and
  utility-plus-compression optimization steps respectively.
  Source/licensing: the base templates come from microsoft/acon @ d63f9ae
  under its MIT License; the UT and UT+CO templates are transcribed from
  Appendix D of arXiv:2510.00615v3, whose arXiv HTML is CC BY 4.0.
  History/observation are evaluated SEPARATELY by the paper (ruling 6):
  arms keep the two faces and guideline stages explicit in their names.
  Summary is embedded into the first user prompt as a <HISTORY_SUMMARY>
  block (the original memory.py:481-498 shape), not a standalone message.

Token counts are estimated as chars/4 (no tokenizer in the proxy); the
estimate is recorded in stats.

Trigger note (2026-09-05 rewrite): faithful to the upstream — the
history compression fires when est(prev_summary + messages since the last
compression) exceeds T_hist (the first compression measures the raw
history alone, tool_calls included in the estimate); between compressions
the policy SEES the summary block + every new raw turn + the preserved
action/observation pair.  An earlier port compressed every turn past the
threshold and dropped the raw recent context — that was NOT conservative,
it removed ACON's recent-context signal.  Summaries are cached per content hash; an
EMPTY compressor result is a failure and is never cached.
"""
from __future__ import annotations

import hashlib
import json
import threading
from typing import Any, Callable, Dict, List, Optional, Tuple

# ---- compressor plumbing -----------------------------------------------------

# decode parameters per policy (papers' originals + NPU serving reality:
# Qwen3 thinking must be off, the server has no reasoning parser)


COMPRESSOR_DECODE: Dict[str, Dict[str, Any]] = {
    "hiagent": {
        "max_tokens": 100, "stop": ["\n\n"],
        "temperature": 0.0, "top_p": 1.0, "seed": 42,
        "chat_template_kwargs": {"enable_thinking": False},
    },
    "acon": {
        "max_tokens": 2048, "temperature": 0.0, "seed": 42,
        "chat_template_kwargs": {"enable_thinking": False},
    },
}


class TextarmCompressorError(RuntimeError):
    """The summarizer/refiner call failed (transport, abort finish, or an
    empty completion).  Must surface as an infrastructure error, never as
    an empty summary that gets cached for the rest of the conversation."""

    kind = "textarm_error"


def compressor_payload(policy: str, model: str, system_text: str,
                       user_text: str) -> Dict[str, Any]:
    """Full chat request for one compressor call (decode params above)."""
    payload: Dict[str, Any] = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_text},
            {"role": "user", "content": user_text},
        ],
    }
    payload.update(COMPRESSOR_DECODE[policy])
    return payload


# compress(payload_dict) -> str : performs the POST, validates the finish
# reason and non-empty content, raises TextarmCompressorError otherwise.
Compress = Callable[[Dict[str, Any]], str]

_LOCK = threading.Lock()
_SUMMARY_CACHE: Dict[str, str] = {}
_ACON_STATE: Dict[Tuple[str, str, str], Tuple[int, str, str]] = {}


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _est_tokens(chars: int) -> int:
    return max(1, chars // 4)


def _content_of(message: Dict[str, Any]) -> str:
    content = message.get("content")
    if isinstance(content, str):
        return content
    if content is None:
        return ""
    return json.dumps(content, ensure_ascii=False)


def _message_chars(messages: List[Dict[str, Any]]) -> int:
    return sum(
        len(_content_of(m)) + len(json.dumps(m.get("tool_calls") or []))
        for m in messages
    )


def _render_line(message: Dict[str, Any], action_dialect) -> str:
    role = message.get("role") or "user"
    if role == "assistant":
        if message.get("tool_calls"):
            return "Action: " + action_dialect(message)
        return "Action: " + _content_of(message)
    if role == "tool":
        return "Observation: " + _content_of(message)
    if role == "user":
        return "User: " + _content_of(message)
    return f"{role}: " + _content_of(message)


def _summarize(cache_key: str, policy: str, model: str, system_text: str,
               user_text: str, compress: Compress,
               stats: Dict[str, Any]) -> str:
    """Cached summary with failure semantics: an empty result raises and is
    never stored."""
    with _LOCK:
        cached = _SUMMARY_CACHE.get(cache_key)
    if cached is not None:
        return cached
    summary = compress(compressor_payload(policy, model, system_text,
                                          user_text)).strip()
    if not summary:
        raise TextarmCompressorError(
            f"{policy} compressor returned an empty summary "
            f"(key={cache_key[:12]})")
    with _LOCK:
        summary = _SUMMARY_CACHE.setdefault(cache_key, summary)
    stats["n_compressor_calls"] = stats.get("n_compressor_calls", 0) + 1
    return summary


# ---- HiAgent -----------------------------------------------------------------

# == train_data.py:14 / proxy.DEFAULT_SYSTEM_PROMPT (kept in sync by
# test_hiagent_default_system_prefix; proxy passes it explicitly)
TRAINING_DEFAULT_SYSTEM_PROMPT = "You are a helpful assistant."

HIAGENT_SUBGOAL_NOTE = """
Note: A subgoal is a milestone goal that you need to complete in order to achieve the final goal.
When there is an unfinished subgoal and another environment action is needed, continue it by issuing the corresponding native tool call.
When there is no current subgoal or you believe the previous subgoal has been completed (based on past actions and observations), and another environment action is needed, put the next subgoal in assistant content as exactly one line in the format \"Subgoal: {subgoal}\" and issue its first action as a native tool call in the same response.
When no environment action is needed because you can answer, finish, or must ask for clarification, reply naturally in assistant content without inventing a tool call or a Subgoal line.
Instructions:
1. You cannot output two subgoals consecutively.
2. Subgoal must be one line of text and does not print any newline characters.
3. Each declared subgoal must be accompanied in the same response by at least one valid native tool call.
4. Every environment action must use the API's native function-calling interface in the structured tool_calls field. Do not serialize an action into assistant content.
"""

HIAGENT_RETRIEVE_TOOL_NAME = "hiagent_retrieve"
HIAGENT_RETRIEVAL_NOTE = """
5. Detailed action-observation trajectories for completed subgoals are hidden. If a hidden trajectory is needed, call hiagent_retrieve with its one-based subgoal id through the same native function-calling interface. This is a context-retrieval action, not an environment action. Do not serialize the retrieval request into assistant content.
"""

# Paper §3.3, verbatim (the repo's summarize.py is a 2026 rebuild and is
# NOT the prompt the paper's numbers were produced with — audit ruling 1).
HIAGENT_SUMMARY_USER_TEMPLATE = """You are an advanced AI system tasked with summarizing and analyzing a series of action-observation pairs (trajectories) and determining whether a specific subgoal has been met.

Your goal is to create a summary that captures all essential information, decisions, and outcomes from the given trajectories, and indicate whether the subgoal has been met based on the summarized observations.

If there are no valid actions taken, you need to analyze the reason.

### Instructions:

1. Provide a summarized observation related to the subgoal in a concise manner.

2. Determine whether the subgoal has been met.

3. Do not output anything except whether summary and subgoal are met. Your output should be only one line. Do not output things like '##Summary', '##Summary and Analysis'.

{example}

##Trajectory

{formatted_trajectory}

##Subgoal:

{subgoal}

###Output:"""

HIAGENT_SUMMARY_SYSTEM = "You are a helpful assistant."
# per-task in-context example from the paper's setup; none transcribed for
# these benchmarks — the slot stays for provenance
HIAGENT_EXAMPLE = ""


def _subgoal_of(message: Dict[str, Any]) -> Optional[str]:
    if (message.get("role") or "") != "assistant":
        return None
    text = _content_of(message).strip()
    if text.startswith("Subgoal:"):
        return text.split("\n", 1)[0][len("Subgoal:"):].strip()
    return None


def hiagent_retrieval_tool() -> Dict[str, Any]:
    """Native tool schema for HiAgent's internal Trajectory Retrieval.

    This is a proxy meta-tool: the proxy consumes it and retries generation;
    the benchmark environment must never receive the call.
    """
    return {
        "type": "function",
        "function": {
            "name": HIAGENT_RETRIEVE_TOOL_NAME,
            "description": (
                "Reveal the full action-observation trajectory of one or more "
                "completed HiAgent subgoals before choosing an environment action."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "subgoal_ids": {
                        "type": "array",
                        "items": {"type": "integer", "minimum": 1},
                        "minItems": 1,
                    }
                },
                "required": ["subgoal_ids"],
                "additionalProperties": False,
            },
        },
    }


def hiagent_retrieval_request(message: Dict[str, Any]) -> Optional[List[int]]:
    """Parse one native retrieval meta-call from an assistant response.

    ``None`` means no retrieval call. A malformed retrieval call raises
    ``ValueError`` instead of leaking through as an environment action.
    """
    matches: List[int] = []
    for call in message.get("tool_calls") or []:
        function = call.get("function") or {}
        if function.get("name") != HIAGENT_RETRIEVE_TOOL_NAME:
            continue
        try:
            arguments = function.get("arguments") or "{}"
            arguments = (json.loads(arguments) if isinstance(arguments, str)
                         else dict(arguments))
            values = arguments["subgoal_ids"]
            if (not isinstance(values, list) or not values
                    or any(isinstance(v, bool) or not isinstance(v, int) or v < 1
                           for v in values)):
                raise ValueError
        except (KeyError, TypeError, json.JSONDecodeError, ValueError) as error:
            raise ValueError("malformed hiagent_retrieve subgoal_ids") from error
        matches.extend(values)
    if not matches:
        return None
    return list(dict.fromkeys(matches))


def hiagent_transform(messages: List[Dict[str, Any]], compress: Compress,
                      action_dialect, model: str = "c2kv-agent",
                      default_system: str = TRAINING_DEFAULT_SYSTEM_PROMPT,
                      variant: str = "summary",
                      retrieve_subgoals: Optional[List[int]] = None,
                      ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Subgoal-protocol note into the system message; segment history by
    assistant 'Subgoal:' declarations; completed segments become their
    surviving user turns + one paper-prompt summary; the current segment
    stays raw.  Passthrough (degenerate=True) when no subgoal is ever
    declared — e.g. pure tool-call replies with null content.

    ``variant='full'`` includes the retrieval component only when the caller
    also advertises :func:`hiagent_retrieval_tool`, intercepts the returned
    meta-call with :func:`hiagent_retrieval_request`, and retries with those
    ids in ``retrieve_subgoals``.
    """
    if variant not in ("summary", "full"):
        raise ValueError(f"unknown HiAgent variant {variant!r}")
    requested = set(retrieve_subgoals or [])
    if any(isinstance(v, bool) or not isinstance(v, int) or v < 1
           for v in requested):
        raise ValueError("retrieve_subgoals must contain positive integer ids")
    if requested and variant != "full":
        raise ValueError("trajectory retrieval requires variant='full'")
    note = HIAGENT_SUBGOAL_NOTE
    if variant == "full":
        note = note.rstrip() + "\n" + HIAGENT_RETRIEVAL_NOTE.strip() + "\n"
    stats: Dict[str, Any] = {
        "policy": "hiagent", "variant": variant, "n_compressor_calls": 0,
    }
    messages = [dict(m) for m in messages]
    for m in messages:
        if m.get("role") == "system":
            m["content"] = (_content_of(m).rstrip() + "\n" + note)
            break
    else:
        # audit 2026-09-05: inserting a system with ONLY the note made
        # _assemble skip the training default, so hiagent differed from
        # full by more than the note — insert default + note instead
        messages.insert(0, {"role": "system",
                            "content": (default_system.rstrip() + "\n"
                                        + note.strip())})

    segments: List[Dict[str, Any]] = [{"start": 0, "subgoal": None, "messages": []}]
    for m in messages:
        subgoal = _subgoal_of(m)
        if subgoal is not None and segments[-1]["messages"]:
            segments.append({"start": 0, "subgoal": subgoal, "messages": []})
        segments[-1]["messages"].append(m)
    pre = segments[0]["messages"] if segments[0]["subgoal"] is None else []
    segs = segments if segments[0]["subgoal"] is not None else segments[1:]
    completed, current = segs[:-1], segs[-1:]

    stats["n_segments"] = len(segs)
    stats["n_summarized"] = len(completed)
    stats["degenerate"] = bool(
        len(segs) == 0 and any(m.get("role") == "assistant" for m in messages))
    raw_chars = _message_chars(messages)

    out: List[Dict[str, Any]] = list(pre)
    for k, seg in enumerate(completed, start=1):
        if k in requested:
            out.extend(dict(m) for m in seg["messages"])
            continue
        # user turns are instructions, not trajectory: survive verbatim
        for m in seg["messages"]:
            if m.get("role") == "user":
                out.append(dict(m))
        trajectory = "\n".join(_render_line(m, action_dialect)
                               for m in seg["messages"])
        subgoal = seg["subgoal"] or f"(segment {k})"
        summary = _summarize(
            _sha(f"hiagent|{subgoal}|{trajectory}"),
            "hiagent", model, HIAGENT_SUMMARY_SYSTEM,
            HIAGENT_SUMMARY_USER_TEMPLATE.format(
                example=HIAGENT_EXAMPLE,
                formatted_trajectory=trajectory, subgoal=subgoal),
            compress, stats)
        out.append({"role": "user",
                    "content": f"Subgoal {k}: {subgoal}\nSummary: {summary}"})
    out.extend(m for seg in current for m in seg["messages"])

    valid_ids = set(range(1, len(completed) + 1))
    stats["retrieved_subgoals"] = sorted(requested & valid_ids)
    stats["invalid_retrieval_subgoals"] = sorted(requested - valid_ids)
    stats["n_summarized"] = len(completed) - len(requested & valid_ids)

    out_chars = _message_chars(out)
    stats["raw_chars"] = raw_chars
    stats["out_chars"] = out_chars
    stats["raw_est_tokens"] = _est_tokens(raw_chars)
    stats["out_est_tokens"] = _est_tokens(out_chars)
    return out, stats


# ---- ACON --------------------------------------------------------------------

ACON_SYSTEM = (
    "You are an agent tasked with extracting and refining a concise and "
    "optimized version of the context based on the user instruction and "
    "other provided information."
)

ACON_HISTORY_PROMPT_BASE = """You are maintaining a structured context-aware summary for a productivity agent. You will be given the user instruction for the agent, a list of interactions corresponding to actions taken by the agent, and the most recent previous summary if one exists. Produce the following:

### REASONING
Summarize key progress, decisions made, important observed outcomes, and rationale behind actions taken so far. Include how earlier steps influenced later ones and why certain data is retained in the summary.

### COMPLETED
List completed subtasks or successful outcomes, with brief results if applicable.

---

## [Information Source]

### USER INSTRUCTION

{task}

## [PREVIOUS SUMMARY] (if any)

{prev_summary}

## [HISTORY OF INTERACTIONS]

{history}

---

## PRIORITIZE

1. Keep all sections relevant and concise.
2. Use reusable structured formats when summarizing artifacts.
3. Ensure agent can resume task with no loss of information.
4. Include key info from errors or failed attempts to prevent repeated mistakes.
5. Preserve all essential artifacts and data needed to complete the task.

---

### [Output Format]

Do **not** include the input or any additional explanation. Only return the formatted summary."""

ACON_HISTORY_PROMPT_UT = """You maintain a compact, state-preserving HISTORY_SUMMARY for a multi-session agent.

Input:
[USER INSTRUCTION] {task}
[PREVIOUS SUMMARY] {prev_summary}
[HISTORY OF INTERACTIONS] {history}

Create the following sections-use the exact headings and order:
<HISTORY_SUMMARY>
1. REASONING
- Key progress, decisions, outcomes, and their rationale.
- Note how earlier steps influence later ones.
2. VARS
| name | value | purpose |
|---|---|---|
Record every runtime value the next session must re-declare (tokens, ids, lists, last page_index/page_limit, etc.).
3. TODO
List pending actions with enough detail to execute directly.
4. COMPLETED
Bullet list of finished subtasks with brief results.
5. GUARDRAILS
Short reminders that prevent repeat errors, e.g.
- Memory resets; re-create VARS before use.
- Paginate until empty page.
- Validate API parameters against spec.
- Avoid redundant logins or doc look-ups.

Requirements:
- Be concise-bullets and tables preferred; no extraneous prose.
- Preserve all essential facts, parameters, and artifacts; omit nothing critical.
- Include errors only if they inform future avoidance.
- Do not output the input or any commentary-return only <HISTORY_SUMMARY>."""

ACON_HISTORY_PROMPT_UT_CO = """You maintain a compact, state-preserving HISTORY_SUMMARY for a multi-session agent.

Input:
[USER INSTRUCTION] {task}
[PREVIOUS SUMMARY] {prev_summary}
[HISTORY OF INTERACTIONS] {history}

Summary Compression Rules:
- Collapse multi-bullet narratives into <=2 concise sentences.
- Replace repetitive step logs with one summarizing phrase.
- Truncate long token/credential strings to "<token>" unless verbatim reuse is required.
- Remove unused/expired credentials, page_index/page_limit, verbose API dumps, and table borders.
- Shrink GUARDRAILS to one bullet unless multiple items are still critical.
- Delete tool/API log output, greetings, meta prose, and section headers that no longer contain content.
- Keep only variables actively referenced in upcoming steps; list each once in VARS.
- Reference removal categories [repetition], [tool-logs], [meta], [formatting] to prune similar lines.
- Preserve factual continuity; never invent or alter state variables.
- Target summaries well under 1500 characters.

Critical Essentials:
Always keep evidence-driven items required next session (e.g., tokens, ids, emails, amounts, lists, paths, description strings, brief task status).

Output EXACTLY the following structure-nothing more:
<HISTORY_SUMMARY>
1. REASONING
One brief paragraph on key progress and rationale.
2. VARS
key=value pairs, comma-separated; only still-needed runtime values.
3. TODO
Bulleted next actions (<=5).
4. COMPLETED
Bulleted finished subtasks (<=5).
5. GUARDRAILS
Single concise bullet, or omit if none.
Return only the <HISTORY_SUMMARY> block-no additional commentary or input echoes."""

ACON_OBS_PROMPT_BASE = """Your task is to generate a "Reasoning" and a "Refined Observation" based on the inputs below.

In the "Reasoning", analyze the user instruction and history to identify what information from the current observation is necessary to complete the remaining steps.
Think about what parts can be summarized or transformed to reduce length, while ensuring that future actions can still be executed based on the refined observation alone.

In the "Refined Observation", include only the information that is minimal but sufficient for the next steps.

[Information source]
# User Instruction
{task}

# History of interactions
{history}

# Observation at the current time step
{observation}

[Output format]
# Reasoning
... your reasoning for what matters and how to optimize it ...
# Refined Observation
... reduced and actionable observation ..."""

ACON_OBS_PROMPT_UT = """Your task: write two sections-"Reasoning" and "Refined Observation".

1. Reasoning
- Examine task, history, and observation.
- Decide exactly which parts of the observation must be kept so the next agent step can succeed.
- Note any need to paginate (page_limit default = 5, page_index).
- Justify any data you drop.

2. Refined Observation
- Contain only the minimal yet sufficient info for the next step.
- Always preserve:
  - Every endpoint that may be called, plus its full parameter list and defaults (especially page_limit/page_index, auth tokens).
  - Response-schema fields referenced or likely needed later (e.g., play_count, release_date, like_count, position, ids).
  - Raw data rows required for future comparisons or loops; if summarising, keep at least all positive-match examples.
- Never:
  - Omit defaults that affect behaviour.
  - Declare parameters "not critical" without proof.
  - Hallucinate endpoints or fields.
  - Replace machine-readable data with vague prose.

[Information source]
# User Instruction
{task}
# History of interactions
{history}
# Observation at the current time step
{observation}

[Output format]
# Reasoning
...concise analysis explaining what is kept/removed...
# Refined Observation
...trimmed yet complete observation ensuring future steps remain possible..."""

ACON_OBS_PROMPT_UT_CO = """Your task: create two sections-"Reasoning" and "Refined Observation".

1. Reasoning (<=40 words)
- Briefly state what was kept and why; note dropped categories and any pagination needs.

2. Refined Observation (use ONLY the current observation)
- Keep strictly necessary data for the next step.
- Always preserve, when present:
  - access_token or other auth values
  - page_limit, page_index and other defaults that affect calls
  - every endpoint name; include parameters only if required; description <= 4 words
  - identifiers/fields needed for comparisons, loops, or API calls (ids, titles, counts, paths, etc.)
- Minimise length:
  - Delete unused fields and narrative text.
  - Compress JSON/arrays (one object per line, no extra spaces).
  - Summarise long uniform lists with a range/pattern when individual rows are not needed.
- Never invent, alter, or omit a required literal.
- Exclude history, prior summaries, and meta comments.

[Information source]
# User Instruction
{task}
# History of interactions
{history}
# Observation at the current time step
{observation}

[Output format]
# Reasoning
...
# Refined Observation
..."""

ACON_GUIDELINES = ("base", "ut", "ut_co")
ACON_HISTORY_PROMPTS = {
    "base": ACON_HISTORY_PROMPT_BASE,
    "ut": ACON_HISTORY_PROMPT_UT,
    "ut_co": ACON_HISTORY_PROMPT_UT_CO,
}
ACON_OBS_PROMPTS = {
    "base": ACON_OBS_PROMPT_BASE,
    "ut": ACON_OBS_PROMPT_UT,
    "ut_co": ACON_OBS_PROMPT_UT_CO,
}
# Compatibility aliases for callers that inspect the original base prompt.
ACON_HISTORY_PROMPT = ACON_HISTORY_PROMPT_BASE
ACON_OBS_PROMPT = ACON_OBS_PROMPT_BASE

# paper §8.3 (ruling 4): T_obs = 1024 tok; history = 4096 tok; chars = tok*4
ACON_OBS_THRESHOLD_CHARS = 1024 * 4
ACON_HISTORY_THRESHOLD_CHARS = 4096 * 4
# one preserved action/observation PAIR = 2 messages
ACON_PRESERVE_LAST_K_MESSAGES = 2
# the original feeds the full history to the obs compressor; we window it
# (documented approximation)
ACON_OBS_HISTORY_MESSAGES = 12
ACON_OBS_HISTORY_CHARS = 8000


def _task_text(messages: List[Dict[str, Any]]) -> str:
    for m in messages:
        if m.get("role") == "user":
            return _content_of(m)
    return "(no user instruction)"


def _history_summary_body(summary: str) -> str:
    """Normalize optimized ACON prompts that emit their own XML wrapper."""
    start, end = "<HISTORY_SUMMARY>", "</HISTORY_SUMMARY>"
    if start in summary and end in summary:
        return summary.split(start, 1)[1].split(end, 1)[0].strip()
    return summary.strip()


def _acompress_obs(observation: str, task: str, history: str, compress,
                   cache_key: str, stats: Dict[str, Any], model: str,
                   guideline: str) -> str:
    with _LOCK:
        refined = _SUMMARY_CACHE.get(cache_key)
    if refined is not None:
        return refined
    out = compress(compressor_payload(
        "acon", model, ACON_SYSTEM,
        ACON_OBS_PROMPTS[guideline].format(
            task=task, history=history, observation=observation)))
    marker = "# Refined Observation"
    if marker in out:
        out = out.split(marker, 1)[1].strip()
    refined = out.strip()
    if not refined:
        raise TextarmCompressorError(
            f"acon obs compressor returned empty (key={cache_key[:12]})")
    with _LOCK:
        refined = _SUMMARY_CACHE.setdefault(cache_key, refined)
    stats["n_compressor_calls"] = stats.get("n_compressor_calls", 0) + 1
    return refined


def acon_transform(messages: List[Dict[str, Any]], compress: Compress,
                   action_dialect, conv: str, mode: str = "both",
                   model: str = "c2kv-agent", guideline: str = "base",
                   ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Apply one fixed ACON compression guideline. ``mode='obs'`` refines oversized tool
    observations in place; 'hist' replaces the covered prefix with the
    rolling structured summary embedded in the first user prompt
    (<HISTORY_SUMMARY> block, the original memory.py:481-498 shape);
    'both' applies obs first then hist, as the original pipeline does.

    ``guideline`` identifies the paper's actual stages: ``base`` (official
    repository initial prompt), ``ut`` (utility optimization), or ``ut_co``
    (utility followed by compression optimization).
    """
    assert mode in ("obs", "hist", "both")
    if guideline not in ACON_GUIDELINES:
        raise ValueError(f"unknown ACON guideline {guideline!r}")
    stats: Dict[str, Any] = {"policy": "acon", "mode": mode,
                             "guideline": guideline,
                             "n_compressor_calls": 0, "n_obs_compressed": 0}
    messages = [dict(m) for m in messages]
    task = _task_text(messages)
    raw_chars = _message_chars(messages)

    if mode in ("obs", "both"):
        for idx, m in enumerate(messages):
            if (m.get("role") or "") != "tool":
                continue
            obs = _content_of(m)
            if len(obs) <= ACON_OBS_THRESHOLD_CHARS:
                continue
            window = messages[max(0, idx - ACON_OBS_HISTORY_MESSAGES):idx]
            history = ("\n".join(_render_line(h, action_dialect) for h in window)
                       )[:ACON_OBS_HISTORY_CHARS]
            key = _sha(f"acon-obs|{guideline}|{task}|{history}|{obs}")
            m["content"] = _acompress_obs(obs, task, history, compress, key,
                                          stats, model, guideline)
            stats["n_obs_compressed"] += 1

    out: List[Dict[str, Any]] = list(messages)
    stats["history_compressed"] = False
    if mode in ("hist", "both"):
        system_msgs = [m for m in out if m.get("role") == "system"]
        nonsystem = [m for m in out if m.get("role") != "system"]
        k = ACON_PRESERVE_LAST_K_MESSAGES
        prefix, tail = ((nonsystem[:-k], nonsystem[-k:])
                        if len(nonsystem) > k else ([], nonsystem))
        # token estimates INCLUDE tool_calls (audit 2026-09-05: counting
        # content only left assistant tool-call turns — where tau2/BFCL
        # keep every action — invisible to the trigger; the arm then never
        # fired and silently became a full arm)
        prefix_chars = _message_chars(prefix)
        stats["prefix_est_tokens"] = prefix_chars // 4

        # FAITHFUL upstream trigger (audit 2026-09-05): tiktoken(
        # prev_summary + messages since the last compression) > T_hist.
        # The FIRST compression measures the raw history alone.  Between
        # compressions the policy SEES the summary block + EVERY new raw
        # turn + the preserved pair — the raw recent context is never
        # dropped turn-by-turn (the old port did exactly that, wrongly).
        def _nonsystem_digest(until: int) -> str:
            return _sha(json.dumps(
                [{"role": m.get("role"), "content": _content_of(m),
                  "tool_calls": m.get("tool_calls") or []}
                 for m in nonsystem[:until]],
                ensure_ascii=False, sort_keys=True))

        with _LOCK:
            state_key = (conv, mode, guideline)
            state = _ACON_STATE.get(state_key)
        if (state is None or state[0] > len(prefix)
                or state[1] != _nonsystem_digest(state[0])):
            # fresh conversation, benchmark rewrote history, or a
            # conv-id collision (different content under the same id)
            covered_until, prev_summary = 0, ""
        else:
            covered_until, prev_summary = state[0], state[2]

        new_msgs = nonsystem[covered_until:len(prefix)]
        new_chars = _message_chars(new_msgs)
        trigger_chars = new_chars if not prev_summary else (
            len(prev_summary) + new_chars)

        if prefix and trigger_chars > ACON_HISTORY_THRESHOLD_CHARS:
            history_text = "\n".join(_render_line(m, action_dialect)
                                     for m in new_msgs)
            user_text = ACON_HISTORY_PROMPTS[guideline].format(
                task=task, prev_summary=prev_summary or "(none)",
                history=history_text)
            # content-addressed cache key: the positional key
            # (conv|covered|len) let a DIFFERENT task's same-length
            # segment hit the wrong cached summary under a shared id
            summary = _history_summary_body(_summarize(
                _sha(f"acon-hist|{guideline}|{user_text}"),
                "acon", model, ACON_SYSTEM, user_text,
                compress, stats))
            with _LOCK:
                _ACON_STATE[state_key] = (
                    len(prefix), _nonsystem_digest(len(prefix)), summary)
            stats["new_raw_messages_folded"] = len(new_msgs)
            block = (f"\n<HISTORY_SUMMARY>\n{summary}\n</HISTORY_SUMMARY>")
            # compressed view: keep system messages and the first user
            # prompt (the task instruction) carrying the block — the
            # memory.py:481-498 shape — plus the preserved tail; the
            # folded prefix messages are dropped (that IS the compression)
            placed = False
            rebuilt = list(system_msgs)
            for m in prefix:
                if (not placed and m.get("role") == "user"
                        and not m.get("tool_calls")):
                    rebuilt.append({"role": "user",
                                    "content": _content_of(m) + block})
                    placed = True
            for m in tail:
                rebuilt.append(dict(m))
            if not placed:  # no user instruction anywhere (should not happen)
                rebuilt.insert(len(system_msgs),
                               {"role": "user", "content": block.strip()})
            out = rebuilt
            stats["history_compressed"] = True
        elif prev_summary:
            # NOT triggered but a summary exists: the policy sees the
            # previous summary block + EVERY new raw turn since the last
            # compression + the preserved pair (upstream semantics — the
            # raw recent context is only folded at the next compression)
            block = (f"\n<HISTORY_SUMMARY>\n{prev_summary}\n</HISTORY_SUMMARY>")
            rebuilt = list(system_msgs)
            placed = False
            for m in nonsystem[:covered_until]:
                if (not placed and m.get("role") == "user"
                        and not m.get("tool_calls")):
                    rebuilt.append({"role": "user",
                                    "content": _content_of(m) + block})
                    placed = True
            if not placed:
                rebuilt.append({"role": "user", "content": block.strip()})
            rebuilt.extend(dict(m) for m in new_msgs)
            rebuilt.extend(dict(m) for m in tail)
            out = rebuilt
            stats["history_compressed"] = False
            stats["new_raw_messages_visible"] = len(new_msgs)
        # else: never compressed and under threshold — plain passthrough

    out_chars = _message_chars(out)
    stats["raw_chars"] = raw_chars
    stats["out_chars"] = out_chars
    stats["raw_est_tokens"] = _est_tokens(raw_chars)
    stats["out_est_tokens"] = _est_tokens(out_chars)
    return out, stats


def reset_state() -> None:
    """Test isolation hook (also used by proxy startup)."""
    with _LOCK:
        _SUMMARY_CACHE.clear()
        _ACON_STATE.clear()
