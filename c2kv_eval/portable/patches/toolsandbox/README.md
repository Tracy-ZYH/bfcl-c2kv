# ToolSandbox deployment patches

Tested source state: the local shallow imported tree at
`165848b9a78cead7ca7fe7c89c688b58e6501219` (`init`). Its configured `origin`
is `https://github.com/apple/ToolSandbox.git`, but the commit is a root commit
with no parent in this checkout. Local evidence therefore does not establish
that this hash is a published Apple upstream commit; treat the imported tree's
content as the pin.

The patch context is independently verifiable from that tree:

```text
1d1e8d001f50ea1225885efebda140d07837e927  tool_sandbox/roles/openai_api_agent.py
d82e15c540dced179db59c582bcbad1179d68a60  tool_sandbox/roles/openai_api_user.py
```

These are the `HEAD:<path>` blob IDs before patching. Both files explicitly
construct the OpenAI client with the public OpenAI base URL, and both use
`tool_calls is None` before `0002` is applied.

`0001-openai-base-url-env.patch` makes the repo reproducible for the bench
(audit BLOCKER: the TS column previously depended on a server-side edit
that existed nowhere in this repository):

* `openai_api_agent.py` — the agent client reads `OPENAI_BASE_URL`
  (upstream hard-codes `https://api.openai.com/v1` and a comment saying it
  deliberately IGNORES the env var, so a vanilla clone can produce no TS
  numbers for any arm).
* `openai_api_user.py` — the USER SIMULATOR reads its own
  `TOOLSANDBOX_USER_BASE_URL` (falling back to `OPENAI_BASE_URL` for
  standalone use).  Both roles reading the same variable routed the
  simulator through the arm proxy, making every historical TS number an
  agent+user joint degradation. `c2kv_eval/portable/run.py` exports the raw
  upstream endpoint here (`--user-upstream`).

From a pristine copy of the tested imported tree, all patches pass plain
`git apply --check` and sequential application; no whitespace-relaxation
option is required.

```bash
git apply --check /path/to/bfcl-c2kv/c2kv_eval/portable/patches/toolsandbox/0001-openai-base-url-env.patch
git apply --check /path/to/bfcl-c2kv/c2kv_eval/portable/patches/toolsandbox/0002-empty-tool-calls.patch

git apply /path/to/bfcl-c2kv/c2kv_eval/portable/patches/toolsandbox/0001-openai-base-url-env.patch
git apply /path/to/bfcl-c2kv/c2kv_eval/portable/patches/toolsandbox/0002-empty-tool-calls.patch
git apply /path/to/bfcl-c2kv/c2kv_eval/portable/patches/toolsandbox/0003-text-end-conversation.patch
git apply /path/to/bfcl-c2kv/c2kv_eval/portable/patches/toolsandbox/0004-qwen-tool-result-name.patch
```

Apply `0002-empty-tool-calls.patch` as well. OpenAI-compatible servers may
return `tool_calls: []` for a normal text reply. Both roles must treat that
like `null`: upstream otherwise appends no messages, so its message-count
limit never advances and the scenario loops indefinitely. The patch changes
only this response-shape normalization; prompts and scoring remain unchanged.

Apply `0003-text-end-conversation.patch` for OpenAI-compatible servers that
occasionally render the user-only stop tool as the literal text
`end_conversation` instead of a structured tool call. The normalization is
deliberately limited to the exact tool name (with optional `()`); arbitrary
natural-language stop requests are not rewritten.

Apply `0004-qwen-tool-result-name.patch` when serving Qwen with its bundled
chat template, and set `TOOLSANDBOX_QWEN_TOOL_RESULT_COMPAT=1`.  That
template renders a tool message's content but ignores its OpenAI `name`
field.  The compatibility layer therefore prefixes the content with the
function name, preventing opaque values such as a successful messaging UUID
from being misread as a contact/person ID.  It changes only the model-facing
serialization; ToolSandbox execution, state and official scoring are not
modified.  The behavior is opt-in and is disabled for other model families.

TS test-mode's "n=3" is ONE base scenario (`send_message_with_contact_
content_cellular_off`) plus two perturbations (distractor tools / scrambled
arg descriptions) — not three independent tasks; only the full suite is a
real column (audit finding).
