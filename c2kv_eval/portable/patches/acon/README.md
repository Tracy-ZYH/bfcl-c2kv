# ACON deployment patches

Tested source state: the pristine tree at
`microsoft/acon@d63f9ae18959dc7215ff62899c94c5e8c56847ae` (`add: dummy private
config`). The local verification checkout had
`https://github.com/microsoft/acon.git` configured as `origin`. This is a
source-state pin; it does not claim that this commit is still the current
upstream tip.

From a pristine ACON checkout at that commit, both the dry checks and the
sequential applications below pass. No whitespace-relaxation option is
required for this exact tree. A repeated check against an already patched
runtime tree is expected to fail, so patch tests such as those using
`ACON_ROOT` must point at pristine source.

```bash
git apply --check /path/to/bfcl-c2kv/c2kv_eval/portable/patches/acon/0001-openai-base-url-env.patch
git apply --check /path/to/bfcl-c2kv/c2kv_eval/portable/patches/acon/0002-unknown-api-cost.patch
git apply --check /path/to/bfcl-c2kv/c2kv_eval/portable/patches/acon/0003-bm25-lazy-imports.patch
git apply --check /path/to/bfcl-c2kv/c2kv_eval/portable/patches/acon/0005-smolagents-error-feedback.patch

git apply /path/to/bfcl-c2kv/c2kv_eval/portable/patches/acon/0001-openai-base-url-env.patch
git apply /path/to/bfcl-c2kv/c2kv_eval/portable/patches/acon/0002-unknown-api-cost.patch
git apply /path/to/bfcl-c2kv/c2kv_eval/portable/patches/acon/0003-bm25-lazy-imports.patch
git apply /path/to/bfcl-c2kv/c2kv_eval/portable/patches/acon/0005-smolagents-error-feedback.patch
```

`0001-openai-base-url-env.patch` makes the ACON runners reproducible for the
bench: `productive_agents.llm.vLLM` (the client every non-OpenAI model name
falls to in `agents/utils.py:LLMManager.create_llm`) hard-codes
`http://localhost:8000/v1` / `token-abc`; the patch makes it read
`ACON_OPENAI_BASE_URL` / `ACON_OPENAI_API_KEY` (same defaults when unset).
`c2kv_eval/portable/adapters/acon_adapter.py` exports the arm proxy there. Nothing
else in the runners changes: prompts, memory, decoding options
(`presence_penalty 0.5`, `enable_thinking false`, seed 42) stay ACON's.

`0002-unknown-api-cost.patch` makes unknown model tariffs return
JSON `null`; AppWorld prints that API cost is unavailable and still records
token counts. This fixes the upstream missing `gpt-4o` fallback without
assigning a hosted-model price to local inference.

`0003-bm25-lazy-imports.patch` must be applied before launching the 8-objective QA
retriever. The shipped server eagerly imports its optional dense-retrieval
stack even when `retrieval_method="bm25"`. The patch defers those imports to
the dense and external-corpus paths, so a stored-document Lucene index needs
only Pyserini and the FastAPI server dependencies. Dense retrieval behavior is
unchanged when its optional dependencies are installed.

Pyserini `0.44.0` also exports its impact and HNSW searchers eagerly from
`pyserini.search.lucene`, which imports the neural encoder stack before the
BM25 `LuceneSearcher` can be used. This patch targets Pyserini rather than the
ACON repository. The verification machine did not have Pyserini installed;
an offline fixture reproducing the `0.44.0` export block passed
`git apply --check` and application. Verify and apply it from the virtual
environment's `site-packages` directory:

```bash
git apply --check /path/to/bfcl-c2kv/c2kv_eval/portable/patches/acon/0004-pyserini-sparse-imports.patch
git apply /path/to/bfcl-c2kv/c2kv_eval/portable/patches/acon/0004-pyserini-sparse-imports.patch
```

The expected preimage contains the three adjacent exports
`_impact_searcher`, `_searcher`, and `_hnsw_searcher` in
`pyserini/search/lucene/__init__.py`; the patch keeps `_searcher` and removes
the other two. This task-local sparse installation deliberately stops
exporting those two optional searcher
families; its BM25 API and Lucene implementation are unchanged.

The Smolagents action processor already extracts fenced Python before
`SmolagentsEnv.step` executes it. If that execution raises, upstream returns
the error but does not store it in `env.observation` or `env.trajectory`.
Consequently the next prompt is rebuilt as an initial turn and the same task
is sent again. `0005-smolagents-error-feedback.patch` records the failed action and its actual executor
error so the next model turn receives that feedback; it does not rewrite the
action, supply missing tool arguments, or change the task prompt.

Both runners then need only the served model name to NOT contain `gpt`,
`o1`, `o3`, `o4` or `gemini` (those names are routed to the OpenAI / Gemini
clients instead).  `c2kv-agent` is fine.

Runner prerequisites (see the upstream READMEs under `experiments/`):

* 8-objective QA — `pip install smolagents`; the Search-R1 wiki-18 BM25 index
  (`PeterJinGo/wiki-18-bm25-index`, `PeterJinGo/wiki-18-corpus`) under
  `experiments/smolagents/search/database/wikipedia`; the retriever server
  `python search/retriever_server.py --index_path search/database/wikipedia/bm25`
  running before the run.  Data = the shipped `data/nq_multi_8` (100/100).
* AppWorld — install `appworld` (validated with `0.1.3.post1`), run
  `appworld install` and `appworld download data`. Set `APPWORLD_ROOT` to
  the directory containing `data/`, or place it in `experiments/appworld/`.
  The adapter creates a private harness per run, links immutable data, and
  gives generation and official scoring the same selected task list.
  The official scorer is the `appworld` CLI beside the runner Python.

ACON's own compression arm (`--co_config_path`) is NOT wired: with
`model_type: local` its compressor is an in-process vLLM, not the served
endpoint, and the bench's ruling is one served 4B for policy and compressor.
The bench text arm for ACON stays `--arm acon_hist` / `acon_obs` in
`c2kv_eval/portable/textarms.py`.
