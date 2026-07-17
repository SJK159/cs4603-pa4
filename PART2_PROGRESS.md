# Part 2 (Databricks Deployment) — Progress Notes

Status as of this session: **the endpoint is deployed and READY, and works
correctly end-to-end.** Notebook polish for Task 2.4 is the only thing left.

## Current live state (verify before resuming — this can drift)

- Registered model: `cs4603.default.document_analyst`, **version 5**
- Serving endpoint: `27100159-document-analyst` — confirmed `READY` at the end
  of this session:
  ```bash
  databricks serving-endpoints get 27100159-document-analyst
  # state: {"config_update": "NOT_UPDATING", "ready": "READY"}
  # served_entities[0]: entity_version "5", deployment "DEPLOYMENT_READY"
  ```
- Verified with a real curl call and a real OpenAI-SDK call — both returned
  the correct final answer, matching local `graph.invoke()` output.

## What was broken and fixed (3 separate real bugs, in order encountered)

Getting from "endpoint stuck in `DEPLOYMENT_FAILED`" to "READY" took three
independent fixes. All three are already applied in the repo.

### 1. Unpinned / partially-pinned `pip_requirements` (deployment/deploy.py)

- **Symptom:** container build failed after ~2.4 hours with pip backtracking
  through hundreds of ancient `regex` releases (transitive dep of `tiktoken`
  via `langchain-openai`), which can't build under the container's Python
  3.13 (`No unicodedata_db.h could be prepared`).
- Tried pinning only the 11 direct dependencies — still failed with
  `resolution-too-deep` (dozens of transitive deps still unconstrained).
- **Fix:** `PIP_REQUIREMENTS` in `deploy.py` is now a fully-pinned, flattened
  list of **all ~153 transitive packages** (generated via `uv pip freeze`
  from a clean venv containing only the direct deps), so pip does zero
  solving in the container.

### 2. `databricks-connect` version deadlock (deployment/stub_wheels/)

- **Symptom:** even fully pinned, `databricks-connect==17.0.10` isn't on this
  workspace's serving-container package mirror (which tops out at 16.1.7).
- **Root cause (genuine upstream conflict, not fixable by re-pinning):**
  `databricks-langchain` unconditionally requires
  `unitycatalog-langchain[databricks]`, which unconditionally requires
  `databricks-connect<17.1,>=15.1.0` (for Spark-Connect-backed UC
  function-calling tools this agent never uses — we only use
  `databricks_langchain.DatabricksVectorSearch`). Every `databricks-connect`
  version in that range actually available on the mirror (<=16.1.7) pins
  `numpy<2`, but `langchain-community>=0.4` (pulled in unconditionally by
  `unitycatalog-langchain>=0.3.0`) requires `numpy>=2.1`. No version
  combination satisfies both — verified by trying to downgrade the whole
  stack, which resolves but lands on a `langchain-core` too old for
  `langchain-mcp-adapters` (breaks MCP tool loading — worse).
- **Fix:** `deployment/stub_wheels/databricks_connect-17.0.10-py3-none-any.whl`
  is a hand-built, dependency-free stub wheel (just `dist-info/METADATA` +
  `WHEEL` + `RECORD`, no actual `databricks/connect/` package content) that
  satisfies the version constraint without requiring numpy at all.
  `deploy.py` ships it via `code_paths` and points pip at it with
  `--find-links=/model/code/stub_wheels` (the first entry in
  `PIP_REQUIREMENTS`). Verified safe: `unitycatalog/ai/core/databricks.py`
  only imports `databricks.connect.session` lazily inside
  `initialize_spark_session()`, which we never call — confirmed by running
  the full graph (retrieval + MCP tool call) locally with the real package
  not installed at all, and again with only the stub installed via real
  `pip` (not `uv`, to match the container).

### 3. `sys.stderr` has no `fileno()` in the serving container (agent/graph.py)

- **Symptom:** endpoint reached `DEPLOYMENT_FAILED` again with a *different*
  error after fixes 1+2: `"A library raised an error during model load"` →
  `AttributeError: 'StreamToLogger' object has no attribute 'fileno'`.
- **Root cause:** `mcp.client.stdio.stdio_client`'s `errlog` parameter
  defaults to `sys.stderr`, evaluated **once, when that module is first
  imported** — not at call time. MLflow replaces `sys.stderr` with a
  `StreamToLogger` shim (no real file descriptor) before loading the model.
  `mcp.client.stdio` gets imported transitively the first time anything
  touches MCP — which turned out to be **`rag.store.get_vector_store()`**
  (`from databricks_langchain import DatabricksVectorSearch` →
  `databricks_langchain/__init__.py` imports its own MCP client wrapper),
  not `load_mcp_tools()` as originally assumed, since `get_retriever()` runs
  first in `build_graph()`. Whichever import happens first bakes the shim in
  permanently (Python caches imports), and asyncio's subprocess machinery
  later calls `.fileno()` on it when launching the MCP server subprocess.
- **Fix:** `agent/graph.py` now has `_real_stderr_for_first_mcp_import()`, a
  context manager wrapping the **entire dependency-construction section** of
  `build_graph()` (llm + retriever + tools, in that order) so it doesn't
  matter which one triggers the first MCP-related import. Verified locally
  by monkey-patching `sys.stderr` to a fileno-less object matching the
  container's shim and re-running the full graph end-to-end (including an
  actual MCP tool call) — works.

## Task 2.4 nuance (documented, not a bug — see Analysis.md)

Calling the deployed endpoint via the OpenAI SDK
(`client.chat.completions.create(...)`) succeeds at the HTTP level (hits
Databricks' `/serving-endpoints/chat/completions` gateway correctly), but the
response is **not** auto-wrapped into a standard `ChatCompletion` object with
populated `.choices` — because our `AnalystState` has extra fields beyond
`messages` (`plan`, `step_results`, `current_step_index`, `next_agent`,
`final_answer` — required by Task 1.1), Databricks' gateway doesn't recognize
the shape as a pure chat completion. The SDK returns a `list` containing one
object with `choices=None` but our own fields exposed as extra attributes.
Correct parsing: `resp[0].final_answer` or `resp[0].messages[-1]["content"]`
— **not** `resp.choices[0].message.content`. The README's Task 2.4 spec only
says "show the parsed response," not that the literal `.choices` shape must
resolve — the `DEPLOYMENT_GUIDE.md` correction about `.choices[0].message.content`
was written for the simpler wk5/15 reference agent (plain `MessagesState`,
no extra fields), not PA4's deliberately richer state. `pa4.ipynb`'s Task 2.4
cells already use the correct parsing pattern with an explanatory markdown
cell.

## What's left to do tomorrow

1. **Finish executing `pa4.ipynb` end-to-end.** Last run hit a transient
   `429 RateLimitError` (`REQUEST_LIMIT_EXCEEDED` on
   `databricks-meta-llama-3-3-70b-instruct`) from too many back-to-back test
   calls during this debugging session — not a real bug. Retry:
   ```bash
   cd /Users/saadjamshaidkhan/LUMS/MLops/PA4/cs4603-pa4
   uv run --with nbconvert jupyter nbconvert --to notebook --execute --inplace pa4.ipynb
   ```
   If it rate-limits again, wait a few minutes between attempts (or reduce
   how many test calls run back-to-back).
2. **Sanity-check the endpoint is still READY** before relying on it (state
   can change): `databricks serving-endpoints get 27100159-document-analyst`.
3. Part 2 write-up in `Analysis.md` is already filled in (Deployment section,
   Task 2.1 + Task 2.3 analysis questions, including the debugging story).
   Nothing else needed there for Part 2.
4. Everything above is uncommitted (`git status` shows all the modified/new
   files). Not committed yet — commit when ready, including
   `deployment/stub_wheels/*.whl` (not excluded by `.gitignore` — verified).
5. Part 1 was already complete going into this session (all tests pass,
   `pa4.ipynb` Task 1.7 cells done). Part 3 (Client SDK) not started yet.
