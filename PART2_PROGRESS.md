# Part 2 (Databricks Deployment) — Progress Notes

Status as of this session: **the endpoint is deployed and READY, and works
correctly end-to-end.** Notebook polish for Task 2.4 is the only thing left.

## Current live state (verify before resuming — this can drift)

- Registered model: `cs4603.default.document_analyst`, **version 6** — bumped
  from version 5 (the version current when this file's debugging narrative
  below was written) by the Bonus A GitHub Actions pipeline
  (`.github/workflows/deploy.yml`, run
  [29639702020](https://github.com/SJK159/cs4603-pa4/actions/runs/29639702020)),
  triggered manually via `workflow_dispatch` to verify the CI/CD deploy job
  end-to-end. Same code as version 5 (no `agent`/`rag`/`tools` changes since),
  so the version bump is purely from re-running `deploy.py` through CI.
- Serving endpoint: `27100159-document-analyst` — confirmed `READY` after the
  CI-triggered deploy, and re-verified directly against `DocumentAnalystClient`
  (`health_check()` → `True`, `ask()` → correct answer).
- Verified with a real curl call and a real OpenAI-SDK call in the earlier
  (version 5) session — both returned the correct final answer, matching
  local `graph.invoke()` output. The 3 debugging fixes below are unaffected
  by the version bump; they're baked into the code, not the artifact version.

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

## What's left to do tomorrow (as of the session this file was written)

All items below are now **done**, in a later session:

1. ~~Finish executing `pa4.ipynb` end-to-end.~~ Done — the rate limit was hit
   again on a second attempt, but retrying with `--allow-errors` (and not
   masking the exit code through a `| tail` pipe) got a clean run: all 19
   code cells across Task 1.7 / 2.4 / 3.2 executed with zero errors against
   the live workspace.
2. ~~Sanity-check the endpoint is still READY.~~ Done, repeatedly — most
   recently re-verified after the Bonus A CI deploy bumped it to version 6
   (see "Current live state" above).
3. Part 2 write-up in `Analysis.md` — unchanged, still complete.
4. ~~Everything above is uncommitted.~~ Done — committed (`e9746c0`) and
   pushed to `origin/main`, including `deployment/stub_wheels/*.whl`.
5. ~~Part 3 (Client SDK) not started yet.~~ Done — `client/sdk.py` fully
   implemented, unit-tested against a mocked transport, and demonstrated live
   in `pa4.ipynb` Task 3.2. Bonus A (CI/CD pipeline) was also implemented and
   verified with a real GitHub Actions run
   ([29639702020](https://github.com/SJK159/cs4603-pa4/actions/runs/29639702020)).
   Bonus B/C remain unstarted.
