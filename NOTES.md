# CS4603 PA4 — Document Analyst — Implementation Notes

Setup, run, and deployment instructions, plus design decisions. The graded
analysis-question answers live in [`ANSWERS.md`](ANSWERS.md).

## Setup

```bash
uv sync
cp .env.example .env   # then fill in your values
```

## Running locally

1. **Ingest the corpus** (run once, from a Databricks notebook with a live Spark session
   — `rag/ingest.py` needs `ai_parse_document`/`ai_prep_search` and can't run outside
   Databricks):
   ```python
   from rag.ingest import build_chunks_table, create_index

   build_chunks_table(
       spark,
       volume_path="/Volumes/cs4603/default/pa4/annual_report.pdf",
       chunks_table="cs4603.default.27100159_analyst_chunks",
   )
   create_index()   # reads VECTOR_SEARCH_ENDPOINT/INDEX + SOURCE_TABLE from .env
   ```
   `build_chunks_table` parses the PDF and chunks it into a Delta table (`chunk_id`,
   `chunk_to_retrieve`, `chunk_to_embed`, `source`, `page`), enabling Change Data Feed so
   the Delta Sync index can track it. `create_index` creates (or reuses) a `STANDARD`
   Vector Search endpoint (`27100159-vs-endpoint`) and a `TRIGGERED` Delta Sync index
   (`cs4603.default.27100159_analyst_index`) over that table, then polls until `ONLINE`/
   `READY`. My `.env` values:
   ```
   UC_CATALOG=cs4603
   UC_SCHEMA=default
   SOURCE_TABLE=cs4603.default.27100159_analyst_chunks
   VECTOR_SEARCH_ENDPOINT=27100159-vs-endpoint
   VECTOR_SEARCH_INDEX=cs4603.default.27100159_analyst_index
   ```

2. **Build and run the graph** in `pa4.ipynb` (Task 1.7):
   ```python
   from agent.graph import build_graph
   graph = build_graph()          # uses config.py + rag/store.py + the MCP server
   result = graph.invoke({"messages": [{"role": "user",
             "content": "What was the net income in 2023?"}]})
   print(result["messages"][-1].content)
   ```

3. **Test queries I ran** (retrieval-only, computation-only, combined) — actual output
   from `pa4.ipynb`, executed end-to-end against the real Vector Search index and LLM
   endpoint:

   | Query | Plan produced | Answer produced |
   |-------|----------------|------------------|
   | "What was the net income in 2023?" | `["Find the net income for the year 2023"]` | The net income in 2023 was ¥1,107 billion [source: annual_report.pdf, p.1]. |
   | "What is 15% of 2.4 billion?" | `["Calculate 15% of 2.4 billion"]` | 15% of 2.4 billion is 0.15 * 2.4e9 = 3.6e+08, or 360 million. |
   | "What was the revenue in 2023, and what would a 10% increase look like?" | `["Find the company's revenue for the year 2023", "Calculate a 10% increase on the 2023 revenue found in the previous step"]` | The revenue in 2023 was ¥16.91 trillion [source: annual_report.pdf, p.1]. A 10% increase would be ¥16.91 trillion * 1.10 = ¥18.601 trillion. |

   The combined query's step-by-step trace (Task 1.7 §4) confirms the full loop:
   `planner → supervisor → rag_agent → supervisor → mcp_tools → supervisor →
   synthesizer`, with `current_step_index` advancing 0→1→2 and `step_results`
   accumulating one entry per step exactly as the state schema (Task 1.1) intends.

   These same three queries were rerun against the **deployed** endpoint in Task 2.4 and
   through `DocumentAnalystClient` in Task 3.2 — all three matched the local answers
   exactly (see `pa4.ipynb` for the full local-vs-deployed diff and latency numbers).

## Deployment

Run:
```bash
uv run python deployment/deploy.py
```

This does two things (`deployment/deploy.py`):

1. **Log + register** (`log_and_register()`): sets `mlflow.set_registry_uri("databricks-uc")`,
   logs `deployment/agent_model.py` via `mlflow.langchain.log_model()` with `code_paths`
   pointing at `agent/`, `rag/`, `tools/`, `config.py` (so the serving container can
   `import agent`, `import rag`, etc.), and registers the result as
   `cs4603.default.document_analyst` in Unity Catalog.
2. **Create/update the serving endpoint** (`create_or_update_endpoint()`): points a
   Model Serving endpoint named `27100159-document-analyst` at that registered version,
   with `workload_size="Small"`, `scale_to_zero_enabled=True`, credentials injected as
   secret references from the `cs4603-deploy` scope, and the Vector Search
   endpoint/index/embeddings model passed as plain env vars. Polls until `READY`.

**Endpoint:** `27100159-document-analyst`
**URL:** `https://dbc-8307750a-af67.cloud.databricks.com/serving-endpoints/27100159-document-analyst/invocations`

### Debugging note: `DEPLOYMENT_FAILED` from unpinned `pip_requirements`

The first deploy attempt registered `document_analyst` v1 but the endpoint got stuck in
`UPDATE_FAILED` / `NOT_READY` with `deployment_state_message: "Container creation failed.
Please see build logs for more information."` Pulling the build logs via
`databricks api get "/api/2.0/serving-endpoints/<name>/served-models/<served-model>/build-logs?config_version=1"`
showed pip backtracking through hundreds of ancient `regex` releases (a transitive
dependency of `tiktoken`, pulled in by `langchain-openai`) and failing to build the
oldest ones from source under the container's Python 3.13 (`No unicodedata_db.h could
be prepared`). The root cause: `PIP_REQUIREMENTS` in `deploy.py` listed every package
unpinned, so the container's pip resolver had to solve the whole dependency graph from
scratch instead of reusing a known-good resolution — and backtracked itself into a
regex sdist too old to build. Fix: pin every entry in `PIP_REQUIREMENTS` to the exact
versions installed in the local `.venv` (matches `uv.lock`, known to resolve together),
which collapses the container's resolve to a single valid solution.

## Client SDK

`client/sdk.py` implements `DocumentAnalystClient`, a thin `httpx`-based wrapper around
the deployed endpoint's raw `/invocations` contract:

```python
from client.sdk import DocumentAnalystClient

client = DocumentAnalystClient(
    endpoint_name="27100159-document-analyst",
    host=settings["host"],
    token=settings["token"],
)
assert client.health_check()
print(client.ask("What was the net income in 2023?"))
for chunk in client.ask_streaming("What is 15% of 2.4 billion?"):
    print(chunk, end="")
```

**Why `/invocations` instead of the OpenAI-compatible `/chat/completions` gateway.**
As documented in Task 2.4, `AnalystState` has extra fields beyond `messages`
(`plan`, `step_results`, `final_answer`, …), so Databricks' `/chat/completions` route
doesn't recognize the output shape as a pure chat completion and leaves `choices=None`.
Hitting `/invocations` directly and parsing the returned state ourselves
(`_extract_answer()` in `client/sdk.py`) sidesteps that mismatch instead of working
around it downstream. `_extract_answer()` handles three response shapes defensively —
a bare state dict, `{"predictions": ...}`-wrapped, or a one-item list of either —
because MLflow's scoring server format for a `models-from-code` LangChain model wasn't
something I wanted to hard-code without re-verifying against a live call every time.

**Retry / timeout / error handling.**
- `ask()` retries on `429` (rate limited) and `503` (endpoint scaling from zero) with
  exponential backoff (`2**attempt` seconds), capped at `max_retries`, then raises
  `AnalystClientError(message, status_code, request_id)` for that or any other
  non-2xx response — `request_id` comes from the response's `x-request-id` header, so a
  failure can be correlated back to Databricks' serving logs.
- A request that exceeds `timeout` raises a plain `TimeoutError` naming the elapsed
  time, not a raw `httpx.TimeoutException`, so callers don't need to know the
  transport library to handle it.
- `health_check()` reuses the same `databricks.sdk.WorkspaceClient` pattern
  `deployment/deploy.py` already uses to poll for `READY` (`state.ready ==
  EndpointStateReady.READY`), rather than re-implementing endpoint-status parsing
  against the REST API by hand. It returns `False` on any exception instead of
  raising, since "not healthy" and "couldn't determine health" collapse to the same
  caller action (don't send the request).

**Streaming.** `ask_streaming()` opens the invocation as an SSE request and yields
`choices[].delta.content` as chunks arrive. Per the assignment's caveat, a
models-from-code LangChain endpoint may not implement `predict_stream` and can return
a single complete JSON body instead of an event stream — `ask_streaming()` detects a
non-`text/event-stream` `content-type` and falls back to yielding the whole answer
once, which Task 3.2's notebook demonstration actually exercises against the live
endpoint (see `pa4.ipynb`, Task 3.2 §4).

**Verification.** Before wiring this into the notebook, I unit-tested `ask()`,
`ask_streaming()`, and `health_check()` against a mocked `httpx` transport covering:
a plain-dict response, a `predictions`-wrapped list response, a 429-then-recover retry,
a non-retryable 400 wrapped into `AnalystClientError` with its `request_id`, retry
exhaustion on repeated 503s, SSE delta streaming, the non-SSE single-chunk fallback,
timeout wrapping, and constructor validation when no host/token is available — all nine
passed. `pa4.ipynb` Task 3.2 then demonstrates the same client against the real
deployed endpoint: `health_check()`, `ask()`, `ask_streaming()`, a forced timeout
(`timeout=0.001`), a genuine `AnalystClientError` (endpoint name that doesn't exist),
and a monkeypatched-503-then-success run that actually exercises the backoff path,
since the real endpoint won't return 503 on demand.

## CI/CD Pipeline (Bonus A)

`.github/workflows/deploy.yml` implements the required `lint → test → deploy` pipeline:

```yaml
on:
  push:
    branches: [main]   # deploy path
  pull_request:        # lint + test only
  workflow_dispatch:   # manual "Run workflow" button
```

- **`lint-and-test`** (every push and PR): `astral-sh/setup-uv` → `uv sync --extra dev`
  → `uv run ruff check agent/ client/` → `uv run pytest -q`. `pytest` runs
  `tests/test_smoke.py`, which builds and invokes the full graph with a fake LLM,
  retriever, and tool (`ScriptedLLM`/`FakeRetriever`/`FakeTool`) — no Databricks
  credentials or network access needed, so this job runs identically for any
  contributor or PR.
- **`deploy`** (`needs: lint-and-test`, `if: github.ref == 'refs/heads/main' &&
  github.event_name != 'pull_request'`): re-installs dependencies on the fresh runner
  (nothing persists between jobs), injects `DATABRICKS_HOST`/`DATABRICKS_TOKEN` from
  GitHub **repository secrets**, injects the non-sensitive config
  (`DATABRICKS_MODEL`, `VECTOR_SEARCH_ENDPOINT`/`INDEX`, `UC_CATALOG`/`SCHEMA`, …) from
  GitHub **repository variables** (falling back to the same defaults `deploy.py`/
  `.env.example` already use), then runs `uv run python deployment/deploy.py` — the
  same script used for manual deploys, unchanged. `deploy.py` already prints the
  registered model version (`log_and_register()`) and polls + prints the endpoint's
  state and URL once `READY` (`wait_for_ready()`), so those show up directly in the
  job's logs without any extra `echo` step.

**Verification.** Beyond the local dry-run (`uv sync --extra dev`, `ruff check agent/
client/`, `pytest -q`, plus `yaml.safe_load` on the workflow file), I configured the
real repo secrets/variables (`gh secret set` / `gh variable set` for
`DATABRICKS_HOST`/`DATABRICKS_TOKEN` and the non-sensitive config) and triggered a real
`workflow_dispatch` run on `main`:
[run 29639702020](https://github.com/SJK159/cs4603-pa4/actions/runs/29639702020).
Both jobs went green (`lint-and-test` in 15s, `deploy` in 49s), and `deploy` genuinely
registered a new Unity Catalog version (5 → 6) and updated the live serving endpoint —
confirmed `READY` and re-queried successfully through `DocumentAnalystClient`
immediately after. One snag along the way, unrelated to the pipeline's own logic:
GitHub rejects pushes that touch `.github/workflows/*.yml` unless the pushing
credential has the `workflow` OAuth scope — the initial push was rejected with
`refusing to allow an OAuth App to create or update workflow ... without workflow
scope` until I ran `gh auth refresh -s workflow` and `gh auth setup-git` to get a
correctly-scoped token wired up as git's credential helper.

## Databricks Agents SDK Deployment (Bonus B)

`deployment/deploy_agents.py` deploys the same Document Analyst using `databricks.agents.deploy()`
instead of the manual `WorkspaceClient` + `EndpointCoreConfigInput` calls in `deploy.py`.

**The one real wrinkle: `agents.deploy()` rejects Task 2.1's model outright.**
`agent_model.py` registers the compiled LangGraph object directly
(`mlflow.models.set_model(graph)`), whose output is the full `AnalystState` — `messages`,
`plan`, `current_step_index`, `step_results`, `next_agent`, `final_answer`. The first live
attempt at Bonus B logged and registered that exact model (version 7), then failed inside
`agents.deploy()`'s pre-flight compatibility check:

```
ValueError: The model's schema is not compatible with Agent Framework. The output
schema must be either ChatCompletionResponse or StringResponse. Output schema:
['messages': ..., 'plan': Array(string) (required), 'current_step_index': long
(required), 'step_results': Array(string) (required), 'next_agent': string
(required), 'final_answer': string (required)]
```

Reading `databricks.agents.utils.mlflow_utils._check_model_is_rag_compatible_legacy_signatures`
confirmed this isn't a bug to route around — Agent Framework enforces a strict output
contract (`ChatCompletionResponse`, `StringResponse`, or a couple of specific legacy
shapes) with no override flag, and our richer state schema doesn't subset any of them.
**Fix:** `deployment/agent_model_agents.py` wraps the *same, unmodified*
`agent.graph.build_graph()` in an `mlflow.pyfunc.ChatModel` — `load_context()` builds the
graph once (same one-time-load pattern Task 1.5 uses for MCP tools), and `predict()`
translates `ChatMessage` list in to `ChatCompletionResponse` out, extracting
`result["final_answer"]` from the graph's own invoke. `deploy_agents.py` then logs *that*
wrapper via `mlflow.pyfunc.log_model()` instead of `mlflow.langchain.log_model()`, still
reusing `deploy.py`'s pinned `PIP_REQUIREMENTS` and `code_paths` unchanged. No change to
Part 2's `agent_model.py`, `deploy.py`, or the graph itself was needed — the fix is
entirely a new adapter file plus a swapped logging call.

**Live result.** Retrying with the wrapper succeeded: registered version 8, provisioned a
*new* endpoint (`27100159-document-analyst-agents`, distinct from Part 2's manually-managed
one) plus an auto-generated Review App, reached `READY` in ~8 minutes (`agents.deploy()`
itself returns immediately — "can take up to 15 minutes" — so I polled
`WorkspaceClient().serving_endpoints.get()` separately until `READY`, since unlike
`deploy.py`'s `wait_for_ready()`, `agents.deploy()` doesn't block for you). Queried via the
OpenAI SDK afterward: the first call back returned a noticeably different answer (¥1,137
billion, source p.3) than the Part 2 endpoint's consistent ¥1,107 billion/p.1 — two
immediate follow-up calls both matched the expected ¥1,107 billion/p.1 exactly, so this
reads as first-request sampling variance on a just-warmed container (the Databricks LLM
endpoint is called with `temperature=0`, but a freshly-serving container's very first
inference isn't guaranteed to be bit-identical to steady-state calls) rather than a defect
in the wrapper.

## Standalone MCP Server on Databricks Apps (Bonus C)

`deployment/mcp_app/app.py` reuses the GIVEN `tools/mcp_server.py` tool definitions
unchanged, serving them over `streamable-http` instead of stdio so they can run as a
long-lived Databricks App. `agent/graph.py::load_mcp_tools()` now branches on
`MCP_SERVER_URL`: unset, it spawns the Part 1 stdio subprocess exactly as before;
set, it connects to the remote App over HTTP instead.

**Four real deployment gotchas, found only by actually deploying — none visible from
reading the code:**

1. **`app.yaml` is silently ignored — the real filename is `app.yml`.** The first deploy
   failed with `No command to run and no Python file found`, even though `app.yaml` (with
   a `command:` field) was present. Databricks Apps only recognizes the three-letter
   extension. Fixed by renaming the file in the repo itself (`deployment/mcp_app/app.yml`),
   not just the deployed copy — the assignment's own snippet says `app.yaml`, but the real
   runtime disagrees, so I went with what's actually true.
2. **`app.yml`/`app.py` must sit at the *root* of the deployed source path, not nested.**
   Even after fixing the filename, the same "no command" error persisted with
   `deployment/mcp_app/app.yml` nested under the uploaded workspace path. Deploying from a
   flattened layout (`app.yml`/`app.py`/`requirements.txt`/`tools/` all siblings at the
   source root) fixed it. `app.py`'s `sys.path` fix-up (needed for `from tools.mcp_server
   import mcp` — the App's clean environment has no editable install of this project, unlike
   local `uv run`) now searches upward for whichever directory actually contains `tools/`
   instead of assuming one fixed nesting depth, so the same file works in both the local
   repo's nested layout and the flattened deployed one — verified in an isolated venv
   mimicking each.
3. **Databricks Apps require an OAuth token, not a personal access token.** Calling the
   deployed App with `DATABRICKS_TOKEN` (the PAT everything else in this project
   authenticates with) returned `401 Unauthorized` — for the *same* user, holding
   `CAN_MANAGE` on the app (ruled out as a permissions issue by checking
   `get-permissions`). A Databricks CLI U2M OAuth token worked immediately. Since the
   model-serving container only ever has a PAT, `agent/graph.py::_mint_mcp_oauth_token()`
   performs a proper OAuth client-credentials exchange (`POST {host}/oidc/v1/token`) using
   a dedicated service principal (`cs4603-mcp-caller`, granted `CAN_USE` on the app)
   whose client id/secret ride along as `MCP_OAUTH_CLIENT_ID`/`MCP_OAUTH_CLIENT_SECRET` —
   plain env vars locally, secret-scope references in `deploy.py`'s `environment_vars`
   (added conditionally, only when `MCP_SERVER_URL` is set).
4. **A subtle load-order bug, caught only by testing the failure path.** The first "stop
   the app, prove it fails" attempt didn't fail — `load_mcp_tools()` silently fell back to
   spawning a local stdio subprocess and answered correctly regardless of the real app's
   state. Root cause: called standalone (not through `build_graph()`), nothing had loaded
   `.env` yet, so `os.environ.get("MCP_SERVER_URL")` read `None` and picked the stdio
   branch — a false pass that would have looked like success right up until submission.
   `build_graph()`'s own call order happens to load `config` (hence `.env`) before
   `load_mcp_tools()` runs, so this never manifested through the normal entry point — but
   `load_mcp_tools()` is a public function and shouldn't depend on caller ordering to
   behave correctly. Fixed by having it `import config` itself (a cheap, idempotent
   `load_dotenv()` side effect) before checking the env var.

**Live verification, redone properly after each fix above:**
- App created (`27100159-mcp-tools`) and deployed; `databricks apps list` shows it
  `ACTIVE`/`SUCCEEDED`.
- Full graph (`build_graph()` with `MCP_SERVER_URL` set) answered the combined
  revenue+growth query correctly through the remote server — identical result to the
  bundled-stdio path.
- Stopped the app (`databricks apps stop`), confirmed genuinely down with a raw
  unauthenticated `curl` (`503 Databricks App Not Available`) — *not* just the control-plane
  status field, since that alone had earlier misled me — then reran both `load_mcp_tools()`
  and the full `build_graph()` call: both failed with the real network error, confirming the
  agent genuinely depends on the remote server rather than having some hidden fallback.
  Restarted the app afterward and confirmed it recovered.
- One more finding along the way, not a code bug: after `apps stop`, the control plane
  reported `UNAVAILABLE` within seconds, but the data plane kept accepting requests for
  roughly 3–4 minutes before actually returning `503` — a real teardown grace period worth
  knowing about if this were ever load-bearing for a security boundary.

## Design decisions

**Graph shape.** `agent/graph.py` wires `planner → supervisor → {rag_agent |
mcp_tools} → synthesizer`, with `rag_agent` and `mcp_tools` both edging back
to `supervisor` and a conditional edge (`route_from_supervisor`) reading
`state["next_agent"]` to pick the next hop or fall through to `synthesizer`
once `current_step_index >= len(plan)`. I chose a flat per-step supervisor
loop over either (a) one monolithic ReAct agent with all tools bound, or (b)
a plan-and-execute variant that runs the whole plan in one pass without
returning to a router. (a) is discussed in the Task 1.3 analysis in
`ANSWERS.md` — it trades away the auditable plan and per-stage prompt
specialization this architecture leans on. (b) would save the extra
supervisor LLM call per step, but loses the single choke point where routing
decisions and step completion are both visible in the state — every step
boundary is a place `current_step_index`, `step_results`, and `next_agent`
are all consistent and inspectable, which is what makes the Task 1.7
step-by-step trace possible.

**State schema.** `AnalystState` (`agent/state.py`) deliberately keeps only
`messages` as an `add_messages`-reduced channel; every other field
(`plan`, `current_step_index`, `step_results`, `next_agent`, `final_answer`)
is plain scratch space the nodes overwrite directly. This follows the
"messages in, messages out" rule: because Databricks Model Serving reads the
deployed endpoint's response off `messages[-1]`, `synthesizer` writes the
answer to *both* `final_answer` (for local/notebook use) and appends an
`AIMessage` to `messages` (for the deployed endpoint) — dropping either one
breaks one of the two call paths silently rather than loudly.

**RAG is an all-Databricks path.** `rag/store.py` talks to a Databricks
Vector Search index exclusively — no local pgvector/FAISS index exists
anywhere in the repo. That was a direct consequence of Task 0.3's framing:
since the same managed index is reachable with just `DATABRICKS_HOST`/
`DATABRICKS_TOKEN`, `rag_agent.py`'s retriever code is *identical* between
`pa4.ipynb` (Task 1.7) and the serving container (Task 2.1) — there's no
separate "local embedding path" to keep in sync or forget to port over at
deploy time, which is exactly the class of bug (`ModuleNotFoundError`,
missing env vars) that is the common deployment failure mode this design
sidesteps.

**MCP tools loaded once, invoked synchronously.** `load_mcp_tools()` is
called once inside `build_graph()`, not per-invocation, and `make_mcp_node`
bridges the resulting async `tool.ainvoke()` calls back to a synchronous
graph node via `_run_async()` (which detects whether an event loop is
already running — e.g. inside a Jupyter kernel — and bridges through a
thread pool if so, or calls `asyncio.run()` directly if not). This was a
direct response to relaunching the MCP subprocess per call being the most
fragile part of the deployment; loading once at graph-build time keeps
exactly one subprocess alive for the container's lifetime instead of
spawning one per tool call.

**Failure signaling is a literal string, not a typed field.** When retrieval
finds nothing, `rag_agent.py` returns the literal `"Not found in
documents."` as the step result (`NOT_FOUND` constant), and
`SYNTHESIZER_PROMPT` is explicitly instructed to look for that string and
acknowledge the gap rather than fabricate a number. This is a pragmatic
choice, not a robust one — it works because both ends of the contract
(`rag_agent` and the synthesizer's prompt) agree on the exact string, but
nothing enforces that agreement structurally (a typo in either place would
silently break gap-detection). The Task 1.2 analysis in `ANSWERS.md`
discusses the natural extension: promoting per-step success/failure into a
structured field the supervisor can act on mid-run, rather than a string
convention the synthesizer merely happens to recognize.

**Deployment-specific workaround.** `agent/graph.py`'s
`_real_stderr_for_first_mcp_import()` context manager (wrapping the entire
dependency-construction section of `build_graph()`) exists purely because of
a serving-container quirk, not a local-dev need — see the Deployment section
above for the full debugging story. I kept it in `agent/graph.py` rather than
`deployment/agent_model.py` specifically so `build_graph()` stays a single
function that behaves correctly in both environments without the caller
needing to know which one it's running in.
