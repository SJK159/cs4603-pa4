# PA4 - Document Analyst Deployment

You already deployed a LangGraph agent in **`wk5_langgraph/15.databricks_deployment/`**.
PA4's deployment is the **same pipeline** — you are not learning a new method. This
guide starts from what you already did and points out the handful of things that are
*different* because your PA4 agent is bigger and modular.

> **Related helper — [`GITHUB_PIPELINE.md`](GITHUB_PIPELINE.md).** That document is a
> helper for two sections of the main assignment ([`PA-4-CS4603.md`](PA-4-CS4603.md)):
> the **Deployment** section covered here (which produces `deployment/deploy.py`) and
> **Bonus A** (the CI/CD pipeline that runs it). Read it if you're new to GitHub Actions
> or want to automate the deploy steps below.

> **PA4 deploys the same graph two ways.**
> - **Part 2 (from wk5/15) — the manual path.** You call `log_model` → `register_model`
>   → `WorkspaceClient.serving_endpoints` yourself, wire a secret scope, and poll for
>   `READY`. **This first task deliberately walks you through the internals of
>   deployment** — every step the platform normally hides (packaging, the serving
>   container, credential injection, endpoint config, the READY lifecycle) is something
>   *you* do by hand, so you understand what actually happens when a model goes live.
> - **Bonus B (from wk5/16) — the `agents.deploy()` path.** Once you've seen the
>   internals, the `databricks-agents` SDK collapses all of that into **one call** that
>   provisions the endpoint (plus a Review App) and handles auth for you — no secret
>   scope, no manual endpoint config.
>
> This guide covers the manual path (the internals) in detail. The `agents.deploy()`
> path **reuses everything below unchanged** — you do not rebuild the model, you only
> swap the final `WorkspaceClient` deploy step for a single `agents.deploy(...)` call.

---

## 1. The mental model you already have

In wk5/15 you ran this exact chain, and PA4 is identical at this level:

```
your agent file  ──log_model──▶  MLflow (models-from-code)
                 ──register───▶  Unity Catalog  (databricks-uc)
                 ──create─────▶  Model Serving endpoint
                 ──POST───────▶  openai.OpenAI(...).chat.completions
```

Everything you learned still applies:
- `mlflow.models.set_model(graph)` at the bottom of the deployable file.
- `mlflow.set_registry_uri("databricks-uc")` before registering.
- `WorkspaceClient().serving_endpoints.create(...)` with `workload_size="Small"` and
  `scale_to_zero_enabled=True`.
- Credentials injected as **secret references** (`{{secrets/cs4603-deploy/...}}`).
- Test with `openai.OpenAI(base_url=f"{host}/serving-endpoints")`.

If you understood wk5/15, you already understand 80% of PA4 deployment. The rest is
the five differences below.

> **Databricks container:**
> When you create a Model Serving endpoint, Databricks doesn't just run your code on
> your laptop or notebook. It builds an isolated, self-contained runtime environment —
> a **Docker container** — on Databricks' own compute, installs the Python packages you
> listed in `pip_requirements`, copies in the files you listed in `code_paths`, loads
> your model, and starts a web server that answers requests. That sandboxed environment
> is "the serving container." It has **none** of your local files or installed packages
> unless you explicitly ship them — which is exactly why `code_paths` and
> `pip_requirements` (the differences below) matter so much.

---

## 2. PA4 Deployment Add-Ons

| # | wk5/15 | PA4 | Why it changes |
|---|--------|-----|----------------|
| 1 | One self-contained `agent.py` | `agent_model.py` **imports** your `agent/`, `rag/`, `config.py` | Your Part 1 code is modular (8 files); you don't want to copy-paste it all into one file |
| 2 | `log_model(lc_model=..., name=..., input_example=...)` | same **+ `code_paths=[...]`** | The serving container needs your local packages, so you ship them |
| 3 | Requirements inferred | same **+ explicit `pip_requirements`** | PA4 adds `databricks-vectorsearch` etc. that inference can miss |
| 4 | Tools are inline functions | Tools come from the **MCP server** | You must bundle `tools/mcp_server.py` too |
| 5 | `MessagesState` handles messages for you | You use a custom `AnalystState` | **You** must append the final answer as an `AIMessage` |

Each of these is explained with a concrete step below.

---

## 3. Difference #1 + #2 — Ship your package with `code_paths`

**This is the single most important change.** In wk5/15 `agent.py` had *everything*
inline, so `log_model` only needed the one file. Your PA4 `agent_model.py` looks like:

```python
# deployment/agent_model.py  — still ends with set_model, but imports your code
from agent.graph import build_graph, load_mcp_tools
from config import get_chat_llm
from rag.store import get_retriever

graph = build_graph(llm=get_chat_llm(), retriever=get_retriever(), tools=load_mcp_tools(...))
mlflow.models.set_model(graph)
```

Because it imports `agent`, `rag`, `config`, the serving container must receive those
files. You do that with `code_paths`:

```python
mlflow.set_registry_uri("databricks-uc")
with mlflow.start_run():
    model_info = mlflow.langchain.log_model(
        lc_model="deployment/agent_model.py",     # same idea as wk5/15
        name="agent",
        code_paths=[                              # NEW: ship your local packages
            "agent", "rag", "tools", "config.py",
        ],
        pip_requirements=[                        # NEW: pin the extras inference misses
            "langgraph", "langchain-openai",
            "databricks-langchain", "databricks-vectorsearch",
            "langchain-mcp-adapters", "mcp", "mlflow", "openai",
        ],
        input_example={"messages": [{"role": "user", "content": "What was the revenue?"}]},
    )
```

> **If you skip `code_paths`,** the endpoint will fail at startup with
> `ModuleNotFoundError: No module named 'agent'` in the Serving **Logs** tab. That is
> the #1 PA4 deployment error — and it never happened in wk5/15 because that agent was
> a single file.

---

## 4. Difference #4 — The MCP server travels with your model

In wk5/15 the tools were plain functions defined right in `agent.py`. In PA4 they live
in `tools/mcp_server.py` and your graph launches it. For deployment:

- Include `"tools"` in `code_paths` (done above) so the file is in the container.
- In `agent_model.py`, resolve the server path relative to the packaged code and load
  the tools at startup:

```python
_mcp_server = os.path.join(os.path.dirname(os.path.dirname(__file__)), "tools", "mcp_server.py")
tools = load_mcp_tools(_mcp_server)
```

> **Sanity check first.** Before deploying, confirm the server runs locally:
> `uv run python tools/mcp_server.py` (Task 0.2). If it can't start on your laptop, it
> won't start in the container either.

> **Async caveat.** MCP tools are async, so you call them via `asyncio.run(...)`. This
> works in MLflow's synchronous serving path, but `asyncio.run()` raises *"cannot be
> called from a running event loop"* if a loop is already active, and each stdio call may
> relaunch the subprocess. Load the tools once at graph-build time and keep invocation
> synchronous — this is the most fragile part of the deployment, so verify it end-to-end
> against the live endpoint (Task 2.4).

---

## 5. Difference #5 — Make the endpoint OpenAI-compatible

wk5/15 used `MessagesState`, so the assistant node automatically put its reply on the
`messages` channel — and the endpoint returned it as a normal chat completion. Your
PA4 `AnalystState` has extra fields (`plan`, `step_results`, `final_answer`). If your
**synthesizer** only sets `final_answer`, the endpoint returns an *empty* completion.

Fix (this is also why Task 1.6 insists on it):

```python
# synthesizer returns BOTH — messages is what the endpoint reads back
return {"final_answer": answer, "messages": [AIMessage(content=answer)]}
```

Rule of thumb: **messages in → messages out.** Keep `messages` (with `add_messages`) as
the entry and exit channel; the other fields are internal scratch space.

---

## 6. Difference you might *expect* but don't have — the vector store

PA4 uses a **Databricks Vector Search index**, which is a managed service reachable from anywhere with `DATABRICKS_HOST` + `DATABRICKS_TOKEN`. The *same* `rag/store.py::get_retriever()` runs locally and in the
container — no code change. 

---

## 7. Everything else is wk5/15, unchanged

**Secrets** — identical to wk5/15. Create the scope once, then reference it:

```bash
databricks secrets create-scope cs4603-deploy
databricks secrets put-secret cs4603-deploy DATABRICKS_TOKEN --string-value "dapi..."
databricks secrets put-secret cs4603-deploy DATABRICKS_HOST  --string-value "https://<workspace>.databricks.com"
databricks secrets put-secret cs4603-deploy DATABRICKS_MODEL --string-value "databricks-meta-llama-3-3-70b-instruct"
```

```python
environment_vars={
    "DATABRICKS_HOST":  "{{secrets/cs4603-deploy/DATABRICKS_HOST}}",
    "DATABRICKS_TOKEN": "{{secrets/cs4603-deploy/DATABRICKS_TOKEN}}",
    "DATABRICKS_MODEL": "{{secrets/cs4603-deploy/DATABRICKS_MODEL}}",
    # plus your Vector Search vars so the retriever can find the index:
    "VECTOR_SEARCH_ENDPOINT": "<your-vs-endpoint>",
    "VECTOR_SEARCH_INDEX":    "<your-index>",
}
```

**Testing** — same OpenAI SDK call. One correction: the wk5/15 notebook's last cell
had a bug (`response[0].messages[-1]`). Use the standard OpenAI shape:

```python
resp = client.chat.completions.create(
    model="<your-endpoint-name>",
    messages=[{"role": "user", "content": "What was the net income in 2023?"}],
)
print(resp.choices[0].message.content)   # not response[0].messages[-1]
```

**Where you run it** — wk5/15 ran the deploy cells in a Databricks notebook. In PA4 you
may either run the same cells in `pa4.ipynb`, or run `deployment/deploy.py`. Both do the
identical three MLflow/SDK calls.

---

## 8. Pre-flight checklist

Before you click deploy, verify — in this order:

- [ ] `uv run python tools/mcp_server.py` starts (Task 0.2).
- [ ] Your Vector Search index is `READY` and a similarity query returns chunks (Task 0.3).
- [ ] `python -c "import deployment.agent_model"` imports cleanly with your `.env` loaded.
- [ ] Your graph answers a **combined** query locally (Task 1.7) — the final answer is
      on `messages[-1]`, not just `final_answer`.
- [ ] Secret scope `cs4603-deploy` exists with all three secrets.
- [ ] `code_paths` lists `agent`, `rag`, `tools`, `config.py`.
- [ ] `pip_requirements` includes `databricks-vectorsearch` and `langchain-mcp-adapters`.

## 9. If the endpoint goes `DEPLOYMENT_FAILED`

Open **Serving → your endpoint → Logs** (same as wk5/15) and match the traceback:

| Log message | Cause | Fix |
|-------------|-------|-----|
| `ModuleNotFoundError: 'agent'` | forgot `code_paths` | add your packages to `code_paths` |
| `Missing required environment variables` | secret scope not wired | check `environment_vars` secret refs |
| `ModuleNotFoundError: 'databricks.vector_search'` | requirement not inferred | add it to `pip_requirements` |
| endpoint READY but empty answers | synthesizer didn't append `AIMessage` | see §5 |
| retrieval errors at inference | VS index/endpoint env vars missing | add `VECTOR_SEARCH_*` to `environment_vars` |

You've done this loop before in wk5/15 — read the traceback, fix the one line, re-log,
re-deploy.
