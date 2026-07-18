# CS4603 PA4 — Document Analyst — Analysis Questions

Answers to every ANALYSIS QUESTION from the assignment. Setup, deployment
instructions, and design decisions live in [`NOTES.md`](NOTES.md).

## Analysis Questions

### Task 1.2 — Planner
1. There's no explicit dependency graph between steps — `plan` is a flat
   `list[str]` and `current_step_index` just walks it in order — so
   cross-step dependencies are handled entirely through natural language,
   not structured data. `PLANNER_PROMPT` instructs the LLM to phrase a
   dependent step so the dependency is explicit in the text itself (e.g.
   "Calculate 8% compound annual growth on the FY2023 net revenue *found in
   the previous step*" rather than assuming shared context). On the
   consuming side, `make_mcp_node` in `agent/graph.py` reads that phrasing:
   before invoking the LLM for a computation step it prepends every prior
   `step_results` entry as `"Step {i}: {result}"` context (see the "Prior
   step results" block), so the LLM can substitute the real number into the
   tool call instead of passing a literal placeholder like `"2023_revenue"`.
   This only runs one direction, though: `make_rag_agent` retrieves using
   *only* the current step's text — it never sees prior `step_results`. So a
   retrieval step that depended on an earlier step's output (e.g. "look up
   the division that had the highest computed growth rate") wouldn't work;
   the architecture only handles computation-depends-on-retrieval, not
   retrieval-depends-on-computation, because that's the only direction
   `PLANNER_PROMPT`'s own examples and the assignment's example queries
   exercise.
2. For this use case — 2 to 5 step plans over a single financial report,
   mixing fixed-shape retrieval and computation steps — full replanning
   after every step would mostly hurt: it adds an LLM round-trip per step
   purely to re-derive a plan shape that's usually stable (find X, compute Y,
   present Z), and it introduces a new failure surface where the replanner
   could second-guess a perfectly good remaining plan into something worse,
   or hallucinate new steps that drift from the original question. It would
   help in exactly one situation the current architecture handles poorly:
   when a retrieval step comes back `"Not found in documents."`, the
   supervisor loop just advances to the next step anyway (`current_step_index
   + 1` happens unconditionally in `make_rag_agent`), so a downstream
   computation step still gets attempted against a missing value and the
   synthesizer is left to notice the gap after the fact. A targeted
   replan — not a full re-plan, but a conditional "if this step failed,
   ask the planner for an alternative retrieval phrasing or a step to skip"
   — would let the graph recover mid-run (e.g. retry with different search
   terms, or drop the now-unanswerable computation step) instead of running
   the full plan to completion and only then explaining the gap.

### Task 1.3 — Supervisor
1. `_classify_step` in `agent/supervisor.py` asks the LLM for exactly
   `"rag_agent"` or `"mcp_tools"`, and only falls back to keyword-matching
   the step text (`"calculat"`, `"growth"`, `"percent"`, …) when the reply
   contains neither literal word — there's no validation against what the
   step actually needs. If it misroutes a computation step to `rag_agent`,
   the retriever runs a similarity search against a numeric/computation
   phrase, `RAG_EXTRACT_PROMPT` either returns `"Not found in documents."`
   (relatively safe — synthesizer is instructed to surface that gap) or,
   worse, confidently extracts an unrelated "fact" from whatever chunks came
   back, citing a real source that doesn't actually support the claim. If it
   misroutes a retrieval step to `mcp_tools`, `MCP_STEP_PROMPT` tells the LLM
   to *always* call a tool, so it will call something like `calculate` with
   fabricated or nonsensical arguments and produce a confident-looking wrong
   number — this is the worse failure mode because nothing in the pipeline
   flags it as suspect; `step_results` just gets a plausible-looking bad
   entry and `current_step_index` advances regardless (same unconditional
   advance noted above). **Detection/recovery I'd add:** have the planner
   tag each step with its intended type (`{"step": ..., "type": "retrieval"
   | "computation"}`) instead of a bare string, so the supervisor's LLM
   classification can be checked against the planner's own label and a
   mismatch logged or escalated rather than silently trusted; and add a
   lightweight validator between each specialist and the supervisor (the
   graph already routes `rag_agent`/`mcp_tools` back to `supervisor`, so
   this is one more node on an edge that already exists) that flags a
   `step_results` entry as suspect if a "retrieval" step returned no
   citation, or a "computation" step's result contains no digits — and
   re-routes that step index to the other specialist once before advancing,
   rather than trusting the first classification unconditionally.
2. A single ReAct agent with every tool bound would resolve simple,
   single-fact queries (e.g. "What was net income in 2023?") in one tool
   call with no planning overhead — strictly cheaper than this architecture,
   which always pays for a planner call plus one supervisor call per step
   even when the question needs only one step. The supervisor pattern earns
   its complexity when queries are genuinely multi-step and heterogeneous,
   which is this assignment's whole premise: (a) it produces an **auditable
   plan** up front (`plan`, `current_step_index`) rather than an opaque
   sequence of tool calls buried in a ReAct scratchpad — useful for
   debugging and for exactly the kind of step-by-step trace Task 1.7 asks
   the notebook to show; (b) it lets each stage use a **narrowly-scoped
   prompt** tuned to its one job — `RAG_EXTRACT_PROMPT` is written entirely
   around citation-grounded extraction and refusing to guess,
   `MCP_STEP_PROMPT` is written entirely around *never* doing arithmetic
   itself and always delegating to a tool — rather than one general-purpose
   ReAct system prompt trying to instill both disciplines at once; (c) the
   loop has a **deterministic termination bound**, `len(plan)` steps, fixed
   once by the planner, rather than a ReAct agent's tool-calling loop, which
   needs its own separate max-iterations guard to avoid running away. Below
   the complexity of a multi-step analytical query, none of that is worth
   the extra LLM round-trips.

### Task 1.4 — RAG Agent
1. `rag_agent.py` retrieves with `retriever.invoke(step)` — `step` is the
   single atomic string the planner produced (e.g. "Find Meridian's net
   revenue for fiscal year 2023"), never the original compound question. For
   a query like "What was net revenue in FY2023, and what would it be after
   3 years of 8% compound annual growth?", embedding the whole question
   would mix a retrieval intent with a calculation intent in one vector,
   pulling the match toward whichever half has more distinctive vocabulary
   ("8% compound annual growth" is generic finance language that could match
   growth-rate boilerplate anywhere in the report, diluting the FY2023
   revenue-specific signal). Retrieving per decomposed step keeps each
   embedding search **intent-pure**, which should improve precision on
   compound questions specifically. The tradeoff runs the other way when the
   planner's step isn't as self-contained as `PLANNER_PROMPT` asks for — if
   it drops context the original question had (e.g. omits "Meridian" or
   "FY2023" because the LLM assumed it was implied), the atomic step
   actually retrieves *worse* than the original question would have,
   because it's lost specificity the compound question had. And because
   retrieval happens once per step with no re-ranking against the rest of
   the plan, the RAG agent can't use information gathered by a later step to
   refine an earlier retrieval — the pass is one-shot per step, forward-only.
2. The current code sends `step` to the retriever as-is with a fixed
   `top_k`, with no query-quality check before searching. For a vague step
   like "find relevant financial data," I'd add a rewrite pass before
   `retriever.invoke(step)`: send the vague step plus the original question
   (still available in `state["messages"][0]`, just not currently plumbed
   into `rag_agent`) to a small LLM call that expands it into concrete,
   document-grounded terms — e.g. "financial data" → "net revenue, net
   income, and total assets for fiscal year 2023" — since the vague step
   alone doesn't give the embedding model enough to anchor on. Failing that,
   a cheaper fallback already fits the existing failure-signaling
   convention: widen `top_k` for a first pass, and if `RAG_EXTRACT_PROMPT`
   comes back with `"Not found in documents."` (the same literal signal the
   synthesizer already knows how to surface), retry once with the step
   text concatenated to the original question as the query instead of the
   step alone, so the retrieval gets the full context back rather than
   only the possibly-underspecified atomic instruction.

### Task 2.1 — Model Definition
1. `models-from-code` serializes `agent_model.py` as a *script*, not a pickle of
   live Python objects — MLflow re-runs the file inside the serving container to
   rebuild the model from scratch. That container is a fresh process with none of
   the state that existed on my laptop when I ran `deploy.py`: no open DB
   connections, no in-memory objects, no filesystem paths outside what
   `code_paths` shipped. If `agent_model.py` referenced external state — e.g. a
   local pgvector container, a Python object built earlier in a notebook cell, or
   a relative path only valid on my machine — the import would either raise
   immediately (module/connection not found) or silently produce a broken model
   that fails on first invocation. Self-containment forces every dependency to be
   either an importable package (via `pip_requirements`/`code_paths`) or a
   network call authenticated by env vars (`DATABRICKS_HOST`/`DATABRICKS_TOKEN`),
   both of which *do* exist inside the container.
2. Querying a managed Vector Search index at inference time (vs. baking the
   corpus into the container image) trades:
   - **Freshness:** external index wins — re-ingesting `rag/ingest.py` updates the
     live index immediately; a baked-in corpus requires re-logging and
     redeploying the model to pick up new documents.
   - **Cold-start size:** external index wins — the container image only carries
     code and small pip packages, not embeddings/vectors, so it builds and starts
     faster.
   - **Latency:** baked-in wins per-request — an external call adds one network
     round trip per retrieval step; an in-container index has no network hop.
   - **Failure modes:** external index adds a new failure surface — the endpoint
     can be "ready" but retrieval can still fail at runtime if the Vector Search
     endpoint is down, misconfigured (wrong `VECTOR_SEARCH_INDEX`), or the
     container's env vars are wrong — a class of error a self-contained baked-in
     index can't have (it either loads at startup or doesn't).

### Task 2.3 — Serving Endpoint
1. The endpoint being "authenticated to serve models" only covers the inbound
   side — Databricks authenticates the *caller* invoking the endpoint. Inside the
   handler, our code makes its own *outbound* calls: `ChatOpenAI` calling the LLM
   serving endpoint, and `DatabricksVectorSearch` calling the Vector Search
   endpoint. Those are separate Databricks services the container must
   authenticate to on its own, and the container has no ambient credentials for
   that — it needs an explicit `DATABRICKS_TOKEN`/`DATABRICKS_HOST` pair injected
   as env vars (here, via secret references) so `config.get_settings()` can
   construct those clients.
2. Databricks Model Serving performs a **rolling/blue-green update**: the new
   served-entity version is built and deployed alongside the existing one, and
   only after the new version passes its readiness checks does traffic cut over
   to it (this is why `create_or_update_endpoint()` polls for `READY` rather than
   returning immediately after `update_config`). In-flight requests issued against
   the old version continue to be served by it until it's ready to be torn down;
   Databricks does not hard-kill the old container mid-request. This is also why
   an endpoint can be left in a "config is applied but state is
   `UPDATE_FAILED`" state, as ours was after the unpinned-`pip_requirements` bug —
   the update never became ready, so the endpoint kept (or, on the very first
   deploy, never had) a serving version.

### Task 3.2 — Client
1. Fixed-interval retries hit a scaling-up endpoint with the same request rate that
   likely caused the `429`/`503` in the first place — if the endpoint is rate-limiting
   because it's overloaded, or still cold-starting after `scale_to_zero`, retrying every
   1s just adds more load exactly while it's least able to handle it, and every
   concurrent client backs off in lockstep, so their retries stay synchronized and
   collide again on the next attempt. Exponential backoff (`2**attempt` in
   `DocumentAnalystClient.ask()`) spaces retries out further apart as failures persist,
   giving the endpoint's autoscaler or rate limiter time to actually resolve the
   underlying condition (finish scaling from zero, let the request queue drain) instead
   of being repeatedly hammered at a fixed cadence. It also naturally desynchronizes
   concurrent clients over successive attempts, spreading out the retry load instead of
   presenting it as another burst.
2. A high `max_retries` combined with exponential backoff means a single caller can hold
   a connection/thread open retrying for a very long time (`2**attempt` grows fast — 10
   retries is over 17 minutes of backoff alone), which under load from many concurrent
   users compounds into a **retry storm**: every one of those callers is still
   consuming a client-side connection slot and re-issuing requests against an endpoint
   that is already struggling (that's *why* it returned 429/503 in the first place), so
   the retries themselves add load that delays recovery — the opposite of what a client
   should do when a service signals it's overwhelmed. It also hides a real outage from
   the caller for a long time (they just look "stuck") instead of failing fast enough
   for the caller's own layer (a UI, an upstream service) to degrade gracefully or fall
   back.
3. `ask_streaming()` is worth it whenever the *perceived* latency to first content
   matters more than total completion time — the canonical case here is a chat UI: our
   Document Analyst's full pipeline (planner → supervisor loop → RAG/MCP tool calls →
   synthesizer) can take several seconds end-to-end, and a user staring at a blank
   response for that long reads as broken, whereas the same total latency feels
   responsive if text starts appearing incrementally. `ask()` is the right choice for
   anything programmatic that needs the complete, parseable answer before proceeding —
   e.g. a downstream service calling the Document Analyst as one step in its own
   pipeline, where partial text has no use until the full answer is available anyway.

### Bonus A — CI/CD Pipeline
1. `main` is the single reviewed source of truth — merging to it is the deliberate
   "this is ready to ship" signal, whereas feature branches are by definition
   in-progress and may not even build. If `deploy` ran on every branch push, two
   developers iterating on separate feature branches would race to overwrite the same
   live serving endpoint with whichever branch happened to push last, and a
   half-finished branch could push a broken model straight to production before it's
   even been reviewed. Restricting `deploy` to `main` (via the `if:` guard in
   `deploy.yml`) means the only thing that reaches the endpoint is code that passed
   `lint-and-test` *and* went through the PR/merge process — `pull_request` events
   still run lint+test so reviewers see green/red before merging, but never trigger a
   deploy.
2. I'd add an **evaluation gate** as a job between `lint-and-test` and `deploy`: after
   registering the new model version (but before `create_or_update_endpoint()` cuts
   traffic to it), run it against a held-out set of question/expected-answer pairs
   built from `data/annual_report.md` (e.g. the same three test queries from Task 1.7,
   plus a few more with known answers) via `mlflow.evaluate`, score it (answer
   correctness / faithfulness to retrieved chunks), and compare that score against the
   metric already logged for the currently-serving version (fetchable via
   `mlflow.search_runs` on the registered model's aliases). If the new version's score
   regresses past some threshold, the job should fail — which, given `deploy` is a
   single job that both registers and serves, means splitting it into a `register` step
   followed by a conditional `promote-and-serve` step, so a failing eval can register
   the new version (visible in the UC model registry) without ever making it the one
   the endpoint actually serves.

### Bonus B — `databricks-agents` SDK
1. The manual approach (`deploy.py`) gives full control at the cost of writing every
   step yourself: I chose the exact registered-model output shape (the raw `AnalystState`
   dict via `mlflow.models.set_model(graph)`), the exact endpoint config
   (`workload_size`, `scale_to_zero_enabled`, which env vars are secrets vs. plaintext),
   and I own the READY-polling loop (`wait_for_ready()`), so I can see precisely what's
   happening at each step and log/handle failures however I want. `agents.deploy()`
   trades that control for one call that provisions the endpoint *and* a Review App for
   human feedback together — but that convenience comes with an opinionated contract I
   don't get to negotiate: it silently requires the model's output to be
   `ChatCompletionResponse`/`StringResponse`-shaped (see the `ValueError` in `NOTES.md`),
   which meant giving up the rich, auditable `AnalystState` output at the served boundary
   — the internal graph state is unchanged, but external callers of the Agents-SDK
   endpoint only ever see a plain chat completion, not
   `plan`/`step_results`/`current_step_index`. I also lose visibility into *when* the
   endpoint is actually ready (`agents.deploy()` returns immediately with "can take up
   to 15 minutes"; I had to write my own separate poll loop against `WorkspaceClient` to
   know when it was safe to query), and I gain, for free, the Review App — something
   `deploy.py`'s manual path doesn't offer at all without building it myself.
2. The Review App gives named reviewers a chat UI against the live endpoint where they
   can rate individual responses (thumbs up/down, free-text feedback) tied back to the
   specific trace that produced them. A concrete feedback loop: have a few people run the
   three Task 1.7-style queries (and edge cases — ambiguous questions, facts not in the
   report) through the Review App over a week; export the logged feedback via
   `mlflow.search_traces`/the feedback API; use the thumbs-down traces as a diagnostic
   set — for each one, look at whether the failure was in retrieval (wrong or missing
   chunk), routing (Task 1.3's misroute failure mode), or synthesis (hallucinated a
   number instead of citing "Not found in documents.") — then turn the clearest recurring
   pattern into either a prompt fix (`agent/prompts.py`) or a held-out eval case
   (feeding into the Bonus A analysis question above: a future deploy that regresses on
   one of these specific traces should fail the evaluation gate before promotion).

### Bonus C — Standalone MCP Server
1. **Gained:** the tool server can now be redeployed, restarted, and scaled independently
   of the model — I proved this directly by stopping and restarting
   `27100159-mcp-tools` without touching the serving endpoint at all. It also removes the
   most fragile part of the container deployment: the model container no longer needs to
   spawn and manage a stdio subprocess at graph-build time, and the
   `_real_stderr_for_first_mcp_import()` workaround becomes unnecessary for the MCP half
   of the problem (it's still needed for `DatabricksVectorSearch`'s own transitive MCP
   import, per the docstring, but the subprocess-launch codepath itself is no longer
   exercised). One tool service could also now serve multiple agents. **Lost/gained new
   failure modes:** every tool call is now a network hop instead of a local pipe, so
   latency goes up and a class of failure that simply couldn't happen before now can —
   the app can be down, slow, or unreachable independent of the model endpoint's own
   health, and callers need real timeout/retry handling for it (the same class of
   problem `client/sdk.py` already solves for the model endpoint, but nothing here reuses
   that — `load_mcp_tools()` has no retry logic at all). Auth is a wholly new surface
   too: I hit this directly (`NOTES.md`) — a PAT that authenticates everywhere else in
   this project doesn't authenticate to the App at all, requiring a dedicated
   service-principal OAuth flow that didn't exist before.
2. Two layers: **network** and **identity**. Network — Databricks Apps can be restricted
   with IP access lists / private networking at the workspace level so the App's ingress
   isn't reachable from the open internet at all, only from within the workspace's network
   boundary. Identity — even reachable, the App should only accept calls from a specific
   principal: exactly what I built here. The service principal `cs4603-mcp-caller` was
   deliberately created *only* for this purpose (not reused from anything else in the
   project) and granted the minimum permission level that works (`CAN_USE`, not
   `CAN_MANAGE`) via `apps update-permissions` — so a leaked `MCP_OAUTH_CLIENT_SECRET`
   can call the tools but can't reconfigure or delete the app. In production I'd rotate
   that secret regularly (the same way `MCP_OAUTH_CLIENT_SECRET`/`DATABRICKS_TOKEN` sit in
   the `cs4603-deploy` secret scope rather than in plaintext env vars) and avoid granting
   the underlying service principal any permissions beyond that one app.
3. Bundling (Part 1) wins when the tool set is small, stable, and used by exactly one
   agent — no network hop, no separate deployment to manage, no second auth surface to
   secure, and it's what a single-container serving deploy naturally supports without
   extra infrastructure. A standalone service (Bonus C) is worth the extra moving parts
   once any of that stops being true: multiple agents sharing one tool server, the tools
   needing to scale or redeploy on a different cadence than the model (e.g. a new
   `calculate` bug fix shouldn't require re-registering and re-serving the whole
   LangGraph model), or the tools needing their own monitoring/rate-limiting independent
   of the model endpoint's. Given how small and stable `tools/mcp_server.py` actually is
   here — five deterministic, dependency-free functions — Part 1's bundled approach is
   arguably the *more* appropriate choice for this specific project; Bonus C is the right
   call once a tool server needs to earn its keep as shared infrastructure, which this
   one doesn't yet.
