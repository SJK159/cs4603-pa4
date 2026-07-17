"""MLflow models-from-code definition (Task 2.1).

This file is what `mlflow.langchain.log_model(lc_model=...)` points at in
deploy.py. MLflow serialises it independently of the rest of the repo, so it
must be self-contained: everything it imports comes from this repo's
`agent`/`rag`/`tools`/`config` packages (shipped alongside it via
`code_paths` in deploy.py) or from `pip_requirements` — nothing else is
reachable once this runs inside the serving container.

Must import cleanly:  python -c "import deployment.agent_model"
"""

from __future__ import annotations

import mlflow

from config import get_settings

# Validate required env vars at import time — get_settings() raises a clear
# OSError naming the missing variable — so a misconfigured serving endpoint
# fails with a readable cause in the Logs tab, not a bare traceback or a
# cryptic DEPLOYMENT_FAILED. Importing config here also triggers its
# load_dotenv() call before we touch any Databricks-related env var.
get_settings()

from agent.graph import build_graph  # noqa: E402

# Rebuild the graph with production clients: the real Databricks LLM, the
# real Vector Search retriever, and the real MCP tools. build_graph()'s
# defaults (llm=None, retriever=None, tools=None) construct all three from
# config.py / rag/store.py / agent/graph.py's own MCP loader — the same
# path already exercised locally end-to-end in Task 1.7.
graph = build_graph()

mlflow.models.set_model(graph)
