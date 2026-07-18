"""Standalone HTTP-transport MCP entrypoint (Bonus C).

Reuses the GIVEN tool definitions from `tools/mcp_server.py` unchanged —
same `calculate`/`percentage_change`/`growth_rate`/`compare_values`/
`unit_convert` tools, same safe-eval implementation — and serves them over
the streamable-http transport instead of stdio, so this can run as a
long-lived Databricks App reachable over HTTPS instead of a subprocess
bundled inside the model container.

Run standalone (for a local smoke test):

    uv run python deployment/mcp_app/app.py
"""

from __future__ import annotations

import os
import sys

# `python <this file>` only puts this file's own directory on sys.path, not
# the repo root — so `import tools` fails in the App's clean environment
# (unlike a local `uv run`, where this project is installed as an editable
# package and `tools` is importable regardless of cwd). Databricks Apps also
# requires app.yml/app.py to sit at the *root* of the deployed source path
# (a nested deployment/mcp_app/app.yml is silently ignored — confirmed by an
# actual failed deploy attempt), whereas the local repo keeps this file
# nested under deployment/mcp_app/ two levels below tools/ — so search
# upward for whichever directory actually contains `tools/` instead of
# assuming one fixed nesting depth.
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
for _candidate in (_THIS_DIR, os.path.dirname(_THIS_DIR), os.path.dirname(os.path.dirname(_THIS_DIR))):
    if os.path.isdir(os.path.join(_candidate, "tools")):
        sys.path.insert(0, _candidate)
        break

from tools.mcp_server import mcp  # noqa: E402

if __name__ == "__main__":
    # Databricks Apps injects the port to listen on via $DATABRICKS_APP_PORT
    # and proxies external HTTPS traffic to it — binding 127.0.0.1 (FastMCP's
    # default) would only accept connections from inside the same container,
    # so this must bind all interfaces to be reachable at all.
    mcp.settings.host = "0.0.0.0"
    mcp.settings.port = int(os.environ.get("DATABRICKS_APP_PORT", 8000))
    mcp.run(transport="streamable-http")
