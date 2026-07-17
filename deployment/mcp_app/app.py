"""Standalone HTTP-transport MCP entrypoint (Bonus C).

TODO:
  - Import the GIVEN tool definitions from `tools.mcp_server` (do not redefine them).
  - Run the MCP server with the streamable-http transport instead of stdio, so it
    can be deployed as a long-lived Databricks App and reached over HTTPS.
"""

from __future__ import annotations

# TODO: from tools.mcp_server import mcp
# TODO: if __name__ == "__main__": mcp.run(transport="streamable-http")
