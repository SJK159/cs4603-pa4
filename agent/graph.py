"""Full Document Analyst graph (Tasks 1.5 + 1.7)."""

from __future__ import annotations

import asyncio
import contextlib
import os
import sys
from concurrent.futures import ThreadPoolExecutor

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph

from agent.planner import make_planner
from agent.prompts import MCP_STEP_PROMPT
from agent.rag_agent import make_rag_agent
from agent.state import AnalystState
from agent.supervisor import MCP, RAG, SYNTH, make_supervisor, route_from_supervisor
from agent.synthesizer import make_synthesizer


@contextlib.contextmanager
def _real_stderr_for_first_mcp_import():
    """Ensure `sys.stderr` has a real file descriptor while MCP-related modules
    first get imported.

    `mcp.client.stdio.stdio_client`'s `errlog` parameter defaults to
    `sys.stderr`, evaluated once — when that module is *first imported*, not
    at call time. In the Databricks serving container, MLflow replaces
    `sys.stderr` with a `StreamToLogger` shim (no `fileno()`) before loading
    the model, and `mcp.client.stdio` gets imported transitively as soon as
    anything touches MCP — either `load_mcp_tools()` below, or, earlier,
    `rag.store.get_vector_store()`'s `from databricks_langchain import
    DatabricksVectorSearch`, since `databricks_langchain/__init__.py` itself
    imports its own MCP client wrapper. Whichever happens first bakes the
    shim into that default permanently (Python caches the import), and
    asyncio's subprocess machinery later crashes with `AttributeError:
    'StreamToLogger' object has no attribute 'fileno'` when it tries to wire
    up the MCP server subprocess's stderr. Wrap the *whole* dependency-
    construction section of `build_graph()` in this so it doesn't matter
    which import happens first; harmless locally, where sys.stderr already
    has a real fd.
    """
    real_stderr = sys.__stderr__ or sys.stderr
    saved_stderr, sys.stderr = sys.stderr, real_stderr
    try:
        yield
    finally:
        sys.stderr = saved_stderr


def _run_async(coro):
    """Run `coro` to completion, whether or not a loop is already running.

    `asyncio.run()` raises "cannot be called from a running event loop" in
    contexts that already have one (e.g. a Jupyter kernel), while a plain
    script or MLflow's synchronous serving path has none. Detect which case
    we're in and bridge accordingly instead of assuming one or the other.
    """
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)
    else:
        with ThreadPoolExecutor(max_workers=1) as executor:
            return executor.submit(asyncio.run, coro).result()


def _mint_mcp_oauth_token() -> str:
    """Mint an M2M OAuth access token to call the standalone MCP Databricks App.

    Databricks Apps' ingress requires a Databricks OAuth token, not a plain
    personal access token — confirmed empirically: the same user, with
    CAN_MANAGE on the app, gets a 401 calling it with `DATABRICKS_TOKEN` (a
    PAT) as the bearer, but succeeds with a Databricks-issued OAuth token.
    `DATABRICKS_TOKEN` is the only credential the model-serving container
    otherwise has, so the app needs its own OAuth-capable identity: a
    service principal (`MCP_OAUTH_CLIENT_ID`/`MCP_OAUTH_CLIENT_SECRET`,
    granted CAN_USE on the app) that this exchanges for a short-lived access
    token via the standard OAuth client-credentials grant.
    """
    import httpx

    from config import get_settings

    settings = get_settings()
    client_id = os.environ["MCP_OAUTH_CLIENT_ID"]
    client_secret = os.environ["MCP_OAUTH_CLIENT_SECRET"]

    response = httpx.post(
        f"{settings['host']}/oidc/v1/token",
        data={"grant_type": "client_credentials", "scope": "all-apis"},
        auth=(client_id, client_secret),
        timeout=30.0,
    )
    response.raise_for_status()
    return response.json()["access_token"]


def load_mcp_tools(server_path: str | None = None):
    """Connect to the MCP server and return its LangChain tools.

    Loaded once at graph-build time (not per-invocation) per the deployment
    guide: stdio MCP is bundled inside the serving container, so tool
    invocation must stay synchronous and the subprocess shouldn't be
    relaunched on every call.

    If `MCP_SERVER_URL` is set (Bonus C — the standalone Databricks App in
    deployment/mcp_app/), connects over streamable HTTP instead of spawning
    a stdio subprocess, decoupling the tool server from the model container.
    Falls back to the Part 1 stdio subprocess behavior when it's unset.
    """
    # Importing config runs its module-level load_dotenv() as a side effect,
    # so MCP_SERVER_URL is populated even if this is called standalone,
    # before build_graph()'s LLM/retriever setup would otherwise load it.
    from langchain_mcp_adapters.client import MultiServerMCPClient

    import config as _config  # noqa: F401

    mcp_url = os.environ.get("MCP_SERVER_URL")
    if mcp_url:
        client = MultiServerMCPClient(
            {
                "analyst": {
                    "url": f"{mcp_url.rstrip('/')}/mcp",
                    "transport": "streamable_http",
                    "headers": {"Authorization": f"Bearer {_mint_mcp_oauth_token()}"},
                }
            }
        )
        return _run_async(client.get_tools())

    if server_path is None:
        server_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "tools",
            "mcp_server.py",
        )

    client = MultiServerMCPClient(
        {
            "analyst": {
                "transport": "stdio",
                "command": "python",
                "args": [server_path],
            }
        }
    )
    return _run_async(client.get_tools())


def _tool_result_to_text(result) -> str:
    """Flatten an MCP tool result into plain text.

    langchain-mcp-adapters returns a list of content blocks, e.g.
    [{"type": "text", "text": "...", "id": "..."}], rather than a plain
    string — extract the text instead of stringifying the raw structure.
    """
    if isinstance(result, str):
        return result
    if isinstance(result, list):
        parts = [
            item["text"] if isinstance(item, dict) and "text" in item else str(item)
            for item in result
        ]
        return "\n".join(parts)
    return str(result)


def make_mcp_node(tools, llm):
    llm_with_tools = llm.bind_tools(tools)
    tools_by_name = {tool.name: tool for tool in tools}

    def mcp_tools(state: AnalystState) -> dict:
        step = state["plan"][state["current_step_index"]]

        # A step can reference an earlier step's result in words (e.g. "the
        # revenue found in the previous step") without the actual number —
        # the planner writes steps before any of them have run. Pass prior
        # results as context so the LLM can substitute real values instead
        # of a literal placeholder when calling the tool.
        if state["step_results"]:
            prior = "\n".join(
                f"Step {i}: {r}" for i, r in enumerate(state["step_results"], start=1)
            )
            human_content = f"Prior step results:\n{prior}\n\nCurrent step: {step}"
        else:
            human_content = step

        response = llm_with_tools.invoke(
            [
                SystemMessage(content=MCP_STEP_PROMPT),
                HumanMessage(content=human_content),
            ]
        )

        if response.tool_calls:
            call = response.tool_calls[0]
            tool = tools_by_name[call["name"]]
            # MCP tools are async-only (langchain-mcp-adapters), so a
            # synchronous graph node must bridge to the event loop.
            result = _run_async(tool.ainvoke(call["args"]))
        else:
            # The LLM should always call a tool per MCP_STEP_PROMPT; if it
            # didn't, fall back to its text rather than crashing the step.
            result = response.content

        return {
            "step_results": [*state["step_results"], _tool_result_to_text(result)],
            "current_step_index": state["current_step_index"] + 1,
        }

    return mcp_tools


def build_graph(llm=None, retriever=None, tools=None):
    """Assemble planner -> supervisor -> {rag_agent | mcp_tools} -> synthesizer.

    Dependencies are injectable so the graph can be unit-tested offline with
    fakes (see tests/test_smoke.py) and reused unchanged by
    deployment/agent_model.py with production clients.
    """
    with _real_stderr_for_first_mcp_import():
        if llm is None:
            from config import get_chat_llm

            llm = get_chat_llm()
        if retriever is None:
            from rag.store import get_retriever

            retriever = get_retriever()
        if tools is None:
            tools = load_mcp_tools()

    builder = StateGraph(AnalystState)
    builder.add_node("planner", make_planner(llm))
    builder.add_node("supervisor", make_supervisor(llm))
    builder.add_node("rag_agent", make_rag_agent(retriever, llm))
    builder.add_node("mcp_tools", make_mcp_node(tools, llm))
    builder.add_node("synthesizer", make_synthesizer(llm))

    builder.add_edge(START, "planner")
    builder.add_edge("planner", "supervisor")
    builder.add_conditional_edges(
        "supervisor",
        route_from_supervisor,
        {RAG: "rag_agent", MCP: "mcp_tools", SYNTH: "synthesizer"},
    )
    builder.add_edge("rag_agent", "supervisor")
    builder.add_edge("mcp_tools", "supervisor")
    builder.add_edge("synthesizer", END)

    return builder.compile()
