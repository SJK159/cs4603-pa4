"""Supervisor node + routing edge (Task 1.3)."""

from __future__ import annotations

from langchain_core.messages import HumanMessage, SystemMessage

from agent.prompts import SUPERVISOR_PROMPT
from agent.state import AnalystState

RAG = "rag_agent"
MCP = "mcp_tools"
SYNTH = "synthesizer"


def _classify_step(llm, step: str) -> str:
    """Ask the LLM to route `step` to RAG or MCP; keyword-match the reply."""
    response = llm.invoke(
        [
            SystemMessage(content=SUPERVISOR_PROMPT),
            HumanMessage(content=step),
        ]
    )
    reply = response.content.strip().lower()
    if MCP in reply:
        return MCP
    if RAG in reply:
        return RAG
    # Ambiguous reply: keyword-match the step text itself as a defensive
    # fallback so a malformed LLM reply doesn't crash routing.
    calc_keywords = ("calculat", "growth", "percent", "compar", "convert", "%")
    if any(kw in step.lower() for kw in calc_keywords):
        return MCP
    return RAG


def make_supervisor(llm):
    def supervisor(state: AnalystState) -> dict:
        plan = state["plan"]
        index = state["current_step_index"]
        if index >= len(plan):
            return {"next_agent": SYNTH}
        step = plan[index]
        return {"next_agent": _classify_step(llm, step)}

    return supervisor


def route_from_supervisor(state: AnalystState) -> str:
    return state["next_agent"]
