"""Offline smoke test for the Document Analyst graph (Bonus A test target).

This is the target the Bonus A CI pipeline runs to prove the graph wires up
before any deploy. No Databricks, no network — everything is faked.

Run:  uv run pytest -q
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from langchain_core.messages import AIMessage  # noqa: E402

from agent.graph import build_graph  # noqa: E402
from agent.prompts import (  # noqa: E402
    MCP_STEP_PROMPT,
    PLANNER_PROMPT,
    RAG_EXTRACT_PROMPT,
    SUPERVISOR_PROMPT,
    SYNTHESIZER_PROMPT,
)


def test_graph_module_imports():
    """Minimal collection guard: the graph module must import cleanly."""
    from agent.graph import build_graph  # noqa: F401


class FakeDoc:
    def __init__(self, content, metadata):
        self.page_content = content
        self.metadata = metadata


class FakeRetriever:
    def invoke(self, query):
        return [
            FakeDoc(
                "Net revenue in FY2023 was 16.91 trillion yen.",
                {"source": "annual_report.pdf", "page": 4},
            )
        ]


class FakeTool:
    name = "calculate"

    async def ainvoke(self, args):
        return f"{args['expression']} = 21.3017"


class ScriptedLLM:
    """Routes a canned reply per node based on which system prompt it receives."""

    def __init__(self):
        self._tools = []

    def bind_tools(self, tools):
        self._tools = tools
        return self

    def invoke(self, messages):
        system = messages[0].content
        human = messages[1].content if len(messages) > 1 else ""

        if system == PLANNER_PROMPT:
            return AIMessage(
                content=(
                    '["Find Meridian net revenue for FY2023", '
                    '"Calculate 8% growth for 3 years on that revenue"]'
                )
            )
        if system == SUPERVISOR_PROMPT:
            if "growth" in human.lower() or "calculate" in human.lower():
                return AIMessage(content="mcp_tools")
            return AIMessage(content="rag_agent")
        if system == RAG_EXTRACT_PROMPT:
            return AIMessage(
                content="Net revenue FY2023: 16.91 trillion yen [source: annual_report.pdf, p.4]"
            )
        if system == MCP_STEP_PROMPT:
            tool = self._tools[0]
            return AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": tool.name,
                        "args": {"expression": "16.91*(1.08**3)"},
                        "id": "call_1",
                    }
                ],
            )
        if system == SYNTHESIZER_PROMPT:
            return AIMessage(
                content=(
                    "Revenue was 16.91T [source: annual_report.pdf, p.4]; "
                    "projected 21.30T after 8% growth for 3 years."
                )
            )
        return AIMessage(content="")


def test_graph_end_to_end_with_fakes():
    graph = build_graph(llm=ScriptedLLM(), retriever=FakeRetriever(), tools=[FakeTool()])
    result = graph.invoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": "What was revenue in FY2023 and after 8% growth for 3 years?",
                }
            ]
        }
    )

    assert len(result["plan"]) == 2
    assert len(result["step_results"]) == 2
    assert result["final_answer"]
    assert result["messages"][-1].content == result["final_answer"]
