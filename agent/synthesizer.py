"""Synthesizer node (Task 1.6)."""

from __future__ import annotations

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from agent.prompts import SYNTHESIZER_PROMPT
from agent.state import AnalystState


def _original_question(state: AnalystState) -> str:
    first = state["messages"][0]
    return first.content if hasattr(first, "content") else first["content"]


def make_synthesizer(llm):
    def synthesizer(state: AnalystState) -> dict:
        question = _original_question(state)
        step_results = state["step_results"]

        results_block = "\n".join(
            f"Step {i}: {result}" for i, result in enumerate(step_results, start=1)
        )
        user_content = (
            f"Original question: {question}\n\nStep results:\n{results_block}"
        )

        response = llm.invoke(
            [
                SystemMessage(content=SYNTHESIZER_PROMPT),
                HumanMessage(content=user_content),
            ]
        )
        answer = response.content

        return {
            "final_answer": answer,
            "messages": [AIMessage(content=answer)],
        }

    return synthesizer
