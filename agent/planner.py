"""Planner node (Task 1.2)."""

from __future__ import annotations

import json
import re

from langchain_core.messages import HumanMessage, SystemMessage

from agent.prompts import PLANNER_PROMPT
from agent.state import AnalystState


def _last_user_question(state: AnalystState) -> str:
    messages = state["messages"]
    last = messages[-1]
    content = last.content if hasattr(last, "content") else last["content"]
    return content


def _parse_plan(raw: str, fallback_question: str) -> list[str]:
    """Parse the planner LLM's response into a list of step strings.

    Falls back to a single step (the original question) if the response
    isn't valid JSON or isn't a non-empty list of strings.
    """
    text = raw.strip()
    # Strip markdown code fences if the model added them anyway.
    fence_match = re.search(r"```(?:json)?\s*(.*?)\s*```", text, re.DOTALL)
    if fence_match:
        text = fence_match.group(1).strip()

    try:
        parsed = json.loads(text)
        steps = [str(s).strip() for s in parsed if str(s).strip()]
        if steps:
            return steps
    except (json.JSONDecodeError, TypeError):
        pass

    return [fallback_question]


def make_planner(llm):
    def planner(state: AnalystState) -> dict:
        question = _last_user_question(state)
        response = llm.invoke(
            [
                SystemMessage(content=PLANNER_PROMPT),
                HumanMessage(content=question),
            ]
        )
        plan = _parse_plan(response.content, fallback_question=question)
        return {"plan": plan, "current_step_index": 0, "step_results": []}

    return planner
