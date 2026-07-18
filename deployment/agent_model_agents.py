"""ChatModel adapter around the Document Analyst graph, for Bonus B.

`agents.deploy()` (the `databricks-agents` SDK) runs a pre-flight schema
check before it will provision anything: the logged model's *inferred*
output schema must be either `ChatCompletionResponse`- or
`StringResponse`-shaped. Task 2.1's `agent_model.py` registers the compiled
LangGraph object directly (`mlflow.models.set_model(graph)`), whose output is
the full `AnalystState` — `messages`, `plan`, `current_step_index`,
`step_results`, `next_agent`, `final_answer` — which is richer than Agent
Framework's contract allows and gets rejected outright:

    ValueError: The model's schema is not compatible with Agent Framework.
    The output schema must be either ChatCompletionResponse or StringResponse.

This file does not change the graph or Part 2's deployment path at all — it
wraps the *same* `agent.graph.build_graph()` in an `mlflow.pyfunc.ChatModel`,
translating between the two contracts (`ChatMessage` list in,
`ChatCompletionResponse` out) so `agents.deploy()`'s schema check passes,
while every actual planning/routing/retrieval/tool-calling decision is still
made by the unmodified graph.

Must import cleanly:  python -c "import deployment.agent_model_agents"
"""

from __future__ import annotations

import mlflow
from mlflow.pyfunc import ChatModel
from mlflow.types.llm import ChatChoice, ChatCompletionResponse, ChatMessage, ChatParams

from config import get_settings

# Same import-time validation as agent_model.py — fail loudly in serving
# logs rather than with a cryptic DEPLOYMENT_FAILED.
get_settings()

from agent.graph import build_graph  # noqa: E402


class DocumentAnalystChatModel(ChatModel):
    def load_context(self, context) -> None:
        # Built once per container lifetime, exactly like agent_model.py —
        # not per-request — for the same reasons Task 1.5 loads MCP tools
        # once at graph-build time.
        self.graph = build_graph()

    def predict(
        self, messages: list[ChatMessage], params: ChatParams
    ) -> ChatCompletionResponse:
        result = self.graph.invoke(
            {"messages": [{"role": m.role, "content": m.content} for m in messages]}
        )
        answer = result["final_answer"]
        return ChatCompletionResponse(
            choices=[ChatChoice(index=0, message=ChatMessage(role="assistant", content=answer))],
            model="document-analyst",
        )


model = DocumentAnalystChatModel()
mlflow.models.set_model(model)
