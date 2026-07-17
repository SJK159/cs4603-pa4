"""RAG agent node (Task 1.4) — retrieves from Databricks Vector Search."""

from __future__ import annotations

from langchain_core.messages import HumanMessage, SystemMessage

from agent.prompts import RAG_EXTRACT_PROMPT
from agent.state import AnalystState

NOT_FOUND = "Not found in documents."


def format_docs(docs) -> str:
    """Format retrieved docs into a citation-tagged context block.

    Each doc becomes:
        [source: <file>, p.<page>]
        <chunk text>
    """
    blocks = []
    for doc in docs:
        source = doc.metadata.get("source", "unknown")
        page = doc.metadata.get("page", "?")
        if isinstance(page, float) and page.is_integer():
            page = int(page)
        blocks.append(f"[source: {source}, p.{page}]\n{doc.page_content}")
    return "\n\n".join(blocks)


def make_rag_agent(retriever, llm):
    def rag_agent(state: AnalystState) -> dict:
        step = state["plan"][state["current_step_index"]]
        docs = retriever.invoke(step)

        if not docs:
            fact = NOT_FOUND
        else:
            context = format_docs(docs)
            response = llm.invoke(
                [
                    SystemMessage(content=RAG_EXTRACT_PROMPT),
                    HumanMessage(content=f"Step: {step}\n\nExcerpts:\n{context}"),
                ]
            )
            fact = response.content

        return {
            "step_results": [*state["step_results"], fact],
            "current_step_index": state["current_step_index"] + 1,
        }

    return rag_agent
