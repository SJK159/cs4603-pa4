"""Vector Search retriever factory (Task 1.4 support / rag/store.py).

Returns a LangChain retriever over the Databricks Vector Search index built
by `ingest.py` (Task 0.3). This exact retriever is reused by both local runs
and the deployed model — the index is a managed Databricks service reachable
with DATABRICKS_HOST/DATABRICKS_TOKEN, so there's no separate embedding path
for deployment.
"""

from __future__ import annotations

from config import get_settings

CITATION_COLUMNS = ["chunk_id", "source", "page"]


def get_vector_store():
    """Return a DatabricksVectorSearch handle over the Task 0.3 index.

    The index was created with managed embeddings (an
    `embedding_model_endpoint_name`) and a configured source column
    (`chunk_to_retrieve`), so no local `Embeddings` object or `text_column`
    is passed here — Databricks infers the text column from the index
    config and embeds the query server-side.
    """
    from databricks_langchain import DatabricksVectorSearch

    settings = get_settings()
    if not settings["vs_endpoint"] or not settings["vs_index"]:
        raise OSError(
            "Missing VECTOR_SEARCH_ENDPOINT / VECTOR_SEARCH_INDEX. "
            "Set them in your .env (local) or the endpoint's environment_vars (deployed)."
        )

    return DatabricksVectorSearch(
        endpoint=settings["vs_endpoint"],
        index_name=settings["vs_index"],
        columns=CITATION_COLUMNS,
    )


def get_retriever(k: int = 4):
    """Return a top-k retriever over the Task 0.3 Vector Search index."""
    return get_vector_store().as_retriever(search_kwargs={"k": k})
