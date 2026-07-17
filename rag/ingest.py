"""Corpus ingestion into Databricks Vector Search (Task 0.3 / rag/ingest.py).

Run inside a Databricks notebook or job (needs a live Spark session plus the
ai_parse_document/ai_prep_search SQL functions and databricks-vectorsearch).
This mirrors the pipeline worked out and verified interactively against the
real workspace before being ported here.
"""

from __future__ import annotations

import os
import time

from config import get_settings


def build_chunks_table(spark, volume_path: str, chunks_table: str) -> None:
    """Parse `volume_path` and chunk it into the Delta table `chunks_table`.

    Produces columns chunk_id, chunk_to_retrieve, chunk_to_embed, source,
    page, and enables Change Data Feed (required for the Delta Sync index
    to stay in sync with this table).
    """
    parsed_df = spark.sql(f"""
        SELECT ai_parse_document(content) AS parsed
        FROM READ_FILES('{volume_path}', format => 'binaryFile')
    """)
    parsed_df.createOrReplaceTempView("parsed_docs")

    # ai_parse_document/ai_prep_search return VARIANT, not ARRAY, so the
    # chunk array must be cast explicitly before EXPLODE can iterate it.
    spark.sql(f"""
        CREATE OR REPLACE TABLE {chunks_table} AS
        SELECT
            chunk_value:chunk_id::string          AS chunk_id,
            chunk_value:chunk_to_retrieve::string AS chunk_to_retrieve,
            chunk_value:chunk_to_embed::string    AS chunk_to_embed,
            'annual_report.pdf'                   AS source,
            chunk_value:pages[0]:page_id::int     AS page
        FROM parsed_docs
        LATERAL VIEW EXPLODE(
            CAST(ai_prep_search(parsed):document:contents AS ARRAY<VARIANT>)
        ) AS chunk_value
    """)

    spark.sql(f"ALTER TABLE {chunks_table} SET TBLPROPERTIES (delta.enableChangeDataFeed = true)")

    row_count = spark.table(chunks_table).count()
    if row_count == 0:
        raise ValueError(f"No chunks were produced in {chunks_table} — check the parsed PDF.")

    null_count = spark.sql(f"""
        SELECT count(*) AS n FROM {chunks_table}
        WHERE chunk_id IS NULL OR chunk_to_retrieve IS NULL
    """).collect()[0]["n"]
    if null_count:
        raise ValueError(
            f"{null_count} rows in {chunks_table} have NULL chunk_id/chunk_to_retrieve."
        )


def create_index() -> None:
    """Create (or reuse) a STANDARD Vector Search endpoint and a TRIGGERED
    Delta Sync index over SOURCE_TABLE, then wait for it to become ready.

    Reads VECTOR_SEARCH_ENDPOINT / VECTOR_SEARCH_INDEX / EMBEDDINGS_ENDPOINT
    from config.get_settings() and SOURCE_TABLE (the table build_chunks_table
    just created) from the environment.
    """
    from databricks.vector_search.client import VectorSearchClient

    settings = get_settings()
    source_table = os.environ.get("SOURCE_TABLE")
    if not source_table or not settings["vs_endpoint"] or not settings["vs_index"]:
        raise OSError(
            "Missing SOURCE_TABLE / VECTOR_SEARCH_ENDPOINT / VECTOR_SEARCH_INDEX. "
            "Set them in your .env before running create_index()."
        )

    vsc = VectorSearchClient()

    existing_endpoints = [e["name"] for e in vsc.list_endpoints().get("endpoints", [])]
    if settings["vs_endpoint"] not in existing_endpoints:
        vsc.create_endpoint(name=settings["vs_endpoint"], endpoint_type="STANDARD")

    while True:
        status = vsc.get_endpoint(settings["vs_endpoint"])
        if status.get("endpoint_status", {}).get("state") == "ONLINE":
            break
        time.sleep(30)

    existing_indexes = [
        i["name"] for i in vsc.list_indexes(settings["vs_endpoint"]).get("vector_indexes", [])
    ]
    if settings["vs_index"] not in existing_indexes:
        # A freshly created TRIGGERED index runs its initial sync
        # automatically — no manual .sync() call needed here.
        vsc.create_delta_sync_index(
            endpoint_name=settings["vs_endpoint"],
            source_table_name=source_table,
            index_name=settings["vs_index"],
            pipeline_type="TRIGGERED",
            primary_key="chunk_id",
            embedding_source_column="chunk_to_retrieve",
            embedding_model_endpoint_name=settings["embeddings"],
        )

    while True:
        desc = vsc.get_index(
            endpoint_name=settings["vs_endpoint"],
            index_name=settings["vs_index"],
        ).describe()
        if desc.get("status", {}).get("ready", False):
            break
        time.sleep(30)
