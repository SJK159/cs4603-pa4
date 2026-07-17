# Databricks notebook source
# MAGIC %md
# MAGIC # PA4 — Task 0.3: Ingest `annual_report.pdf` into Databricks Vector Search
# MAGIC
# MAGIC Ad-hoc exploration notebook. Run cell by cell, checking output before moving on.
# MAGIC Once everything below works end-to-end, we'll port the working code into
# MAGIC `rag/ingest.py` (`build_chunks_table` / `create_index`) as the graded deliverable.
# MAGIC
# MAGIC **Before running:** fill in the `<your-name>` placeholders in the widgets cell below.

# COMMAND ----------

# MAGIC %md ## 0. Config — EDIT THESE

# COMMAND ----------

CATALOG = "cs4603"
SCHEMA = "default"

YOUR_NAME = "27100159"  # matches the notebook already run in Databricks

VOLUME_PATH = f"/Volumes/{CATALOG}/{SCHEMA}/pa4/annual_report.pdf"
CHUNKS_TABLE = f"{CATALOG}.{SCHEMA}.{YOUR_NAME}_analyst_chunks"
VS_ENDPOINT = f"{YOUR_NAME}-vs-endpoint"
VS_INDEX = f"{CATALOG}.{SCHEMA}.{YOUR_NAME}_analyst_index"
EMBEDDINGS_ENDPOINT = "databricks-gte-large-en"

print(f"Volume path:   {VOLUME_PATH}")
print(f"Chunks table:  {CHUNKS_TABLE}")
print(f"VS endpoint:   {VS_ENDPOINT}")
print(f"VS index:      {VS_INDEX}")

# COMMAND ----------

# MAGIC %md ## 1. Create the volume (if needed) and upload the PDF
# MAGIC
# MAGIC If the volume doesn't exist yet, create it, then upload `annual_report.pdf`
# MAGIC either via the Catalog UI (drag-and-drop) or `dbutils.fs.cp` below.

# COMMAND ----------

spark.sql(f"CREATE VOLUME IF NOT EXISTS {CATALOG}.{SCHEMA}.pa4")

# COMMAND ----------

# MAGIC %md
# MAGIC Upload the file, then confirm it landed correctly:

# COMMAND ----------

display(dbutils.fs.ls(f"/Volumes/{CATALOG}/{SCHEMA}/pa4/"))

# COMMAND ----------

# MAGIC %md ## 2. Parse the PDF with `ai_parse_document`

# COMMAND ----------

parsed_df = spark.sql(f"""
    SELECT ai_parse_document(content) AS parsed
    FROM READ_FILES('{VOLUME_PATH}', format => 'binaryFile')
""")
parsed_df.createOrReplaceTempView("parsed_docs")
parsed_df.printSchema()

# COMMAND ----------

# MAGIC %md
# MAGIC **Inspect the parsed output before chunking** — the exact field names inside
# MAGIC `ai_prep_search`'s result depend on the DBR/SQL functions version. Run this and
# MAGIC look at the schema before writing the chunking query in the next cell.

# COMMAND ----------

prep_preview = spark.sql("SELECT ai_prep_search(parsed) AS prepped FROM parsed_docs")
prep_preview.printSchema()
prep_preview.show(2, truncate=80)

# COMMAND ----------

# MAGIC %md ## 3. Chunk with `ai_prep_search` into a Delta table
# MAGIC
# MAGIC Adjust the `chunk.xxx` field references below to match whatever schema you saw
# MAGIC printed above — this is the part most likely to need tweaking.

# COMMAND ----------

spark.sql(f"""
    CREATE OR REPLACE TABLE {CHUNKS_TABLE} AS
    SELECT
        chunk_value:chunk_id::string          AS chunk_id,
        chunk_value:chunk_to_retrieve::string AS chunk_to_retrieve,
        chunk_value:chunk_to_embed::string    AS chunk_to_embed,
        'annual_report.pdf'                   AS source,
        chunk_value:pages[0]:page_id::int     AS page
    FROM parsed_docs
    LATERAL VIEW EXPLODE(CAST(ai_prep_search(parsed):document:contents AS ARRAY<VARIANT>)) AS chunk_value
""")

spark.sql(f"ALTER TABLE {CHUNKS_TABLE} SET TBLPROPERTIES (delta.enableChangeDataFeed = true)")

display(spark.table(CHUNKS_TABLE))

# COMMAND ----------

# MAGIC %md
# MAGIC Sanity checks before moving on:

# COMMAND ----------

n = spark.table(CHUNKS_TABLE).count()
print(f"Row count: {n}")
assert n > 0, "No chunks were produced — check the ai_parse_document / ai_prep_search output above."

nulls = spark.sql(f"""
    SELECT count(*) AS n
    FROM {CHUNKS_TABLE}
    WHERE chunk_id IS NULL OR chunk_to_retrieve IS NULL
""").collect()[0]["n"]
print(f"Rows with NULL chunk_id/chunk_to_retrieve: {nulls}")
assert nulls == 0, "Found NULL chunk_id or chunk_to_retrieve — fix the chunking query above."

# COMMAND ----------

# MAGIC %md ## 4. Create the Vector Search endpoint

# COMMAND ----------

# MAGIC %pip install databricks-vectorsearch

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %md
# MAGIC `restartPython()` wipes all Python variables (Spark tables/views survive fine).
# MAGIC Re-declare the config before continuing.

# COMMAND ----------

CATALOG = "cs4603"
SCHEMA = "default"

YOUR_NAME = "27100159"  # matches the notebook already run in Databricks

VOLUME_PATH = f"/Volumes/{CATALOG}/{SCHEMA}/pa4/annual_report.pdf"
CHUNKS_TABLE = f"{CATALOG}.{SCHEMA}.{YOUR_NAME}_analyst_chunks"
VS_ENDPOINT = f"{YOUR_NAME}-vs-endpoint"
VS_INDEX = f"{CATALOG}.{SCHEMA}.{YOUR_NAME}_analyst_index"
EMBEDDINGS_ENDPOINT = "databricks-gte-large-en"

# COMMAND ----------

from databricks.vector_search.client import VectorSearchClient

vsc = VectorSearchClient()

existing_endpoints = [e["name"] for e in vsc.list_endpoints().get("endpoints", [])]
if VS_ENDPOINT not in existing_endpoints:
    vsc.create_endpoint(name=VS_ENDPOINT, endpoint_type="STANDARD")
else:
    print(f"Endpoint {VS_ENDPOINT} already exists, skipping creation.")

# COMMAND ----------

# MAGIC %md
# MAGIC Poll until the endpoint is online (takes a few minutes on first creation):

# COMMAND ----------

import time

while True:
    status = vsc.get_endpoint(VS_ENDPOINT)
    state = status.get("endpoint_status", {}).get("state", "UNKNOWN")
    print(state)
    if state == "ONLINE":
        break
    time.sleep(30)

# COMMAND ----------

# MAGIC %md ## 5. Create the Delta Sync index with managed embeddings

# COMMAND ----------

existing_indexes = [i["name"] for i in vsc.list_indexes(VS_ENDPOINT).get("vector_indexes", [])]
if VS_INDEX not in existing_indexes:
    index = vsc.create_delta_sync_index(
        endpoint_name=VS_ENDPOINT,
        source_table_name=CHUNKS_TABLE,
        index_name=VS_INDEX,
        pipeline_type="TRIGGERED",
        primary_key="chunk_id",
        embedding_source_column="chunk_to_retrieve",
        embedding_model_endpoint_name=EMBEDDINGS_ENDPOINT,
    )
else:
    print(f"Index {VS_INDEX} already exists, fetching handle.")
    index = vsc.get_index(endpoint_name=VS_ENDPOINT, index_name=VS_INDEX)

# COMMAND ----------

# MAGIC %md ## 6. Wait for the index to reach `READY` / `ONLINE`
# MAGIC
# MAGIC A freshly created `TRIGGERED` index runs its **initial sync automatically** as
# MAGIC part of provisioning — it starts in `PROVISIONING` and isn't ready to accept a
# MAGIC manual `.sync()` call yet. Just wait here; don't call `.sync()` on a brand new
# MAGIC index.

# COMMAND ----------

while True:
    desc = vsc.get_index(endpoint_name=VS_ENDPOINT, index_name=VS_INDEX).describe()
    state = desc.get("status", {}).get("detailed_state", "UNKNOWN")
    ready = desc.get("status", {}).get("ready", False)
    print(state, "| ready =", ready)
    if ready:
        break
    time.sleep(30)

# COMMAND ----------

# MAGIC %md
# MAGIC **Only needed later** — if you rebuild `CHUNKS_TABLE` after the index is already
# MAGIC `READY`, re-run this cell to push the changes (a `TRIGGERED` pipeline does not
# MAGIC auto-sync on its own after the initial creation):
# MAGIC
# MAGIC ```python
# MAGIC index.sync()
# MAGIC ```

# COMMAND ----------

# MAGIC %md ## 7. Similarity search smoke test

# COMMAND ----------

results = vsc.get_index(endpoint_name=VS_ENDPOINT, index_name=VS_INDEX).similarity_search(
    query_text="What was the net revenue in fiscal year 2023?",
    columns=["chunk_id", "chunk_to_retrieve", "source", "page"],
    num_results=3,
)
results

# COMMAND ----------

# MAGIC %md
# MAGIC If that returned relevant chunks with sensible `source`/`page` metadata, Task 0.3
# MAGIC is done. Copy these values into your local `.env`:
# MAGIC
# MAGIC ```
# MAGIC VECTOR_SEARCH_ENDPOINT=<VS_ENDPOINT printed above>
# MAGIC VECTOR_SEARCH_INDEX=<VS_INDEX printed above>
# MAGIC SOURCE_TABLE=<CHUNKS_TABLE printed above>
# MAGIC ```
