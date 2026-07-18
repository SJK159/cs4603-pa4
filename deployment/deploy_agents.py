"""Bonus B — deploy via the databricks-agents SDK (deployment/deploy_agents.py).

Reuses deploy.py's pinned pip_requirements and code_paths, and swaps the
final "create/update a serving endpoint by hand" step for a single
`agents.deploy()` call, which additionally auto-provisions a Review App for
human feedback. One real difference from deploy.py, discovered the hard
way: `agents.deploy()` runs a pre-flight schema check that Task 2.1's
`agent_model.py` (registering the raw graph, output = full AnalystState)
fails outright —

    ValueError: The model's schema is not compatible with Agent Framework.
    The output schema must be either ChatCompletionResponse or StringResponse.

— so this logs `agent_model_agents.py` (a ChatModel-wrapped version of the
exact same graph) instead of `agent_model.py`. See that file's docstring for
the full story. Everything else — the graph, the pinned dependency list, the
Vector Search/MCP wiring — is unchanged from Part 2.

Run:  uv run python deployment/deploy_agents.py
"""

from __future__ import annotations

import os

import mlflow
from databricks import agents

from config import get_settings
from deployment.deploy import (
    _REPO_ROOT,
    MODEL_NAME,
    PIP_REQUIREMENTS,
    SECRET_SCOPE,
    UC_CATALOG,
    UC_SCHEMA,
)

AGENTS_ENDPOINT_NAME = "27100159-document-analyst-agents"


def log_and_register_chat_model() -> tuple[str, str]:
    """Log agent_model_agents.py (the ChatModel wrapper) and register it in UC.

    Same registered model name as deploy.py's log_and_register() — this is
    still "the Document Analyst," just packaged for Agent Framework's
    stricter output-schema contract — so it lands as a new version of the
    same cs4603.default.document_analyst entry.
    """
    mlflow.set_registry_uri("databricks-uc")
    uc_model_name = f"{UC_CATALOG}.{UC_SCHEMA}.{MODEL_NAME}"

    with mlflow.start_run():
        model_info = mlflow.pyfunc.log_model(
            python_model=os.path.join(_REPO_ROOT, "deployment", "agent_model_agents.py"),
            name="agent",
            code_paths=[
                os.path.join(_REPO_ROOT, "agent"),
                os.path.join(_REPO_ROOT, "rag"),
                os.path.join(_REPO_ROOT, "tools"),
                os.path.join(_REPO_ROOT, "config.py"),
                os.path.join(_REPO_ROOT, "deployment", "stub_wheels"),
            ],
            pip_requirements=PIP_REQUIREMENTS,
            input_example={"messages": [{"role": "user", "content": "What was the revenue?"}]},
        )

    registered = mlflow.register_model(model_info.model_uri, uc_model_name)
    print(f"Registered model: {uc_model_name}, version {registered.version}")
    return uc_model_name, registered.version


def main() -> None:
    uc_name, version = log_and_register_chat_model()

    settings = get_settings()
    deployment = agents.deploy(
        model_name=uc_name,
        model_version=int(version),
        endpoint_name=AGENTS_ENDPOINT_NAME,
        scale_to_zero=True,
        environment_vars={
            # Secrets — never plaintext, same secret scope deploy.py uses.
            "DATABRICKS_HOST": f"{{{{secrets/{SECRET_SCOPE}/DATABRICKS_HOST}}}}",
            "DATABRICKS_TOKEN": f"{{{{secrets/{SECRET_SCOPE}/DATABRICKS_TOKEN}}}}",
            "DATABRICKS_MODEL": f"{{{{secrets/{SECRET_SCOPE}/DATABRICKS_MODEL}}}}",
            # Not secrets — the retriever needs these to reach the index.
            "VECTOR_SEARCH_ENDPOINT": settings["vs_endpoint"],
            "VECTOR_SEARCH_INDEX": settings["vs_index"],
            "EMBEDDINGS_ENDPOINT": settings["embeddings"],
        },
    )

    print(f"Endpoint: {deployment.endpoint_name}")
    print(f"Endpoint URL: {deployment.endpoint_url}")
    print(f"Review App URL: {deployment.review_app_url}")


if __name__ == "__main__":
    main()
