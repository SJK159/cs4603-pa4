"""Python client SDK for the deployed Document Analyst (Part 3).

Talks to the raw MLflow `/invocations` contract (messages in, state dict
out) rather than the OpenAI-compatible `/chat/completions` gateway: the
`AnalystState` carries extra fields beyond `messages` (`plan`,
`step_results`, `final_answer`, ...), which the gateway does not recognize
as a pure chat completion (see DEPLOYMENT_GUIDE.md / Analysis.md for the
`resp[0].final_answer` vs. `resp.choices[0].message.content` discussion).
Hitting `/invocations` directly and parsing the state ourselves sidesteps
that gateway-shape mismatch entirely.
"""

from __future__ import annotations

import json
import os
import time
from collections.abc import Iterator

import httpx

_RETRYABLE_STATUS_CODES = {429, 503}


class AnalystClientError(Exception):
    """Raised for any non-retryable (or retry-exhausted) HTTP error."""

    def __init__(self, message: str, status_code: int | None = None, request_id: str | None = None):
        super().__init__(message)
        self.status_code = status_code
        self.request_id = request_id

    def __str__(self) -> str:
        detail = super().__str__()
        extra = ", ".join(
            f"{k}={v}"
            for k, v in (("status_code", self.status_code), ("request_id", self.request_id))
            if v is not None
        )
        return f"{detail} ({extra})" if extra else detail


class DocumentAnalystClient:
    def __init__(
        self,
        endpoint_name: str,
        host: str | None = None,
        token: str | None = None,
        timeout: float = 120.0,
        max_retries: int = 3,
    ) -> None:
        host = (host or os.environ.get("DATABRICKS_HOST") or "").rstrip("/")
        token = token or os.environ.get("DATABRICKS_TOKEN")
        if not host:
            raise ValueError("host not provided and DATABRICKS_HOST is not set")
        if not token:
            raise ValueError("token not provided and DATABRICKS_TOKEN is not set")

        self.endpoint_name = endpoint_name
        self.host = host
        self.token = token
        self.timeout = timeout
        self.max_retries = max_retries
        self._invocations_url = f"{self.host}/serving-endpoints/{endpoint_name}/invocations"

    def _headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self.token}", "Content-Type": "application/json"}

    def _payload(self, question: str) -> dict:
        return {"messages": [{"role": "user", "content": question}]}

    def _raise_for_status(self, response: httpx.Response) -> None:
        if response.status_code < 400:
            return
        request_id = response.headers.get("x-request-id")
        try:
            body = response.json()
            message = body.get("message") or body.get("error") or response.text
        except (ValueError, json.JSONDecodeError):
            message = response.text
        raise AnalystClientError(message, status_code=response.status_code, request_id=request_id)

    def ask(self, question: str) -> str:
        start = time.monotonic()
        attempt = 0
        while True:
            try:
                with httpx.Client(timeout=self.timeout) as client:
                    response = client.post(
                        self._invocations_url, headers=self._headers(), json=self._payload(question)
                    )
            except httpx.TimeoutException as exc:
                elapsed = time.monotonic() - start
                raise TimeoutError(
                    f"Request to endpoint '{self.endpoint_name}' timed out after {elapsed:.2f}s"
                ) from exc

            if response.status_code in _RETRYABLE_STATUS_CODES and attempt < self.max_retries:
                time.sleep(2**attempt)
                attempt += 1
                continue

            self._raise_for_status(response)
            return _extract_answer(response.json())

    def ask_streaming(self, question: str) -> Iterator[str]:
        start = time.monotonic()
        attempt = 0
        while True:
            try:
                with httpx.Client(timeout=self.timeout) as client, client.stream(
                    "POST",
                    self._invocations_url,
                    headers={**self._headers(), "Accept": "text/event-stream"},
                    json=self._payload(question),
                ) as response:
                    if response.status_code in _RETRYABLE_STATUS_CODES and attempt < self.max_retries:
                        response.close()
                        time.sleep(2**attempt)
                        attempt += 1
                        continue

                    self._raise_for_status(response)

                    if "text/event-stream" not in response.headers.get("content-type", ""):
                        # Not an SSE stream — a single complete state came back.
                        # Treat that as a valid outcome and yield it once.
                        body = json.loads(response.read())
                        yield _extract_answer(body)
                        return

                    yielded_any = False
                    for line in response.iter_lines():
                        if not line or not line.startswith("data:"):
                            continue
                        data = line[len("data:") :].strip()
                        if data == "[DONE]":
                            break
                        chunk = json.loads(data)
                        delta = _extract_delta(chunk)
                        if delta:
                            yielded_any = True
                            yield delta
                    if not yielded_any:
                        return
            except httpx.TimeoutException as exc:
                elapsed = time.monotonic() - start
                raise TimeoutError(
                    f"Request to endpoint '{self.endpoint_name}' timed out after {elapsed:.2f}s"
                ) from exc
            return

    def health_check(self) -> bool:
        try:
            from databricks.sdk import WorkspaceClient
            from databricks.sdk.service.serving import EndpointStateReady

            w = WorkspaceClient(host=self.host, token=self.token)
            status = w.serving_endpoints.get(self.endpoint_name)
            return bool(status.state and status.state.ready == EndpointStateReady.READY)
        except Exception:
            return False


def _extract_answer(body: dict) -> str:
    """Pull the final answer out of a raw `/invocations` response body.

    Handles the shapes MLflow's scoring server may return: the graph's state
    dict directly, that dict wrapped in `{"predictions": ...}`, or a
    single-item list of either.
    """
    state = body.get("predictions", body) if isinstance(body, dict) else body
    if isinstance(state, list):
        state = state[0] if state else {}

    final_answer = state.get("final_answer") if isinstance(state, dict) else None
    if final_answer:
        return final_answer

    messages = state.get("messages", []) if isinstance(state, dict) else []
    if messages:
        last = messages[-1]
        if isinstance(last, dict):
            return last.get("content", "")
        return str(last)

    return ""


def _extract_delta(chunk: dict) -> str:
    """Pull incremental text out of one SSE chunk, OpenAI-style if present."""
    choices = chunk.get("choices") or []
    if choices:
        delta = choices[0].get("delta") or choices[0].get("message") or {}
        content = delta.get("content")
        if content:
            return content
    return chunk.get("final_answer") or ""
