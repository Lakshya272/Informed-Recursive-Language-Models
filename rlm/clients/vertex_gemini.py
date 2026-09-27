"""Vertex AI Gemini client using Google's OpenAI-compatible Chat Completions API.

This module keeps the RLM-facing ``BaseLM`` contract unchanged, but delegates
message conversion and response parsing to the official ``openai`` package.
Vertex's OpenAI-compatible endpoint is different from the native
``generateContent`` endpoint: it uses an OAuth access token and a project/
location-scoped ``endpoints/openapi`` URL.
"""

from __future__ import annotations

import asyncio
import os
import threading
import time
from collections import defaultdict
from typing import Any

import httpx
import openai
from dotenv import load_dotenv

from rlm.clients.base_lm import BaseLM
from rlm.core.types import ModelUsageSummary, UsageSummary

load_dotenv()

DEFAULT_VERTEX_PROJECT_ID = os.getenv("VERTEX_PROJECT_ID", "dark-bindery-507417-t6")
DEFAULT_VERTEX_LOCATION = os.getenv("VERTEX_LOCATION", "global")
DEFAULT_VERTEX_MODEL = "gemini-3.8-flash"
DEFAULT_VERTEX_ACCESS_TOKEN = os.getenv("VERTEX_ACCESS_TOKEN") or os.getenv(
    "VERTEX_OPENAI_API_KEY"
)
VERTEX_CLOUD_PLATFORM_SCOPE = "https://www.googleapis.com/auth/cloud-platform"

# Gemini 3.8 Flash published rates for this project (USD per 1M tokens).
INPUT_USD_PER_MILLION = 0.75
OUTPUT_USD_PER_MILLION = 3.75

VALID_THINKING_LEVELS = frozenset({"low", "medium", "high"})
DEFAULT_MALFORMED_RETRIES = 2
DEFAULT_MALFORMED_RETRY_DELAY = 0.5


def vertex_openai_base_url(
    project_id: str, location: str, api_version: str = "v1"
) -> str:
    """Return Vertex's official OpenAI-compatible Chat Completions base URL."""
    host = (
        "aiplatform.googleapis.com"
        if location == "global"
        else f"{location}-aiplatform.googleapis.com"
    )
    return (
        f"https://{host}/{api_version}/projects/{project_id}/locations/{location}/"
        "endpoints/openapi"
    )


def vertex_openai_model(model: str) -> str:
    """Vertex's OpenAI-compatible endpoint uses the ``google/`` model prefix."""
    return model if model.startswith("google/") else f"google/{model}"


def _as_int(value: Any) -> int:
    if value is None:
        return 0
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _field(value: Any, name: str, default: Any = None) -> Any:
    if isinstance(value, dict):
        return value.get(name, default)
    return getattr(value, name, default)


def parse_usage_metadata(usage: Any) -> dict[str, Any]:
    """Normalize OpenAI-compatible usage into the repository's usage schema."""
    prompt = _as_int(_field(usage, "prompt_tokens"))
    completion = _as_int(_field(usage, "completion_tokens"))
    total = _as_int(_field(usage, "total_tokens"))
    details = _field(usage, "completion_tokens_details", {}) or {}
    thoughts = _as_int(_field(details, "reasoning_tokens"))
    if total <= 0:
        total = prompt + completion
    prompt_details = _field(usage, "prompt_tokens_details", {}) or {}
    cached = _as_int(_field(prompt_details, "cached_tokens"))
    return {
        "prompt_tokens": prompt,
        "candidates_tokens": max(0, completion - thoughts),
        "thoughts_tokens": thoughts,
        "thoughts_source": "completion_tokens_details.reasoning_tokens"
        if thoughts > 0
        else "none",
        "cached_tokens": cached,
        "total_tokens": total,
        # OpenAI-compatible completion_tokens already includes reasoning tokens
        # when the provider reports them, so do not add thoughts a second time.
        "output_tokens": completion,
    }


def cost_usd(prompt_tokens: int, output_tokens: int) -> float:
    return (prompt_tokens / 1_000_000) * INPUT_USD_PER_MILLION + (
        output_tokens / 1_000_000
    ) * OUTPUT_USD_PER_MILLION


def _response_dump(response: Any) -> Any:
    try:
        return response.model_dump()
    except AttributeError:
        return repr(response)


def extract_candidate_text(response: Any) -> str:
    """Extract the assistant text returned by Chat Completions."""
    choices = _field(response, "choices", []) or []
    if not choices:
        raise ValueError(
            "Vertex OpenAI-compatible response had no choices: "
            f"{_response_dump(response)}"
        )
    choice = choices[0]
    message = _field(choice, "message", {}) or {}
    content = _field(message, "content")
    if content:
        return str(content).strip()
    finish = _field(choice, "finish_reason")
    if finish and str(finish).upper() == "MALFORMED_FUNCTION_CALL":
        raise ValueError(
            "Vertex OpenAI-compatible Chat Completions finished with "
            f"{finish}: {_response_dump(response)}"
        )
    raise ValueError(
        "Vertex OpenAI-compatible response had no text content: "
        f"{_response_dump(response)}"
    )


def _is_retryable_malformed_response(response: Any) -> bool:
    choices = _field(response, "choices", []) or []
    if not choices:
        return False
    finish = _field(choices[0], "finish_reason")
    return str(finish or "").upper() == "MALFORMED_FUNCTION_CALL"


class VertexGeminiClient(BaseLM):
    """Drop-in BaseLM using the official Vertex OpenAI-compatible endpoint."""

    def __init__(
        self,
        api_key: str | None = None,
        model_name: str | None = None,
        project_id: str | None = None,
        location: str | None = None,
        thinking_level: str | None = None,
        malformed_retries: int = DEFAULT_MALFORMED_RETRIES,
        malformed_retry_delay: float = DEFAULT_MALFORMED_RETRY_DELAY,
        api_version: str = "v1",
        sampling_args: dict[str, Any] | None = None,
        **kwargs,
    ):
        super().__init__(
            model_name=model_name or DEFAULT_VERTEX_MODEL,
            sampling_args=sampling_args,
            **kwargs,
        )
        self.project_id = project_id or DEFAULT_VERTEX_PROJECT_ID
        self.location = location or DEFAULT_VERTEX_LOCATION
        self.model_name = model_name or DEFAULT_VERTEX_MODEL
        self.base_url = vertex_openai_base_url(self.project_id, self.location, api_version)

        level = thinking_level or (self.sampling_args or {}).get("thinking_level") or "low"
        level = str(level).lower()
        if level not in VALID_THINKING_LEVELS:
            raise ValueError(
                f"thinking_level must be one of {sorted(VALID_THINKING_LEVELS)}, got {level!r}"
            )
        self.thinking_level = level
        self.malformed_retries = max(0, int(malformed_retries))
        self.malformed_retry_delay = max(0.0, float(malformed_retry_delay))

        # An explicit token is useful for a short smoke test or API-key based auth.
        # If api_key is explicitly passed, use it. Otherwise, default to ADC (OAuth)
        # to avoid picking up mismatched API keys from ambient .env files.
        self._static_access_token = api_key

        self._credentials: Any | None = None
        self._credential_lock = threading.Lock()
        self._client: openai.OpenAI | None = None
        self._client_token: str | None = None

        self.model_call_counts: dict[str, int] = defaultdict(int)
        self.model_input_tokens: dict[str, int] = defaultdict(int)
        self.model_output_tokens: dict[str, int] = defaultdict(int)
        self.model_thought_tokens: dict[str, int] = defaultdict(int)
        self.model_total_tokens: dict[str, int] = defaultdict(int)
        self.model_costs: dict[str, float] = defaultdict(float)
        self.last_prompt_tokens = 0
        self.last_completion_tokens = 0
        self.last_thought_tokens = 0
        self.last_thoughts_source: str | None = None
        self.last_cost: float | None = None
        self.last_model_version: str | None = None
        self.last_raw_usage: dict[str, Any] = {}
        self._last_logical_call_count = 1

    def _get_access_token(self) -> str:
        if self._static_access_token:
            return self._static_access_token
        with self._credential_lock:
            now = time.time()
            if hasattr(self, "_cached_gcloud_token") and self._cached_gcloud_token and (now - getattr(self, "_gcloud_token_time", 0)) < 2400:
                return self._cached_gcloud_token
            try:
                import subprocess
                res = subprocess.run(
                    ["gcloud", "auth", "print-access-token"],
                    capture_output=True,
                    text=True,
                    shell=True,
                )
                tok = res.stdout.strip()
                if tok and tok.startswith("ya29."):
                    self._cached_gcloud_token = tok
                    self._gcloud_token_time = now
                    return tok
            except Exception:
                pass

            if self._credentials is None:
                try:
                    import google.auth
                    from google.auth.transport.requests import Request

                    self._credentials, _ = google.auth.default(
                        scopes=[VERTEX_CLOUD_PLATFORM_SCOPE]
                    )
                    self._credential_request = Request()
                except Exception as exc:
                    raise ValueError(
                        "Vertex OpenAI-compatible Chat Completions needs either "
                        "VERTEX_ACCESS_TOKEN/VERTEX_OPENAI_API_KEY or Google ADC. "
                        "Run `gcloud auth application-default login` for ADC."
                    ) from exc
            if not getattr(self._credentials, "token", None) or getattr(
                self._credentials, "expired", True
            ):
                self._credentials.refresh(self._credential_request)
            if not self._credentials.token:
                raise ValueError("Google credentials did not provide an access token.")
            return str(self._credentials.token)

    def _get_client(self) -> openai.OpenAI:
        token = self._get_access_token()
        is_api_key = token and not token.startswith("ya29.")
        with self._credential_lock:
            if self._client is None or self._client_token != token:
                timeout = httpx.Timeout(
                    timeout=self.timeout if isinstance(self.timeout, (int, float)) else 180.0,
                    connect=15.0,
                    read=180.0,
                    write=30.0,
                    pool=10.0,
                )
                if is_api_key:
                    def _strip_auth(request):
                        request.headers.pop("authorization", None)

                    http_client = httpx.Client(
                        headers={"X-Goog-Api-Key": token},
                        event_hooks={"request": [_strip_auth]},
                    )
                    self._client = openai.OpenAI(
                        api_key="none",
                        base_url=self.base_url,
                        timeout=timeout,
                        max_retries=3,
                        http_client=http_client,
                    )
                else:
                    self._client = openai.OpenAI(
                        api_key=token,
                        base_url=self.base_url,
                        timeout=timeout,
                        max_retries=3,
                    )
                self._client_token = token
            return self._client

    def _request_kwargs(self, model: str | None = None) -> dict[str, Any]:
        target_model = model or self.model_name
        args = dict(self.sampling_args or {})
        args.pop("thinking_level", None)
        extra_body = dict(args.pop("extra_body", {}) or {})
        google_body = dict(extra_body.get("google", {}) or {})
        # Inject thinking parameters based on model family
        if self.thinking_level:
            thinking_config = dict(google_body.get("thinking_config", {}) or {})
            if "2.5" in str(target_model):
                budget_map = {
                    "low": 4096,
                    "medium": 8192,
                    "high": 16384,
                }
                thinking_config["thinking_budget"] = budget_map.get(self.thinking_level, 4096)
            elif not any(v in str(target_model) for v in ["1.5", "2.0"]):
                thinking_config["thinking_level"] = self.thinking_level
            google_body["thinking_config"] = thinking_config
        if google_body:
            extra_body["google"] = google_body
        if extra_body:
            args["extra_body"] = extra_body
        return {key: value for key, value in args.items() if value is not None}

    def completion(self, prompt: str | list[dict[str, Any]], model: str | None = None) -> str:
        model = model or self.model_name
        if not model:
            raise ValueError("Model name is required for Vertex Gemini client.")
        if isinstance(prompt, str):
            messages = [{"role": "user", "content": prompt}]
        elif isinstance(prompt, list) and all(isinstance(item, dict) for item in prompt):
            messages = prompt
        else:
            raise ValueError(f"Invalid prompt type: {type(prompt)}")

        client = self._get_client()
        logical_attempt_usage: list[dict[str, Any]] = []
        max_rate_retries = 15
        for attempt in range(self.malformed_retries + 1):
            response = None
            for rl_attempt in range(max_rate_retries):
                try:
                    response = client.chat.completions.create(
                        model=vertex_openai_model(model),
                        messages=messages,
                        **self._request_kwargs(model),
                    )
                    break
                except Exception as exc:
                    err_msg = str(exc).lower()
                    is_rate_limit = ("429" in err_msg or "resource_exhausted" in err_msg or "rate" in err_msg)
                    is_server_err = any(code in err_msg for code in ["500", "502", "503", "504", "internal error", "unavailable", "bad gateway", "connection error", "timeout"])
                    is_auth_err = ("401" in err_msg or "unauthenticated" in err_msg or "access_token" in err_msg)

                    if (is_rate_limit or is_server_err or is_auth_err) and rl_attempt < max_rate_retries - 1:
                        if is_auth_err:
                            with self._credential_lock:
                                self._cached_gcloud_token = None
                                self._client = None
                            client = self._get_client()
                        if is_rate_limit:
                            sleep_s = min(90.0, max(15.0, 10.0 * (1.5**rl_attempt)))
                        else:
                            sleep_s = min(60.0, 3.0 * (1.8**rl_attempt))
                        time.sleep(sleep_s)
                    else:
                        raise

            logical_attempt_usage.append(self._track_cost(response, model))
            try:
                text = extract_candidate_text(response)
                self._set_last_logical_usage(logical_attempt_usage)
                return text
            except ValueError:
                if not _is_retryable_malformed_response(response) or attempt >= self.malformed_retries:
                    self._set_last_logical_usage(logical_attempt_usage)
                    raise
                if self.malformed_retry_delay:
                    time.sleep(self.malformed_retry_delay * (2**attempt))
        raise AssertionError("unreachable")

    async def acompletion(self, prompt: str | list[dict[str, Any]], model: str | None = None) -> str:
        return await asyncio.to_thread(self.completion, prompt, model)

    def _track_cost(self, response: Any, model: str) -> dict[str, Any]:
        parsed = parse_usage_metadata(_field(response, "usage"))
        self.last_raw_usage = parsed
        self.last_model_version = _field(response, "model")
        prompt_tokens = parsed["prompt_tokens"]
        output_tokens = parsed["output_tokens"]
        thoughts = parsed["thoughts_tokens"]
        total = parsed["total_tokens"]
        usd = cost_usd(prompt_tokens, output_tokens)

        self.model_call_counts[model] += 1
        self.model_input_tokens[model] += prompt_tokens
        self.model_output_tokens[model] += output_tokens
        self.model_thought_tokens[model] += thoughts
        self.model_total_tokens[model] += total
        self.model_costs[model] += usd
        self.last_prompt_tokens = prompt_tokens
        self.last_completion_tokens = output_tokens
        self.last_thought_tokens = thoughts
        self.last_thoughts_source = str(parsed.get("thoughts_source") or "none")
        self.last_cost = usd
        return parsed

    def _set_last_logical_usage(self, attempts: list[dict[str, Any]]) -> None:
        """Expose aggregate usage for one logical completion, including retries."""
        if not attempts:
            return
        self._last_logical_call_count = len(attempts)
        self.last_prompt_tokens = sum(item["prompt_tokens"] for item in attempts)
        self.last_completion_tokens = sum(item["output_tokens"] for item in attempts)
        self.last_thought_tokens = sum(item["thoughts_tokens"] for item in attempts)
        self.last_cost = sum(
            cost_usd(item["prompt_tokens"], item["output_tokens"]) for item in attempts
        )
        self.last_raw_usage = dict(attempts[-1])
        self.last_raw_usage["attempt_count"] = len(attempts)
        self.last_thoughts_source = str(attempts[-1].get("thoughts_source") or "none")

    def get_usage_summary(self) -> UsageSummary:
        model_summaries = {}
        for model in self.model_call_counts:
            model_summaries[model] = ModelUsageSummary(
                total_calls=self.model_call_counts[model],
                total_input_tokens=self.model_input_tokens[model],
                total_output_tokens=self.model_output_tokens[model],
                total_cost=self.model_costs.get(model),
                total_thought_tokens=self.model_thought_tokens[model],
                thoughts_source=self.last_thoughts_source,
            )
        return UsageSummary(model_usage_summaries=model_summaries)

    def get_last_usage(self) -> ModelUsageSummary:
        return ModelUsageSummary(
            total_calls=self._last_logical_call_count,
            total_input_tokens=self.last_prompt_tokens,
            total_output_tokens=self.last_completion_tokens,
            total_cost=self.last_cost,
            total_thought_tokens=self.last_thought_tokens,
            thoughts_source=self.last_thoughts_source,
        )
