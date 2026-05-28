from __future__ import annotations

import json
import os
import socket
import urllib.error
import urllib.request
from dataclasses import dataclass


DEFAULT_LLM_PROVIDER = "openai_compatible"
DEFAULT_LLM_TIMEOUT = 60.0


@dataclass(frozen=True)
class OpenAICompatibleClient:
    api_key: str
    base_url: str
    model: str
    timeout: float = DEFAULT_LLM_TIMEOUT

    @classmethod
    def from_env(
        cls,
        model: str | None = None,
        base_url: str | None = None,
        timeout: float = DEFAULT_LLM_TIMEOUT,
    ) -> "OpenAICompatibleClient":
        api_key = os.environ.get("LABMATE_LLM_API_KEY")
        resolved_base_url = base_url or os.environ.get("LABMATE_LLM_BASE_URL")
        resolved_model = model or os.environ.get("LABMATE_LLM_MODEL")

        missing = []
        if not api_key:
            missing.append("LABMATE_LLM_API_KEY")
        if not resolved_base_url:
            missing.append("LABMATE_LLM_BASE_URL")
        if not resolved_model:
            missing.append("LABMATE_LLM_MODEL or --llm_model")
        if missing:
            raise RuntimeError(
                "Missing LLM configuration: "
                + ", ".join(missing)
                + ". Configure an OpenAI-compatible endpoint with environment variables "
                "or CLI overrides for model/base URL."
            )

        if timeout <= 0:
            raise ValueError("LLM timeout must be greater than 0.")

        return cls(api_key=api_key, base_url=resolved_base_url, model=resolved_model, timeout=timeout)

    @property
    def chat_completions_url(self) -> str:
        base = self.base_url.rstrip("/")
        if base.endswith("/chat/completions"):
            return base
        if base.endswith("/v1"):
            return f"{base}/chat/completions"
        return f"{base}/v1/chat/completions"

    def chat(self, messages: list[dict[str, str]], temperature: float = 0.2) -> str:
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
        }
        request = urllib.request.Request(
            self.chat_completions_url,
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )

        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                data = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"LLM request failed with HTTP {exc.code}: {body}") from exc
        except TimeoutError as exc:
            raise RuntimeError(f"LLM request timed out after {self.timeout:g} seconds.") from exc
        except socket.timeout as exc:
            raise RuntimeError(f"LLM request timed out after {self.timeout:g} seconds.") from exc
        except urllib.error.URLError as exc:
            if isinstance(exc.reason, socket.timeout):
                raise RuntimeError(f"LLM request timed out after {self.timeout:g} seconds.") from exc
            raise RuntimeError(f"LLM request failed: {exc.reason}") from exc

        try:
            return str(data["choices"][0]["message"]["content"]).strip()
        except (KeyError, IndexError, TypeError) as exc:
            raise RuntimeError(f"Unexpected LLM response format: {data}") from exc
