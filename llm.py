"""
minima — Pure-stdlib LLM client for OpenAI-compatible chat APIs.

Works on i686 Debian with no compiled dependencies.
Supports llama.cpp server, vLLM, and any API with /v1/chat/completions.
"""

import json
import urllib.request
import urllib.error


class LLMError(Exception):
    """Raised on API or connection failures."""
    pass


class LLMClient:
    """
    OpenAI-compatible chat completion client via stdlib HTTP.

    Defaults to llama.cpp server at localhost:8080 but any
    OpenAI-compatible endpoint works (vLLM, remote APIs, etc.).
    """

    def __init__(
        self,
        base_url: str = "http://localhost:8080",
        model: str = "qwen2.5-0.5b-q4",
        timeout: float = 120.0,
    ):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout

    def _chat_url(self) -> str:
        return f"{self.base_url}/v1/chat/completions"

    def chat(
        self,
        messages: list[dict],
        temperature: float = 0.0,
        max_tokens: int = 4096,
        stop: list[str] | None = None,
    ) -> str:
        """Send a chat request and return the assistant's message content."""
        body = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if stop:
            body["stop"] = stop

        payload = json.dumps(body).encode("utf-8")
        req = urllib.request.Request(
            self._chat_url(),
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        try:
            resp = urllib.request.urlopen(req, timeout=self.timeout)
        except urllib.error.HTTPError as e:
            detail = e.read().decode("utf-8", errors="replace")[:500]
            raise LLMError(
                f"HTTP {e.code} from {self._chat_url()}: {detail}"
            ) from e
        except (urllib.error.URLError, OSError) as e:
            raise LLMError(
                f"Connection to {self.base_url} failed: {e}"
            ) from e

        raw = resp.read().decode("utf-8")
        data = json.loads(raw)
        choices = data.get("choices", [])
        if not choices:
            raise LLMError(f"LLM returned empty choices: {raw[:500]}")

        content = choices[0].get("message", {}).get("content", "")
        return content.strip() if content else ""
