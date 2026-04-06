from __future__ import annotations

from typing import Any

from app.config import get_settings
from app.logging import get_logger


logger = get_logger(__name__)

try:
    from openai import OpenAI
except ImportError:  # pragma: no cover - dependency should exist in app environments
    OpenAI = None  # type: ignore[assignment]


class OptionalLLM:
    def __init__(self) -> None:
        self.settings = get_settings()
        self._client = self._build_client()

    def _build_client(self) -> Any | None:
        if self.settings.llm_backend != "openai":
            logger.info("LLM backend %s is configured but not implemented locally; using deterministic fallback.", self.settings.llm_backend)
            return None
        if not self.settings.openai_api_key or OpenAI is None:
            return None
        return OpenAI(api_key=self.settings.openai_api_key)

    @property
    def available(self) -> bool:
        return self._client is not None

    def complete(self, *, system_prompt: str, user_prompt: str, max_tokens: int = 500) -> str | None:
        if self._client is None:
            return None

        try:
            response = self._client.responses.create(
                model=self.settings.openai_model,
                input=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                max_output_tokens=max_tokens,
            )
            return getattr(response, "output_text", None) or None
        except Exception as exc:  # pragma: no cover - network/model failures are environment-specific
            logger.warning("LLM completion failed, falling back to deterministic path: %s", exc)
            return None
