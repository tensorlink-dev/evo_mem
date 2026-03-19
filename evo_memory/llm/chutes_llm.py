"""Chutes AI LLM backend.

Wraps the OpenAI-compatible Chutes AI API for access to
decentralised GPU models (DeepSeek-R1, Llama, Qwen, etc.).

Usage:
    export CHUTES_API_KEY="cpk_..."
    llm = ChutesLLM(model_name="deepseek-ai/DeepSeek-R1")
"""

import os
from typing import List, Dict, Optional, Any

from .base import BaseLLM, LLMResponse


CHUTES_API_BASE = "https://api.chutes.ai/v1"


class ChutesLLM(BaseLLM):
    """Chutes AI backend (OpenAI-compatible API on decentralised GPU)."""

    def __init__(
        self,
        model_name: str = "deepseek-ai/DeepSeek-R1",
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
        **kwargs,
    ):
        """
        Initialize Chutes AI LLM.

        Args:
            model_name: Model identifier on Chutes (e.g. deepseek-ai/DeepSeek-R1)
            api_key: Chutes API key (defaults to CHUTES_API_KEY env var)
            api_base: Optional custom API base URL (defaults to Chutes endpoint)
            **kwargs: Additional BaseLLM parameters
        """
        api_key = api_key or os.environ.get("CHUTES_API_KEY")
        if not api_key:
            raise ValueError(
                "Chutes API key required. Set CHUTES_API_KEY environment variable "
                "or pass api_key parameter."
            )
        api_base = api_base or CHUTES_API_BASE
        super().__init__(model_name, api_key, api_base, **kwargs)
        self._client = None

    @property
    def client(self):
        """Lazy load OpenAI client pointed at Chutes."""
        if self._client is None:
            try:
                from openai import OpenAI
                self._client = OpenAI(
                    api_key=self.api_key,
                    base_url=self.api_base,
                    timeout=self.timeout,
                )
            except ImportError:
                raise ImportError(
                    "openai package is required for Chutes backend. "
                    "Install with: pip install openai>=1.0"
                )
        return self._client

    def _generate(
        self,
        messages: List[Dict[str, str]],
        **kwargs,
    ) -> LLMResponse:
        """Generate response using Chutes AI API."""
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            temperature=kwargs.get("temperature", self.temperature),
            max_tokens=kwargs.get("max_tokens", self.max_tokens),
            top_p=kwargs.get("top_p", self.top_p),
        )

        choice = response.choices[0]
        usage = response.usage

        return LLMResponse(
            content=choice.message.content or "",
            model=response.model,
            usage={
                "prompt_tokens": usage.prompt_tokens if usage else 0,
                "completion_tokens": usage.completion_tokens if usage else 0,
                "total_tokens": usage.total_tokens if usage else 0,
            },
            finish_reason=choice.finish_reason,
            raw_response=response,
        )
