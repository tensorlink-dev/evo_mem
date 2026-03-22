"""Base LLM interface for Evo-Memory."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
import logging
import random
import time

logger = logging.getLogger(__name__)


@dataclass
class LLMResponse:
    """Response from LLM."""
    content: str
    model: str
    usage: Dict[str, int] = field(default_factory=dict)
    finish_reason: Optional[str] = None
    latency: float = 0.0
    raw_response: Optional[Any] = None

    @property
    def total_tokens(self) -> int:
        return self.usage.get("total_tokens", 0)


class BaseLLM(ABC):
    """
    Abstract base class for LLM backends.

    Implements the base LLM F that produces output: ŷ_t = F(C̃_t)
    """

    def __init__(
        self,
        model_name: str,
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: int = 4096,
        top_p: float = 1.0,
        timeout: int = 300,
        retry_attempts: int = 6,
        retry_delay: float = 2.0,
    ):
        """
        Initialize LLM.

        Args:
            model_name: Model identifier
            api_key: API key for authentication
            api_base: Base URL for API
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            top_p: Top-p sampling parameter
            timeout: Request timeout in seconds
            retry_attempts: Number of retry attempts
            retry_delay: Delay between retries in seconds
        """
        self.model_name = model_name
        self.api_key = api_key
        self.api_base = api_base
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.top_p = top_p
        self.timeout = timeout
        self.retry_attempts = retry_attempts
        self.retry_delay = retry_delay

        # Usage tracking
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0
        self.total_requests = 0

    @abstractmethod
    def _generate(
        self,
        messages: List[Dict[str, str]],
        **kwargs,
    ) -> LLMResponse:
        """
        Internal generation method to be implemented by subclasses.

        Args:
            messages: List of message dicts with 'role' and 'content'
            **kwargs: Additional generation parameters

        Returns:
            LLMResponse object
        """
        pass

    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        **kwargs,
    ) -> LLMResponse:
        """
        Generate response from prompt.

        Args:
            prompt: User prompt/query
            system_prompt: Optional system prompt
            **kwargs: Additional generation parameters

        Returns:
            LLMResponse object
        """
        messages = []

        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})

        messages.append({"role": "user", "content": prompt})

        return self.chat(messages, **kwargs)

    def chat(
        self,
        messages: List[Dict[str, str]],
        **kwargs,
    ) -> LLMResponse:
        """
        Generate response from chat messages with retry logic.

        Args:
            messages: List of message dicts
            **kwargs: Additional generation parameters

        Returns:
            LLMResponse object
        """
        last_error = None

        for attempt in range(self.retry_attempts):
            try:
                start_time = time.time()
                response = self._generate(messages, **kwargs)
                response.latency = time.time() - start_time

                # Update usage tracking
                self.total_requests += 1
                self.total_prompt_tokens += response.usage.get("prompt_tokens", 0)
                self.total_completion_tokens += response.usage.get("completion_tokens", 0)

                return response

            except Exception as e:
                last_error = e
                if attempt < self.retry_attempts - 1:
                    # Exponential backoff with jitter; longer wait for rate limits
                    is_rate_limit = "429" in str(e) or "rate" in str(e).lower()
                    base = self.retry_delay * (2 ** attempt)
                    if is_rate_limit:
                        base = max(base, 10.0)  # at least 10s for 429s
                    wait = base + random.uniform(0, base * 0.5)
                    logger.warning(
                        "LLM request failed (attempt %d/%d): %s — retrying in %.1fs",
                        attempt + 1, self.retry_attempts, type(e).__name__, wait,
                    )
                    time.sleep(wait)

        raise last_error

    def get_usage_stats(self) -> Dict[str, Any]:
        """Get usage statistics."""
        return {
            "total_requests": self.total_requests,
            "total_prompt_tokens": self.total_prompt_tokens,
            "total_completion_tokens": self.total_completion_tokens,
            "total_tokens": self.total_prompt_tokens + self.total_completion_tokens,
        }

    def reset_usage_stats(self) -> None:
        """Reset usage statistics."""
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0
        self.total_requests = 0
