"""Groq client: one function that sends a chat prompt to a model and never raises.

Every LLM call in the pipeline (grounded answers, prediction explanations, evaluation)
goes through `complete`. A failed call is reported as `error = config.API_ERROR`, the only
text the interface shows; the real exception goes to the Python logger.
"""
import logging
import time
from dataclasses import dataclass

from groq import Groq

from milestone3 import config

log = logging.getLogger(__name__)


@dataclass
class LLMResult:
    model: str
    text: str | None        # final answer, None when the call failed
    error: str | None       # config.API_ERROR when the call failed
    latency_s: float
    input_tokens: int
    output_tokens: int


def complete(model: str, messages: list[dict], temperature: float = 0) -> LLMResult:
    """Send `messages` to `model` on Groq and return the answer with latency and token usage."""
    start = time.perf_counter()
    try:
        # The SDK retries rate-limit (429) errors with exponential backoff.
        client = Groq(api_key=config.GROQ_API_KEY, max_retries=4)
        extra = {"reasoning_effort": "low"} if model.startswith("openai/gpt-oss") else {}
        response = client.chat.completions.create(
            model=model, messages=messages, temperature=temperature,
            max_completion_tokens=1024, **extra)
        # `content` is the final answer only; gpt-oss puts its reasoning in a separate field.
        text = (response.choices[0].message.content or "").strip()
        if not text:
            raise ValueError("empty answer")
        usage = response.usage
        return LLMResult(model, text, None, time.perf_counter() - start,
                         usage.prompt_tokens, usage.completion_tokens)
    except Exception as exc:  # any failure becomes the single user-facing error
        log.warning("LLM call to %s failed: %s", model, type(exc).__name__)
        return LLMResult(model, None, config.API_ERROR, time.perf_counter() - start, 0, 0)
