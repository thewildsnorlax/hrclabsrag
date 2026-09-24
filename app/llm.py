"""Thin streaming wrapper around the Claude Messages API.

Kept behind a small interface so the RAG pipeline can be tested with a fake.
"""

from dataclasses import dataclass, field
from typing import AsyncIterator, Dict, List, Optional, Protocol, Union

import anthropic

from app.config import Settings


@dataclass(frozen=True)
class TextDelta:
    text: str


@dataclass(frozen=True)
class Completed:
    stop_reason: Optional[str]
    usage: Dict[str, int] = field(default_factory=dict)


class LLMError(Exception):
    """An LLM failure with a message that is safe to show to end users."""


StreamEvent = Union[TextDelta, Completed]


class AnswerLLM(Protocol):
    def stream(self, system: str, messages: List[dict]) -> AsyncIterator[StreamEvent]: ...


def _api_error_message(exc: anthropic.APIStatusError) -> str:
    """The API's human-readable message, without the SDK's status/dict wrapper."""
    body = exc.body
    if isinstance(body, dict):
        error = body.get("error")
        if isinstance(error, dict) and isinstance(error.get("message"), str):
            return error["message"]
    return exc.message


class ClaudeLLM:
    def __init__(self, settings: Settings):
        self._settings = settings
        self._client: Optional[anthropic.AsyncAnthropic] = None

    def _get_client(self) -> anthropic.AsyncAnthropic:
        if self._client is None:
            # With no explicit key the SDK falls back to its own credential chain
            # (ANTHROPIC_API_KEY env var, `ant auth login` profile, ...).
            key = self._settings.anthropic_api_key.strip() or None
            self._client = anthropic.AsyncAnthropic(api_key=key)
        return self._client

    async def stream(self, system: str, messages: List[dict]) -> AsyncIterator[StreamEvent]:
        s = self._settings
        try:
            async with self._get_client().messages.stream(
                model=s.llm_model,
                max_tokens=s.llm_max_tokens,
                system=system,
                messages=messages,
                output_config={"effort": s.llm_effort},
            ) as stream:
                async for text in stream.text_stream:
                    yield TextDelta(text)
                final = await stream.get_final_message()
        except anthropic.AuthenticationError as exc:
            raise LLMError("The Anthropic API key is missing or invalid.") from exc
        except anthropic.NotFoundError as exc:
            raise LLMError(f"Model '{s.llm_model}' was not found. Check LLM_MODEL.") from exc
        except anthropic.RateLimitError as exc:
            raise LLMError("The LLM is rate limited. Please retry shortly.") from exc
        except anthropic.BadRequestError as exc:
            raise LLMError(f"The LLM rejected the request: {_api_error_message(exc)}") from exc
        except anthropic.APIStatusError as exc:
            raise LLMError(f"The LLM service returned an error ({exc.status_code}).") from exc
        except anthropic.APIConnectionError as exc:
            raise LLMError("Could not reach the LLM service.") from exc
        except anthropic.AnthropicError as exc:  # e.g. no credentials configured at all
            raise LLMError(f"LLM is not configured: {exc}") from exc

        usage = {
            "input_tokens": final.usage.input_tokens,
            "output_tokens": final.usage.output_tokens,
        }
        yield Completed(stop_reason=final.stop_reason, usage=usage)
