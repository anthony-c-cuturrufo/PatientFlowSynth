"""Small, secret-safe OpenAI Responses API integration."""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from typing import Any, Generic, TypeVar
from urllib.parse import urlsplit

from openai import (
    APIConnectionError,
    APIStatusError,
    APITimeoutError,
    ContentFilterFinishReasonError,
    LengthFinishReasonError,
    OpenAI,
    OpenAIError,
)
from pydantic import BaseModel, ValidationError

DEFAULT_OPENAI_BASE_URL = "https://api.openai.com/v1"
SMOKE_EXPECTED_TEXT = "PATIENT_TOOL_GYM_LLM_OK"
SMOKE_INSTRUCTIONS = (
    "This is an automated API connectivity check. Return only the exact token requested, "
    "without quotation marks or punctuation."
)
SMOKE_INPUT = f"Reply with exactly: {SMOKE_EXPECTED_TEXT}"

StructuredOutputT = TypeVar("StructuredOutputT", bound=BaseModel)


class LLMError(RuntimeError):
    """An error whose public fields are safe to print from the CLI."""

    def __init__(
        self,
        code: str,
        public_message: str,
        *,
        status_code: int | None = None,
        request_id: str | None = None,
    ) -> None:
        super().__init__(public_message)
        self.code = code
        self.public_message = public_message
        self.status_code = status_code
        self.request_id = _safe_request_id(request_id)

    def public_dict(self) -> dict[str, str | int]:
        result: dict[str, str | int] = {
            "code": self.code,
            "message": self.public_message,
        }
        if self.status_code is not None:
            result["status_code"] = self.status_code
        if self.request_id is not None:
            result["request_id"] = self.request_id
        return result


class LLMConfigurationError(LLMError):
    """The local OpenAI configuration is missing or invalid."""


class LLMRequestError(LLMError):
    """The OpenAI request did not complete successfully."""


class LLMResponseError(LLMError):
    """The OpenAI response was successful but did not contain usable text."""


@dataclass(frozen=True)
class OpenAISettings:
    """Validated OpenAI settings with the credential excluded from ``repr``."""

    api_key: str = field(repr=False)
    model: str
    base_url: str = DEFAULT_OPENAI_BASE_URL
    timeout_seconds: float = 30.0

    def __post_init__(self) -> None:
        api_key = self.api_key.strip()
        model = self.model.strip()
        base_url = self.base_url.strip().rstrip("/")

        if not api_key:
            raise LLMConfigurationError(
                "missing_api_key",
                "OPENAI_API_KEY is required.",
            )
        if not model:
            raise LLMConfigurationError(
                "missing_model",
                "Set OPENAI_MODEL or pass --model.",
            )
        if any(character.isspace() for character in base_url):
            raise LLMConfigurationError(
                "invalid_base_url",
                "OPENAI_BASE_URL must be a valid HTTPS URL.",
            )

        parsed = urlsplit(base_url)
        if (
            parsed.scheme != "https"
            or not parsed.hostname
            or parsed.username is not None
            or parsed.password is not None
            or parsed.query
            or parsed.fragment
        ):
            raise LLMConfigurationError(
                "invalid_base_url",
                "OPENAI_BASE_URL must be a valid HTTPS URL.",
            )
        if self.timeout_seconds <= 0:
            raise LLMConfigurationError(
                "invalid_timeout",
                "The OpenAI timeout must be greater than zero.",
            )

        object.__setattr__(self, "api_key", api_key)
        object.__setattr__(self, "model", model)
        object.__setattr__(self, "base_url", base_url)

    @classmethod
    def from_environment(
        cls,
        *,
        model: str | None = None,
        timeout_seconds: float = 30.0,
        environment: Mapping[str, str] | None = None,
    ) -> OpenAISettings:
        values = os.environ if environment is None else environment
        return cls(
            api_key=values.get("OPENAI_API_KEY", ""),
            model=model or values.get("OPENAI_MODEL", ""),
            base_url=values.get("OPENAI_BASE_URL", DEFAULT_OPENAI_BASE_URL),
            timeout_seconds=timeout_seconds,
        )

    @property
    def endpoint_host(self) -> str:
        host = urlsplit(self.base_url).hostname
        assert host is not None  # Validated in __post_init__.
        return host


@dataclass(frozen=True)
class TokenUsage:
    input_tokens: int | None
    output_tokens: int | None
    total_tokens: int | None


@dataclass(frozen=True)
class GeneratedText:
    response_id: str
    model: str
    text: str
    usage: TokenUsage


@dataclass(frozen=True)
class GeneratedStructuredOutput(Generic[StructuredOutputT]):
    response_id: str
    model: str
    parsed: StructuredOutputT
    usage: TokenUsage


@dataclass(frozen=True)
class SmokeTestReport:
    ok: bool
    endpoint_host: str
    expected_text: str
    response: GeneratedText

    def public_dict(self) -> dict[str, Any]:
        return {
            "endpoint_host": self.endpoint_host,
            "expected_text": self.expected_text,
            "model": self.response.model,
            "ok": self.ok,
            "provider": "openai",
            "response_id": self.response.response_id,
            "text": self.response.text,
            "usage": asdict(self.response.usage),
        }


class OpenAITextClient:
    """Narrow wrapper around text generation with injectable SDK transport."""

    def __init__(self, settings: OpenAISettings, *, sdk_client: Any | None = None) -> None:
        self.settings = settings
        self._sdk_client = (
            sdk_client
            if sdk_client is not None
            else OpenAI(
                api_key=settings.api_key,
                base_url=settings.base_url,
                timeout=settings.timeout_seconds,
            )
        )

    def generate_text(
        self,
        *,
        instructions: str,
        input_text: str,
        max_output_tokens: int,
    ) -> GeneratedText:
        if not instructions.strip() or not input_text.strip():
            raise ValueError("instructions and input_text must not be empty")
        if max_output_tokens <= 0:
            raise ValueError("max_output_tokens must be greater than zero")

        response = self._call(
            self._sdk_client.responses.create,
            model=self.settings.model,
            instructions=instructions,
            input=input_text,
            max_output_tokens=max_output_tokens,
            store=False,
        )

        text = str(getattr(response, "output_text", "") or "").strip()
        if not text:
            raise LLMResponseError(
                "empty_response",
                "The OpenAI response did not contain text output.",
            )

        return GeneratedText(
            response_id=str(getattr(response, "id", "")),
            model=str(getattr(response, "model", self.settings.model)),
            text=text,
            usage=_token_usage(response),
        )

    def generate_structured(
        self,
        *,
        instructions: str,
        input_text: str,
        response_format: type[StructuredOutputT],
        max_output_tokens: int,
    ) -> GeneratedStructuredOutput[StructuredOutputT]:
        """Generate one Pydantic-validated object through Responses structured output."""

        if not instructions.strip() or not input_text.strip():
            raise ValueError("instructions and input_text must not be empty")
        if max_output_tokens <= 0:
            raise ValueError("max_output_tokens must be greater than zero")
        try:
            response = self._call(
                self._sdk_client.responses.parse,
                model=self.settings.model,
                instructions=instructions,
                input=input_text,
                max_output_tokens=max_output_tokens,
                store=False,
                text_format=response_format,
            )
        except ValidationError as exc:
            raise LLMResponseError(
                "invalid_structured_response",
                "The OpenAI response did not match the required output schema.",
            ) from exc

        status = getattr(response, "status", "completed")
        if status != "completed":
            raise LLMResponseError(
                "incomplete_response",
                "The OpenAI response was incomplete.",
            )

        parsed = getattr(response, "output_parsed", None)
        if parsed is None:
            if _contains_refusal(response):
                raise LLMResponseError(
                    "model_refusal",
                    "The model declined to produce the requested output.",
                )
            raise LLMResponseError(
                "missing_structured_response",
                "The OpenAI response did not contain structured output.",
            )
        if not isinstance(parsed, response_format):
            raise LLMResponseError(
                "invalid_structured_response",
                "The OpenAI response did not match the required output schema.",
            )

        return GeneratedStructuredOutput(
            response_id=str(getattr(response, "id", "")),
            model=str(getattr(response, "model", self.settings.model)),
            parsed=parsed,
            usage=_token_usage(response),
        )

    @staticmethod
    def _call(method: Any, **kwargs: Any) -> Any:
        try:
            return method(**kwargs)
        except APITimeoutError as exc:
            raise LLMRequestError(
                "request_timeout",
                "The OpenAI request timed out.",
            ) from exc
        except APIConnectionError as exc:
            raise LLMRequestError(
                "connection_error",
                "Could not connect to the configured OpenAI endpoint.",
            ) from exc
        except APIStatusError as exc:
            raise LLMRequestError(
                "api_status_error",
                "The OpenAI endpoint rejected the request.",
                status_code=exc.status_code,
                request_id=getattr(exc, "request_id", None),
            ) from exc
        except LengthFinishReasonError as exc:
            raise LLMResponseError(
                "incomplete_response",
                "The OpenAI response was incomplete.",
            ) from exc
        except ContentFilterFinishReasonError as exc:
            raise LLMResponseError(
                "content_filter",
                "The OpenAI response was stopped by a content filter.",
            ) from exc
        except OpenAIError as exc:
            raise LLMRequestError(
                "openai_error",
                "The OpenAI request failed.",
            ) from exc


def _safe_request_id(value: Any) -> str | None:
    """Keep only a bounded printable provider request ID in public diagnostics."""

    if not isinstance(value, str):
        return None
    request_id = value.strip()
    if not request_id or len(request_id) > 255:
        return None
    allowed_punctuation = "-_.:"
    if any(
        not (character.isascii() and (character.isalnum() or character in allowed_punctuation))
        for character in request_id
    ):
        return None
    return request_id


def _token_usage(response: Any) -> TokenUsage:
    usage = getattr(response, "usage", None)
    return TokenUsage(
        input_tokens=getattr(usage, "input_tokens", None),
        output_tokens=getattr(usage, "output_tokens", None),
        total_tokens=getattr(usage, "total_tokens", None),
    )


def _contains_refusal(response: Any) -> bool:
    for output in getattr(response, "output", ()):
        for content in getattr(output, "content", ()):
            if getattr(content, "type", None) == "refusal":
                return True
    return False


def run_smoke_test(
    *,
    model: str | None = None,
    max_output_tokens: int = 128,
    environment: Mapping[str, str] | None = None,
    sdk_client: Any | None = None,
) -> SmokeTestReport:
    """Send one fixed, non-clinical prompt and validate the returned sentinel."""

    settings = OpenAISettings.from_environment(model=model, environment=environment)
    response = OpenAITextClient(settings, sdk_client=sdk_client).generate_text(
        instructions=SMOKE_INSTRUCTIONS,
        input_text=SMOKE_INPUT,
        max_output_tokens=max_output_tokens,
    )
    return SmokeTestReport(
        ok=response.text == SMOKE_EXPECTED_TEXT,
        endpoint_host=settings.endpoint_host,
        expected_text=SMOKE_EXPECTED_TEXT,
        response=response,
    )
