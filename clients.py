import abc
import asyncio
import logging
from collections.abc import AsyncGenerator, Callable, Coroutine
from typing import Any, TypeVar

from anthropic import NOT_GIVEN, AsyncAnthropic
from anthropic import APIError as AnthropicError
from google import genai
from google.genai import errors as genai_errors
from google.genai import types as genai_types
from openai import APIError as OpenAIError
from openai import AsyncOpenAI

from schemas import ChatMessage, ModelConfig, ModelResponse

logger = logging.getLogger(__name__)
MAX_RETRIES = 3
T = TypeVar("T")


def _is_server_error(error: Exception) -> bool:
    status = getattr(error, "status_code", None)
    if status is None:
        status = getattr(error, "code", None)

    try:
        status_code = int(status)
    except (TypeError, ValueError):
        return False

    return 500 <= status_code <= 599


async def _retry_async(operation: Callable[[], Coroutine[Any, Any, T]]) -> T:
    for attempt in range(MAX_RETRIES + 1):
        try:
            return await operation()
        except Exception as error:
            if not _is_server_error(error) or attempt == MAX_RETRIES:
                raise

            delay = min(2**attempt, 8)
            logger.warning(
                "Error HTTP 5xx; reintento %d/%d en %d segundos: %s",
                attempt + 1,
                MAX_RETRIES,
                delay,
                error,
            )
            await asyncio.sleep(delay)

    raise RuntimeError("Retry loop ended unexpectedly")


async def _retry_stream(
    stream_factory: Callable[[], AsyncGenerator[str, None]],
) -> AsyncGenerator[str, None]:
    for attempt in range(MAX_RETRIES + 1):
        yielded_content = False
        try:
            async for chunk in stream_factory():
                yielded_content = True
                yield chunk
            return
        except Exception as error:
            if (
                yielded_content
                or not _is_server_error(error)
                or attempt == MAX_RETRIES
            ):
                raise

            delay = min(2**attempt, 8)
            logger.warning(
                "Error HTTP 5xx en streaming; reintento %d/%d en %d segundos: %s",
                attempt + 1,
                MAX_RETRIES,
                delay,
                error,
            )
            await asyncio.sleep(delay)


class BaseLLMClient(abc.ABC):
    def __init__(self, config: ModelConfig):
        self.config = config

    @abc.abstractmethod
    async def generate(self, messages: list[ChatMessage]) -> ModelResponse:
        pass

    @abc.abstractmethod
    async def stream(self, messages: list[ChatMessage]) -> AsyncGenerator[str, None]:
        pass


class OpenAIClient(BaseLLMClient):
    def __init__(self, api_key: str, config: ModelConfig):
        super().__init__(config)
        self.client = AsyncOpenAI(api_key=api_key)

    async def generate(self, messages: list[ChatMessage]) -> ModelResponse:
        try:
            formatted_messages = [msg.model_dump() for msg in messages]
            response = await _retry_async(
                lambda: self.client.chat.completions.create(
                    model=self.config.model,
                    messages=formatted_messages,
                    temperature=self.config.temperature,
                    max_tokens=self.config.max_tokens,
                )
            )
            return ModelResponse(
                content=response.choices[0].message.content or "",
                model=self.config.model,
                usage=response.usage.model_dump() if response.usage else None,
            )
        except OpenAIError as e:
            logger.error(f"OpenAI API Error: {e}")
            return ModelResponse(content="", model=self.config.model, error=str(e))
        except Exception:
            logger.exception("Error inesperado en OpenAIClient")
            return ModelResponse(content="", model=self.config.model, error="Internal Error")

    async def stream(self, messages: list[ChatMessage]) -> AsyncGenerator[str, None]:
        try:
            formatted_messages = [msg.model_dump() for msg in messages]
            async def request_stream() -> AsyncGenerator[str, None]:
                response_stream = await self.client.chat.completions.create(
                    model=self.config.model,
                    messages=formatted_messages,
                    temperature=self.config.temperature,
                    max_tokens=self.config.max_tokens,
                    stream=True,
                )
                async for chunk in response_stream:
                    if chunk.choices and chunk.choices[0].delta.content:
                        yield chunk.choices[0].delta.content

            async for text in _retry_stream(request_stream):
                yield text
        except OpenAIError as e:
            logger.error(f"OpenAI Stream Error: {e}")
            yield f"\n[Stream interrumpido por error: {str(e)}]"
        except Exception:
            logger.exception("Error inesperado en stream de OpenAIClient")
            yield "\n[Stream interrumpido por error interno]"


class AnthropicClient(BaseLLMClient):
    def __init__(self, api_key: str, config: ModelConfig):
        super().__init__(config)
        self.client = AsyncAnthropic(api_key=api_key)

    def _format_messages(
        self, messages: list[ChatMessage]
    ) -> tuple[str, list[dict[str, str]]]:
        system_prompt = ""
        anthropic_messages = []
        for msg in messages:
            if msg.role == "system":
                system_prompt = msg.content
            else:
                anthropic_messages.append({"role": msg.role, "content": msg.content})
        return system_prompt, anthropic_messages

    async def generate(self, messages: list[ChatMessage]) -> ModelResponse:
        try:
            system, formatted_messages = self._format_messages(messages)
            response = await _retry_async(
                lambda: self.client.messages.create(
                    model=self.config.model,
                    system=system or NOT_GIVEN,
                    messages=formatted_messages,
                    temperature=self.config.temperature,
                    max_tokens=self.config.max_tokens,
                )
            )
            content = "".join(
                block.text for block in response.content if block.type == "text"
            )
            return ModelResponse(
                content=content,
                model=self.config.model,
                usage={
                    "input_tokens": response.usage.input_tokens,
                    "output_tokens": response.usage.output_tokens,
                },
            )
        except AnthropicError as e:
            logger.error(f"Anthropic API Error: {e}")
            return ModelResponse(content="", model=self.config.model, error=str(e))
        except Exception:
            logger.exception("Error inesperado en AnthropicClient")
            return ModelResponse(content="", model=self.config.model, error="Internal Error")

    async def stream(self, messages: list[ChatMessage]) -> AsyncGenerator[str, None]:
        try:
            system, formatted_messages = self._format_messages(messages)
            async def request_stream() -> AsyncGenerator[str, None]:
                async with self.client.messages.stream(
                    model=self.config.model,
                    system=system or NOT_GIVEN,
                    messages=formatted_messages,
                    temperature=self.config.temperature,
                    max_tokens=self.config.max_tokens,
                ) as stream:
                    async for text in stream.text_stream:
                        yield text

            async for text in _retry_stream(request_stream):
                yield text
        except AnthropicError as e:
            logger.error(f"Anthropic Stream Error: {e}")
            yield f"\n[Stream interrumpido por error: {str(e)}]"
        except Exception:
            logger.exception("Error inesperado en stream de AnthropicClient")
            yield "\n[Stream interrumpido por error interno]"


class GeminiClient(BaseLLMClient):
    def __init__(self, api_key: str, config: ModelConfig):
        super().__init__(config)
        self.client = genai.Client(api_key=api_key)

    def _format_messages(
        self, messages: list[ChatMessage]
    ) -> tuple[str | None, list[genai_types.Content]]:
        """Gemini recibe el system prompt por config y usa los roles 'user' y 'model'."""
        system_instruction = None
        contents: list[genai_types.Content] = []

        for msg in messages:
            if msg.role == "system":
                system_instruction = msg.content
            else:
                # Mapear "assistant" de nuestro schema unificado al "model" de Gemini
                role = "model" if msg.role == "assistant" else "user"
                contents.append(
                    genai_types.Content(
                        role=role,
                        parts=[genai_types.Part.from_text(text=msg.content)],
                    )
                )

        return system_instruction, contents

    def _build_config(self, system_instruction: str | None) -> genai_types.GenerateContentConfig:
        return genai_types.GenerateContentConfig(
            system_instruction=system_instruction,
            temperature=self.config.temperature,
            max_output_tokens=self.config.max_tokens,
        )

    async def generate(self, messages: list[ChatMessage]) -> ModelResponse:
        try:
            system_instruction, contents = self._format_messages(messages)

            response = await _retry_async(
                lambda: self.client.aio.models.generate_content(
                    model=self.config.model,
                    contents=contents,
                    config=self._build_config(system_instruction),
                )
            )

            usage_data = None
            if response.usage_metadata:
                usage_data = {
                    "prompt_tokens": response.usage_metadata.prompt_token_count,
                    "candidates_tokens": response.usage_metadata.candidates_token_count,
                    "total_tokens": response.usage_metadata.total_token_count,
                }

            return ModelResponse(
                content=response.text or "",
                model=self.config.model,
                usage=usage_data,
            )
        except genai_errors.APIError as e:
            logger.error(f"Gemini API Error: {e}")
            return ModelResponse(content="", model=self.config.model, error=str(e))
        except Exception:
            logger.exception("Error inesperado en GeminiClient")
            return ModelResponse(content="", model=self.config.model, error="Internal Error")

    async def stream(self, messages: list[ChatMessage]) -> AsyncGenerator[str, None]:
        try:
            system_instruction, contents = self._format_messages(messages)

            async def request_stream() -> AsyncGenerator[str, None]:
                response_stream = await self.client.aio.models.generate_content_stream(
                    model=self.config.model,
                    contents=contents,
                    config=self._build_config(system_instruction),
                )
                async for chunk in response_stream:
                    if chunk.text:
                        yield chunk.text

            async for text in _retry_stream(request_stream):
                yield text

        except genai_errors.APIError as e:
            logger.error(f"Gemini Stream Error: {e}")
            yield f"\n[Stream interrumpido por error: {str(e)}]"
        except Exception:
            logger.exception("Error inesperado en stream de GeminiClient")
            yield "\n[Stream interrumpido por error interno]"
