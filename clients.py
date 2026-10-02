import abc
import logging
from typing import AsyncGenerator
from openai import AsyncOpenAI, APIError as OpenAIError
from anthropic import AsyncAnthropic, APIError as AnthropicError

from schemas import ChatMessage, ModelConfig, ModelResponse

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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
            response = await self.client.chat.completions.create(
                model=self.config.model,
                messages=formatted_messages,
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
            )
            return ModelResponse(
                content=response.choices[0].message.content or "",
                model=self.config.model,
                usage=response.usage.model_dump() if response.usage else None
            )
        except OpenAIError as e:
            logger.error(f"OpenAI API Error: {e}")
            return ModelResponse(content="", model=self.config.model, error=str(e))
        except Exception as e:
            logger.exception("Error inesperado en OpenAIClient")
            return ModelResponse(content="", model=self.config.model, error="Internal Error")

    async def stream(self, messages: list[ChatMessage]) -> AsyncGenerator[str, None]:
        try:
            formatted_messages = [msg.model_dump() for msg in messages]
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
        except OpenAIError as e:
            logger.error(f"OpenAI Stream Error: {e}")
            yield f"\n[Stream interrumpido por error: {str(e)}]"

class AnthropicClient(BaseLLMClient):
    def __init__(self, api_key: str, config: ModelConfig):
        super().__init__(config)
        self.client = AsyncAnthropic(api_key=api_key)

    def _format_messages(self, messages: list[ChatMessage]) -> tuple[str, list[dict[str, str]]]:
        system_prompt = ""
        anthropic_messages = []
        for msg in messages:
            if msg.role == "system":
                system_prompt = msg.content
            else:
                anthropic_messages.append({
                    "role": msg.role,
                    "content": msg.content
                })
        return system_prompt, anthropic_messages

    async def generate(self, messages: list[ChatMessage]) -> ModelResponse:
        try:
            system, formatted_messages = self._format_messages(messages)
            response = await self.client.messages.create(
                model=self.config.model,
                system=system,
                messages=formatted_messages,
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
            )
            return ModelResponse(
                content=response.content[0].text,
                model=self.config.model,
                usage={
                    "input_tokens": response.usage.input_tokens, 
                    "output_tokens": response.usage.output_tokens
                }
            )
        except AnthropicError as e:
            logger.error(f"Anthropic API Error: {e}")
            return ModelResponse(content="", model=self.config.model, error=str(e))

    async def stream(self, messages: list[ChatMessage]) -> AsyncGenerator[str, None]:
        try:
            system, formatted_messages = self._format_messages(messages)
            async with self.client.messages.stream(
                model=self.config.model,
                system=system,
                messages=formatted_messages,
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
            ) as stream:
                async for text in stream.text_stream:
                    yield text
        except AnthropicError as e:
            logger.error(f"Anthropic Stream Error: {e}")
            yield f"\n[Stream interrumpido por error: {str(e)}]"
