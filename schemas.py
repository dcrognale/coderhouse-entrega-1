from typing import Literal, Any
from pydantic import BaseModel, Field

class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str

class ModelConfig(BaseModel):
    model: str
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    max_tokens: int = Field(default=1024, gt=0)

class ModelResponse(BaseModel):
    content: str
    model: str
    usage: dict[str, Any] | None = None
    error: str | None = None
