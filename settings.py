from typing import Any, Literal

import dspy
from pydantic import BaseModel, HttpUrl
from pydantic_settings import (
    BaseSettings,
)

class LLMSettings(BaseModel):
    api_url: HttpUrl = HttpUrl("")
    api_key: str = ""
    model_name: Literal[
        "azure-gpt-4o",
        "llama-31-70b",
        "llama-31-405b",
        "llama-33-70b",
        "phi-4-17b",
    ] = "azure-gpt-4o"
    temperature: float = 1.0
    max_tokens: int = 10000
    cache: bool = True
    max_retries: int = 3
    seed: int | None = None


class Settings(BaseSettings):
    llm_settings: LLMSettings = LLMSettings()

    @property
    def llm_client(self) -> dspy.LM:
        config: dict[str, Any] = {
            "model": f"openai/{self.llm_settings.model_name}",
            "api_base": str(self.llm_settings.api_url),
            "api_key": self.llm_settings.api_key,
            "temperature": self.llm_settings.temperature,
            "max_tokens": self.llm_settings.max_tokens,
            "max_retries": self.llm_settings.max_retries,
            "cache": self.llm_settings.cache
        }

        if self.llm_settings.seed:
            config["seed"] = self.llm_settings.seed

        return dspy.LM(**config)


settings = Settings()
