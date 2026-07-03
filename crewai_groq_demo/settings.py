from functools import lru_cache
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict

# Published pricing, used to estimate (not bill) run cost. Named constants
# rather than inline literals since providers change pricing independently
# of this codebase — update here when they do.
GROQ_INPUT_PRICE_PER_MILLION_TOKENS = 0.59
GROQ_OUTPUT_PRICE_PER_MILLION_TOKENS = 0.79
TAVILY_PRICE_PER_CREDIT = 0.008  # 1 basic Tavily search ≈ 1 credit


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # Empty-string defaults (rather than required fields) so construction never
    # raises pydantic's generic ValidationError. Callers check for "" and raise
    # our own MissingAPIKeyError instead — TAVILY_API_KEY is only actually
    # needed when the researcher agent is used, so it can't be required here.
    groq_api_key: str = ""
    tavily_api_key: str = ""
    groq_model: str = "groq/llama-3.3-70b-versatile"
    groq_temperature: float = 0.2
    tavily_max_results: int = 5
    tavily_search_depth: Literal["basic", "advanced"] = "basic"


@lru_cache
def get_settings() -> Settings:
    return Settings()
