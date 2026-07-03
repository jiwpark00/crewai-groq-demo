from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


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


@lru_cache
def get_settings() -> Settings:
    return Settings()
