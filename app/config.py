from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Settings for the application."""

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")
    API_TITLE: str = "Langchain Async API Server"
    API_VERSION: str = "0.1.0"
    DATABASE_URL: str
    OPENAI_API_KEY: str


@lru_cache()
def get_settings() -> Settings:
    return Settings()  # type: ignore
