"""Configuration management for the Financial Filing & Market Intelligence Agent."""

from __future__ import annotations

import os
from pathlib import Path
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Typed application settings loaded from environment variables or .env file."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # LLM Settings
    google_api_key: str = Field(default="", alias="GOOGLE_API_KEY")
    gemini_model: str = Field(default="gemini-2.5-flash", alias="GEMINI_MODEL")

    # Market API Settings
    alphavantage_api_key: str = Field(default="demo", alias="ALPHAVANTAGE_API_KEY")
    alphavantage_base_url: str = Field(
        default="https://www.alphavantage.co/query",
        alias="ALPHAVANTAGE_BASE_URL",
    )
    api_timeout_seconds: float = Field(default=10.0)
    max_api_retries: int = Field(default=3)

    # Agent Reliability & Validation Settings
    max_extraction_retries: int = Field(default=3)
    circuit_breaker_failure_threshold: int = Field(default=3)
    circuit_breaker_recovery_timeout: float = Field(default=30.0)

    # Observability (OpenTelemetry + LangSmith)
    otel_service_name: str = Field(
        default="finagent-langgraph-agent", alias="OTEL_SERVICE_NAME"
    )
    otel_traces_export_path: Path = Field(
        default=Path("logs/otel_traces.jsonl"), alias="OTEL_TRACES_EXPORT_PATH"
    )
    langchain_tracing_v2: bool = Field(default=False, alias="LANGCHAIN_TRACING_V2")
    langchain_api_key: str = Field(default="", alias="LANGCHAIN_API_KEY")
    langchain_project: str = Field(
        default="finagent-reliable-prod", alias="LANGCHAIN_PROJECT"
    )

    def configure_langsmith(self) -> bool:
        """Enable LangSmith environment flags if API key and flag are configured."""
        if self.langchain_tracing_v2 and self.langchain_api_key:
            os.environ["LANGCHAIN_TRACING_V2"] = "true"
            os.environ["LANGCHAIN_API_KEY"] = self.langchain_api_key
            os.environ["LANGCHAIN_PROJECT"] = self.langchain_project
            return True
        return False


def get_settings() -> Settings:
    """Return a fresh Settings instance."""
    return Settings()
