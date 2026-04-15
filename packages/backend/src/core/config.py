from functools import lru_cache
from typing import Annotated, List

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict


class Settings(BaseSettings):
    app_name: str = "AetherForecast Backend"
    app_env: str = "dev"
    log_level: str = "INFO"

    aws_region: str = "ap-southeast-1"
    aws_endpoint_url: str | None = None

    data_bucket: str = ""
    model_bucket: str = ""
    model_prefix: str = "chronos/models"
    model_s3_uri: str = "s3://aetherforecast-models/chronos-v1/model/"
    hf_model_fallback_id: str = "amazon/chronos-2"
    hf_tokenizer_fallback_id: str = "amazon/chronos-2"
    model_cache_dir: str = "/tmp/aetherforecast-model-cache"
    model_torch_dtype: str = "auto"
    inference_num_samples: int = 20
    require_s3_model: bool = True

    cognito_user_pool_id: str = ""
    cognito_client_id: str = ""
    cognito_region: str = "ap-southeast-1"

    cors_origins: Annotated[List[str], NoDecode] = Field(default_factory=lambda: ["*"])

    symbols_source: str = "binance"
    binance_base_url: str = "https://api.binance.com"
    binance_ws_url: str = "wss://stream.binance.com:9443"
    realtime_kline_interval: str = "1m"

    fetch_concurrency: int = 32
    fetch_symbol_limit: int = 0
    parquet_prefix: str = "market/klines"

    external_sentiment_enabled: bool = True
    external_sentiment_refresh_seconds: int = 900
    external_sentiment_force_refresh_per_request: bool = True
    external_sentiment_require_live_sources: bool = True
    sentiment_mode: str = "simple"
    sentiment_model_id: str = "ProsusAI/finbert"
    sentiment_cache_dir: str = "/tmp/aetherforecast-sentiment-cache"
    external_sentiment_news_rss_urls: Annotated[List[str], NoDecode] = Field(
        default_factory=lambda: [
            "https://www.coindesk.com/arc/outboundfeeds/rss/",
            "https://cointelegraph.com/rss",
            "https://www.reuters.com/markets/currencies/rss",
        ]
    )
    external_x_sentiment_endpoint: str | None = None
    external_geopolitical_sentiment_endpoint: str | None = None

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )

    @field_validator("cors_origins", mode="before")
    @classmethod
    def parse_cors_origins(cls, value: str | list[str]) -> list[str]:
        if isinstance(value, list):
            return value
        if isinstance(value, str):
            if not value.strip():
                return ["*"]
            return [item.strip() for item in value.split(",") if item.strip()]
        return ["*"]

    @field_validator("external_sentiment_news_rss_urls", mode="before")
    @classmethod
    def parse_news_rss_urls(cls, value: str | list[str]) -> list[str]:
        if isinstance(value, list):
            return [item.strip() for item in value if isinstance(item, str) and item.strip()]
        if isinstance(value, str):
            return [item.strip() for item in value.split(",") if item.strip()]
        return []

    @property
    def cognito_issuer(self) -> str:
        return (
            f"https://cognito-idp.{self.cognito_region}.amazonaws.com/"
            f"{self.cognito_user_pool_id}"
        )


@lru_cache
def get_settings() -> Settings:
    return Settings()
