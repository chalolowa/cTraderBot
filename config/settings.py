"""Application configuration loaded from environment variables."""

from pathlib import Path

from pydantic import AliasChoices, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=Path(__file__).resolve().parent.parent / ".env",
        env_file_encoding="utf-8",
        env_ignore_empty=True,
        extra="ignore",
    )

    # --- cTrader Open API ---------------------------------------------
    # Obtained once via scripts/ctrader_oauth_setup.py
    CTRADER_CLIENT_ID: str = Field(
        default="", description="Open API application client ID"
    )
    CTRADER_CLIENT_SECRET: str = Field(
        default="", description="Open API application client secret"
    )
    CTRADER_ACCOUNT_ID: int | None = Field(
        default=None,
        description="cTID trader account ID, resolved once by ctrader_oauth_setup.py",
    )
    CTRADER_TRADER_LOGIN: int | None = Field(
        default=None,
        description="Numeric trader login, used only during initial OAuth setup",
    )
    CTRADER_ENVIRONMENT: str = Field(default="demo", description="'demo' or 'live'")
    CTRADER_PORT: int = Field(default=5035)
    # First-run fallback only; after the first successful connect, rotated
    # tokens live in the SQLite token store (see ctrader_token_store.py) and
    # these values are no longer read.
    CTRADER_ACCESS_TOKEN: str | None = Field(default=None)
    CTRADER_REFRESH_TOKEN: str | None = Field(default=None)
    CTRADER_TOKEN_EXPIRES_AT: int | None = Field(
        default=None, description="Unix timestamp"
    )

    @property
    def CTRADER_HOST(self) -> str:
        return (
            "live.ctraderapi.com"
            if self.CTRADER_ENVIRONMENT.lower() == "live"
            else "demo.ctraderapi.com"
        )

    # Telegram
    TELEGRAM_BOT_TOKEN: str = ""
    TELEGRAM_CHAT_ID: str = ""

    # Trading Parameters
    RISK_PER_TRADE_PCT: float = Field(1.0, description="Risk % per trade")
    MAX_OPEN_TRADES: int = Field(3, description="Max concurrent day trades")
    BASE_CURRENCY: str = Field("USD", description="Account currency")

    # Forex weekly session and the bot's weekday day-trading close (US/Eastern).
    FOREX_WEEK_OPEN_HOUR: int = 17  # Sunday 5:00 PM Eastern
    FOREX_WEEK_CLOSE_HOUR: int = 17  # Friday 5:00 PM Eastern
    MARKET_OPEN_HOUR: int = (
        8  # Legacy setting; retained for configuration compatibility
    )
    MARKET_CLOSE_HOUR: int = 16  # 4:00 PM Eastern
    CLOSE_BUFFER_MINUTES: int = (
        30  # Close all 30 min before market close (3:30 PM Eastern)
    )

    # Scanner
    SCANNER_INTERVAL_MIN: int = 30
    MIN_ATR_PIPS: float = 10.0
    MAX_SPREAD_PIPS: float = 3.0
    MIN_ADX: float = 20.0

    # Database
    DB_PATH: str = Field(
        str(Path(__file__).resolve().parent.parent / "data" / "trades.db"),
        description="SQLite path",
    )
    MONGO_URI: str | None = Field(
        default=None,
        validation_alias=AliasChoices("MONGO_URI", "MONGODB_URI"),
    )
    MONGO_DATABASE: str = "mt5_tradebot"
    MONGO_COLLECTION: str = "ctraderbot_events"
    OUTBOX_SYNC_INTERVAL_SECONDS: float = 5.0
    OUTBOX_BATCH_SIZE: int = 100
    API_HOST: str = "127.0.0.1"
    API_PORT: int = 8000


settings = Settings()
