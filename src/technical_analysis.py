"""
Technical Analysis module with pandas-ta support and pure pandas fallbacks.
Calculates SMA, EMA, RSI, MACD, ATR, ADX and detects crossovers.
"""
from typing import Any

import numpy as np
import pandas as pd

try:
    import pandas_ta as ta
except ImportError:
    ta = None


class TechnicalAnalyzer:
    """
    Computes technical indicators and generates signal states.
    All methods accept a DataFrame and return enriched DataFrame.
    """

    @staticmethod
    def add_sma(df: pd.DataFrame, period: int = 20) -> pd.DataFrame:
        df = df.copy()
        if ta is not None:
            res = ta.sma(df["close"], length=period)
            if res is not None:
                df[f"SMA_{period}"] = res
                return df
        df[f"SMA_{period}"] = df["close"].rolling(window=period).mean()
        return df

    @staticmethod
    def add_ema(df: pd.DataFrame, period: int = 20) -> pd.DataFrame:
        df = df.copy()
        if ta is not None:
            res = ta.ema(df["close"], length=period)
            if res is not None:
                df[f"EMA_{period}"] = res
                return df
        df[f"EMA_{period}"] = df["close"].ewm(span=period, adjust=False).mean()
        return df

    @staticmethod
    def add_rsi(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
        df = df.copy()
        if ta is not None:
            res = ta.rsi(df["close"], length=period)
            if res is not None:
                df[f"RSI_{period}"] = res
                return df
        delta = df["close"].diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        avg_gain = gain.rolling(window=period, min_periods=period).mean()
        avg_loss = loss.rolling(window=period, min_periods=period).mean()
        rs = avg_gain / avg_loss.replace(0, np.nan)
        df[f"RSI_{period}"] = 100 - (100 / (1 + rs))
        return df

    @staticmethod
    def add_macd(df: pd.DataFrame, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
        df = df.copy()
        if ta is not None:
            macd = ta.macd(df["close"], fast=fast, slow=slow, signal=signal)
            if macd is not None and not macd.empty:
                df = pd.concat([df, macd], axis=1)
                return df
        fast_ema = df["close"].ewm(span=fast, adjust=False).mean()
        slow_ema = df["close"].ewm(span=slow, adjust=False).mean()
        macd_line = fast_ema - slow_ema
        signal_line = macd_line.ewm(span=signal, adjust=False).mean()
        hist = macd_line - signal_line
        df[f"MACD_{fast}_{slow}_{signal}"] = macd_line
        df[f"MACDh_{fast}_{slow}_{signal}"] = hist
        df[f"MACDs_{fast}_{slow}_{signal}"] = signal_line
        return df

    @staticmethod
    def add_atr(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
        df = df.copy()
        if ta is not None:
            res = ta.atr(df["high"], df["low"], df["close"], length=period)
            if res is not None:
                df[f"ATR_{period}"] = res
                return df
        high_low = df["high"] - df["low"]
        high_close = (df["high"] - df["close"].shift()).abs()
        low_close = (df["low"] - df["close"].shift()).abs()
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        df[f"ATR_{period}"] = tr.rolling(window=period).mean()
        return df

    @staticmethod
    def add_adx(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
        df = df.copy()
        if ta is not None:
            adx = ta.adx(df["high"], df["low"], df["close"], length=period)
            if adx is not None and not adx.empty:
                df = pd.concat([df, adx], axis=1)
                return df
        # Simplified ADX fallback
        up = df["high"].diff()
        down = -df["low"].diff()
        plus_dm = np.where((up > down) & (up > 0), up, 0.0)
        minus_dm = np.where((down > up) & (down > 0), down, 0.0)
        high_low = df["high"] - df["low"]
        high_close = (df["high"] - df["close"].shift()).abs()
        low_close = (df["low"] - df["close"].shift()).abs()
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        tr_smooth = pd.Series(tr).rolling(window=period).mean()
        plus_di = 100 * (pd.Series(plus_dm, index=df.index).rolling(window=period).mean() / tr_smooth)
        minus_di = 100 * (pd.Series(minus_dm, index=df.index).rolling(window=period).mean() / tr_smooth)
        dx = (100 * (plus_di - minus_di).abs() / (plus_di + minus_di)).fillna(0)
        df[f"ADX_{period}"] = dx.rolling(window=period).mean()
        df[f"DMP_{period}"] = plus_di
        df[f"DMN_{period}"] = minus_di
        return df

    @staticmethod
    def add_bollinger(df: pd.DataFrame, period: int = 20, std: float = 2.0) -> pd.DataFrame:
        df = df.copy()
        if ta is not None:
            bbands = ta.bbands(df["close"], length=period, std=std)
            if bbands is not None and not bbands.empty:
                df = pd.concat([df, bbands], axis=1)
                return df
        mid = df["close"].rolling(window=period).mean()
        rstd = df["close"].rolling(window=period).std()
        df[f"BBL_{period}_{std}"] = mid - (std * rstd)
        df[f"BBM_{period}_{std}"] = mid
        df[f"BBU_{period}_{std}"] = mid + (std * rstd)
        return df

    @staticmethod
    def detect_crossover(
            df: pd.DataFrame,
            fast_col: str,
            slow_col: str
    ) -> pd.DataFrame:
        """
        Detects crossovers between two series.
        Adds columns: crossover_up, crossover_down
        """
        df = df.copy()
        if fast_col not in df.columns or slow_col not in df.columns:
            df["crossover_up"] = False
            df["crossover_down"] = False
            return df

        df["crossover_up"] = (
                (df[fast_col] > df[slow_col]) &
                (df[fast_col].shift(1) <= df[slow_col].shift(1))
        )
        df["crossover_down"] = (
                (df[fast_col] < df[slow_col]) &
                (df[fast_col].shift(1) >= df[slow_col].shift(1))
        )
        return df

    @classmethod
    def analyze_instrument(
            cls,
            df: pd.DataFrame,
            strategy_config: dict[str, Any] | None = None
    ) -> pd.DataFrame:
        """
        Full pipeline: compute indicators required by strategy_config or defaults.
        """
        if df.empty or len(df) < 5:
            return df.copy()

        df = df.copy()

        # Parse indicator periods from config if available
        sma_periods = {9, 21}
        if strategy_config:
            rules = strategy_config.get("rules", {})
            for direction_rules in rules.values():
                for cond in direction_rules:
                    ind = cond.get("indicator", "")
                    if ind == "sma" and "period" in cond:
                        sma_periods.add(int(cond["period"]))
                    if "reference" in cond and cond.get("reference") == "sma" and "ref_period" in cond:
                        sma_periods.add(int(cond["ref_period"]))

        for period in sorted(sma_periods):
            df = cls.add_sma(df, period)

        df = cls.add_rsi(df, 14)
        df = cls.add_atr(df, 14)
        df = cls.add_adx(df, 14)
        df = cls.add_macd(df)

        # Detect standard SMA 9/21 crossovers
        df = cls.detect_crossover(df, "SMA_9", "SMA_21")

        return df

    @classmethod
    def get_latest_signal(cls, df: pd.DataFrame) -> dict[str, Any]:
        """Extract latest indicator values and signal states."""
        if df.empty:
            return {}

        latest = df.iloc[-1]

        return {
            "sma_9": latest.get("SMA_9"),
            "sma_21": latest.get("SMA_21"),
            "rsi_14": latest.get("RSI_14"),
            "atr_14": latest.get("ATR_14"),
            "adx_14": latest.get("ADX_14"),
            "macd": latest.get("MACD_12_26_9"),
            "macd_signal": latest.get("MACDs_12_26_9"),
            "crossover_up": bool(latest.get("crossover_up", False)),
            "crossover_down": bool(latest.get("crossover_down", False)),
            "close": float(latest["close"]) if "close" in latest else 0.0,
            "timestamp": latest.name.isoformat() if hasattr(latest.name, 'isoformat') else str(latest.name)
        }