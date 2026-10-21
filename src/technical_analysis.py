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
        rsi = 100 - (100 / (1 + rs))
        rsi = rsi.mask((avg_loss == 0) & (avg_gain > 0), 100.0)
        rsi = rsi.mask((avg_loss == 0) & (avg_gain == 0), 50.0)
        df[f"RSI_{period}"] = rsi
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

        sma_periods = {9, 21}
        ema_periods: set[int] = set()
        rsi_periods = {14}
        adx_periods = {14}
        atr_periods = {14}
        if strategy_config:
            entry_rules = strategy_config.get("entry_rules", {})
            for direction_rules in entry_rules.values():
                for cond in direction_rules.get("conditions", []):
                    indicator = str(cond.get("indicator", "")).upper()
                    period = int(cond.get("period", 14))
                    if indicator == "SMA":
                        sma_periods.add(period)
                        if str(cond.get("reference", "")).upper() == "SMA":
                            sma_periods.add(int(cond.get("ref_period", 21)))
                    elif indicator == "EMA":
                        ema_periods.add(period)
                    elif indicator == "RSI":
                        rsi_periods.add(period)
                    elif indicator == "ADX":
                        adx_periods.add(period)
                    elif indicator == "ATR":
                        atr_periods.add(period)

            stop_loss = strategy_config.get("exit_rules", {}).get("stop_loss", {})
            if stop_loss.get("type") == "ATR_MULTIPLE":
                atr_periods.add(int(stop_loss.get("atr_period", 14)))

        for period in sorted(sma_periods):
            df = cls.add_sma(df, period)
        for period in sorted(ema_periods):
            df = cls.add_ema(df, period)
        for period in sorted(rsi_periods):
            df = cls.add_rsi(df, period)
        for period in sorted(atr_periods):
            df = cls.add_atr(df, period)
        for period in sorted(adx_periods):
            df = cls.add_adx(df, period)
        df = cls.add_macd(df)

        # Detect standard SMA 9/21 crossovers
        df = cls.detect_crossover(df, "SMA_9", "SMA_21")

        return df

    @classmethod
    def compute_trend_bias(
            cls,
            df: pd.DataFrame,
            ema_fast: int = 20,
            ema_slow: int = 50,
            slope_bars: int = 5,
            adx_period: int = 14,
            min_adx: float = 20.0,
            drift_lookback_bars: int = 12,
    ) -> dict[str, Any]:
        """
        Higher-timeframe bias (meant for H1 candles).

        bias is "long" / "short" / "neutral". A direction needs ALL of:
          * fast EMA on the right side of slow EMA
          * close on the right side of slow EMA
          * slow EMA sloping the same way over `slope_bars`
          * ADX >= min_adx (otherwise the market is ranging -> neutral)
        drift_atr is the net move over `drift_lookback_bars` in ATR units, i.e. what the
        "day so far" has done; the caller uses it to veto counter-drift entries.
        """
        need = max(ema_slow, adx_period * 2, drift_lookback_bars) + slope_bars + 2
        if df is None or df.empty or len(df) < need:
            return {"bias": "neutral", "reason": "insufficient_htf_data", "drift_atr": 0.0}

        d = cls.add_ema(df, ema_fast)
        d = cls.add_ema(d, ema_slow)
        d = cls.add_adx(d, adx_period)
        d = cls.add_atr(d, adx_period)

        last = d.iloc[-1]
        fast_v = last.get(f"EMA_{ema_fast}")
        slow_v = last.get(f"EMA_{ema_slow}")
        adx_v = last.get(f"ADX_{adx_period}")
        atr_v = last.get(f"ATR_{adx_period}")
        slow_prev = d[f"EMA_{ema_slow}"].iloc[-1 - slope_bars]
        if any(v is None or pd.isna(v) for v in (fast_v, slow_v, adx_v, atr_v, slow_prev)):
            return {"bias": "neutral", "reason": "htf_indicators_not_ready", "drift_atr": 0.0}

        close = float(last["close"])
        fast_v, slow_v, adx_v, atr_v = float(fast_v), float(slow_v), float(adx_v), float(atr_v)
        slope = slow_v - float(slow_prev)

        long_ok = fast_v > slow_v and close > slow_v and slope > 0
        short_ok = fast_v < slow_v and close < slow_v and slope < 0
        trending = adx_v >= min_adx

        bias = "neutral"
        reason = "ranging_or_mixed"
        if trending and long_ok:
            bias, reason = "long", "trend_up"
        elif trending and short_ok:
            bias, reason = "short", "trend_down"
        elif not trending:
            reason = f"htf_adx_{adx_v:.1f}_below_{min_adx:g}"

        ref_close = float(d["close"].iloc[-1 - drift_lookback_bars])
        drift_atr = (close - ref_close) / atr_v if atr_v > 0 else 0.0

        return {
            "bias": bias,
            "reason": reason,
            "adx": round(adx_v, 1),
            "ema_fast": fast_v,
            "ema_slow": slow_v,
            "slope": slope,
            "drift_atr": round(drift_atr, 2),
            "bar_time": str(d.index[-1]),
        }

    @classmethod
    def detect_pullback(
            cls,
            df: pd.DataFrame,
            direction: str,
            lookback_bars: int = 3,
            tolerance_atr: float = 0.25,
            min_adx: float = 20.0,
            rsi_min: float = 45.0,
            rsi_max: float = 85.0,
            slope_bars: int = 3,
            fast: int = 9,
            slow: int = 21,
            atr_period: int = 14,
            adx_period: int = 14,
            rsi_period: int = 14,
    ) -> dict[str, Any]:
        """
        Trend-continuation entry: an established M15 trend dips to the fast SMA and the
        latest closed candle resumes in the trend direction. Returns {"matched", "checks"}.
        This is the entry a discretionary trader takes when a trend is already running
        and no fresh crossover will ever appear.
        """
        cols = [f"SMA_{fast}", f"SMA_{slow}", f"ATR_{atr_period}", f"ADX_{adx_period}", f"RSI_{rsi_period}"]
        need = lookback_bars + slope_bars + 2
        if df is None or len(df) < need or any(c not in df.columns for c in cols):
            return {"matched": False, "checks": [("data", False, "insufficient/missing indicators")]}

        last = df.iloc[-1]
        prior = df.iloc[-1 - lookback_bars:-1]
        sma_f, sma_s = float(last[f"SMA_{fast}"]), float(last[f"SMA_{slow}"])
        sma_s_prev = float(df[f"SMA_{slow}"].iloc[-1 - slope_bars])
        atr, adx, rsi = float(last[f"ATR_{atr_period}"]), float(last[f"ADX_{adx_period}"]), float(last[f"RSI_{rsi_period}"])
        if any(pd.isna(v) for v in (sma_f, sma_s, sma_s_prev, atr, adx, rsi)) or atr <= 0:
            return {"matched": False, "checks": [("data", False, "indicators not ready")]}

        tol = tolerance_atr * atr
        is_long = direction == "long"
        if is_long:
            trend = sma_f > sma_s > sma_s_prev
            touched = bool((prior["low"] <= prior[f"SMA_{fast}"] + tol).any())
            resumed = bool(last["close"] > last["open"] and last["close"] > sma_f)
            rsi_ok = rsi_min <= rsi <= rsi_max
        else:
            trend = sma_f < sma_s < sma_s_prev
            touched = bool((prior["high"] >= prior[f"SMA_{fast}"] - tol).any())
            resumed = bool(last["close"] < last["open"] and last["close"] < sma_f)
            rsi_ok = (100 - rsi_max) <= rsi <= (100 - rsi_min)

        checks = [
            ("trend", trend, f"SMA{fast} {'>' if is_long else '<'} SMA{slow} and SMA{slow} sloping"),
            ("adx", adx >= min_adx, f"ADX {adx:.1f} >= {min_adx:g}"),
            ("pullback", touched, f"price touched SMA{fast} zone in last {lookback_bars} bars"),
            ("resume", resumed, "latest candle closed back in trend direction beyond the fast SMA"),
            ("rsi", rsi_ok, f"RSI {rsi:.1f} in range"),
        ]
        return {"matched": all(ok for _, ok, _ in checks), "checks": checks}

    @classmethod
    def reversal_votes(
            cls,
            df: pd.DataFrame,
            is_long: bool,
            body_atr: float = 1.0,
            fast: int = 9,
            atr_period: int = 14,
            rsi_period: int = 14,
    ) -> dict[str, Any]:
        """
        Count independent signs that the latest CLOSED candle turned against an open trade:
          ma     - closed back through the fast SMA against the trade
          rsi    - RSI crossed to the wrong side of 50
          candle - opposite candle with a body of at least `body_atr` x ATR
        """
        cols = [f"SMA_{fast}", f"ATR_{atr_period}", f"RSI_{rsi_period}"]
        if df is None or df.empty or any(c not in df.columns for c in cols):
            return {"votes": 0, "detail": {}}
        last = df.iloc[-1]
        sma, atr, rsi = float(last[f"SMA_{fast}"]), float(last[f"ATR_{atr_period}"]), float(last[f"RSI_{rsi_period}"])
        if any(pd.isna(v) for v in (sma, atr, rsi)) or atr <= 0:
            return {"votes": 0, "detail": {}}
        body = float(last["close"]) - float(last["open"])
        if is_long:
            detail = {"ma": float(last["close"]) < sma, "rsi": rsi < 50, "candle": -body >= body_atr * atr}
        else:
            detail = {"ma": float(last["close"]) > sma, "rsi": rsi > 50, "candle": body >= body_atr * atr}
        return {"votes": sum(detail.values()), "detail": detail}

    @classmethod
    def momentum_fading(
            cls,
            df: pd.DataFrame,
            is_long: bool,
            bars: int = 3,
            max_progress_atr: float = 0.5,
            adx_period: int = 14,
            atr_period: int = 14,
    ) -> bool:
        """True when ADX is lower than `bars` candles ago AND price made little progress with the trade."""
        adx_col, atr_col = f"ADX_{adx_period}", f"ATR_{atr_period}"
        if df is None or len(df) < bars + 2 or adx_col not in df.columns or atr_col not in df.columns:
            return False
        adx_now, adx_then = df[adx_col].iloc[-1], df[adx_col].iloc[-1 - bars]
        atr = df[atr_col].iloc[-1]
        if any(pd.isna(v) for v in (adx_now, adx_then, atr)) or float(atr) <= 0:
            return False
        sign = 1.0 if is_long else -1.0
        progress = sign * (float(df["close"].iloc[-1]) - float(df["close"].iloc[-1 - bars])) / float(atr)
        return bool(float(adx_now) < float(adx_then) and progress <= max_progress_atr)

    @classmethod
    def get_latest_signal(cls, df: pd.DataFrame) -> dict[str, Any]:
        """Extract latest indicator values and signal states."""
        if df.empty:
            return {}

        latest = df.iloc[-1]

        signal = {
            column.lower(): latest[column]
            for column in df.columns
            if column.startswith(("SMA_", "EMA_", "RSI_", "ATR_", "ADX_"))
        }
        signal.update(
            {
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
                "timestamp": latest.name.isoformat()
                if hasattr(latest.name, "isoformat")
                else str(latest.name),
            }
        )
        return signal