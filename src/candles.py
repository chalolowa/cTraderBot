"""
Closed-candle fetcher.

`copy_rates_from_pos(symbol, tf, 1, n)` was returning candles ONE BAR BEHIND:
at 22:15 UTC the newest M15 bar was the one opened at 21:45, not 22:00. Signals were
therefore always a full candle late. Instead of trusting the `start_pos` semantics of
the broker wrapper, this helper asks for start_pos=0, then keeps only bars whose
(open time + timeframe) is already in the past. That is correct whether or not the
wrapper includes the still-forming candle, and it reports any remaining feed lag.
"""

import logging
from datetime import UTC, datetime

import pandas as pd

from config.ctrader_client import ctrader_client
from src.logger import _console

_last_logged_lag: dict[tuple[str, str], int] = {}


def timeframe_minutes(timeframe: str) -> int:
    tf = str(timeframe).upper()
    try:
        n = int(tf[1:])
    except ValueError:
        return 15
    return {"M": n, "H": n * 60, "D": n * 1440}.get(tf[:1], 15)


async def fetch_closed_candles(
    symbol: str, timeframe: str, count: int, now: datetime | None = None
) -> pd.DataFrame:
    """Return up to `count` fully closed candles, newest last. df.attrs['lag_bars'] is set."""
    now = now or datetime.now(UTC)
    tf_min = timeframe_minutes(timeframe)

    try:
        df = await ctrader_client.copy_rates_from_pos(symbol, timeframe, 0, count + 2)
    except (ValueError, TypeError):
        df = await ctrader_client.copy_rates_from_pos(symbol, timeframe, 1, count + 2)

    if df is None or df.empty or not isinstance(df.index, pd.DatetimeIndex):
        return df

    idx = df.index.tz_localize("UTC") if df.index.tz is None else df.index.tz_convert("UTC")
    now_ts = pd.Timestamp(now)
    closed = (idx + pd.Timedelta(minutes=tf_min)) <= now_ts
    df = df[closed].tail(count)
    if df.empty:
        return df

    lag_bars = 0
    if tf_min < 1440:
        expected_open = now_ts.floor(f"{tf_min}min") - pd.Timedelta(minutes=tf_min)
        last_open = idx[closed][-1]
        lag_bars = max(0, int(round((expected_open - last_open).total_seconds() / 60 / tf_min)))
    df.attrs["lag_bars"] = lag_bars

    key = (symbol, timeframe)
    if _last_logged_lag.get(key) != lag_bars:
        _last_logged_lag[key] = lag_bars
        level = logging.WARNING if lag_bars else logging.INFO
        _console(
            f"Candle feed {symbol} {timeframe}: last closed bar opened {idx[closed][-1]} "
            f"({'FEED IS ' + str(lag_bars) + ' BAR(S) BEHIND' if lag_bars else 'up to date'})",
            level=level,
        )
    return df