"""
Market Scanner: Filters currency pairs every 30 minutes.
Ranks instruments by setup quality using ATR, spread, ADX, and trend alignment.
"""

import json
import logging
import traceback
from datetime import UTC, datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from config.ctrader_client import ctrader_client, normalize_symbol
from config.settings import settings
from src.logger import _console
from src.market_hours import market_hours
from src.strategy_engine import strategy_engine
from src.technical_analysis import TechnicalAnalyzer

CACHE_FILE = Path(settings.DB_PATH).parent / "scan_cache.json"


class MarketScanner:
    def __init__(self):
        self.ta = TechnicalAnalyzer()
        self.instruments = strategy_engine.get_instruments()
        self.last_results: list[dict[str, Any]] = []
        self._subscribed = False

    async def _ensure_subscribed(self) -> None:
        # cTrader prices arrive by push subscription, not on-demand request,
        # so every instrument we might scan has to be subscribed once.
        if not self._subscribed:
            await ctrader_client.ensure_subscribed(self.instruments)
            self._subscribed = True

    def get_cached_scan(
        self, max_age_seconds: int = 1800
    ) -> tuple[datetime, list[dict[str, Any]]] | None:
        """Load scan results from durable JSON cache if valid within max_age_seconds."""
        if not CACHE_FILE.exists():
            return None
        try:
            with open(CACHE_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
            ts_str = data.get("timestamp")
            if not ts_str:
                return None
            ts = datetime.fromisoformat(ts_str)
            now = datetime.now(UTC)
            if ts.tzinfo is None:
                ts = ts.replace(tzinfo=UTC)
            age = (now - ts).total_seconds()
            if age < max_age_seconds:
                return ts, data.get("results", [])
        except Exception as exc:
            _console(f"Failed to read scan cache: {exc}", level=logging.WARNING)
        return None

    def save_cached_scan(self, results: list[dict[str, Any]]) -> None:
        """Persist scan results to durable JSON cache."""
        try:
            CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
            payload = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "results": results,
            }
            with open(CACHE_FILE, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2)
        except Exception as exc:
            _console(f"Failed to write scan cache: {exc}", level=logging.WARNING)

    async def scan_all(self) -> list[dict[str, Any]]:
        """
        Scan all configured instruments and return ranked opportunities.
        Runs every 30 minutes.
        """
        if not market_hours.is_market_open():
            _console(
                "Forex market closed (weekly session) — scan skipped",
                level=logging.INFO,
            )
            return []

        await self._ensure_subscribed()

        _console(
            f"Starting market scan across {len(self.instruments)} configured instruments...",
            level=logging.INFO,
        )
        results = []

        for instrument in self.instruments:
            try:
                score = await self._analyze_instrument(instrument)
                if score.get("tradable"):
                    results.append(score)
            except Exception as exc:
                err_tb = traceback.format_exc()
                _console(
                    f"Error scanning instrument {instrument}: {exc}\n{err_tb}",
                    level=logging.ERROR,
                )
                continue

        # Sort by composite score (higher = better setup)
        results.sort(key=lambda x: x["composite_score"], reverse=True)
        self.last_results = results
        self.save_cached_scan(results)

        # A9 logging
        tradable_summary = ", ".join(
            f"{r['instrument']}(score={r['composite_score']})" for r in results
        )
        _console(
            f"{len(results)}/{len(self.instruments)} instruments tradable this scan"
            + (f": [{tradable_summary}]" if results else " (none passed filters)"),
            level=logging.INFO,
        )
        return results

    async def _analyze_instrument(self, instrument: str) -> dict[str, Any]:
        """Analyze single instrument for trade readiness with non-blocking ctrader calls."""
        norm_sym = normalize_symbol(instrument)
        try:
            df = await ctrader_client.copy_rates_from_pos(instrument, "M15", 1, 100)
        except Exception as exc:
            _console(
                f"Failed to fetch rates for {instrument}: {exc}", level=logging.ERROR
            )
            raise

        if df.empty or len(df) < 30:
            _console(
                f"Scanner rejected {instrument}: insufficient candle data (count={len(df)})",
                level=logging.INFO,
            )
            return {
                "tradable": False,
                "instrument": norm_sym,
                "reason": "insufficient_data",
            }

        # Calculate indicators
        df = self.ta.analyze_instrument(df, {})
        latest = df.iloc[-1]

        # Get current spread asynchronously
        pip_factor = 100.0 if "JPY" in norm_sym else 10000.0
        try:
            tick = ctrader_client.get_tick(instrument)
            spread_pips = (float(tick.ask) - float(tick.bid)) * pip_factor
        except Exception as exc:
            _console(
                f"Failed to fetch tick for spread check on {instrument}: {exc}",
                level=logging.WARNING,
            )
            spread_pips = 999.0

        # Extract indicators with NaN safety (B28)
        atr_raw = latest.get("ATR_14")
        adx_raw = latest.get("ADX_14")
        rsi_raw = latest.get("RSI_14")

        atr = float(atr_raw) if (atr_raw is not None and not pd.isna(atr_raw)) else None
        adx = float(adx_raw) if (adx_raw is not None and not pd.isna(adx_raw)) else None
        rsi = float(rsi_raw) if (rsi_raw is not None and not pd.isna(rsi_raw)) else 50.0

        tradable = True
        reasons = []

        # Check filters with explicit NaN checks
        if pd.isna(spread_pips) or spread_pips > settings.MAX_SPREAD_PIPS:
            tradable = False
            reasons.append(
                f"Spread too high: actual={spread_pips:.1f} pips, max={settings.MAX_SPREAD_PIPS:.1f}"
            )

        if atr is None or (atr * pip_factor < settings.MIN_ATR_PIPS):
            tradable = False
            atr_pips_str = f"{atr * pip_factor:.1f}" if atr is not None else "NaN"
            reasons.append(
                f"ATR too low: actual={atr_pips_str} pips, min={settings.MIN_ATR_PIPS:.1f}"
            )

        if adx is None or (adx < settings.MIN_ADX):
            tradable = False
            adx_str = f"{adx:.1f}" if adx is not None else "NaN"
            reasons.append(f"ADX too low: actual={adx_str}, min={settings.MIN_ADX:.1f}")

        # A8 logging: Log any filter failure with actual vs threshold
        if not tradable:
            _console(
                f"Instrument {norm_sym} filtered out: {'; '.join(reasons)}",
                level=logging.INFO,
            )

        # Composite scoring (0-100)
        score = 0
        if adx is not None:
            if adx > 25:
                score += 30
            elif adx > 20:
                score += 20

        if 40 < rsi < 60:
            score += 20
        elif 30 < rsi < 70:
            score += 10

        if spread_pips < 2.0:
            score += 25
        elif spread_pips < 3.0:
            score += 15

        if atr is not None and (atr * pip_factor > 15):
            score += 25

        sma_9 = (
            float(latest.get("SMA_9", 0))
            if not pd.isna(latest.get("SMA_9", 0))
            else 0.0
        )
        sma_21 = (
            float(latest.get("SMA_21", 0))
            if not pd.isna(latest.get("SMA_21", 0))
            else 0.0
        )
        trend = (
            "bullish"
            if sma_9 > sma_21
            else ("bearish" if sma_9 < sma_21 else "neutral")
        )

        return {
            "tradable": tradable,
            "instrument": norm_sym,
            "composite_score": score,
            "trend": trend,
            "adx": round(adx, 1) if adx is not None else 0.0,
            "rsi": round(rsi, 1),
            "atr_pips": round(atr * pip_factor, 1) if atr is not None else 0.0,
            "spread_pips": round(spread_pips, 1),
            "sma_9": round(sma_9, 5),
            "sma_21": round(sma_21, 5),
            "reasons": reasons,
            "timestamp": datetime.now(UTC).isoformat(),
        }

    def get_top_opportunities(self, n: int = 3) -> list[dict[str, Any]]:
        """Get top N opportunities from memory or persistent cache."""
        if self.last_results:
            return self.last_results[:n]
        cached = self.get_cached_scan()
        if cached:
            _, results = cached
            return results[:n]
        return []


scanner = MarketScanner()
