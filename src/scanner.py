"""Filter configured instruments using current spread and candle/indicator data."""

import hashlib
import json
import logging
import math
import traceback
from datetime import UTC, datetime
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
SCAN_CACHE_VERSION = 2


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
        """Load a compatible scan result from the durable cache."""
        if not CACHE_FILE.exists():
            return None
        try:
            with open(CACHE_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
            if (
                data.get("version") != SCAN_CACHE_VERSION
                or data.get("strategy_signature") != self._strategy_signature()
            ):
                return None
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
                "version": SCAN_CACHE_VERSION,
                "strategy_signature": self._strategy_signature(),
                "timestamp": datetime.now(UTC).isoformat(),
                "results": results,
            }
            with open(CACHE_FILE, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2)
        except Exception as exc:
            _console(f"Failed to write scan cache: {exc}", level=logging.WARNING)

    @staticmethod
    def _strategy_signature() -> str:
        serialized = json.dumps(
            strategy_engine.config, sort_keys=True, separators=(",", ":")
        )
        return hashlib.sha256(serialized.encode("utf-8")).hexdigest()

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

        self.instruments = strategy_engine.get_instruments()
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
            f"{len(results)}/{len(self.instruments)} instruments pass market filters"
            + (f": [{tradable_summary}]" if results else " (none passed filters)"),
            level=logging.INFO,
        )
        return results

    async def _analyze_instrument(self, instrument: str) -> dict[str, Any]:
        """Apply market filters and compute configured strategy indicators."""
        norm_sym = normalize_symbol(instrument)
        timeframe = strategy_engine.config.get("timeframe", "M15")
        candle_count = int(strategy_engine.config.get("candle_count", 200))
        try:
            df = await ctrader_client.copy_rates_from_pos(
                instrument, timeframe, 1, candle_count
            )
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

        analyzed_df = self.ta.analyze_instrument(df, strategy_engine.config)
        latest = analyzed_df.iloc[-1]

        symbol_info = await ctrader_client.get_symbol_info(instrument)
        pip_factor = 10**symbol_info.pip_position
        spread_pips: float | None = None
        reasons = []
        try:
            tick = ctrader_client.get_tick(instrument)
            spread_pips = (float(tick.ask) - float(tick.bid)) * pip_factor
        except (RuntimeError, TypeError, ValueError) as exc:
            reasons.append(f"Live spread unavailable: {exc}")
            _console(
                f"Failed to fetch tick for spread check on {instrument}: {exc}",
                level=logging.WARNING,
            )
        filters = strategy_engine.config.get("filters", {})
        max_spread = float(filters.get("spread_max_pips", settings.MAX_SPREAD_PIPS))
        if spread_pips is not None and (
            not math.isfinite(spread_pips) or spread_pips < 0
        ):
            reasons.append(f"Invalid live spread: {spread_pips}")
            spread_pips = None
        elif spread_pips is not None and spread_pips > max_spread:
            reasons.append(
                f"Spread too high: actual={spread_pips:.1f} pips, max={max_spread:.1f}"
            )

        # Keep ATR and ADX as diagnostics; entry thresholds belong to strategy rules.
        atr_raw = latest.get("ATR_14")
        adx_raw = latest.get("ADX_14")
        rsi_raw = latest.get("RSI_14")

        atr = float(atr_raw) if (atr_raw is not None and not pd.isna(atr_raw)) else None
        adx = float(adx_raw) if (adx_raw is not None and not pd.isna(adx_raw)) else None
        rsi = float(rsi_raw) if (rsi_raw is not None and not pd.isna(rsi_raw)) else 50.0

        stop_loss = strategy_engine.config.get("exit_rules", {}).get("stop_loss", {})
        atr_period = int(stop_loss.get("atr_period", 14))
        atr_for_risk = latest.get(f"ATR_{atr_period}")
        if stop_loss.get("type") == "ATR_MULTIPLE" and (
            atr_for_risk is None or pd.isna(atr_for_risk) or float(atr_for_risk) <= 0
        ):
            reasons.append(f"ATR_{atr_period} unavailable for stop-loss sizing")

        tradable = not reasons
        if not tradable:
            _console(
                f"Instrument {norm_sym} filtered out: {'; '.join(reasons)}",
                level=logging.INFO,
            )

        score = 0.0
        if adx is not None:
            if adx > 25:
                score += 30
            elif adx > 20:
                score += 20

        if 40 < rsi < 60:
            score += 20
        elif 30 < rsi < 70:
            score += 10

        if spread_pips is not None:
            if spread_pips < 2.0:
                score += 25
            elif spread_pips < max_spread:
                score += 15

        if atr is not None and atr * pip_factor > 15:
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

        result = {
            "tradable": tradable,
            "instrument": norm_sym,
            "composite_score": score,
            "trend": trend,
            "adx": round(adx, 1) if adx is not None else 0.0,
            "rsi": round(rsi, 1),
            "atr_pips": round(atr * pip_factor, 1) if atr is not None else 0.0,
            "spread_pips": round(spread_pips, 1) if spread_pips is not None else None,
            "sma_9": round(sma_9, 5),
            "sma_21": round(sma_21, 5),
            "reasons": reasons,
            "timestamp": datetime.now(UTC).isoformat(),
        }
        return result

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
