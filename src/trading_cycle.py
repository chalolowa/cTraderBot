"""Autonomous cTrader trading cycle."""

import asyncio
import logging
import math
import sqlite3
from collections import defaultdict
from datetime import UTC, datetime, timedelta
from typing import Any

import pandas as pd
from ctrader_api_client.exceptions import CTraderError

from config.ctrader_client import ctrader_client, normalize_symbol
from config.settings import settings
from src.logger import _console
from src.market_hours import market_hours
from src.risk_manager import Trade, risk_manager
from src.scanner import scanner
from src.strategy_engine import strategy_engine
from src.telegram_bot import telegram


class TradingCycle:
    def __init__(self) -> None:
        self.running = False
        self.last_scan_time: datetime | None = None
        self.scan_results: list[dict] = []
        # Last closed candle already decided on, per symbol. Entries are evaluated
        # once per closed candle, so a stale crossover is never re-traded every minute.
        self._evaluated_bar: dict[str, str] = {}
        # Short-lived cache of higher-timeframe bias per symbol.
        self._htf_cache: dict[str, tuple[datetime, dict[str, Any]]] = {}

    async def start(self) -> None:
        self.running = True
        _console("cTraderbot loop started", level=logging.INFO)
        await telegram.send_message_async("auto trading initiated....")
        while self.running:
            try:
                await self._tick()
            except Exception as exc:
                _console(f"Trading cycle tick error: {exc}", level=logging.ERROR)
                await telegram.send_message_async(f"Cycle error: {exc}")
            await asyncio.sleep(60)

    async def _tick(self) -> None:
        await self._reconcile_exits()

        if market_hours.is_close_time():
            _console(
                "Close window active — flattening all positions", level=logging.INFO
            )
            await self._close_all_positions("market_close")
            return

        # Keep analysing positions that are already open (breakeven, trail,
        # profit lock, reversal exits). Runs even when new entries are blocked.
        await self._manage_open_positions()

        can_open = market_hours.can_open_new_trade()
        if not can_open.get("allowed", False):
            _console(
                f"Trading cycle: {can_open.get('reason', 'Market closed or cutoff active')}",
                level=logging.INFO,
            )
            return

        # Daily loss limit check (B21)
        max_daily_loss_pct = strategy_engine.config.get("risk_management", {}).get(
            "max_daily_loss_pct", 3.0
        )
        try:
            account = await ctrader_client.get_account_info()
            balance = float(getattr(account, "balance", 0.0) or 0.0)
            equity = getattr(account, "equity", None)
            floating_pnl = (float(equity) - balance) if equity is not None else 0.0
        except (CTraderError, RuntimeError, ValueError, sqlite3.Error) as exc:
            _console(
                f"Unable to fetch account info for loss limit check: {exc}",
                level=logging.WARNING,
            )
            return

        if risk_manager.daily_loss_limit_reached(
            balance, max_daily_loss_pct, floating_pnl
        ):
            return

        guards = strategy_engine.get_guard_config()
        max_streak = int(guards["max_consecutive_losses"])
        if max_streak > 0 and risk_manager.consecutive_losses_today() >= max_streak:
            _console(
                f"{max_streak} consecutive losses today — no new entries until tomorrow",
                level=logging.WARNING,
            )
            return

        minutes_to_flatten = market_hours.time_until_close().total_seconds() / 60.0
        if minutes_to_flatten < float(guards["no_new_entries_minutes_before_flatten"]):
            _console(
                f"Entry window closed ({minutes_to_flatten:.0f} min to flatten, "
                f"cutoff {guards['no_new_entries_minutes_before_flatten']} min)",
                level=logging.INFO,
            )
            return

        # Scan cache evaluation across fresh process invocations (A4, B8)
        scan_interval_seconds = settings.SCANNER_INTERVAL_MIN * 60
        cached_scan = scanner.get_cached_scan(max_age_seconds=scan_interval_seconds)
        now_utc = datetime.now(UTC)
        last_scan = self.last_scan_time
        if last_scan is not None and last_scan.tzinfo is None:
            last_scan = last_scan.replace(tzinfo=UTC)

        if self.scan_results and (
            last_scan is None
            or (now_utc - last_scan).total_seconds() < scan_interval_seconds
        ):
            ts_str = last_scan.isoformat() if last_scan else now_utc.isoformat()
            _console(
                f"Reusing in-memory scan from {ts_str}, {len(self.scan_results)} results",
                level=logging.INFO,
            )
        elif cached_scan:
            cached_ts, cached_results = cached_scan
            self.last_scan_time = cached_ts
            self.scan_results = cached_results
            _console(
                f"Reusing cached scan from {cached_ts.isoformat()}, {len(self.scan_results)} results",
                level=logging.INFO,
            )
        else:
            _console("Running fresh scan", level=logging.INFO)
            self.scan_results = await scanner.scan_all()
            self.last_scan_time = now_utc

        positions = await ctrader_client.get_bot_positions()
        max_open = strategy_engine.config.get("risk_management", {}).get(
            "max_open_trades", settings.MAX_OPEN_TRADES
        )
        if len(positions) >= max_open:
            _console(
                f"Open position count {len(positions)} >= limit {max_open}",
                level=logging.INFO,
            )
            return

        open_symbols = {
            ctrader_client.symbol_name(position.symbol_id) for position in positions
        }
        for opportunity in self.scan_results:
            try:
                current_positions = await ctrader_client.get_bot_positions()
                if len(current_positions) >= max_open:
                    _console(
                        f"Position limit reached during scan evaluation ({len(current_positions)} >= {max_open})",
                        level=logging.INFO,
                    )
                    break
                symbol = normalize_symbol(opportunity["instrument"])
                if symbol not in open_symbols:
                    await self._evaluate_and_trade(symbol)
                    open_symbols.add(symbol)
            except (CTraderError, RuntimeError, sqlite3.Error) as exc:
                _console(
                    f"Error evaluating candidate {opportunity.get('instrument')}: {exc}",
                    level=logging.ERROR,
                )
                continue

    async def _evaluate_and_trade(self, symbol: str) -> None:
        timeframe = strategy_engine.config.get("timeframe", "M15")
        candle_count = strategy_engine.config.get("candle_count", 200)

        df = await ctrader_client.copy_rates_from_pos(
            symbol, timeframe, 1, candle_count
        )
        if df.empty or len(df) < 5:
            _console(
                f"Insufficient candle data for {symbol} (count={len(df)})",
                level=logging.INFO,
            )
            return

        # Decide once per closed candle. Without this, a crossover on the last
        # closed bar stays "true" for 15 minutes: the bot would enter late and
        # re-enter the same signal right after being stopped out.
        bar_key = str(df.index[-1])
        if self._evaluated_bar.get(symbol) == bar_key:
            return
        self._evaluated_bar[symbol] = bar_key

        _console(
            f"Evaluating {symbol}: timeframe={timeframe}, candles={len(df)}, bar={bar_key}",
            level=logging.INFO,
        )
        analyzed_df = strategy_engine.ta.analyze_instrument(df, strategy_engine.config)
        entry_matches = []
        direction_results = {}
        for direction in ("long", "short"):
            should_enter, metadata = strategy_engine.evaluate_entry(
                df, direction, analyzed_df=analyzed_df
            )
            direction_results[direction] = should_enter
            if should_enter:
                entry_matches.append((direction, metadata))
        should_enter = len(entry_matches) == 1
        direction = entry_matches[0][0] if should_enter else None
        risk_manager.record_event(
            "trade_decision",
            {
                "instrument": symbol,
                "direction": direction,
                "should_enter": should_enter,
                "direction_results": direction_results,
                "timeframe": timeframe,
                "candle_count": len(df),
                "bar": bar_key,
                "strategy": strategy_engine.config.get("name"),
            },
        )
        if not should_enter:
            if len(entry_matches) > 1:
                _console(
                    f"Skipping {symbol}: both long and short strategy rules matched",
                    level=logging.WARNING,
                )
            return

        direction, metadata = entry_matches[0]

        blocked = await self._entry_blocked_reason(
            symbol, direction, df.index[-1], timeframe
        )
        if blocked:
            _console(
                f"Entry blocked {symbol} {direction}: {blocked}", level=logging.INFO
            )
            risk_manager.record_event(
                "trade_blocked",
                {
                    "instrument": symbol,
                    "direction": direction,
                    "reason": blocked,
                    "bar": bar_key,
                },
            )
            return

        tick = ctrader_client.get_tick(symbol)
        symbol_info = await ctrader_client.get_symbol_info(symbol)
        pip_factor = 10**symbol_info.pip_position
        spread_pips = (float(tick.ask) - float(tick.bid)) * pip_factor
        max_spread = float(
            strategy_engine.config.get("filters", {}).get(
                "spread_max_pips", settings.MAX_SPREAD_PIPS
            )
        )
        if (
            not math.isfinite(spread_pips)
            or spread_pips < 0
            or spread_pips > max_spread
        ):
            _console(
                f"Skipping {symbol} {direction}: live spread {spread_pips:.1f} pips "
                f"exceeds limit {max_spread:.1f}",
                level=logging.INFO,
            )
            return
        entry_price = float(tick.ask if direction == "long" else tick.bid)
        atr_val = metadata.get("atr")
        enriched_df = metadata.get("enriched_df", df)

        # Chase guard: if price already ran away from the signal candle's close,
        # the entry is late and the stop would sit behind a stretched move.
        guards = strategy_engine.get_guard_config()
        signal_close = float(metadata.get("signal", {}).get("close", 0.0) or 0.0)
        if (
            atr_val is not None
            and not pd.isna(atr_val)
            and float(atr_val) > 0
            and signal_close
        ):
            sign = 1.0 if direction == "long" else -1.0
            chased_atr = sign * (entry_price - signal_close) / float(atr_val)
            if chased_atr > float(guards["max_chase_atr"]):
                _console(
                    f"Skipping {symbol} {direction}: price already moved {chased_atr:.2f} ATR "
                    f"past the signal close (max {guards['max_chase_atr']})",
                    level=logging.INFO,
                )
                return

        stop_loss = strategy_engine.calculate_stop_loss(
            entry_price, direction, df=enriched_df, atr=atr_val
        )
        take_profit = strategy_engine.calculate_take_profit(
            entry_price, stop_loss, direction
        )

        risk = abs(entry_price - stop_loss)
        min_rr = strategy_engine.config.get("risk_management", {}).get(
            "min_risk_reward", 1.5
        )
        reward = abs(take_profit - entry_price)
        rr_ratio = (reward / risk) if risk > 0 else 0.0

        if risk == 0 or rr_ratio < min_rr:
            # A13 logging: Risk/reward check fails (< min_risk_reward)
            _console(
                f"Risk/reward check failed for {symbol}: computed R:R={rr_ratio:.2f} < min_risk_reward={min_rr:.2f}",
                level=logging.INFO,
            )
            return

        account = await ctrader_client.get_account_info()
        balance = float(account.balance)
        risk_pct = strategy_engine.config.get("risk_management", {}).get(
            "risk_per_trade_pct", settings.RISK_PER_TRADE_PCT
        )
        lots = await ctrader_client.lots_from_risk(
            symbol, balance * float(risk_pct) / 100.0, entry_price, stop_loss
        )
        if lots <= 0:
            _console(
                f"Position sizing rejected for {symbol}: balance={balance:.2f}, "
                f"entry={entry_price:.5f}, stop_dist={risk:.5f}, risk_pct={risk_pct}",
                level=logging.WARNING,
            )
            return

        # A16 logging: Immediately before place_market_order()
        _console(
            f"Placing market order: symbol={symbol}, direction={direction}, "
            f"volume={lots:.2f}, SL={stop_loss:.5f}, TP={take_profit:.5f}",
            level=logging.INFO,
        )

        result = await ctrader_client.place_market_order(
            symbol, direction, lots, stop_loss, take_profit
        )
        if result.position_id is None:
            raise RuntimeError("cTrader order fill did not include a position ID")
        position_id = int(result.position_id)
        # B19: Use actual fill price from result
        actual_fill_price = result.fill_price or entry_price

        risk_manager.record_entry(
            Trade(
                trade_id=position_id,
                instrument=symbol,
                direction=direction,
                entry_price=actual_fill_price,
                stop_loss=stop_loss,
                take_profit=take_profit,
                position_size=lots,
                entry_time=datetime.now(UTC),
            )
        )

        # A18 logging: Order placed successfully
        _console(
            f"Order placed successfully: position_id={position_id}, fill_price={actual_fill_price:.5f}, "
            f"SL={stop_loss:.5f}, TP={take_profit:.5f}",
            level=logging.INFO,
        )

        await telegram.send_message_async(
            f"{'BUY' if direction == 'long' else 'SELL'} {symbol} "
            f"position_id={position_id} entry={actual_fill_price:.5f} SL={stop_loss:.5f} TP={take_profit:.5f}"
        )

    # ------------------------------------------------------------------
    # Entry guards
    # ------------------------------------------------------------------
    @staticmethod
    def _tf_minutes(timeframe: str) -> int:
        tf = str(timeframe).upper()
        try:
            n = int(tf[1:])
        except ValueError:
            return 15
        return {"M": n, "H": n * 60, "D": n * 1440}.get(tf[:1], 15)

    @staticmethod
    def _to_utc(ts: Any) -> datetime | None:
        if not hasattr(ts, "to_pydatetime") and not isinstance(ts, datetime):
            return None
        value = ts.to_pydatetime() if hasattr(ts, "to_pydatetime") else ts
        return (
            value.replace(tzinfo=UTC) if value.tzinfo is None else value.astimezone(UTC)
        )

    async def _get_htf_bias(
        self, symbol: str, max_age_s: float = 120.0
    ) -> dict[str, Any]:
        cfg = strategy_engine.get_trend_filter_config()
        now = datetime.now(UTC)
        cached = self._htf_cache.get(symbol)
        if cached and (now - cached[0]).total_seconds() < max_age_s:
            return cached[1]
        try:
            htf_df = await ctrader_client.copy_rates_from_pos(
                symbol, cfg["timeframe"], 1, int(cfg["candle_count"])
            )
            bias = strategy_engine.compute_htf_bias(htf_df)
        except (CTraderError, RuntimeError, ValueError, KeyError, TypeError) as exc:
            _console(f"HTF bias unavailable for {symbol}: {exc}", level=logging.WARNING)
            return {
                "bias": "neutral",
                "reason": f"htf_fetch_failed: {exc}",
                "drift_atr": 0.0,
            }
        self._htf_cache[symbol] = (now, bias)
        return bias

    @staticmethod
    def _currency_exposure_block(
        symbol: str, direction: str, open_trades: list[dict[str, Any]], limit: int
    ) -> str | None:
        def legs(sym: str, side: str) -> dict[str, int]:
            s = 1 if str(side).lower() in {"long", "buy"} else -1
            return {sym[:3]: s, sym[3:6]: -s}

        net: dict[str, int] = defaultdict(int)
        for trade in open_trades:
            for ccy, val in legs(str(trade["instrument"]), trade["direction"]).items():
                net[ccy] += val
        for ccy, val in legs(symbol, direction).items():
            after = net[ccy] + val
            if abs(after) > limit and abs(after) > abs(net[ccy]):
                return f"{ccy} net exposure would be {after:+d} (limit {limit})"
        return None

    async def _entry_blocked_reason(
        self, symbol: str, direction: str, bar_index: Any, timeframe: str
    ) -> str | None:
        """Return why this signal must not be traded, or None when it may be."""
        guards = strategy_engine.get_guard_config()
        now = datetime.now(UTC)
        bar_open = self._to_utc(bar_index)

        # 1. Freshness: a signal that is minutes old is a late entry.
        if bar_open is not None:
            bar_close = bar_open + timedelta(minutes=self._tf_minutes(timeframe))
            age_min = (now - bar_close).total_seconds() / 60.0
            if age_min > float(guards["max_signal_age_minutes"]):
                return f"stale signal ({age_min:.1f} min after candle close)"

        # 2. Higher-timeframe trend must agree with the direction.
        if strategy_engine.get_trend_filter_config().get("enabled", True):
            bias = await self._get_htf_bias(symbol)
            ok, why = strategy_engine.direction_allowed_by_trend(direction, bias)
            if not ok:
                return f"trend filter: {why}"

        # 3. Per-symbol history: no re-entry on the same candle, cooldown after a
        #    loss (this also stops the buy -> sell -> buy flip-flop), daily cap.
        activity = risk_manager.symbol_activity_today(symbol)
        last_entry = activity["last_entry_time"]
        if bar_open is not None and last_entry is not None and last_entry >= bar_open:
            return "already traded this signal candle"
        if activity["count"] >= int(guards["max_trades_per_symbol_per_day"]):
            return f"{activity['count']} trades on {symbol} today (cap reached)"
        last_closed = activity["last_closed"]
        if last_closed and float(last_closed.get("pnl") or 0.0) < 0:
            exited = risk_manager._parse_ts(last_closed.get("exit_time"))
            cooldown = float(guards["cooldown_minutes_after_loss"])
            if exited and (now - exited).total_seconds() < cooldown * 60:
                return f"cooldown after a loss on {symbol} ({cooldown:.0f} min)"

        # 4. Correlated exposure (e.g. three USD trades are one bet).
        exposure = self._currency_exposure_block(
            symbol,
            direction,
            risk_manager.get_open_trades(),
            int(guards["max_currency_net_exposure"]),
        )
        if exposure:
            return f"correlation: {exposure}"
        return None

    # ------------------------------------------------------------------
    # Management of open positions
    # ------------------------------------------------------------------
    async def _manage_open_positions(self) -> None:
        cfg = strategy_engine.get_management_config()
        if not cfg.get("enabled", True):
            return
        open_trades = risk_manager.get_open_trades()
        if not open_trades:
            return
        try:
            broker_ids = {
                int(p.position_id) for p in await ctrader_client.get_bot_positions()
            }
        except (CTraderError, RuntimeError, ValueError) as exc:
            _console(f"Position management skipped: {exc}", level=logging.WARNING)
            return
        minutes_left = market_hours.time_until_close().total_seconds() / 60.0

        for trade in open_trades:
            if int(trade["trade_id"]) not in broker_ids:
                continue
            try:
                await self._manage_one(trade, cfg, minutes_left)
            except (
                CTraderError,
                RuntimeError,
                ValueError,
                KeyError,
                TypeError,
                sqlite3.Error,
            ) as exc:
                _console(
                    f"Management error for position {trade.get('trade_id')}: {exc}",
                    level=logging.ERROR,
                )

    async def _close_managed(
        self, position_id: int, symbol: str, reason: str, r_now: float, peak_r: float
    ) -> None:
        await ctrader_client.close_position(position_id)
        await self._record_exit(position_id, reason)
        risk_manager.record_event(
            "trade_management",
            {
                "position_id": position_id,
                "instrument": symbol,
                "action": "close",
                "reason": reason,
                "r_now": round(r_now, 2),
                "peak_r": round(peak_r, 2),
            },
            aggregate_id=position_id,
        )
        _console(
            f"Managed exit {symbol} position_id={position_id}: {reason} "
            f"(R={r_now:+.2f}, peak={peak_r:+.2f}R)",
            level=logging.INFO,
        )
        await telegram.send_message_async(
            f"EXIT {symbol} #{position_id}: {reason} (R={r_now:+.2f}, peak {peak_r:+.2f}R)"
        )

    async def _manage_one(
        self, trade: dict[str, Any], cfg: dict[str, Any], minutes_left: float
    ) -> None:
        position_id = int(trade["trade_id"])
        symbol = normalize_symbol(trade["instrument"])
        is_long = str(trade["direction"]).lower() in {"long", "buy"}
        sign = 1.0 if is_long else -1.0
        entry = float(trade["entry_price"])
        initial_stop = float(trade["stop_loss"])
        take_profit = float(trade["take_profit"])
        risk = abs(entry - initial_stop)
        if risk <= 0:
            return
        current_stop = float(trade.get("current_stop") or initial_stop)

        tick = ctrader_client.get_tick(symbol)
        price = float(tick.bid if is_long else tick.ask)  # the side we would exit on
        r_now = sign * (price - entry) / risk
        peak_r = max(float(trade.get("peak_r") or 0.0), r_now)
        prev_peak = trade.get("peak_price")
        if prev_peak is None:
            peak_price = price
        else:
            peak_price = (
                max(float(prev_peak), price)
                if is_long
                else min(float(prev_peak), price)
            )
        risk_manager.update_trade_management(
            position_id, peak_r=peak_r, peak_price=peak_price
        )

        # 1. Late in the day: bank open profit instead of handing it back at the flatten.
        if minutes_left <= float(cfg["late_day_close_minutes"]) and r_now >= float(
            cfg["late_day_close_min_r"]
        ):
            await self._close_managed(
                position_id, symbol, "late_day_profit", r_now, peak_r
            )
            return

        # 2. Fresh view of the market for this open trade.
        timeframe = strategy_engine.config.get("timeframe", "M15")
        candle_count = int(strategy_engine.config.get("candle_count", 200))
        df = await ctrader_client.copy_rates_from_pos(
            symbol, timeframe, 1, candle_count
        )
        atr = None
        opposite_cross = False
        if not df.empty and len(df) >= 30:
            analyzed = strategy_engine.ta.analyze_instrument(df, strategy_engine.config)
            latest = analyzed.iloc[-1]
            atr_period = int(
                strategy_engine.config["exit_rules"]["stop_loss"].get("atr_period", 14)
            )
            atr_raw = latest.get(f"ATR_{atr_period}")
            if atr_raw is not None and not pd.isna(atr_raw) and float(atr_raw) > 0:
                atr = float(atr_raw)
            opposite_cross = bool(
                latest.get("crossover_down" if is_long else "crossover_up", False)
            )

        if (
            cfg["exit_on_opposite_cross_in_profit"]
            and opposite_cross
            and r_now >= float(cfg["min_profit_r_for_signal_exit"])
        ):
            await self._close_managed(
                position_id, symbol, "reversal_in_profit", r_now, peak_r
            )
            return

        if cfg["exit_on_htf_flip"] and strategy_engine.get_trend_filter_config().get(
            "enabled", True
        ):
            bias = await self._get_htf_bias(symbol, max_age_s=300)
            if bias.get("bias") == ("short" if is_long else "long"):
                await self._close_managed(
                    position_id, symbol, "htf_reversal", r_now, peak_r
                )
                return

        # 3. Ratchet the stop (never loosen it).
        def tighter(a: float, b: float) -> float:
            return max(a, b) if is_long else min(a, b)

        new_stop = current_stop
        why: list[str] = []

        def propose(label: str, level: float) -> None:
            nonlocal new_stop
            candidate = tighter(new_stop, level)
            if candidate != new_stop:
                new_stop = candidate
                why.append(label)

        if peak_r >= float(cfg["breakeven_r"]):
            propose("breakeven", entry + sign * float(cfg["breakeven_buffer_r"]) * risk)
        if (
            cfg["trail_enabled"]
            and atr is not None
            and peak_r >= float(cfg["trail_start_r"])
        ):
            propose("atr_trail", peak_price - sign * float(cfg["trail_atr_mult"]) * atr)
        if minutes_left <= float(cfg["late_day_lock_minutes"]) and r_now >= float(
            cfg["late_day_lock_min_r"]
        ):
            propose(
                "late_day_lock",
                entry + float(cfg["late_day_lock_fraction"]) * (price - entry),
            )

        if not why:
            return

        if (is_long and new_stop >= price) or (not is_long and new_stop <= price):
            # Price has already given back to the protective level: exit now.
            await self._close_managed(
                position_id, symbol, "trailing_stop", r_now, peak_r
            )
            return

        symbol_info = await ctrader_client.get_symbol_info(symbol)
        pip_factor = 10**symbol_info.pip_position
        if abs(price - new_stop) * pip_factor < float(cfg["min_stop_distance_pips"]):
            return
        new_stop = round(new_stop, symbol_info.pip_position + 1)
        if abs(new_stop - current_stop) < float(cfg["min_stop_step_r"]) * risk:
            return

        # amend replaces BOTH levels, so always send the existing take profit too.
        await ctrader_client.amend_position_sltp(
            position_id, stop_loss=new_stop, take_profit=take_profit
        )
        risk_manager.update_trade_management(position_id, current_stop=new_stop)
        risk_manager.record_event(
            "trade_management",
            {
                "position_id": position_id,
                "instrument": symbol,
                "action": "amend_stop",
                "old_stop": current_stop,
                "new_stop": new_stop,
                "why": why,
                "r_now": round(r_now, 2),
                "peak_r": round(peak_r, 2),
            },
            aggregate_id=position_id,
        )
        _console(
            f"Stop moved {symbol} #{position_id}: {current_stop:.5f} -> {new_stop:.5f} "
            f"({'+'.join(why)}, R={r_now:+.2f}, peak={peak_r:+.2f}R)",
            level=logging.INFO,
        )

    async def _close_all_positions(self, reason: str) -> None:
        for position in await ctrader_client.get_bot_positions():
            position_id = int(position.position_id)
            try:
                await ctrader_client.close_position(position_id)
                await self._record_exit(position_id, reason)
                _console(
                    f"Closed position position_id={position_id} (reason={reason})",
                    level=logging.INFO,
                )
            except (CTraderError, RuntimeError, ValueError, sqlite3.Error) as exc:
                _console(
                    f"Close failed for position {position_id}: {exc}",
                    level=logging.ERROR,
                )
                await telegram.send_message_async(
                    f"Close failed for position {position_id}: {exc}"
                )

    async def _record_exit(self, position_id: int, reason: str | None) -> None:
        exit_price, pnl = await ctrader_client.get_closed_position_summary(position_id)
        if reason is None:
            reason = self._infer_exit_reason(
                risk_manager.get_trade(position_id), float(exit_price)
            )
        risk_manager.record_exit(position_id, exit_price, pnl, reason)

    @staticmethod
    def _infer_exit_reason(trade: dict[str, Any] | None, exit_price: float) -> str:
        """Label broker-side exits so post-mortems can tell SL / trail / TP apart."""
        if not trade:
            return "broker_exit"
        try:
            entry = float(trade["entry_price"])
            initial_stop = float(trade["stop_loss"])
            tp = float(trade["take_profit"])
            current_stop = float(trade.get("current_stop") or initial_stop)
        except (KeyError, TypeError, ValueError):
            return "broker_exit"
        tol = 0.15 * abs(entry - initial_stop)
        if tol <= 0:
            return "broker_exit"
        if abs(exit_price - tp) <= tol:
            return "take_profit"
        if abs(exit_price - current_stop) <= tol:
            return (
                "stop_loss"
                if abs(current_stop - initial_stop) <= tol
                else "trailing_stop"
            )
        return "broker_exit"

    async def _reconcile_exits(self) -> None:
        broker_positions = await ctrader_client.get_bot_positions()
        broker_position_ids = {position.position_id for position in broker_positions}
        for trade in risk_manager.get_open_trades():
            position_id = int(trade["trade_id"])
            if position_id in broker_position_ids:
                continue
            try:
                await self._record_exit(position_id, None)
                _console(
                    f"Reconciled broker-closed position position_id={position_id}",
                    level=logging.INFO,
                )
            except (CTraderError, RuntimeError, sqlite3.Error) as exc:
                _console(
                    f"Unable to reconcile closed position {position_id}: {exc}",
                    level=logging.WARNING,
                )

    async def send_daily_summary(self) -> None:
        try:
            account = await ctrader_client.get_account_info()
            balance = float(account.balance)
        except Exception:
            balance = 0.0
        stats = risk_manager.get_r_multiple_stats()
        summary_text = (
            f"Daily summary: balance={balance:.2f}, "
            f"trades={stats['total_trades']}, total R={stats['total_r']:+.2f}, "
            f"win_rate={stats['win_rate']}%, max_drawdown={stats['max_r_drawdown']}R"
        )
        _console(summary_text, level=logging.INFO)
        await telegram.send_daily_summary_async(summary_text)
        risk_manager.save_daily_summary()

    def stop(self) -> None:
        self.running = False
        _console("cTraderbot loop stopped", level=logging.INFO)
        telegram.send_message("auto trading stopped.")


trading_cycle = TradingCycle()
