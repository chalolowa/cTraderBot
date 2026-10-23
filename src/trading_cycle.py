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
from src.candles import fetch_closed_candles, timeframe_minutes
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
        # Serialises everything that closes/records positions (tick flatten, reconcile,
        # fast management loop) so one exit can never be recorded twice.
        self._pos_lock = asyncio.Lock()
        self._bar_cache: dict[str, dict[str, Any]] = {}
        self._ret_cache: dict[str, tuple[datetime, pd.Series]] = {}
        self._pip_position: dict[str, int] = {}
        self._amend_warned = False
        self._notification_lock = asyncio.Lock()

    @staticmethod
    def _format_candle_bars(
        candles: pd.DataFrame | None, digits: int, count: int = 3
    ) -> str:
        if candles is None or candles.empty:
            return "unavailable"

        rows: list[str] = []
        for timestamp, candle in candles.tail(count).iterrows():
            parts = [
                f"{timestamp}",
                *(
                    f"{name[0].upper()}={float(candle[name]):.{digits}f}"
                    for name in ("open", "high", "low", "close")
                    if name in candle and pd.notna(candle[name])
                ),
            ]
            volume_name = next(
                (name for name in ("tick_volume", "volume") if name in candle),
                None,
            )
            if volume_name and pd.notna(candle[volume_name]):
                parts.append(f"V={float(candle[volume_name]):g}")
            indicators = [
                name
                for name in candle.index
                if name.upper().startswith(("RSI_", "ATR_", "ADX_"))
                and pd.notna(candle[name])
            ]
            parts.extend(
                f"{name}={float(candle[name]):.3f}" for name in indicators[:6]
            )
            rows.append(" ".join(parts))
        return " ; ".join(rows) if rows else "unavailable"

    @staticmethod
    def _format_entry_conditions(conditions: Any) -> str:
        if not isinstance(conditions, list) or not conditions:
            return "unavailable"
        return "; ".join(
            f"{condition.get('indicator', '?')} "
            f"{condition.get('comparison', '?')} "
            f"expected={condition.get('expected', '?')} "
            f"actual={condition.get('actual', '?')} "
            f"pass={condition.get('result', '?')}"
            for condition in conditions
            if isinstance(condition, dict)
        ) or "unavailable"

    async def start(self) -> None:
        self.running = True
        _console("cTraderbot loop started", level=logging.INFO)
        await telegram.send_message_async("auto trading initiated....")
        await self._deliver_pending_trade_notifications()
        mgmt_task = asyncio.create_task(self._management_loop())
        try:
            while self.running:
                try:
                    await self._tick()
                except Exception as exc:
                    _console(f"Trading cycle tick error: {exc}", level=logging.ERROR)
                    await telegram.send_message_async(f"Cycle error: {exc}")
                await asyncio.sleep(60)
        finally:
            mgmt_task.cancel()

    async def _tick(self) -> None:
        await self._reconcile_exits()

        if market_hours.is_close_time():
            _console(
                "Close window active — flattening all positions", level=logging.INFO
            )
            await self._close_all_positions("market_close")
            await self.send_daily_summary()
            return

        # Open positions are managed by _management_loop (every few seconds), so
        # a slow scan here can never delay a protective exit.

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

        df = await fetch_closed_candles(symbol, timeframe, int(candle_count))
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
            metadata["mode"] = "crossover"
            if not should_enter:
                # No fresh crossover: look for a continuation entry in a running trend.
                pb_ok, pb_meta = strategy_engine.evaluate_pullback(
                    analyzed_df, direction
                )
                if pb_ok:
                    should_enter, metadata = True, pb_meta
            direction_results[direction] = should_enter
            if should_enter:
                entry_matches.append((direction, metadata))

        # One readable line per evaluation (what the bot saw and why it did or did not act).
        htf_line = "htf=n/a"
        if strategy_engine.get_trend_filter_config().get("enabled", True):
            bias_now = await self._get_htf_bias(symbol)
            htf_line = (
                f"htf={bias_now.get('bias')} (adx={bias_now.get('adx', '-')}, "
                f"drift={bias_now.get('drift_atr', 0)} ATR, {bias_now.get('reason', '')})"
            )
        _console(
            f"DECISION {symbol} bar={bar_key[:16]}: long={'ENTRY' if direction_results['long'] else 'no'} "
            f"short={'ENTRY' if direction_results['short'] else 'no'} | {htf_line} | "
            f"modes={[m[1].get('mode') for m in entry_matches]}",
            level=logging.INFO,
        )
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
                "mode": entry_matches[0][1].get("mode") if should_enter else None,
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
            symbol,
            direction,
            df.index[-1],
            timeframe,
            mode=str(metadata.get("mode", "crossover")),
            feed_lag_bars=int(df.attrs.get("lag_bars", 0)),
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
        self._pip_position[symbol] = symbol_info.pip_position
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

        signal_bars = self._format_candle_bars(enriched_df, symbol_info.pip_position + 1)
        signal_time = metadata.get("signal", {}).get("time", bar_key)
        atr_pips = (
            float(atr_val) * pip_factor
            if atr_val is not None and not pd.isna(atr_val)
            else None
        )
        atr_text = f"{atr_pips:.2f}p" if atr_pips is not None else "n/a"
        _console(
            f"ENTRY SIGNAL {symbol} {direction.upper()} "
            f"strategy={metadata.get('strategy', strategy_engine.config.get('name'))} "
            f"mode={metadata.get('mode', 'crossover')} timeframe={timeframe} "
            f"signal_bar={signal_time} spread={spread_pips:.2f}p ATR={atr_text}",
            level=logging.INFO,
        )
        _console(
            f"ENTRY CONTEXT {symbol} conditions=[{self._format_entry_conditions(metadata.get('conditions'))}] "
            f"signal_bars[{len(enriched_df.tail(3))}]=[{signal_bars}]",
            level=logging.INFO,
        )

        # Immediately before submitting, include the planned price distances and risk.
        _console(
            f"Placing market order: symbol={symbol}, direction={direction}, "
            f"volume={lots:.2f} lots, entry_quote={entry_price:.{symbol_info.pip_position + 1}f}, "
            f"SL={stop_loss:.{symbol_info.pip_position + 1}f} "
            f"({risk * pip_factor:.1f}p), "
            f"TP={take_profit:.{symbol_info.pip_position + 1}f} "
            f"({reward * pip_factor:.1f}p), RR={rr_ratio:.2f}, "
            f"risk_budget={balance * float(risk_pct) / 100.0:.2f} "
            f"({float(risk_pct):.3f}% of balance={balance:.2f})",
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
        await self._deliver_pending_trade_notifications()
        fill_sl_pips = abs(actual_fill_price - stop_loss) * pip_factor
        fill_tp_pips = abs(take_profit - actual_fill_price) * pip_factor
        signed_slippage_pips = (
            (actual_fill_price - entry_price)
            * pip_factor
            * (1 if direction == "long" else -1)
        )
        _console(
            f"POSITION OPENED id={position_id} {symbol} {direction.upper()} "
            f"fill={actual_fill_price:.{symbol_info.pip_position + 1}f} "
            f"quote={entry_price:.{symbol_info.pip_position + 1}f} "
            f"adverse_slippage={signed_slippage_pips:+.2f}p "
            f"size={lots:.2f} lots SL={stop_loss:.{symbol_info.pip_position + 1}f} "
            f"({fill_sl_pips:.1f}p) TP={take_profit:.{symbol_info.pip_position + 1}f} "
            f"({fill_tp_pips:.1f}p) spread={spread_pips:.2f}p "
            f"ATR={atr_text}",
            level=logging.INFO,
        )

    # ------------------------------------------------------------------
    # Entry guards
    # ------------------------------------------------------------------
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
            htf_df = await fetch_closed_candles(
                symbol, cfg["timeframe"], int(cfg["candle_count"])
            )
            bias = strategy_engine.compute_htf_bias(htf_df)
        except Exception as exc:  # fail closed: no bias -> no trade
            _console(f"HTF bias unavailable for {symbol}: {exc}", level=logging.WARNING)
            return {
                "bias": "neutral",
                "reason": f"htf_fetch_failed: {exc}",
                "drift_atr": 0.0,
            }
        self._htf_cache[symbol] = (now, bias)
        return bias

    @staticmethod
    def _is_risk_free(trade: dict[str, Any]) -> bool:
        """True once the (enforced) stop sits at or beyond entry: the trade can no longer lose."""
        try:
            entry = float(trade["entry_price"])
            stop = trade.get("current_stop")
            if stop is None:
                return False
            stop = float(stop)
        except (KeyError, TypeError, ValueError):
            return False
        is_long = str(trade["direction"]).lower() in {"long", "buy"}
        return stop >= entry if is_long else stop <= entry

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
        self,
        symbol: str,
        direction: str,
        bar_index: Any,
        timeframe: str,
        mode: str = "crossover",
        feed_lag_bars: int = 0,
    ) -> str | None:
        """Return why this signal must not be traded, or None when it may be."""
        guards = strategy_engine.get_guard_config()
        now = datetime.now(UTC)
        bar_open = self._to_utc(bar_index)

        # 1. Freshness: a signal that is minutes old is a late entry.
        if bar_open is not None:
            bar_close = bar_open + timedelta(minutes=timeframe_minutes(timeframe))
            age_min = (now - bar_close).total_seconds() / 60.0
            if age_min > float(guards["max_signal_age_minutes"]):
                hint = (
                    f" — candle feed is {feed_lag_bars} bar(s) behind, check copy_rates_from_pos"
                    if feed_lag_bars
                    else ""
                )
                return f"stale signal ({age_min:.1f} min after candle close){hint}"

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
        if mode == "pullback" and bar_open is not None and last_entry is not None:
            gap = int(strategy_engine.get_pullback_config()["min_gap_bars"])
            if (now - last_entry) < timedelta(
                minutes=gap * timeframe_minutes(timeframe)
            ):
                return f"pullback re-entry gap ({gap} bars) on {symbol}"
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
            [
                t
                for t in risk_manager.get_open_trades()
                if not (
                    guards["correlation_guard"].get("exclude_risk_free", True)
                    and self._is_risk_free(t)
                )
            ],
            int(guards["max_currency_net_exposure"]),
        )
        if exposure:
            return f"correlation: {exposure}"

        # 5. Measured correlation: same-direction trades on pairs that move together
        #    (e.g. EURCAD + GBPCAD) are one bet taken twice.
        corr_block = await self._correlation_block(symbol, direction)
        if corr_block:
            return f"correlation: {corr_block}"
        return None

    async def _returns(self, symbol: str, cfg: dict[str, Any]) -> pd.Series | None:
        now = datetime.now(UTC)
        cached = self._ret_cache.get(symbol)
        if (
            cached
            and (now - cached[0]).total_seconds() < float(cfg["cache_minutes"]) * 60
        ):
            return cached[1]
        try:
            df = await fetch_closed_candles(
                symbol, cfg["timeframe"], int(cfg["lookback_bars"])
            )
        except Exception as exc:
            _console(
                f"Correlation data unavailable for {symbol}: {exc}",
                level=logging.WARNING,
            )
            return None
        if df is None or len(df) < 30:
            return None
        series = df["close"].astype(float).pct_change().dropna()
        self._ret_cache[symbol] = (now, series)
        return series

    async def _correlation_block(self, symbol: str, direction: str) -> str | None:
        """
        Cap how many open positions may be the same bet. A position counts as correlated
        when the measured return correlation, signed by both trade directions, is at or
        above `max_effective_corr`. A new trade is refused once `max_correlated_positions`
        correlated positions are already open. Positions whose stop is locked at/beyond
        breakeven carry no risk and are not counted (exclude_risk_free).
        """
        cfg = strategy_engine.get_guard_config()["correlation_guard"]
        if not cfg.get("enabled", True):
            return None
        cap = int(cfg.get("max_correlated_positions", 3))
        open_trades = [
            t
            for t in risk_manager.get_open_trades()
            if normalize_symbol(t["instrument"]) != symbol
        ]
        if cfg.get("exclude_risk_free", True):
            open_trades = [t for t in open_trades if not self._is_risk_free(t)]
        if not open_trades:
            return None
        mine = await self._returns(symbol, cfg)
        if mine is None:
            return None
        my_sign = 1.0 if direction == "long" else -1.0
        correlated: list[str] = []
        for trade in open_trades:
            other_sym = normalize_symbol(trade["instrument"])
            other = await self._returns(other_sym, cfg)
            if other is None:
                continue
            joined = pd.concat([mine, other], axis=1, join="inner").dropna()
            if len(joined) < int(cfg["min_overlap_bars"]):
                continue
            corr = float(joined.iloc[:, 0].corr(joined.iloc[:, 1]))
            other_sign = (
                1.0 if str(trade["direction"]).lower() in {"long", "buy"} else -1.0
            )
            effective = (
                corr * my_sign * other_sign
            )  # > 0 means both trades win/lose together
            if effective >= float(cfg["max_effective_corr"]):
                correlated.append(
                    f"{other_sym} {trade['direction']} ({effective:+.2f})"
                )
        if len(correlated) >= cap:
            return (
                f"{symbol} {direction} would be correlated position #{len(correlated) + 1} "
                f"(cap {cap}; open: {', '.join(correlated)})"
            )
        return None

    # ------------------------------------------------------------------
    # Management of open positions
    # ------------------------------------------------------------------
    async def _management_loop(self) -> None:
        """Fast loop (seconds, not minutes) that protects open trades."""
        while self.running:
            try:
                await self._deliver_pending_trade_notifications()
                await self._manage_open_positions()
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # never let one bad cycle kill protection
                _console(f"Management loop error: {exc!r}", level=logging.ERROR)
            interval = float(
                strategy_engine.get_management_config().get(
                    "management_interval_seconds", 10
                )
            )
            await asyncio.sleep(max(2.0, interval))

    async def _manage_open_positions(self) -> None:
        cfg = strategy_engine.get_management_config()
        if not cfg.get("enabled", True):
            return
        open_trades = risk_manager.get_open_trades()
        if not open_trades:
            return
        async with self._pos_lock:
            try:
                broker_ids = {
                    int(p.position_id) for p in await ctrader_client.get_bot_positions()
                }
            except Exception as exc:
                _console(f"Position management skipped: {exc!r}", level=logging.WARNING)
                return
            minutes_left = market_hours.time_until_close().total_seconds() / 60.0
            for trade in open_trades:
                if int(trade["trade_id"]) not in broker_ids:
                    continue
                try:
                    await self._manage_one(trade, cfg, minutes_left)
                except Exception as exc:  # incl. AttributeError: log loudly, keep going
                    _console(
                        f"Management error for position {trade.get('trade_id')}: {exc!r}",
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

    async def _bar_analysis(self, symbol: str) -> dict[str, Any] | None:
        """Indicators on the latest closed candle, refreshed once per candle (not every 10 s)."""
        timeframe = strategy_engine.config.get("timeframe", "M15")
        tf_min = timeframe_minutes(timeframe)
        now = datetime.now(UTC)
        bar_key = pd.Timestamp(now).floor(f"{tf_min}min")
        cached = self._bar_cache.get(symbol)
        if cached and cached["bar_key"] == bar_key and (
            cached["lag_bars"] == 0
            or (now - cached["fetched_at"]).total_seconds() < 30
        ):
            return cached
        candle_count = int(strategy_engine.config.get("candle_count", 200))
        df = await fetch_closed_candles(symbol, timeframe, candle_count)
        if df is None or df.empty or len(df) < 30:
            return None
        analyzed = strategy_engine.ta.analyze_instrument(df, strategy_engine.config)
        atr_period = int(
            strategy_engine.config["exit_rules"]["stop_loss"].get("atr_period", 14)
        )
        atr_raw = analyzed.iloc[-1].get(f"ATR_{atr_period}")
        atr = (
            float(atr_raw)
            if atr_raw is not None and not pd.isna(atr_raw) and float(atr_raw) > 0
            else None
        )
        last_open = self._to_utc(analyzed.index[-1])
        entry = {
            "bar_key": bar_key,
            "fetched_at": now,
            "lag_bars": int(df.attrs.get("lag_bars", 0)),
            "analyzed": analyzed,
            "atr": atr,
            "bar_close": last_open + timedelta(minutes=tf_min) if last_open else None,
        }
        self._bar_cache[symbol] = entry
        return entry

    async def _manage_one(
        self, trade: dict[str, Any], cfg: dict[str, Any], minutes_left: float
    ) -> None:
        position_id = int(trade["trade_id"])
        symbol = normalize_symbol(trade["instrument"])
        is_long = str(trade["direction"]).lower() in {"long", "buy"}
        sign = 1.0 if is_long else -1.0
        entry = float(trade["entry_price"])
        initial_stop = float(trade["stop_loss"])
        risk = abs(entry - initial_stop)
        if risk <= 0:
            return
        current_stop = float(trade.get("current_stop") or initial_stop)
        current_tp = float(trade.get("current_tp") or trade["take_profit"])
        broker_stop = float(trade.get("broker_stop") or initial_stop)
        broker_tp = float(trade.get("broker_tp") or trade["take_profit"])

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

        # 0. Software stop / target: the bot enforces the CURRENT levels itself, so
        #    trailing still protects the trade if the broker amend is missing or failed.
        if cfg.get("software_stops", True):
            if (is_long and price <= current_stop) or (
                not is_long and price >= current_stop
            ):
                label = (
                    "stop_loss"
                    if abs(current_stop - initial_stop) <= 0.05 * risk
                    else "trailing_stop"
                )
                await self._close_managed(position_id, symbol, label, r_now, peak_r)
                return
            if (is_long and price >= current_tp) or (
                not is_long and price <= current_tp
            ):
                await self._close_managed(
                    position_id, symbol, "take_profit", r_now, peak_r
                )
                return

        # 1. Late in the day: bank open profit instead of handing it back at the flatten.
        if minutes_left <= float(cfg["late_day_close_minutes"]) and r_now >= float(
            cfg["late_day_close_min_r"]
        ):
            await self._close_managed(
                position_id, symbol, "late_day_profit", r_now, peak_r
            )
            return

        entry_time = risk_manager._parse_ts(trade.get("entry_time"))
        max_stall_minutes = float(cfg.get("stalled_trade_max_hold_minutes", 0))
        if (
            entry_time is not None
            and max_stall_minutes > 0
            and (datetime.now(UTC) - entry_time).total_seconds()
            >= max_stall_minutes * 60
            and peak_r < float(cfg.get("stalled_trade_min_progress_r", 0.5))
        ):
            await self._close_managed(
                position_id, symbol, "time_stall", r_now, peak_r
            )
            return

        # 2. Reading the market again: reversal signs on candles that closed AFTER entry.
        analysis = await self._bar_analysis(symbol)
        analyzed = analysis["analyzed"] if analysis else None
        atr = analysis["atr"] if analysis else None
        fresh_bar = bool(
            analysis
            and analysis["bar_close"]
            and entry_time
            and analysis["bar_close"] > entry_time
        )

        if analyzed is not None and fresh_bar:
            latest = analyzed.iloc[-1]
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
            if cfg["reversal_exit_enabled"] and r_now >= float(
                cfg["reversal_min_profit_r"]
            ):
                rv = strategy_engine.ta.reversal_votes(
                    analyzed, is_long, body_atr=float(cfg["reversal_body_atr"])
                )
                if rv["votes"] >= int(cfg["reversal_votes_required"]):
                    _console(
                        f"Reversal detected {symbol} #{position_id}: {rv['detail']}",
                        level=logging.INFO,
                    )
                    await self._close_managed(
                        position_id, symbol, "reversal_exit", r_now, peak_r
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

        # 3. Ratchet the stop (never loosen it) and pull the target closer when momentum fades.
        def tighter(a: float, b: float) -> float:
            return max(a, b) if is_long else min(a, b)

        new_stop, new_tp = current_stop, current_tp
        why: list[str] = []

        def propose_stop(label: str, level: float) -> None:
            nonlocal new_stop
            candidate = tighter(new_stop, level)
            if candidate != new_stop:
                new_stop = candidate
                why.append(label)

        if peak_r >= float(cfg["breakeven_r"]):
            propose_stop(
                "breakeven", entry + sign * float(cfg["breakeven_buffer_r"]) * risk
            )
        if cfg["giveback_enabled"] and peak_r >= float(cfg["giveback_arm_r"]):
            keep = peak_r * (1.0 - float(cfg["giveback_fraction"]))
            propose_stop("giveback", entry + sign * keep * risk)
        if (
            cfg["trail_enabled"]
            and atr is not None
            and peak_r >= float(cfg["trail_start_r"])
        ):
            propose_stop(
                "atr_trail", peak_price - sign * float(cfg["trail_atr_mult"]) * atr
            )
        if minutes_left <= float(cfg["late_day_lock_minutes"]) and r_now >= float(
            cfg["late_day_lock_min_r"]
        ):
            propose_stop(
                "late_day_lock",
                entry + float(cfg["late_day_lock_fraction"]) * (price - entry),
            )

        symbol_info = await ctrader_client.get_symbol_info(symbol)
        pip_factor = 10**symbol_info.pip_position
        self._pip_position[symbol] = symbol_info.pip_position
        min_dist = float(cfg["min_stop_distance_pips"]) / pip_factor
        digits = symbol_info.pip_position + 1

        if (
            cfg["tp_adjust_enabled"]
            and analyzed is not None
            and atr is not None
            and peak_r >= float(cfg["tp_adjust_min_peak_r"])
            and strategy_engine.ta.momentum_fading(
                analyzed, is_long, bars=int(cfg["tp_fade_bars"])
            )
        ):
            candidate_tp = price + sign * float(cfg["tp_near_atr"]) * atr
            closer = candidate_tp < new_tp if is_long else candidate_tp > new_tp
            if closer and abs(candidate_tp - price) >= min_dist:
                new_tp = candidate_tp
                why.append("tp_compress")

        if (is_long and new_stop >= price) or (not is_long and new_stop <= price):
            # Price has already given back to the protective level: exit now.
            await self._close_managed(
                position_id, symbol, "trailing_stop", r_now, peak_r
            )
            return

        new_stop, new_tp = round(new_stop, digits), round(new_tp, digits)
        step = float(cfg["min_stop_step_r"]) * risk
        stop_moved = abs(new_stop - current_stop) >= step
        tp_moved = abs(new_tp - current_tp) >= step
        use_stop = new_stop if stop_moved else current_stop
        use_tp = new_tp if tp_moved else current_tp
        broker_update_pending = use_stop != broker_stop or use_tp != broker_tp
        if not (stop_moved or tp_moved or broker_update_pending):
            return

        # Persist intended levels for software enforcement. Broker-confirmed values
        # advance only after a successful amend so failures remain retryable.
        risk_manager.update_trade_management(
            position_id, current_stop=use_stop, current_tp=use_tp
        )
        broker_note = "broker levels already match"
        if broker_update_pending and (
            abs(price - use_stop) < min_dist or abs(price - use_tp) < min_dist
        ):
            broker_note = "too close to price for broker; will retry"
        elif broker_update_pending:
            try:
                # amend replaces BOTH levels, so always send both.
                if "tp_compress" in why:
                    tp_pips = abs(use_tp - price) * pip_factor
                    use_tp = await ctrader_client.modify_take_profit_pips(
                        position_id, tp_pips, stop_loss=use_stop
                    )
                    risk_manager.update_trade_management(
                        position_id, current_tp=use_tp
                    )
                else:
                    await ctrader_client.amend_position_sltp(
                        position_id, stop_loss=use_stop, take_profit=use_tp
                    )
                risk_manager.update_trade_management(
                    position_id, broker_stop=use_stop, broker_tp=use_tp
                )
                broker_stop, broker_tp = use_stop, use_tp
                broker_note = "broker updated"
            except Exception as exc:
                broker_note = (
                    f"BROKER AMEND FAILED ({exc!r}); software enforces level and will retry"
                )
                _console(
                    f"Amend failed for {symbol} #{position_id}: {exc!r}",
                    level=logging.ERROR,
                )
                if not self._amend_warned:
                    self._amend_warned = True
                    await telegram.send_message_async(
                        f"WARNING: stop/TP amend failed ({exc!r}). "
                        "Software protection remains active; broker amend will be retried."
                    )
        if broker_update_pending and not why:
            why.append("broker_retry")
        risk_manager.record_event(
            "trade_management",
            {
                "position_id": position_id,
                "instrument": symbol,
                "action": "amend_levels",
                "old_stop": current_stop,
                "new_stop": use_stop,
                "old_tp": current_tp,
                "new_tp": use_tp,
                "why": why,
                "note": broker_note,
                "r_now": round(r_now, 2),
                "peak_r": round(peak_r, 2),
            },
            aggregate_id=position_id,
        )
        management_atr = f"{atr * pip_factor:.1f}p" if atr is not None else "n/a"
        _console(
            f"POSITION MANAGEMENT {symbol} #{position_id} "
            f"price={price:.{digits}f} R={r_now:+.2f} peak={peak_r:+.2f}R "
            f"SL={current_stop:.{digits}f}->{use_stop:.{digits}f} "
            f"(step={sign * (use_stop - current_stop) * pip_factor:+.1f}p, "
            f"gap={abs(price - use_stop) * pip_factor:.1f}p) "
            f"TP={current_tp:.{digits}f}->{use_tp:.{digits}f} "
            f"(gap={abs(price - use_tp) * pip_factor:.1f}p) ATR={management_atr}",
            level=logging.INFO,
        )
        _console(
            f"MANAGEMENT CANDLE {symbol} #{position_id} "
            f"close_time={analysis.get('bar_close') if analysis else 'unavailable'} "
            f"bar=[{self._format_candle_bars(analyzed, digits, count=1)}] "
            f"reason={'+'.join(why)} broker={broker_note}",
            level=logging.INFO,
        )

    async def _close_all_positions(self, reason: str) -> None:
        async with self._pos_lock:
            await self._close_all_positions_locked(reason)

    async def _close_all_positions_locked(self, reason: str) -> None:
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
        trade = risk_manager.get_trade(position_id)
        if reason is None:
            reason = self._infer_exit_reason(trade, float(exit_price))
        r_multiple = risk_manager.record_exit(position_id, exit_price, pnl, reason)
        if trade is not None:
            symbol = normalize_symbol(str(trade["instrument"]))
            is_long = str(trade["direction"]).lower() in {"long", "buy"}
            entry_price = float(trade["entry_price"])
            pip_position = self._pip_position.get(symbol)
            if pip_position is None:
                try:
                    symbol_info = await ctrader_client.get_symbol_info(symbol)
                    pip_position = int(symbol_info.pip_position)
                    self._pip_position[symbol] = pip_position
                except (
                    CTraderError,
                    OSError,
                    RuntimeError,
                    TypeError,
                    ValueError,
                ) as exc:
                    _console(
                        f"Exit diagnostics missing pip size for {symbol} "
                        f"position_id={position_id}: {exc!r}",
                        level=logging.WARNING,
                    )
            pips_text = "n/a"
            if pip_position is not None:
                signed_pips = (
                    (float(exit_price) - entry_price)
                    * (10**pip_position)
                    * (1 if is_long else -1)
                )
                pips_text = f"{signed_pips:+.1f}p"

            entry_time = risk_manager._parse_ts(trade.get("entry_time"))
            hold_text = "unknown"
            if entry_time is not None:
                held = datetime.now(UTC) - entry_time
                hold_seconds = max(0, int(held.total_seconds()))
                hours, remainder = divmod(hold_seconds, 3600)
                minutes, seconds = divmod(remainder, 60)
                hold_text = f"{hours}h{minutes:02d}m{seconds:02d}s"

            bar_data = self._bar_cache.get(symbol)
            if not bar_data or bar_data.get("analyzed") is None:
                try:
                    bar_data = await self._bar_analysis(symbol)
                except (
                    CTraderError,
                    OSError,
                    RuntimeError,
                    TypeError,
                    ValueError,
                ) as exc:
                    _console(
                        f"Exit candle diagnostics unavailable for {symbol} "
                        f"position_id={position_id}: {exc!r}",
                        level=logging.WARNING,
                    )
            bar_df = bar_data.get("analyzed") if bar_data else None
            digits = (pip_position + 1) if pip_position is not None else 5
            _console(
                f"POSITION CLOSED id={position_id} {symbol} "
                f"{str(trade['direction']).upper()} reason={reason} "
                f"entry={entry_price:.{digits}f} exit={float(exit_price):.{digits}f} "
                f"move={pips_text} gross_pnl={float(pnl):+.2f} "
                f"R={r_multiple:+.2f} peak_R={float(trade.get('peak_r') or 0.0):+.2f} "
                f"held={hold_text} initial_SL={float(trade.get('stop_loss') or 0.0):.{digits}f} "
                f"current_SL={float(trade.get('current_stop') or trade.get('stop_loss') or 0.0):.{digits}f} "
                f"current_TP={float(trade.get('current_tp') or trade.get('take_profit') or 0.0):.{digits}f}",
                level=logging.INFO,
            )
            _console(
                f"EXIT CANDLE CONTEXT id={position_id} {symbol} "
                f"timeframe={strategy_engine.config.get('timeframe', 'M15')} "
                f"bars=[{self._format_candle_bars(bar_df, digits)}]",
                level=logging.INFO,
            )
            await self._deliver_pending_trade_notifications()

    async def _deliver_pending_trade_notifications(self) -> None:
        """Send journal-backed alerts and retain failures for a later retry."""
        async with self._notification_lock:
            await self._deliver_pending_trade_notifications_locked()

    async def _deliver_pending_trade_notifications_locked(self) -> None:
        pending = risk_manager.get_pending_trade_notifications()
        if not pending:
            return

        open_ids: set[int] = set()
        if any(trade["status"] == "open" for trade in pending):
            try:
                open_ids = {
                    int(position.position_id)
                    for position in await ctrader_client.get_bot_positions()
                }
            except (CTraderError, RuntimeError, ValueError) as exc:
                _console(
                    f"Cannot verify open bot positions for Telegram alerts: {exc}",
                    level=logging.WARNING,
                )
                return

        for trade in pending:
            position_id = int(trade["trade_id"])
            status = str(trade["status"])
            if status == "open":
                if position_id not in open_ids:
                    continue
                message = (
                    f"OPEN {trade['instrument']} #{position_id} "
                    f"{str(trade['direction']).upper()} | "
                    f"entry={float(trade['entry_price']):.5f} | "
                    f"size={float(trade['position_size']):g} lots | "
                    f"SL={float(trade.get('current_stop') or trade['stop_loss']):.5f} | "
                    f"TP={float(trade.get('current_tp') or trade['take_profit']):.5f}"
                )
                event = "open"
            else:
                message = (
                    f"CLOSED {trade['instrument']} #{position_id} "
                    f"{str(trade['direction']).upper()} | "
                    f"{trade.get('exit_reason') or 'broker_exit'} | "
                    f"exit={float(trade.get('exit_price') or 0.0):.5f} | "
                    f"P&L={float(trade.get('pnl') or 0.0):+.2f} | "
                    f"R={float(trade.get('r_multiple') or 0.0):+.2f}"
                )
                event = "close"

            if await telegram.send_message_async(message):
                risk_manager.mark_trade_notification_sent(position_id, event)

    @staticmethod
    def _infer_exit_reason(trade: dict[str, Any] | None, exit_price: float) -> str:
        """Label broker-side exits so post-mortems can tell SL / trail / TP apart."""
        if not trade:
            return "broker_exit"
        try:
            entry = float(trade["entry_price"])
            initial_stop = float(trade["stop_loss"])
            tp = float(trade.get("current_tp") or trade["take_profit"])
            original_tp = float(trade["take_profit"])
            current_stop = float(trade.get("current_stop") or initial_stop)
        except (KeyError, TypeError, ValueError):
            return "broker_exit"
        tol = 0.15 * abs(entry - initial_stop)
        if tol <= 0:
            return "broker_exit"
        if abs(exit_price - tp) <= tol or abs(exit_price - original_tp) <= tol:
            return "take_profit"
        if abs(exit_price - current_stop) <= tol:
            return (
                "stop_loss"
                if abs(current_stop - initial_stop) <= tol
                else "trailing_stop"
            )
        return "broker_exit"

    async def _reconcile_exits(self) -> None:
        async with self._pos_lock:
            await self._reconcile_exits_locked()

    async def _reconcile_exits_locked(self) -> None:
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
        summary_day = market_hours.now_est().date().isoformat()
        if risk_manager.daily_summary_exists(summary_day):
            return

        try:
            account = await ctrader_client.get_account_info()
            balance_text = f"{float(account.balance):.2f}"
        except (CTraderError, RuntimeError, ValueError, sqlite3.Error) as exc:
            _console(f"Unable to fetch balance for daily summary: {exc}", level=logging.WARNING)
            balance_text = "unavailable"

        trades = risk_manager.get_today_closed_trades()
        pnl = sum(float(trade.get("pnl") or 0.0) for trade in trades)
        r_values = [float(trade.get("r_multiple") or 0.0) for trade in trades]
        wins = sum(value > 0 for value in r_values)
        losses = sum(value < 0 for value in r_values)
        non_break_even = wins + losses
        win_rate = wins / non_break_even * 100 if non_break_even else 0.0
        cumulative_r = 0.0
        peak_r = 0.0
        max_drawdown = 0.0
        for value in r_values:
            cumulative_r += value
            peak_r = max(peak_r, cumulative_r)
            max_drawdown = max(max_drawdown, peak_r - cumulative_r)
        summary_text = (
            f"Daily Summary · {summary_day} ET\n"
            f"Balance: {balance_text}\n"
            f"Closed bot trades: {len(trades)} (wins {wins}, losses {losses})\n"
            f"Realized P&L: {pnl:+.2f}\n"
            f"Net R: {cumulative_r:+.2f} | Win rate: {win_rate:.1f}%\n"
            f"Max R drawdown: {max_drawdown:.2f}"
        )
        _console(summary_text, level=logging.INFO)
        risk_manager.save_daily_summary()
        if await telegram.send_daily_summary_async(summary_text):
            risk_manager.mark_daily_summary_notified(summary_day)

    def stop(self) -> None:
        self.running = False
        _console("cTraderbot loop stopped", level=logging.INFO)
        telegram.send_message("auto trading stopped.")


trading_cycle = TradingCycle()
