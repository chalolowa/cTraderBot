"""Autonomous cTrader trading cycle."""

import asyncio
import logging
import math
import sqlite3
from datetime import UTC, datetime

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
        except (CTraderError, RuntimeError, ValueError, sqlite3.Error) as exc:
            _console(
                f"Unable to fetch account info for loss limit check: {exc}",
                level=logging.WARNING,
            )
            return

        if risk_manager.daily_loss_limit_reached(balance, max_daily_loss_pct):
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

        _console(
            f"Evaluating {symbol}: timeframe={timeframe}, candles={len(df)}",
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

    async def _record_exit(self, position_id: int, reason: str) -> None:
        exit_price, pnl = await ctrader_client.get_closed_position_summary(position_id)
        risk_manager.record_exit(position_id, exit_price, pnl, reason)

    async def _reconcile_exits(self) -> None:
        broker_positions = await ctrader_client.get_bot_positions()
        broker_position_ids = {position.position_id for position in broker_positions}
        for trade in risk_manager.get_open_trades():
            position_id = int(trade["trade_id"])
            if position_id in broker_position_ids:
                continue
            try:
                await self._record_exit(position_id, "broker_exit")
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
