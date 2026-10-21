from datetime import UTC, datetime, timedelta
from decimal import Decimal
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest import IsolatedAsyncioTestCase, TestCase
from unittest.mock import AsyncMock, call, patch

import pandas as pd
from ctrader_api_client import ExecutionEvent, SpotEvent
from ctrader_api_client.enums import ExecutionType, OrderSide

from config.ctrader_client import CTraderBrokerClient
from dashboard.multiple_tracker import (
    api_live_account,
    api_trade_history,
    dashboard_home,
)
from src.risk_manager import RiskManager, Trade
from src.scanner import MarketScanner
from src.strategy_engine import StrategyEngine
from src.technical_analysis import TechnicalAnalyzer
from src.telegram_bot import TelegramNotifier
from src.trading_cycle import TradingCycle


class SpotQuoteTests(IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        self.client = object.__new__(CTraderBrokerClient)
        self.client._last_spot = {}
        self.client._symbol_name_to_id = {"EURUSD": 1}
        self.timestamp = datetime.now(UTC)

    def test_merges_one_sided_spot_events_into_complete_quote(self) -> None:
        self.client._cache_spot_event(
            SpotEvent(
                account_id=7,
                symbol_id=1,
                bid=Decimal("1.10000"),
                ask=None,
                trendbar=[],
                timestamp=self.timestamp,
            )
        )
        self.client._cache_spot_event(
            SpotEvent(
                account_id=7,
                symbol_id=1,
                bid=None,
                ask=Decimal("1.10012"),
                trendbar=[],
                timestamp=self.timestamp + timedelta(milliseconds=1),
            )
        )

        quote = self.client.get_tick("EURUSD")
        self.assertEqual(quote.bid, Decimal("1.10000"))
        self.assertEqual(quote.ask, Decimal("1.10012"))

    def test_rejects_incomplete_quote_instead_of_returning_none_side(self) -> None:
        self.client._cache_spot_event(
            SpotEvent(
                account_id=7,
                symbol_id=1,
                bid=Decimal("1.10000"),
                ask=None,
                trendbar=[],
                timestamp=self.timestamp,
            )
        )

        with self.assertRaisesRegex(RuntimeError, "ask is not available"):
            self.client.get_tick("EURUSD")

    async def test_rejected_stop_amendment_is_reported_as_failure(self) -> None:
        client = object.__new__(CTraderBrokerClient)
        client._account_id = 7
        client._client = SimpleNamespace(
            trading=SimpleNamespace(
                amend_position=AsyncMock(
                    return_value=ExecutionEvent(
                        account_id=7,
                        execution_type=ExecutionType.ORDER_REJECTED,
                        order_id=0,
                        position_id=12,
                        symbol_id=1,
                        side=OrderSide.BUY,
                        filled_volume=None,
                        fill_price=None,
                        timestamp=datetime.now(UTC),
                        error_code="INVALID_SL",
                    )
                )
            )
        )

        with self.assertRaisesRegex(RuntimeError, "ORDER_REJECTED.*INVALID_SL"):
            await client.amend_position_sltp(12, 1.101, 1.13)


class TelegramNotifierTests(IsolatedAsyncioTestCase):
    async def test_initializes_and_retries_failed_message_delivery(self) -> None:
        bot = SimpleNamespace(
            initialize=AsyncMock(),
            shutdown=AsyncMock(),
            send_message=AsyncMock(
                side_effect=[RuntimeError("temporary network error"), object()]
            ),
        )
        with (
            patch(
                "src.telegram_bot.settings",
                SimpleNamespace(
                    TELEGRAM_BOT_TOKEN="test-token",
                    TELEGRAM_CHAT_ID="123",
                ),
            ),
            patch("telegram.Bot", return_value=bot),
            patch("src.telegram_bot.asyncio.sleep", new=AsyncMock()),
        ):
            notifier = TelegramNotifier()
            delivered = await notifier.send_message_async("trade opened")
            await notifier.shutdown()

        self.assertTrue(delivered)
        bot.initialize.assert_awaited_once()
        self.assertEqual(bot.send_message.await_count, 2)
        bot.shutdown.assert_awaited_once()

    async def test_missing_credentials_reports_delivery_failure(self) -> None:
        with patch(
            "src.telegram_bot.settings",
            SimpleNamespace(TELEGRAM_BOT_TOKEN="", TELEGRAM_CHAT_ID=""),
        ):
            notifier = TelegramNotifier()
            delivered = await notifier.send_message_async("trade opened")

        self.assertFalse(delivered)

class ScannerTests(IsolatedAsyncioTestCase):
    async def test_scanner_does_not_override_looser_strategy_filters(self) -> None:
        scanner = MarketScanner()
        candles = pd.DataFrame(
            [
                {
                    "close": 1.1,
                    "SMA_9": 1.101,
                    "SMA_21": 1.100,
                    "ATR_14": 0.0002,
                    "ADX_14": 16.0,
                    "RSI_14": 52.0,
                }
            ]
        )
        scanner.ta.analyze_instrument = lambda *_: candles
        tick = SimpleNamespace(
            bid=Decimal("1.10000"),
            ask=Decimal("1.10010"),
        )
        with (
            patch(
                "src.scanner.ctrader_client.copy_rates_from_pos",
                new=AsyncMock(return_value=pd.DataFrame({"close": [1.1] * 40})),
            ),
            patch(
                "src.scanner.ctrader_client.get_symbol_info",
                new=AsyncMock(return_value=SimpleNamespace(pip_position=4)),
            ),
            patch("src.scanner.ctrader_client.get_tick", return_value=tick),
            patch("src.scanner.strategy_engine.evaluate_entry") as evaluate_entry,
        ):
            result = await scanner._analyze_instrument("EURUSD")

        self.assertTrue(result["tradable"])
        self.assertAlmostEqual(result["spread_pips"], 1.0)
        self.assertEqual(result["adx"], 16.0)
        self.assertAlmostEqual(result["atr_pips"], 2.0)
        evaluate_entry.assert_not_called()


class TechnicalAnalysisTests(TestCase):
    def test_rsi_fallback_handles_flat_and_one_way_price_series(self) -> None:
        flat = pd.DataFrame({"close": [1.0] * 20})
        rising = pd.DataFrame({"close": [float(index) for index in range(20)]})

        with patch("src.technical_analysis.ta", None):
            flat_rsi = TechnicalAnalyzer.add_rsi(flat, 7)["RSI_7"].iloc[-1]
            rising_rsi = TechnicalAnalyzer.add_rsi(rising, 7)["RSI_7"].iloc[-1]

        self.assertEqual(flat_rsi, 50.0)
        self.assertEqual(rising_rsi, 100.0)

    def test_trend_bias_requires_sufficient_data_and_detects_uptrend(self) -> None:
        close = pd.Series([1 + index * 0.001 for index in range(100)])
        candles = pd.DataFrame(
            {
                "close": close,
                "high": close + 0.002,
                "low": close - 0.002,
            }
        )

        with patch("src.technical_analysis.ta", None):
            insufficient = TechnicalAnalyzer.compute_trend_bias(candles.iloc[:20])
            bias = TechnicalAnalyzer.compute_trend_bias(candles)

        self.assertEqual(insufficient["bias"], "neutral")
        self.assertEqual(insufficient["reason"], "insufficient_htf_data")
        self.assertEqual(bias["bias"], "long")
        self.assertGreater(bias["drift_atr"], 0)

    def test_analyzer_builds_indicator_periods_from_entry_rules(self) -> None:
        close = pd.Series([1 + index * 0.001 for index in range(40)])
        candles = pd.DataFrame(
            {
                "close": close,
                "high": close + 0.002,
                "low": close - 0.002,
            }
        )
        strategy = {
            "entry_rules": {
                "long": {
                    "conditions": [
                        {
                            "indicator": "SMA",
                            "period": 5,
                            "comparison": "crosses_above",
                            "reference": "SMA",
                            "ref_period": 12,
                        },
                        {
                            "indicator": "RSI",
                            "period": 7,
                            "comparison": ">",
                            "value": 50,
                        },
                        {
                            "indicator": "ADX",
                            "period": 8,
                            "comparison": ">",
                            "value": 15,
                        },
                    ]
                }
            },
            "exit_rules": {"stop_loss": {"type": "ATR_MULTIPLE", "atr_period": 9}},
        }

        analyzed = TechnicalAnalyzer.analyze_instrument(candles, strategy)
        signal = TechnicalAnalyzer.get_latest_signal(analyzed)

        self.assertIn("sma_5", signal)
        self.assertIn("sma_12", signal)
        self.assertIn("rsi_7", signal)
        self.assertIn("adx_8", signal)
        self.assertIn("atr_9", signal)

    def test_strategy_evaluates_configured_indicator_periods(self) -> None:
        close = pd.Series([1 + index * 0.001 for index in range(40)])
        candles = pd.DataFrame(
            {
                "close": close,
                "high": close + 0.002,
                "low": close - 0.002,
            }
        )
        engine = object.__new__(StrategyEngine)
        engine.ta = TechnicalAnalyzer()
        engine.config = {
            "name": "dynamic-period-test",
            "entry_rules": {
                "long": {
                    "conditions": [
                        {
                            "indicator": "RSI",
                            "period": 7,
                            "comparison": ">",
                            "value": 50,
                        }
                    ],
                    "require_all": True,
                }
            },
            "exit_rules": {"stop_loss": {"type": "ATR_MULTIPLE", "atr_period": 9}},
        }

        should_enter, metadata = engine.evaluate_entry(candles, "long")

        self.assertTrue(should_enter)
        self.assertGreater(metadata["atr"], 0)
        self.assertEqual(metadata["conditions"][0]["actual"], 100.0)


class RiskManagerTests(TestCase):
    def test_daily_loss_limit_uses_strategy_threshold(self) -> None:
        manager = object.__new__(RiskManager)
        manager.get_daily_summary = lambda: {"daily_pnl": -31.0}

        self.assertTrue(manager.daily_loss_limit_reached(1000.0, 3.0))
        manager.get_daily_summary = lambda: {"daily_pnl": -29.0}
        self.assertFalse(manager.daily_loss_limit_reached(1000.0, 3.0))

    def test_symbol_cooldown_sees_loss_closed_today_from_older_entry(self) -> None:
        with TemporaryDirectory() as temp_dir:
            manager = RiskManager(str(Path(temp_dir) / "trades.db"))
            start, _ = manager._today_window_utc()
            manager.record_entry(
                Trade(
                    trade_id=1,
                    instrument="EURUSD",
                    direction="long",
                    entry_price=1.1,
                    stop_loss=1.09,
                    take_profit=1.13,
                    entry_time=start - timedelta(minutes=1),
                )
            )
            manager.record_exit(1, exit_price=1.09, pnl=-10.0, exit_reason="stop_loss")

            activity = manager.symbol_activity_today("EURUSD")

        self.assertEqual(activity["count"], 0)
        self.assertIsNotNone(activity["last_closed"])
        self.assertEqual(activity["last_closed"]["pnl"], -10.0)


class DashboardTests(IsolatedAsyncioTestCase):
    async def test_dashboard_renders_live_positions_and_history_sections(self) -> None:
        page = await dashboard_home()

        self.assertIn("Open bot positions · live from cTrader", page)
        self.assertIn("Closed bot positions", page)
        self.assertNotIn("Includes trades placed outside the bot", page)
        self.assertIn("/dashboard/api/live", page)
        self.assertIn("/dashboard/api/history", page)

    async def test_live_dashboard_only_includes_bot_positions(self) -> None:
        bot_position = SimpleNamespace(
            position_id=12,
            symbol_id=1,
            side=SimpleNamespace(value="BUY"),
            volume=100,
            entry_price=Decimal("1.10000"),
            stop_loss=Decimal("1.09000"),
            take_profit=Decimal("1.12000"),
        )
        manual_position = SimpleNamespace(
            position_id=13,
            symbol_id=1,
            side=SimpleNamespace(value="SELL"),
            volume=200,
            entry_price=Decimal("1.11000"),
            stop_loss=None,
            take_profit=None,
        )
        symbol_info = SimpleNamespace(
            volume_to_lots=lambda volume: Decimal(volume) / Decimal(100)
        )
        with (
            patch(
                "dashboard.multiple_tracker.ctrader_client.get_account_info",
                new=AsyncMock(return_value=SimpleNamespace(balance=Decimal(1000))),
            ),
            patch(
                "dashboard.multiple_tracker.ctrader_client.get_bot_positions",
                new=AsyncMock(return_value=[bot_position]),
            ),
            patch(
                "dashboard.multiple_tracker.ctrader_client.get_open_positions",
                new=AsyncMock(return_value=[bot_position, manual_position]),
            ) as get_open_positions,
            patch(
                "dashboard.multiple_tracker.ctrader_client.get_unrealized_pnl",
                new=AsyncMock(return_value={12: 25.0}),
            ),
            patch(
                "dashboard.multiple_tracker.ctrader_client.symbol_name",
                return_value="EURUSD",
            ),
            patch(
                "dashboard.multiple_tracker.ctrader_client.get_symbol_info",
                new=AsyncMock(return_value=symbol_info),
            ),
        ):
            response = await api_live_account()

        self.assertEqual(response["balance"], 1000.0)
        self.assertEqual(response["unrealized_pnl"], 25.0)
        self.assertEqual(response["equity"], 1025.0)
        self.assertEqual(len(response["positions"]), 1)
        self.assertEqual(response["positions"][0]["position_id"], 12)
        self.assertEqual(response["positions"][0]["lots"], 1.0)
        get_open_positions.assert_not_awaited()

    async def test_history_dashboard_includes_labeled_bot_deals_not_manual_deals(self) -> None:
        closed_at = datetime.now(UTC)
        bot_deal = SimpleNamespace(
            close_detail=SimpleNamespace(
                closed_volume=100,
                entry_price=Decimal("1.10000"),
                gross_profit=Decimal(20),
                swap=Decimal(-1),
                commission=Decimal(-2),
                pnl_conversion_fee=Decimal(0),
            ),
            position_id=12,
            symbol_id=1,
            side=SimpleNamespace(value="SELL"),
            execution_price=Decimal("1.10200"),
            execution_timestamp=closed_at,
        )
        manual_deal = SimpleNamespace(
            close_detail=SimpleNamespace(
                closed_volume=200,
                entry_price=Decimal("1.11000"),
                gross_profit=Decimal(50),
                swap=Decimal(0),
                commission=Decimal(0),
                pnl_conversion_fee=Decimal(0),
            ),
            position_id=13,
            symbol_id=1,
            side=SimpleNamespace(value="BUY"),
            execution_price=Decimal("1.10800"),
            execution_timestamp=closed_at,
        )
        symbol_info = SimpleNamespace(
            volume_to_lots=lambda volume: Decimal(volume) / Decimal(100)
        )
        with (
            patch(
                "dashboard.multiple_tracker.ctrader_client.get_orders",
                new=AsyncMock(
                    return_value=[
                        SimpleNamespace(position_id=12, label="tradebot"),
                        SimpleNamespace(position_id=13, label="manual"),
                    ]
                ),
            ),
            patch(
                "dashboard.multiple_tracker.ctrader_client.get_deals",
                new=AsyncMock(return_value=[bot_deal, manual_deal]),
            ),
            patch(
                "dashboard.multiple_tracker.ctrader_client.symbol_name",
                return_value="EURUSD",
            ),
            patch(
                "dashboard.multiple_tracker.ctrader_client.get_symbol_info",
                new=AsyncMock(return_value=symbol_info),
            ),
            patch(
                "dashboard.multiple_tracker.risk_manager.get_closed_trades",
                return_value=[],
            ),
            patch(
                "dashboard.multiple_tracker.risk_manager.get_r_multiple_stats",
                return_value={"total_trades": 1, "win_rate": 100},
            ),
        ):
            response = await api_trade_history(days=30)

        self.assertEqual(response["realized_pnl"], 17.0)
        self.assertEqual(len(response["closed_positions"]), 1)
        self.assertEqual(response["closed_positions"][0]["position_id"], 12)
        self.assertEqual(response["closed_positions"][0]["side"], "buy")
        self.assertEqual(response["closed_positions"][0]["pnl"], 17.0)
        self.assertIsNone(response["closed_positions"][0]["r_multiple"])


class TradingCycleNotificationTests(IsolatedAsyncioTestCase):
    async def test_failed_daily_summary_is_retried_until_delivered(self) -> None:
        cycle = TradingCycle()
        with (
            patch(
                "src.trading_cycle.market_hours.now_est",
                return_value=datetime(2026, 10, 21, 15, 30, tzinfo=UTC),
            ),
            patch(
                "src.trading_cycle.risk_manager.daily_summary_exists",
                side_effect=[False, False, True],
            ),
            patch(
                "src.trading_cycle.ctrader_client.get_account_info",
                new=AsyncMock(return_value=SimpleNamespace(balance=1000)),
            ),
            patch(
                "src.trading_cycle.risk_manager.get_today_closed_trades",
                return_value=[],
            ),
            patch("src.trading_cycle.risk_manager.save_daily_summary") as save,
            patch(
                "src.trading_cycle.risk_manager.mark_daily_summary_notified"
            ) as mark_notified,
            patch(
                "src.trading_cycle.telegram.send_daily_summary_async",
                new=AsyncMock(side_effect=[False, True]),
            ) as send_summary,
        ):
            await cycle.send_daily_summary()
            await cycle.send_daily_summary()
            await cycle.send_daily_summary()

        self.assertEqual(send_summary.await_count, 2)
        self.assertEqual(save.call_count, 2)
        mark_notified.assert_called_once_with("2026-10-21")

    async def test_pending_trade_alerts_are_retained_until_delivered(self) -> None:
        cycle = TradingCycle()
        open_trade = {
            "trade_id": "12",
            "status": "open",
            "instrument": "EURUSD",
            "direction": "long",
            "entry_price": 1.1,
            "stop_loss": 1.09,
            "take_profit": 1.13,
            "current_stop": 1.105,
            "current_tp": 1.13,
            "position_size": 1.0,
        }
        closed_trade = {
            "trade_id": "13",
            "status": "closed",
            "instrument": "GBPUSD",
            "direction": "short",
            "entry_price": 1.3,
            "exit_price": 1.29,
            "pnl": 10.0,
            "r_multiple": 1.0,
            "exit_reason": "take_profit",
        }
        with (
            patch(
                "src.trading_cycle.risk_manager.get_pending_trade_notifications",
                side_effect=[[open_trade, closed_trade], [open_trade]],
            ),
            patch(
                "src.trading_cycle.ctrader_client.get_bot_positions",
                new=AsyncMock(
                    return_value=[SimpleNamespace(position_id=12)]
                ),
            ),
            patch(
                "src.trading_cycle.telegram.send_message_async",
                new=AsyncMock(side_effect=[False, True, True]),
            ) as send_message,
            patch(
                "src.trading_cycle.risk_manager.mark_trade_notification_sent"
            ) as mark_sent,
        ):
            await cycle._deliver_pending_trade_notifications()
            mark_sent.assert_called_once_with(13, "close")
            await cycle._deliver_pending_trade_notifications()

        self.assertEqual(send_message.await_count, 3)
        self.assertEqual(
            mark_sent.call_args_list,
            [
                call(13, "close"),
                call(12, "open"),
            ],
        )

    async def test_broker_trailing_amendment_retries_until_confirmed(self) -> None:
        cycle = TradingCycle()
        trade = {
            "trade_id": 12,
            "instrument": "EURUSD",
            "direction": "long",
            "entry_price": 1.1,
            "stop_loss": 1.09,
            "take_profit": 1.13,
            "current_stop": 1.117,
            "current_tp": 1.13,
            "broker_stop": 1.09,
            "broker_tp": 1.13,
            "peak_r": 2.0,
            "peak_price": 1.12,
            "entry_time": datetime.now(UTC).isoformat(),
        }
        cfg = {
            "software_stops": True,
            "late_day_close_minutes": 10,
            "late_day_close_min_r": 5.0,
            "exit_on_opposite_cross_in_profit": False,
            "reversal_exit_enabled": False,
            "exit_on_htf_flip": False,
            "breakeven_r": 1.0,
            "breakeven_buffer_r": 0.1,
            "giveback_enabled": False,
            "trail_enabled": True,
            "trail_start_r": 1.5,
            "trail_atr_mult": 1.5,
            "late_day_lock_minutes": 10,
            "late_day_lock_min_r": 5.0,
            "late_day_lock_fraction": 0.5,
            "min_stop_distance_pips": 1.0,
            "tp_adjust_enabled": False,
            "min_stop_step_r": 0.1,
        }
        with (
            patch(
                "src.trading_cycle.ctrader_client.get_tick",
                return_value=SimpleNamespace(bid=1.12, ask=1.1201),
            ),
            patch(
                "src.trading_cycle.ctrader_client.get_symbol_info",
                new=AsyncMock(return_value=SimpleNamespace(pip_position=4)),
            ),
            patch.object(
                cycle,
                "_bar_analysis",
                new=AsyncMock(
                    return_value={
                        "analyzed": None,
                        "atr": 0.002,
                        "bar_close": None,
                    }
                ),
            ),
            patch(
                "src.trading_cycle.ctrader_client.amend_position_sltp",
                new=AsyncMock(
                    side_effect=[RuntimeError("temporary broker rejection"), None]
                ),
            ) as amend,
            patch(
                "src.trading_cycle.risk_manager.update_trade_management"
            ) as update_management,
            patch("src.trading_cycle.risk_manager.record_event"),
            patch(
                "src.trading_cycle.telegram.send_message_async", new=AsyncMock()
            ),
        ):
            await cycle._manage_one(trade, cfg, minutes_left=500)
            await cycle._manage_one(trade, cfg, minutes_left=500)

        self.assertEqual(amend.await_count, 2)
        self.assertAlmostEqual(amend.await_args.kwargs["stop_loss"], 1.117)
        self.assertTrue(
            any(
                call.kwargs.get("broker_stop") == 1.117
                for call in update_management.call_args_list
            )
        )

    async def test_market_close_flattens_then_sends_daily_summary(self) -> None:
        cycle = TradingCycle()
        with (
            patch("src.trading_cycle.market_hours.is_close_time", return_value=True),
            patch.object(cycle, "_reconcile_exits", new=AsyncMock()) as reconcile,
            patch.object(cycle, "_close_all_positions", new=AsyncMock()) as close_all,
            patch.object(cycle, "send_daily_summary", new=AsyncMock()) as summary,
        ):
            await cycle._tick()

        reconcile.assert_awaited_once()
        close_all.assert_awaited_once_with("market_close")
        summary.assert_awaited_once()

    async def test_reconciled_bot_exit_sends_close_notification(self) -> None:
        cycle = TradingCycle()
        trade = {
            "instrument": "EURUSD",
            "direction": "long",
            "entry_price": 1.1,
            "stop_loss": 1.09,
            "take_profit": 1.13,
        }
        with (
            patch("src.trading_cycle.risk_manager.get_trade", return_value=trade),
            patch(
                "src.trading_cycle.ctrader_client.get_closed_position_summary",
                new=AsyncMock(return_value=(1.12, 20.0)),
            ),
            patch(
                "src.trading_cycle.risk_manager.record_exit", return_value=2.0
            ) as record_exit,
            patch.object(
                cycle, "_deliver_pending_trade_notifications", new=AsyncMock()
            ) as deliver_pending,
            patch(
                "src.trading_cycle.telegram.send_message_async", new=AsyncMock()
            ) as send_message,
        ):
            await cycle._record_exit(12, "take_profit")

        record_exit.assert_called_once_with(12, 1.12, 20.0, "take_profit")
        deliver_pending.assert_awaited_once()
        send_message.assert_not_awaited()

    async def test_daily_summary_uses_today_and_is_only_sent_once(self) -> None:
        cycle = TradingCycle()
        trades = [
            {"pnl": 20.0, "r_multiple": 2.0},
            {"pnl": -10.0, "r_multiple": -1.0},
        ]
        with (
            patch(
                "src.trading_cycle.market_hours.now_est",
                return_value=datetime(2026, 10, 21, 15, 30, tzinfo=UTC),
            ),
            patch(
                "src.trading_cycle.risk_manager.daily_summary_exists",
                side_effect=[False, True],
            ),
            patch(
                "src.trading_cycle.ctrader_client.get_account_info",
                new=AsyncMock(return_value=SimpleNamespace(balance=1000)),
            ),
            patch(
                "src.trading_cycle.risk_manager.get_today_closed_trades",
                return_value=trades,
            ),
            patch("src.trading_cycle.risk_manager.save_daily_summary") as save,
            patch(
                "src.trading_cycle.risk_manager.mark_daily_summary_notified"
            ) as mark_notified,
            patch(
                "src.trading_cycle.telegram.send_daily_summary_async",
                new=AsyncMock(),
            ) as send_summary,
        ):
            await cycle.send_daily_summary()
            await cycle.send_daily_summary()

        save.assert_called_once()
        mark_notified.assert_called_once_with("2026-10-21")
        send_summary.assert_awaited_once()
        summary = send_summary.await_args.args[0]
        self.assertIn("2026-10-21 ET", summary)
        self.assertIn("Closed bot trades: 2", summary)
        self.assertIn("Realized P&L: +10.00", summary)
        self.assertIn("Net R: +1.00", summary)
