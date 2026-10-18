from datetime import UTC, datetime, timedelta
from decimal import Decimal
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest import IsolatedAsyncioTestCase, TestCase
from unittest.mock import AsyncMock, patch

import pandas as pd
from ctrader_api_client import SpotEvent

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

        self.assertIn("Open positions · live from cTrader", page)
        self.assertIn("Closed positions · broker history", page)
        self.assertIn("/dashboard/api/live", page)
        self.assertIn("/dashboard/api/history", page)

    async def test_live_dashboard_uses_broker_pnl_for_equity(self) -> None:
        position = SimpleNamespace(
            position_id=12,
            symbol_id=1,
            side=SimpleNamespace(value="BUY"),
            volume=100,
            entry_price=Decimal("1.10000"),
            stop_loss=Decimal("1.09000"),
            take_profit=Decimal("1.12000"),
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
                "dashboard.multiple_tracker.ctrader_client.get_open_positions",
                new=AsyncMock(return_value=[position]),
            ),
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
        self.assertEqual(response["positions"][0]["lots"], 1.0)

    async def test_history_dashboard_aggregates_broker_closed_deals(self) -> None:
        closed_at = datetime.now(UTC)
        deal = SimpleNamespace(
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
        symbol_info = SimpleNamespace(
            volume_to_lots=lambda volume: Decimal(volume) / Decimal(100)
        )
        with (
            patch(
                "dashboard.multiple_tracker.ctrader_client.get_deals",
                new=AsyncMock(return_value=[deal]),
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
                "dashboard.multiple_tracker.risk_manager.get_trade_history",
                return_value=pd.DataFrame(columns=["trade_id", "pnl", "r_multiple"]),
            ),
            patch(
                "dashboard.multiple_tracker.risk_manager.get_r_multiple_stats",
                return_value={"total_trades": 0, "win_rate": 0},
            ),
        ):
            response = await api_trade_history(days=30)

        self.assertEqual(response["realized_pnl"], 17.0)
        self.assertEqual(response["closed_positions"][0]["side"], "buy")
        self.assertEqual(response["closed_positions"][0]["pnl"], 17.0)
