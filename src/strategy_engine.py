"""
Strategy Engine reads strategy.json and evaluates entry/exit conditions
against live market data. Fully dynamic — edit JSON to change strategy.
"""
import json
import logging
import os
from typing import Any, ClassVar

import pandas as pd

from src.logger import _console
from src.technical_analysis import TechnicalAnalyzer


class StrategyEngine:
    def __init__(self, config_path: str | None = None):
        if config_path is None:
            config_path = os.path.join(
                os.path.dirname(os.path.dirname(__file__)), "config", "strategy.json"
            )
        self.config_path = config_path
        self.ta = TechnicalAnalyzer()
        self.config = self._load_config(config_path)

    def _validate_config(self, config: dict[str, Any]) -> None:
        """Validate strategy configuration schema and required keys."""
        if not isinstance(config, dict):
            raise TypeError("Strategy configuration must be a valid JSON dictionary")

        required_sections = ["name", "entry_rules", "exit_rules", "risk_management"]
        for section in required_sections:
            if section not in config:
                raise ValueError(f"Strategy configuration missing required section: '{section}'")

        entry_rules = config.get("entry_rules", {})
        for direction in ["long", "short"]:
            if direction in entry_rules:
                rules = entry_rules[direction]
                conditions = rules.get("conditions", [])
                for idx, cond in enumerate(conditions):
                    if not isinstance(cond, dict):
                        raise TypeError(f"Malformed condition #{idx} for {direction}: not a dict")
                    if "indicator" not in cond:
                        raise ValueError(f"Condition #{idx} for {direction} missing 'indicator': {cond}")
                    if "comparison" not in cond:
                        raise ValueError(f"Condition #{idx} for {direction} missing 'comparison': {cond}")

                    comparison = cond["comparison"]
                    if comparison in [">", "<", ">=", "<=", "=="]:
                        if "value" not in cond or cond["value"] is None:
                            raise ValueError(
                                f"Condition #{idx} for {direction} with comparison '{comparison}' missing 'value': {cond}"
                            )
                        if not isinstance(cond["value"], (int, float)):
                            raise ValueError(
                                f"Condition #{idx} for {direction} 'value' must be numeric, got {type(cond['value'])}: {cond}"
                            )
                    elif comparison in ["crosses_above", "crosses_below"]:
                        if "period" not in cond or "ref_period" not in cond:
                            raise ValueError(
                                f"Crossover condition #{idx} for {direction} requires 'period' and 'ref_period': {cond}"
                            )

        # Validate exit rules
        exit_rules = config.get("exit_rules", {})
        if "stop_loss" not in exit_rules:
            raise ValueError("exit_rules missing 'stop_loss' definition")
        if "take_profit" not in exit_rules:
            raise ValueError("exit_rules missing 'take_profit' definition")

    def _load_config(self, path: str) -> dict[str, Any]:
        with open(path, 'r', encoding='utf-8') as f:
            cfg = json.load(f)
        self._validate_config(cfg)
        return cfg

    def reload_config(self):
        """Hot-reload strategy without restarting bot."""
        self.config = self._load_config(self.config_path)

    def evaluate_entry(
        self,
        df: pd.DataFrame,
        direction: str,
        *,
        analyzed_df: pd.DataFrame | None = None,
    ) -> tuple[bool, dict[str, Any]]:
        """
        Evaluate if entry conditions are met for given direction (long/short).
        Returns (should_enter, metadata).
        """
        if df.empty or len(df) < 5:
            _console(f"Strategy evaluation for {direction}: insufficient data (len={len(df)})", level=logging.INFO)
            return False, {"reason": "insufficient_data"}

        enriched_df = analyzed_df
        if enriched_df is None:
            enriched_df = self.ta.analyze_instrument(df, self.config)
        signal = self.ta.get_latest_signal(enriched_df)

        if not signal:
            _console(f"Strategy evaluation for {direction}: no signal generated", level=logging.INFO)
            return False, {"reason": "no_signal"}

        rules = self.config["entry_rules"].get(direction, {})
        conditions = rules.get("conditions", [])
        if not conditions:
            _console(
                f"Strategy evaluation for {direction}: no entry conditions configured",
                level=logging.WARNING,
            )
            return False, {"reason": "no_conditions"}
        require_all = rules.get("require_all", True)

        condition_results = []
        for cond in conditions:
            result, actual_val, expected_val = self._check_condition(cond, signal, enriched_df, direction)
            condition_results.append({
                "indicator": cond.get("indicator"),
                "comparison": cond.get("comparison"),
                "expected": expected_val,
                "actual": actual_val,
                "result": result
            })

        met_conditions = [c for c in condition_results if c["result"]]
        total_conditions = len(conditions)
        should_enter = len(met_conditions) == total_conditions if require_all else len(met_conditions) > 0

        # A12 logging
        _console(
            f"Strategy evaluation for {direction}: {len(met_conditions)}/{total_conditions} conditions met, "
            f"entry={should_enter}"
        )

        atr_period = self.config.get("exit_rules", {}).get("stop_loss", {}).get("atr_period", 14)
        atr_val = signal.get(f"atr_{atr_period}", signal.get("atr_14"))

        metadata = {
            "signal": signal,
            "conditions": condition_results,
            "all_met": should_enter,
            "strategy": self.config["name"],
            "atr": atr_val,
            "enriched_df": enriched_df,
        }

        return should_enter, metadata

    def _check_condition(
            self,
            cond: dict[str, Any],
            signal: dict[str, Any],
            df: pd.DataFrame,
            direction: str = ""
    ) -> tuple[bool, Any, Any]:
        """
        Evaluate a single condition from strategy.json.
        Returns (passed, actual_value, expected_value).
        """
        indicator = str(cond["indicator"]).upper()
        period = cond.get("period", 14)
        comparison = cond["comparison"]

        if indicator == "SMA" and comparison in ["crosses_above", "crosses_below"]:
            fast_period = cond.get("period", 9)
            slow_period = cond.get("ref_period", 21)
            fast_col = f"SMA_{fast_period}"
            slow_col = f"SMA_{slow_period}"
            crossover_df = self.ta.detect_crossover(df, fast_col, slow_col)
            latest = crossover_df.iloc[-1]
            if comparison == "crosses_above":
                passed = bool(latest.get("crossover_up", False))
                actual = f"{fast_col}={latest.get(fast_col):.5f}, {slow_col}={latest.get(slow_col):.5f}, up={passed}"
                expected = f"{fast_col} crosses above {slow_col}"
            else:
                passed = bool(latest.get("crossover_down", False))
                actual = f"{fast_col}={latest.get(fast_col):.5f}, {slow_col}={latest.get(slow_col):.5f}, down={passed}"
                expected = f"{fast_col} crosses below {slow_col}"

            # A11 logging
            _console(
                f"Evaluating condition [{direction}]: {indicator} {comparison} "
                f"(fast={fast_period}, slow={slow_period}) -> actual=[{actual}], pass={passed}"
            )
            return passed, actual, expected

        # Key in signal dictionary (e.g. rsi_14, adx_14, sma_9)
        key = f"{indicator.lower()}_{period}"
        value = signal.get(key)
        if value is None or pd.isna(value):
            _console(
                f"Evaluating condition [{direction}]: {indicator}_{period} {comparison} threshold={cond.get('value')} "
                f"-> actual=None/NaN, pass=False"
            )
            return False, value, cond.get("value")

        ref_value = cond.get("value")
        if ref_value is None:
            return False, value, None

        if comparison == ">":
            passed = bool(value > ref_value)
        elif comparison == "<":
            passed = bool(value < ref_value)
        elif comparison == ">=":
            passed = bool(value >= ref_value)
        elif comparison == "<=":
            passed = bool(value <= ref_value)
        elif comparison == "==":
            passed = bool(value == ref_value)
        else:
            passed = False

        # A11 logging
        val_str = f"{value:.2f}" if isinstance(value, float) else str(value)
        _console(
            f"Evaluating condition [{direction}]: {indicator}_{period} {comparison} threshold={ref_value} "
            f"-> actual={val_str}, pass={passed}"
        )
        return passed, value, ref_value

    def calculate_stop_loss(
            self,
            entry_price: float,
            direction: str,
            df: pd.DataFrame | None = None,
            atr: float | None = None
    ) -> float:
        """Calculate stop loss based on strategy.json exit rules."""
        sl_config = self.config["exit_rules"]["stop_loss"]

        if sl_config["type"] == "ATR_MULTIPLE":
            atr_val = atr
            if atr_val is None or pd.isna(atr_val) or atr_val <= 0:
                atr_period = sl_config.get("atr_period", 14)
                if df is not None and not df.empty:
                    # Check if column already present or compute it cleanly
                    if f"ATR_{atr_period}" in df.columns:
                        val = df.iloc[-1].get(f"ATR_{atr_period}")
                        if val is not None and not pd.isna(val) and val > 0:
                            atr_val = float(val)
                    if atr_val is None:
                        enriched = self.ta.add_atr(df, atr_period)
                        val = enriched.iloc[-1].get(f"ATR_{atr_period}")
                        if val is not None and not pd.isna(val) and val > 0:
                            atr_val = float(val)

            if atr_val is None or pd.isna(atr_val) or atr_val <= 0:
                atr_val = entry_price * 0.001

            distance = atr_val * sl_config["atr_multiplier"]

            if direction == "long":
                return entry_price - distance
            else:
                return entry_price + distance

        return entry_price * 0.99 if direction == "long" else entry_price * 1.01

    def calculate_take_profit(
            self,
            entry_price: float,
            stop_loss: float,
            direction: str
    ) -> float:
        """Calculate take profit based on risk:reward ratio."""
        tp_config = self.config["exit_rules"]["take_profit"]
        risk = abs(entry_price - stop_loss)

        if tp_config["type"] == "RISK_REWARD":
            reward = risk * tp_config["risk_reward_ratio"]
            if direction == "long":
                return entry_price + reward
            else:
                return entry_price - reward

        return entry_price * 1.02 if direction == "long" else entry_price * 0.98

    # ------------------------------------------------------------------
    # Higher-timeframe trend filter, management and guard settings
    # ------------------------------------------------------------------
    _TREND_DEFAULTS: ClassVar[dict[str, Any]] = {
        "enabled": True,
        "timeframe": "H1",
        "candle_count": 150,
        "ema_fast": 20,
        "ema_slow": 50,
        "slope_bars": 5,
        "adx_period": 14,
        "min_adx": 20,
        "drift_lookback_bars": 12,
        "max_counter_drift_atr": 1.0,
    }
    _MANAGEMENT_DEFAULTS: ClassVar[dict[str, Any]] = {
        "enabled": True,
        "breakeven_r": 1.0,
        "breakeven_buffer_r": 0.1,
        "min_stop_step_r": 0.1,
        "min_stop_distance_pips": 1.0,
        "exit_on_opposite_cross_in_profit": True,
        "min_profit_r_for_signal_exit": 0.2,
        "exit_on_htf_flip": True,
        "late_day_lock_minutes": 120,
        "late_day_lock_min_r": 0.3,
        "late_day_lock_fraction": 0.5,
        "late_day_close_minutes": 45,
        "late_day_close_min_r": 0.5,
        "management_interval_seconds": 10,
        "software_stops": True,
        "giveback_enabled": True,
        "giveback_arm_r": 0.8,
        "giveback_fraction": 0.5,
        "reversal_exit_enabled": True,
        "reversal_min_profit_r": 0.3,
        "reversal_votes_required": 2,
        "reversal_body_atr": 1.0,
        "tp_adjust_enabled": True,
        "tp_adjust_min_peak_r": 1.0,
        "tp_fade_bars": 3,
        "tp_near_atr": 1.0,
    }
    _GUARD_DEFAULTS: ClassVar[dict[str, Any]] = {
        "max_signal_age_minutes": 4,
        "max_chase_atr": 0.5,
        "cooldown_minutes_after_loss": 90,
        "max_trades_per_symbol_per_day": 2,
        "max_consecutive_losses": 3,
        "max_currency_net_exposure": 3,
        "no_new_entries_minutes_before_flatten": 120,
        "correlation_guard": {
            "enabled": True,
            "timeframe": "M15",
            "lookback_bars": 200,
            "min_overlap_bars": 60,
            "max_effective_corr": 0.65,
            "max_correlated_positions": 3,
            "exclude_risk_free": True,
            "cache_minutes": 15,
        },
    }

    def get_trend_filter_config(self) -> dict[str, Any]:
        cfg = dict(self._TREND_DEFAULTS)
        cfg.update(self.config.get("filters", {}).get("trend_filter", {}))
        return cfg

    def get_management_config(self) -> dict[str, Any]:
        cfg = dict(self._MANAGEMENT_DEFAULTS)
        cfg.update(self.config.get("trade_management", {}))
        # Keep the legacy trailing_stop block meaningful: it now drives the trail.
        trail = self.config.get("exit_rules", {}).get("trailing_stop", {})
        cfg["trail_enabled"] = bool(trail.get("enabled", True))
        cfg["trail_start_r"] = float(trail.get("activation_r_multiple", 1.5))
        cfg["trail_atr_mult"] = float(trail.get("trail_atr_multiplier", 1.5))
        return cfg

    def get_guard_config(self) -> dict[str, Any]:
        cfg = dict(self._GUARD_DEFAULTS)
        user = self.config.get("entry_guards", {})
        cfg.update({k: v for k, v in user.items() if k != "correlation_guard"})
        corr = dict(self._GUARD_DEFAULTS["correlation_guard"])
        corr.update(user.get("correlation_guard", {}))
        cfg["correlation_guard"] = corr
        return cfg

    def compute_htf_bias(self, htf_df: pd.DataFrame) -> dict[str, Any]:
        """Bias of the higher timeframe ("long" / "short" / "neutral")."""
        cfg = self.get_trend_filter_config()
        return self.ta.compute_trend_bias(
            htf_df,
            ema_fast=int(cfg["ema_fast"]),
            ema_slow=int(cfg["ema_slow"]),
            slope_bars=int(cfg["slope_bars"]),
            adx_period=int(cfg["adx_period"]),
            min_adx=float(cfg["min_adx"]),
            drift_lookback_bars=int(cfg["drift_lookback_bars"]),
        )

    def direction_allowed_by_trend(
        self, direction: str, bias: dict[str, Any]
    ) -> tuple[bool, str]:
        """
        Gate an M15 entry with the higher-timeframe bias.
        Fail-closed: no usable bias -> no trade.
        """
        cfg = self.get_trend_filter_config()
        if not cfg.get("enabled", True):
            return True, "trend_filter_disabled"

        htf_bias = bias.get("bias", "neutral")
        if htf_bias != direction:
            return False, f"htf_bias={htf_bias} ({bias.get('reason', '')}) vs {direction}"

        drift = float(bias.get("drift_atr", 0.0))
        limit = float(cfg["max_counter_drift_atr"])
        if direction == "long" and drift < -limit:
            return False, f"day drift {drift:+.2f} ATR is against a long"
        if direction == "short" and drift > limit:
            return False, f"day drift {drift:+.2f} ATR is against a short"
        return True, "aligned"

    _ENTRY_MODE_DEFAULTS: ClassVar[dict[str, Any]] = {
        "pullback": {
            "enabled": True,
            "lookback_bars": 3,
            "tolerance_atr": 0.25,
            "min_adx": 20,
            "rsi_min": 45,
            "rsi_max": 85,
            "min_gap_bars": 4,
        }
    }

    def get_pullback_config(self) -> dict[str, Any]:
        cfg = dict(self._ENTRY_MODE_DEFAULTS["pullback"])
        cfg.update(self.config.get("entry_modes", {}).get("pullback", {}))
        return cfg

    def evaluate_pullback(
        self, analyzed_df: pd.DataFrame, direction: str
    ) -> tuple[bool, dict[str, Any]]:
        """Trend-continuation entry (see TechnicalAnalyzer.detect_pullback)."""
        cfg = self.get_pullback_config()
        if not cfg.get("enabled", True):
            return False, {"reason": "pullback_disabled"}
        result = self.ta.detect_pullback(
            analyzed_df,
            direction,
            lookback_bars=int(cfg["lookback_bars"]),
            tolerance_atr=float(cfg["tolerance_atr"]),
            min_adx=float(cfg["min_adx"]),
            rsi_min=float(cfg["rsi_min"]),
            rsi_max=float(cfg["rsi_max"]),
        )
        signal = self.ta.get_latest_signal(analyzed_df)
        atr_period = self.config.get("exit_rules", {}).get("stop_loss", {}).get("atr_period", 14)
        metadata = {
            "signal": signal,
            "conditions": [
                {"indicator": name, "comparison": "pullback", "expected": desc, "actual": ok, "result": ok}
                for name, ok, desc in result["checks"]
            ],
            "all_met": result["matched"],
            "strategy": self.config["name"],
            "mode": "pullback",
            "atr": signal.get(f"atr_{atr_period}", signal.get("atr_14")),
            "enriched_df": analyzed_df,
        }
        return bool(result["matched"]), metadata

    def get_instruments(self) -> list[str]:
        return [item.replace("_", "").upper() for item in self.config.get("instruments", [])]


# Singleton
strategy_engine = StrategyEngine()