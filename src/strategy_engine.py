"""
Strategy Engine reads strategy.json and evaluates entry/exit conditions
against live market data. Fully dynamic — edit JSON to change strategy.
"""
import json
import logging
import os
from typing import Any

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

    def evaluate_entry(self, df: pd.DataFrame, direction: str) -> tuple[bool, dict[str, Any]]:
        """
        Evaluate if entry conditions are met for given direction (long/short).
        Returns (should_enter, metadata).
        """
        if df.empty or len(df) < 5:
            _console(f"Strategy evaluation for {direction}: insufficient data (len={len(df)})", level=logging.INFO)
            return False, {"reason": "insufficient_data"}

        enriched_df = self.ta.analyze_instrument(df, self.config)
        signal = self.ta.get_latest_signal(enriched_df)

        if not signal:
            _console(f"Strategy evaluation for {direction}: no signal generated", level=logging.INFO)
            return False, {"reason": "no_signal"}

        rules = self.config["entry_rules"].get(direction, {})
        conditions = rules.get("conditions", [])
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
        indicator = cond["indicator"]
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

    def get_instruments(self) -> list[str]:
        return [item.replace("_", "").upper() for item in self.config.get("instruments", [])]


# Singleton
strategy_engine = StrategyEngine()