"""
Market session detection and day-trade enforcement.
Ensures no overnight positions and trades only during active sessions.
"""
from datetime import datetime, time, timedelta
from typing import Any

import pytz

from config.settings import settings


class MarketHours:
    """
    Forex market hours (US/Eastern):
    - Sydney: 5 PM - 2 AM EST
    - Tokyo: 7 PM - 4 AM EST
    - London: 3 AM - 12 PM EST
    - New York: 8 AM - 5 PM EST

    Forex weekly session: Sunday 5:00 PM through Friday 5:00 PM.
    Positions are still flattened daily at 3:30 PM, with new entries cut off
    45 minutes before that time.
    """

    def __init__(self):
        self.est = pytz.timezone("US/Eastern")
        self.utc = pytz.utc

    def now_est(self) -> datetime:
        return datetime.now(self.est)

    def is_market_open(self) -> bool:
        """Check whether the standard Sunday-evening to Friday-evening session is open."""
        now = self.now_est()
        weekday = now.weekday()
        current_time = now.time()
        if weekday == 6:
            return current_time >= time(settings.FOREX_WEEK_OPEN_HOUR, 0)
        if weekday == 4:
            return current_time < time(settings.FOREX_WEEK_CLOSE_HOUR, 0)
        return weekday < 4

    def is_close_time(self) -> bool:
        """Check whether weekday positions should be flattened for the daily close."""
        now = self.now_est()
        if now.weekday() >= 5:
            return False
        market_close = self.est.localize(datetime.combine(
            now.date(), time(settings.MARKET_CLOSE_HOUR, 0)
        ))
        close_start = market_close - timedelta(minutes=settings.CLOSE_BUFFER_MINUTES)
        return now >= close_start

    def time_until_new_trade_cutoff(self) -> timedelta:
        """Get time remaining until the next weekday new-trade cutoff."""
        now = self.now_est()
        cutoff_hour = time(settings.MARKET_CLOSE_HOUR, 0)
        cutoff_offset = timedelta(minutes=settings.CLOSE_BUFFER_MINUTES)

        if now.weekday() < 5:
            today_cutoff = self.est.localize(
                datetime.combine(now.date(), cutoff_hour)
            ) - cutoff_offset
            if now >= today_cutoff:
                return timedelta(0)
            return today_cutoff - now

        for days_ahead in range(1, 8):
            cutoff_date = now.date() + timedelta(days=days_ahead)
            if cutoff_date.weekday() < 5:
                cutoff = self.est.localize(
                    datetime.combine(cutoff_date, cutoff_hour)
                ) - cutoff_offset
                return cutoff - now

        raise RuntimeError("Could not find the next weekday trade cutoff")

    def time_until_close(self) -> timedelta:
        """Get time remaining until forced close (alias for time_until_new_trade_cutoff)."""
        return self.time_until_new_trade_cutoff()

    def can_open_new_trade(self) -> dict[str, Any]:
        """
        Comprehensive check before opening new trades.
        Returns dict with allowed flag and reason.
        """
        # Check the weekly forex session.
        if not self.is_market_open():
            return {
                "allowed": False,
                "reason": (
                    "Forex market closed. Weekly session: Sunday "
                    f"{settings.FOREX_WEEK_OPEN_HOUR}:00-Friday "
                    f"{settings.FOREX_WEEK_CLOSE_HOUR}:00 US/Eastern"
                ),
            }

        # Preserve the bot's weekday day-trading cutoff.
        time_remaining = self.time_until_new_trade_cutoff()
        if time_remaining < timedelta(minutes=45):
            return {
                "allowed": False,
                "reason": f"Too close to market close. {time_remaining} remaining. No new trades."
            }

        return {
            "allowed": True,
            "reason": "OK",
            "time_until_close_minutes": time_remaining.total_seconds() / 60
        }

    def get_session(self) -> str:
        """Identify current forex session."""
        now = self.now_est()
        hour = now.hour

        if 8 <= hour < 12:
            return "NY-London Overlap (High Volatility)"
        elif 12 <= hour < 17:
            return "New York Session"
        elif 3 <= hour < 8:
            return "London Session"
        elif 19 <= hour < 24 or 0 <= hour < 4:
            return "Asia Session"
        else:
            return "Low Liquidity"


market_hours = MarketHours()