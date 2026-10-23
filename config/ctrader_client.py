"""Async broker adapter around cTrader's Open API (ctrader-api-client)."""

from __future__ import annotations

import asyncio
import logging
import math
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from time import monotonic
from typing import Any

import pandas as pd
from ctrader_api_client import (
    AccountCredentials,
    AmendPositionRequest,
    ClientConfig,
    ClosePositionRequest,
    CTraderClient,
    ExecutionEvent,
    NewOrderRequest,
    OrderSide,
    OrderType,
    ReadyEvent,
    ReconnectedEvent,
    SpotEvent,
    SubscriptionRestoreFailedEvent,
    TokenRefreshFailedEvent,
    TrendbarPeriod,
)
from ctrader_api_client.enums import ExecutionType

from config.ctrader_token_store import SQLiteTokenStore
from config.settings import settings

logger = logging.getLogger(__name__)


def _console(message: str) -> None:
    print(f"[tradebot] {message}", flush=True)


# Label every order/position this bot places, so get_bot_positions() can
# filter them out from anything placed manually or by another application —
# the equivalent of MT5's numeric "magic number".
BOT_LABEL = "tradebot"

TIMEFRAMES = {
    "M1": TrendbarPeriod.M1,
    "M2": TrendbarPeriod.M2,
    "M3": TrendbarPeriod.M3,
    "M4": TrendbarPeriod.M4,
    "M5": TrendbarPeriod.M5,
    "M10": TrendbarPeriod.M10,
    "M15": TrendbarPeriod.M15,
    "M30": TrendbarPeriod.M30,
    "H1": TrendbarPeriod.H1,
    "H4": TrendbarPeriod.H4,
    "H12": TrendbarPeriod.H12,
    "D1": TrendbarPeriod.D1,
}

_TIMEFRAME_MINUTES = {
    "M1": 1,
    "M2": 2,
    "M3": 3,
    "M4": 4,
    "M5": 5,
    "M10": 10,
    "M15": 15,
    "M30": 30,
    "H1": 60,
    "H4": 240,
    "H12": 720,
    "D1": 1440,
}

_FILLED_TYPES = (
    ExecutionType.ORDER_PARTIAL_FILL,
    ExecutionType.ORDER_FILLED,
)
_TRENDBAR_REQUEST_INTERVAL_SECONDS = 0.5
_AMEND_SUCCESS_TYPES = (
    ExecutionType.ORDER_ACCEPTED,
    ExecutionType.ORDER_REPLACED,
)


def normalize_symbol(symbol: str) -> str:
    """Return the broker symbol-name format used by this application."""
    return symbol.replace("_", "").upper()


@dataclass(frozen=True, slots=True)
class _SpotQuote:
    account_id: int
    symbol_id: int
    bid: Decimal | None
    ask: Decimal | None
    timestamp: datetime
    bid_timestamp: datetime | None
    ask_timestamp: datetime | None


class OrderResult:
    """Expose cTrader execution identifiers and fill details."""

    def __init__(self, execution: ExecutionEvent, ok: bool):
        self.order_id = execution.order_id
        self.position_id = execution.position_id
        self.symbol_id = execution.symbol_id
        self.fill_price = (
            float(execution.fill_price) if execution.fill_price is not None else 0.0
        )
        self.retcode = 0 if ok else 1
        self.comment = execution.execution_type.value
        self.raw = execution


class CTraderBrokerClient:
    """Thin adapter over ctrader_api_client.CTraderClient.

    Enter once at application startup (`await connect()`), keep it open for
    the life of the process, and call `await disconnect()` at shutdown. Do
    not create a new instance per request.
    """

    def __init__(self) -> None:
        self._client: CTraderClient | None = None
        self._account_id: int | None = None
        self._symbol_name_to_id: dict[str, int] = {}
        self._symbol_id_to_name: dict[int, str] = {}
        self._symbol_cache: dict[int, Any] = {}
        self._last_spot: dict[int, _SpotQuote] = {}
        self._trendbar_request_lock = asyncio.Lock()
        self._last_trendbar_request: float | None = None
        self._execution_waiters: dict[int, asyncio.Future[ExecutionEvent]] = {}
        self._recent_fills: dict[int, ExecutionEvent] = {}
        self._token_store = SQLiteTokenStore(settings.DB_PATH)

    # ---- lifecycle ------------------------------------------------------

    async def connect(self) -> None:
        if self._client is not None:
            return
        if not settings.CTRADER_ACCOUNT_ID:
            raise RuntimeError(
                "CTRADER_ACCOUNT_ID is not set. Run scripts/ctrader_oauth_setup.py "
                "once to authorize an account and discover it, then add it to .env."
            )

        config = ClientConfig(
            client_id=settings.CTRADER_CLIENT_ID,
            client_secret=settings.CTRADER_CLIENT_SECRET,
            host=settings.CTRADER_HOST,
            port=settings.CTRADER_PORT,
            use_ssl=True,
        )
        client = CTraderClient(config, token_store=self._token_store)
        self._client = client
        self._register_handlers()

        await client.__aenter__()
        _console(
            f"cTrader: connected to {settings.CTRADER_HOST}:{settings.CTRADER_PORT}, app authenticated"
        )

        creds = await self._token_store.load(settings.CTRADER_ACCOUNT_ID)
        if creds is None:
            if not (
                settings.CTRADER_ACCESS_TOKEN
                and settings.CTRADER_REFRESH_TOKEN
                and settings.CTRADER_TOKEN_EXPIRES_AT
            ):
                raise RuntimeError(
                    "No tokens in the local store and none provided in settings. "
                    "Run scripts/ctrader_oauth_setup.py first."
                )
            creds = AccountCredentials(
                account_id=settings.CTRADER_ACCOUNT_ID,
                access_token=settings.CTRADER_ACCESS_TOKEN,
                refresh_token=settings.CTRADER_REFRESH_TOKEN,
                expires_at=settings.CTRADER_TOKEN_EXPIRES_AT,
            )
            _console("cTrader: seeding token store from settings (first run only)")

        self._account_id = creds.account_id
        await client.auth.authenticate_trader(creds)
        _console(f"cTrader: account {creds.account_id} authenticated")
        await self._load_symbols()

    def _register_handlers(self) -> None:
        client = self._client

        @client.on(SpotEvent)
        async def _on_spot(event: SpotEvent) -> None:
            self._cache_spot_event(event)

        @client.on(ExecutionEvent)
        async def _on_execution(event: ExecutionEvent) -> None:
            await self._handle_execution_event(event)

        @client.on(ReadyEvent)
        async def _on_ready(event: ReadyEvent) -> None:
            _console(
                f"cTrader: account {event.account_id} ready (is_reconnect={event.is_reconnect})"
            )
            if event.is_reconnect:
                # Positions/fills that happened while disconnected produced
                # no events we ever saw -- reconcile by re-fetching.
                try:
                    positions = await self.get_open_positions()
                    _console(
                        f"cTrader: reconciled {len(positions)} open position(s) after reconnect"
                    )
                except Exception as exc:  # pragma: no cover - defensive
                    _console(f"cTrader: reconciliation after reconnect FAILED: {exc!r}")

        @client.on(ReconnectedEvent)
        async def _on_reconnected(event: ReconnectedEvent) -> None:
            _console(
                f"cTrader: reconnected. restored={event.restored_accounts} failed={event.failed_accounts}"
            )

        @client.on(TokenRefreshFailedEvent)
        async def _on_refresh_failed(event: TokenRefreshFailedEvent) -> None:
            _console(
                f"cTrader: TOKEN REFRESH FAILED for account {event.account_id}: {event.error}. "
                "Retried automatically; if this repeats, re-authorize via scripts/ctrader_oauth_setup.py."
            )

        @client.on(SubscriptionRestoreFailedEvent)
        async def _on_restore_failed(event: SubscriptionRestoreFailedEvent) -> None:
            _console(
                f"cTrader: market data subscription restore FAILED for account "
                f"{event.account_id}: {event.error}. No prices for affected symbols until the next reconnect."
            )

    def _cache_spot_event(self, event: SpotEvent) -> None:
        previous = self._last_spot.get(event.symbol_id)
        bid_is_newer = event.bid is not None and (
            previous is None
            or previous.bid_timestamp is None
            or event.timestamp >= previous.bid_timestamp
        )
        ask_is_newer = event.ask is not None and (
            previous is None
            or previous.ask_timestamp is None
            or event.timestamp >= previous.ask_timestamp
        )
        bid = previous.bid if previous else None
        ask = previous.ask if previous else None
        bid_timestamp = previous.bid_timestamp if previous else None
        ask_timestamp = previous.ask_timestamp if previous else None
        if bid_is_newer:
            bid = event.bid
            bid_timestamp = event.timestamp
        if ask_is_newer:
            ask = event.ask
            ask_timestamp = event.timestamp
        self._last_spot[event.symbol_id] = _SpotQuote(
            account_id=event.account_id,
            symbol_id=event.symbol_id,
            bid=bid,
            ask=ask,
            timestamp=event.timestamp,
            bid_timestamp=bid_timestamp,
            ask_timestamp=ask_timestamp,
        )

    async def _handle_execution_event(self, event: ExecutionEvent) -> None:
        if (
            not event.is_server_event
            or event.order_id <= 0
            or event.execution_type not in _FILLED_TYPES
        ):
            return
        self._recent_fills[event.order_id] = event
        while len(self._recent_fills) > 100:
            self._recent_fills.pop(next(iter(self._recent_fills)))
        waiter = self._execution_waiters.get(event.order_id)
        if waiter is not None and not waiter.done():
            waiter.set_result(event)

    async def disconnect(self) -> None:
        if self._client is None:
            return
        await self._client.__aexit__(None, None, None)
        self._client = None
        self._last_spot.clear()
        _console("cTrader: disconnected")

    @property
    def is_connected(self) -> bool:
        return self._client is not None and self._client.is_connected

    # ---- symbols ----------------------------------------------------------

    async def _load_symbols(self) -> None:
        symbols = await self._client.symbols.list_all(self._account_id)
        self._symbol_name_to_id = {
            normalize_symbol(s.name): s.symbol_id for s in symbols
        }
        self._symbol_id_to_name = {
            s.symbol_id: normalize_symbol(s.name) for s in symbols
        }
        _console(f"cTrader: loaded {len(self._symbol_name_to_id)} symbols")

    async def resolve_symbol_id(self, symbol: str) -> int:
        name = normalize_symbol(symbol)
        if name not in self._symbol_name_to_id:
            await self._load_symbols()
        if name not in self._symbol_name_to_id:
            raise RuntimeError(
                f"cTrader: unknown symbol {name!r} (not in this account's symbol list)"
            )
        return self._symbol_name_to_id[name]

    def symbol_name(self, symbol_id: int) -> str:
        return self._symbol_id_to_name.get(symbol_id, str(symbol_id))

    async def get_symbol_info(self, symbol: str):
        symbol_id = await self.resolve_symbol_id(symbol)
        if symbol_id not in self._symbol_cache:
            self._symbol_cache[symbol_id] = await self._client.symbols.get_by_id(
                self._account_id, symbol_id
            )
        return self._symbol_cache[symbol_id]

    # ---- market data --------------------------------------------------

    async def ensure_subscribed(self, symbols: list[str]) -> None:
        ids = [await self.resolve_symbol_id(s) for s in symbols]
        await self._client.market_data.subscribe_spots(self._account_id, ids)
        _console(f"cTrader: subscribed to live prices for {len(ids)} symbol(s)")

    def get_tick(self, symbol: str) -> _SpotQuote:
        """Return the latest complete quote assembled from spot updates."""
        symbol_id = self._symbol_name_to_id.get(normalize_symbol(symbol))
        if symbol_id is None or symbol_id not in self._last_spot:
            raise RuntimeError(
                f"cTrader: no cached price for {symbol} -- is it subscribed?"
            )
        quote = self._last_spot[symbol_id]
        if quote.bid is None or quote.ask is None:
            missing = "bid" if quote.bid is None else "ask"
            raise RuntimeError(
                f"cTrader: waiting for a complete {symbol} quote; {missing} is not available yet"
            )
        now = datetime.now(UTC)
        side_timestamps = (quote.bid_timestamp, quote.ask_timestamp)
        age = max(
            (now - timestamp).total_seconds()
            for timestamp in side_timestamps
            if timestamp is not None
        )
        if age > 5:
            _console(f"cTrader: WARNING stale price for {symbol} ({age:.1f}s old)")
        return quote

    async def copy_rates_from_pos(
        self, symbol: str, timeframe: str, start_pos: int = 0, count: int = 200
    ) -> pd.DataFrame:
        """Mirrors the MT5Client method's output shape: a DataFrame
        indexed by time with lowercase open/high/low/close columns, oldest
        first. start_pos=1 (skip the still-forming bar) is emulated by
        fetching a wider window and trimming the most recent bar, since the
        Open API has no position-based paging the way MT5's copy_rates does."""
        tf = timeframe.upper()
        period = TIMEFRAMES.get(tf)
        if period is None:
            raise ValueError(f"Unsupported timeframe: {timeframe}")
        symbol_id = await self.resolve_symbol_id(symbol)
        minutes_per_bar = _TIMEFRAME_MINUTES[tf]
        span = timedelta(
            minutes=minutes_per_bar * (count + start_pos) * 3
        )  # padded for weekends/holidays
        to_ts = datetime.now(UTC)
        from_ts = to_ts - span
        async with self._trendbar_request_lock:
            if self._last_trendbar_request is not None:
                wait = _TRENDBAR_REQUEST_INTERVAL_SECONDS - (
                    monotonic() - self._last_trendbar_request
                )
                if wait > 0:
                    await asyncio.sleep(wait)
            self._last_trendbar_request = monotonic()
            bars = await self._client.market_data.get_trendbars(
                self._account_id, symbol_id, period, from_ts, to_ts
            )
        if not bars:
            return pd.DataFrame()
        if start_pos:
            bars = bars[:-start_pos] if start_pos < len(bars) else []
        bars = bars[-count:]
        if not bars:
            return pd.DataFrame()
        frame = pd.DataFrame(
            {
                "time": [b.timestamp for b in bars],
                "open": [float(b.open) for b in bars],
                "high": [float(b.high) for b in bars],
                "low": [float(b.low) for b in bars],
                "close": [float(b.close) for b in bars],
                "tick_volume": [b.volume for b in bars],
            }
        )
        return frame.set_index("time")

    # ---- account / positions -----------------------------------------

    async def get_account_info(self):
        return await self._client.accounts.get_trader(self._account_id)

    async def get_open_positions(self):
        return await self._client.trading.get_open_positions(self._account_id)

    async def get_bot_positions(self):
        """Positions placed by this application (label-tagged)."""
        positions = await self.get_open_positions()
        return [p for p in positions if p.label == BOT_LABEL]

    async def get_closed_position_summary(
        self, position_id: int
    ) -> tuple[float, float]:
        """Return the volume-weighted close price and realized gross PnL."""
        deals = await self._client.trading.get_deals_by_position_id(
            self._account_id, position_id
        )
        closing_deals = [deal for deal in deals if deal.close_detail is not None]
        closed_volume = sum(deal.close_detail.closed_volume for deal in closing_deals)
        if not closing_deals or closed_volume <= 0:
            raise RuntimeError(
                f"cTrader: no closing deals available yet for position {position_id}"
            )
        exit_price = (
            sum(
                float(deal.execution_price) * deal.close_detail.closed_volume
                for deal in closing_deals
            )
            / closed_volume
        )
        realized_pnl = sum(
            float(
                deal.close_detail.gross_profit
                + deal.close_detail.swap
                + deal.close_detail.commission
                - deal.close_detail.pnl_conversion_fee
            )
            for deal in closing_deals
        )
        return exit_price, realized_pnl

    async def get_deals(self, from_timestamp: datetime, to_timestamp: datetime):
        """Return broker deals executed in the requested time range."""
        return await self._client.trading.get_deals(
            self._account_id, from_timestamp, to_timestamp
        )

    async def get_orders(self, from_timestamp: datetime, to_timestamp: datetime):
        """Return broker orders executed in the requested time range."""
        return await self._client.trading.get_orders(
            self._account_id,
            from_timestamp=from_timestamp,
            to_timestamp=to_timestamp,
        )

    async def get_unrealized_pnl(self) -> dict[int, float]:
        """Return broker-reported unrealized PnL keyed by position ID."""
        positions = await self._client.trading.get_unrealized_pnl_per_position(
            self._account_id
        )
        return {
            position.position_id: float(position.net_unrealized_pnl)
            for position in positions
        }

    # ---- sizing ---------------------------------------------------------

    async def lots_from_risk(
        self, symbol: str, risk_amount: float, entry_price: float, stop_loss: float
    ) -> float:
        """Risk-based position size in lots, a too-small
        computed size is rejected with a clear log line instead of silently
        being floored to an invalid order.
        """
        symbol_info = await self.get_symbol_info(symbol)
        price_distance = abs(entry_price - stop_loss)
        if price_distance == 0:
            return 0.0
        contract_units = (
            float(symbol_info.lot_size) / 100
        )  # lot_size is in hundredths of a base unit
        risk_per_lot = price_distance * contract_units
        if risk_per_lot == 0:
            return 0.0
        raw_lots = risk_amount / risk_per_lot

        min_lots = float(symbol_info.volume_to_lots(symbol_info.min_volume))
        max_lots = float(symbol_info.volume_to_lots(symbol_info.max_volume))
        step_lots = float(symbol_info.volume_to_lots(symbol_info.step_volume)) or 0.01

        lots = math.floor(raw_lots / step_lots + 1e-9) * step_lots
        lots = max(0.0, min(lots, max_lots))
        if lots < min_lots:
            _console(
                f"{symbol}: computed size {raw_lots:.4f} lots rounds to {lots:.4f}, "
                f"below broker minimum {min_lots:.4f} -- refusing order"
            )
            return 0.0
        return lots

    # ---- trading --------------------------------------------------------

    async def place_market_order(
        self,
        symbol: str,
        direction: str,
        lots: float,
        stop_loss: float | None = None,
        take_profit: float | None = None,
        comment: str = "tradebot",
    ) -> OrderResult:
        symbol_id = await self.resolve_symbol_id(symbol)
        symbol_info = await self.get_symbol_info(symbol)
        is_buy = direction.lower() in {"long", "buy"}
        volume = symbol_info.lots_to_volume(Decimal(str(lots)))
        if volume < symbol_info.min_volume:
            raise ValueError(
                f"cTrader: volume {volume} for {symbol} is below the broker minimum "
                f"({symbol_info.min_volume}); refusing to send an invalid order"
            )

        request = NewOrderRequest(
            symbol_id=symbol_id,
            order_type=OrderType.MARKET,
            side=OrderSide.BUY if is_buy else OrderSide.SELL,
            volume=volume,
            stop_loss=symbol_info.quantize_price(Decimal(str(stop_loss)))
            if stop_loss
            else None,
            take_profit=symbol_info.quantize_price(Decimal(str(take_profit)))
            if take_profit
            else None,
            label=BOT_LABEL,
            comment=comment[:256],
        )
        _console(
            f"cTrader: placing order symbol={symbol} side={request.side.value} "
            f"volume={volume} ({lots} lots) sl={stop_loss} tp={take_profit}"
        )
        existing_position_ids = {
            int(position.position_id) for position in await self.get_bot_positions()
        }
        execution = await self._client.trading.place_order(self._account_id, request)
        if execution.execution_type in _FILLED_TYPES:
            filled = execution
        elif execution.execution_type == ExecutionType.ORDER_ACCEPTED:
            filled = await self._wait_for_fill(execution.order_id)
            if filled is None:
                positions = await self.get_bot_positions()
                if execution.position_id is not None:
                    candidates = [
                        position
                        for position in positions
                        if position.position_id == execution.position_id
                    ]
                else:
                    candidates = [
                        position
                        for position in positions
                        if int(position.position_id) not in existing_position_ids
                        and position.symbol_id == symbol_id
                        and position.side == request.side
                    ]
                if len(candidates) == 1:
                    position = candidates[0]
                    result = OrderResult(execution, True)
                    result.position_id = int(position.position_id)
                    result.fill_price = float(position.entry_price)
                    return result
            if filled is None:
                raise RuntimeError(
                    f"cTrader accepted market order {execution.order_id}, "
                    "but no fill confirmation arrived; verify the broker account "
                    "before retrying"
                )
        else:
            _console(
                f"cTrader: order failed symbol={symbol} type={execution.execution_type.value} "
                f"error={execution.error_code}"
            )
            raise RuntimeError(
                f"cTrader order rejected ({execution.execution_type.value}): {execution.error_code}"
            )
        _console(
            f"cTrader: order result order_id={filled.order_id} position_id={filled.position_id} "
            f"fill_price={filled.fill_price} type={filled.execution_type.value}"
        )
        return OrderResult(filled, True)

    async def _wait_for_fill(
        self, order_id: int, timeout: float = 10.0
    ) -> ExecutionEvent | None:
        """Wait for cTrader's server-pushed fill after an ORDER_ACCEPTED reply."""
        if order_id <= 0:
            return None
        recent = self._recent_fills.pop(order_id, None)
        if recent is not None:
            return recent
        waiter = asyncio.get_running_loop().create_future()
        self._execution_waiters[order_id] = waiter
        try:
            return await asyncio.wait_for(waiter, timeout=timeout)
        except TimeoutError:
            return None
        finally:
            self._execution_waiters.pop(order_id, None)

    async def amend_position_sltp(
        self, position_id: int, stop_loss: float | None, take_profit: float | None
    ) -> None:
        request = AmendPositionRequest(
            position_id=int(position_id),
            stop_loss=Decimal(str(stop_loss)) if stop_loss is not None else None,
            take_profit=Decimal(str(take_profit)) if take_profit is not None else None,
        )
        execution = await self._client.trading.amend_position(
            self._account_id, request
        )
        if execution.execution_type not in _AMEND_SUCCESS_TYPES:
            raise RuntimeError(
                f"cTrader SL/TP amendment rejected for position {position_id} "
                f"({execution.execution_type.value}): {execution.error_code}"
            )
        _console(
            f"cTrader: amended position_id={position_id} "
            f"SL={stop_loss} TP={take_profit} type={execution.execution_type.value}"
        )

    async def modify_take_profit_pips(
        self,
        position_id: int,
        pips: float,
        *,
        stop_loss: float | None = None,
    ) -> float:
        """Move a position's take profit by a pip distance from its current quote."""
        if not math.isfinite(pips) or pips <= 0:
            raise ValueError("Take-profit distance must be a positive finite pip value")
        positions = await self.get_open_positions()
        position = next(
            (p for p in positions if p.position_id == int(position_id)), None
        )
        if position is None:
            raise ValueError(f"cTrader: open position not found: {position_id}")

        symbol = self.symbol_name(position.symbol_id)
        symbol_info = await self.get_symbol_info(symbol)
        is_long = position.side == OrderSide.BUY
        sign = 1 if is_long else -1
        tick = self.get_tick(symbol)
        exit_quote = float(tick.bid if is_long else tick.ask)
        target = exit_quote + sign * pips / (10**symbol_info.pip_position)
        if position.take_profit is not None:
            current_tp = float(position.take_profit)
            target = min(target, current_tp) if is_long else max(target, current_tp)
        target_price = float(symbol_info.quantize_price(Decimal(str(target))))
        await self.amend_position_sltp(
            position_id,
            stop_loss=(
                stop_loss
                if stop_loss is not None
                else (
                    float(position.stop_loss)
                    if position.stop_loss is not None
                    else None
                )
            ),
            take_profit=target_price,
        )
        return target_price

    async def close_position(self, position_id: int) -> OrderResult:
        positions = await self.get_open_positions()
        match = next((p for p in positions if p.position_id == int(position_id)), None)
        if match is None:
            raise ValueError(f"cTrader: open position not found: {position_id}")
        request = ClosePositionRequest(
            position_id=match.position_id, volume=match.volume
        )
        execution = await self._client.trading.close_position(self._account_id, request)
        filled = execution
        if execution.execution_type == ExecutionType.ORDER_ACCEPTED:
            filled = await self._wait_for_fill(execution.order_id)
            if filled is None:
                remaining = await self.get_open_positions()
                if any(p.position_id == int(position_id) for p in remaining):
                    raise RuntimeError(
                        f"cTrader accepted close for position {position_id}, "
                        "but the position remains open and no fill confirmation arrived"
                    )
                filled = execution
        ok = filled.execution_type in _FILLED_TYPES or (
            execution.execution_type == ExecutionType.ORDER_ACCEPTED
        )
        _console(
            f"cTrader: close position_id={position_id} type={filled.execution_type.value} "
            f"fill_price={filled.fill_price}"
        )
        if not ok:
            raise RuntimeError(
                f"cTrader close rejected ({execution.execution_type.value}): {execution.error_code}"
            )
        return OrderResult(filled, ok)


ctrader_client = CTraderBrokerClient()
