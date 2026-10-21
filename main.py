"""FastAPI entry point for cTraderBot."""

import asyncio
import logging
import sqlite3
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from pathlib import Path

from ctrader_api_client.exceptions import CTraderError
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from config.ctrader_client import ctrader_client, normalize_symbol
from config.settings import settings
from dashboard import dashboard_router
from src.database import init_database
from src.event_outbox import mongo_synchronizer
from src.logger import _console
from src.market_hours import market_hours
from src.risk_manager import Trade, risk_manager
from src.scanner import scanner
from src.strategy_engine import strategy_engine
from src.telegram_bot import telegram
from src.trading_cycle import trading_cycle


class TradeRequest(BaseModel):
    instrument: str
    direction: str = Field(pattern="^(BUY|SELL|LONG|SHORT)$")
    volume: float = Field(gt=0)
    stop_loss: float | None = None
    take_profit: float | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    host_time = datetime.now(UTC).isoformat()
    est_time = market_hours.now_est().strftime("%Y-%m-%d %H:%M:%S %Z")
    _console(
        f"cTraderBot starting on {settings.API_HOST}:{settings.API_PORT} | "
        f"Host local time: {host_time} | US/Eastern time: {est_time}",
        level=logging.INFO,
    )
    init_database()
    app.state.trading_task = None
    try:
        await ctrader_client.connect()
        await mongo_synchronizer.start()
        await telegram.initialize()
        await telegram.send_message_async(
            f"cTraderBot started on {settings.API_HOST}:{settings.API_PORT}"
        )
        yield
    finally:
        trading_cycle.stop()
        task = getattr(app.state, "trading_task", None)
        if task and not task.done():
            try:
                await asyncio.wait_for(task, timeout=5.0)
            except (TimeoutError, asyncio.CancelledError, Exception):
                pass
        await mongo_synchronizer.stop()
        await ctrader_client.disconnect()
        await telegram.send_message_async("cTraderBot stopped.")
        await telegram.shutdown()


app = FastAPI(title="cTraderBot API", version="3.0.0", lifespan=lifespan)
app.mount(
    "/assets",
    StaticFiles(directory=Path(__file__).resolve().parent / "assets"),
    name="assets",
)
app.include_router(dashboard_router)


@app.get("/", response_class=HTMLResponse)
async def root():
    return """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>cTraderBot</title>
        <style>
            :root {
                color-scheme: dark;
                font-family: "Segoe UI", Arial, sans-serif;
                background: #07111f;
                color: #e5eefb;
            }
            * { box-sizing: border-box; }
            body {
                margin: 0;
                min-height: 100vh;
                display: grid;
                place-items: center;
                background:
                    radial-gradient(circle at 15% 20%, #12375a 0, transparent 35%),
                    radial-gradient(circle at 85% 80%, #173c37 0, transparent 35%),
                    #07111f;
            }
            .card {
                width: min(920px, calc(100% - 32px));
                display: grid;
                grid-template-columns: 1fr 1fr;
                overflow: hidden;
                border: 1px solid #29415d;
                border-radius: 24px;
                background: rgba(14, 29, 48, 0.9);
                box-shadow: 0 24px 70px rgba(0, 0, 0, 0.35);
            }
            .content { padding: clamp(32px, 6vw, 72px); }
            .eyebrow {
                margin: 0 0 14px;
                color: #57d7ff;
                font-size: 0.8rem;
                font-weight: 700;
                letter-spacing: 0.16em;
                text-transform: uppercase;
            }
            h1 { margin: 0 0 18px; font-size: clamp(2.2rem, 5vw, 4rem); line-height: 1; }
            p { color: #a9bad0; line-height: 1.7; }
            .button {
                display: inline-block;
                margin-top: 18px;
                padding: 13px 22px;
                border-radius: 10px;
                background: #38bdf8;
                color: #062033;
                font-weight: 700;
                text-decoration: none;
                transition: transform 0.2s, background 0.2s;
            }
            .button:hover { transform: translateY(-2px); background: #7ddcff; }
            .actions { display: flex; flex-wrap: wrap; gap: 12px; }
            button.button {
                border: 0;
                cursor: pointer;
                font: inherit;
            }
            button.button:disabled { cursor: wait; opacity: 0.7; }
            #autotrading-status { min-height: 1.5em; margin-top: 12px; }
            .art { min-height: 360px; background: #0b1b2e; }
            .art img { width: 100%; height: 100%; object-fit: cover; display: block; }
            @media (max-width: 700px) {
                .card { grid-template-columns: 1fr; }
                .art { min-height: 240px; order: -1; }
            }
        </style>
    </head>
    <body>
        <main class="card">
            <section class="content">
                <p class="eyebrow">cTraderBot</p>
                <h1>Trade with clarity.</h1>
                <p>
                    Monitor market activity, review performance, and manage your
                    trading workflow from one focused dashboard.
                </p>
                <div class="actions">
                    <a class="button" href="/dashboard/">Open dashboard</a>
                    <button class="button" id="start-autotrading" type="button">
                        Start auto-trading
                    </button>
                </div>
                <p id="autotrading-status" role="status" aria-live="polite"></p>
            </section>
            <section class="art">
                <img src="/assets/locha%20eng.jpg" alt="cTraderBot">
            </section>
        </main>
        <script>
            const startButton = document.getElementById("start-autotrading");
            const statusMessage = document.getElementById("autotrading-status");

            startButton.addEventListener("click", async () => {
                startButton.disabled = true;
                statusMessage.textContent = "Starting auto-trading...";

                try {
                    const response = await fetch("/api/v1/auto/start", { method: "POST" });
                    const result = await response.json();
                    if (!response.ok) {
                        throw new Error(result.detail || "Unable to start auto-trading.");
                    }
                    statusMessage.textContent = result.status === "already_running"
                        ? "Auto-trading is already running."
                        : "Auto-trading started.";
                } catch (error) {
                    statusMessage.textContent = error.message;
                } finally {
                    startButton.disabled = false;
                }
            });
        </script>
    </body>
    </html>
    """


@app.get("/api/v1/health")
async def health_check():
    try:
        account = await ctrader_client.get_account_info()
        positions = await ctrader_client.get_open_positions()
        return {
            "status": "healthy",
            "ctrader_connected": ctrader_client.is_connected,
            "environment": settings.CTRADER_ENVIRONMENT,
            "balance": float(account.balance),
            "open_positions": len(positions),
            "timestamp": datetime.now(UTC).isoformat(),
        }
    except Exception as exc:
        return {
            "status": "unhealthy",
            "ctrader_connected": False,
            "error": str(exc),
            "timestamp": datetime.now(UTC).isoformat(),
        }


@app.get("/api/v1/account")
async def account_info():
    try:
        account = await ctrader_client.get_account_info()
        positions = await ctrader_client.get_open_positions()
        unrealized_pnl = await ctrader_client.get_unrealized_pnl()
        total_unrealized_pnl = sum(unrealized_pnl.values())
        return {
            "status": "success",
            "data": {
                "balance": float(account.balance),
                "equity": float(account.balance) + total_unrealized_pnl,
                "unrealized_pnl": total_unrealized_pnl,
                "deposit_asset_id": account.deposit_asset_id,
                "open_position_count": len(positions),
            },
        }
    except Exception as exc:
        raise HTTPException(status_code=503, detail=str(exc))


@app.get("/api/v1/positions")
async def get_positions():
    try:
        positions = await ctrader_client.get_open_positions()
        unrealized_pnl = await ctrader_client.get_unrealized_pnl()
        serialized_positions = []
        for position in positions:
            symbol = ctrader_client.symbol_name(position.symbol_id)
            symbol_info = await ctrader_client.get_symbol_info(symbol)
            serialized_positions.append(
                {
                    "position_id": position.position_id,
                    "symbol_id": position.symbol_id,
                    "symbol": symbol,
                    "side": position.side.value.lower(),
                    "volume": float(symbol_info.volume_to_lots(position.volume)),
                    "entry_price": float(position.entry_price),
                    "stop_loss": float(position.stop_loss)
                    if position.stop_loss is not None
                    else None,
                    "take_profit": float(position.take_profit)
                    if position.take_profit is not None
                    else None,
                    "unrealized_pnl": unrealized_pnl.get(position.position_id),
                }
            )
        return {
            "status": "success",
            "count": len(positions),
            "positions": serialized_positions,
        }
    except Exception as exc:
        raise HTTPException(status_code=503, detail=str(exc))


@app.post("/api/v1/scan")
async def run_scanner():
    try:
        results = await scanner.scan_all()
        return {"status": "success", "data": results}
    except Exception as exc:
        raise HTTPException(status_code=503, detail=str(exc))


@app.post("/api/v1/trade/manual")
async def manual_trade(request: TradeRequest):
    try:
        direction = request.direction.lower()
        symbol = normalize_symbol(request.instrument)
        result = await ctrader_client.place_market_order(
            symbol,
            direction,
            request.volume,
            request.stop_loss,
            request.take_profit,
            "manual",
        )
        if result.position_id is None:
            raise RuntimeError("cTrader order fill did not include a position ID")
        position_id = int(result.position_id)
        risk_manager.record_event(
            "manual_order_submitted",
            {
                "position_id": position_id,
                "instrument": symbol,
                "direction": direction,
                "volume": request.volume,
                "stop_loss": request.stop_loss,
                "take_profit": request.take_profit,
            },
            aggregate_id=position_id,
        )
        risk_manager.record_entry(
            Trade(
                trade_id=position_id,
                instrument=symbol,
                direction="long" if direction in {"buy", "long"} else "short",
                entry_price=result.fill_price,
                stop_loss=request.stop_loss or 0.0,
                take_profit=request.take_profit or 0.0,
                position_size=request.volume,
                entry_time=datetime.now(UTC),
            )
        )
        return {
            "status": "success",
            "position_id": position_id,
            "retcode": int(result.retcode),
        }
    except Exception as exc:
        raise HTTPException(status_code=503, detail=str(exc))


@app.post("/api/v1/trade/close/{position_id}")
async def close_trade(position_id: int):
    try:
        result = await ctrader_client.close_position(position_id)
        risk_manager.record_event(
            "manual_close_submitted",
            {"position_id": position_id, "retcode": int(result.retcode)},
            aggregate_id=position_id,
        )
        try:
            await trading_cycle._record_exit(position_id, "manual")
        except (CTraderError, RuntimeError, sqlite3.Error) as exc:
            _console(
                f"Position {position_id} closed; exit journal will be reconciled: {exc}",
                level=logging.WARNING,
            )
        return {
            "status": "success",
            "position_id": position_id,
            "retcode": int(result.retcode),
        }
    except Exception as exc:
        raise HTTPException(status_code=503, detail=str(exc))


@app.post("/api/v1/trade/close-all")
async def close_all_trades():
    await trading_cycle._close_all_positions("manual")
    return {"status": "success"}


@app.post("/api/v1/auto/start")
async def start_autotrading():
    if trading_cycle.running:
        return {"status": "already_running"}
    task = asyncio.create_task(trading_cycle.start(), name="autotrading-cycle")
    app.state.trading_task = task
    return {"status": "auto trading started"}


@app.post("/api/v1/auto/stop")
async def stop_autotrading():
    trading_cycle.stop()
    task = getattr(app.state, "trading_task", None)
    if task and not task.done():
        try:
            await asyncio.wait_for(task, timeout=5.0)
        except (TimeoutError, asyncio.CancelledError, Exception):
            pass
    return {"status": "auto trading stopped"}


@app.get("/api/v1/stats/r-multiple")
async def get_r_stats(days: int = 30):
    return risk_manager.get_r_multiple_stats(days)


@app.get("/api/v1/stats/daily")
async def get_daily_stats():
    return risk_manager.get_daily_summary()


@app.get("/api/v1/strategy/rules")
async def get_strategy_rules():
    return strategy_engine.config


@app.get("/api/v1/historical/{instrument}")
async def get_historical(instrument: str, timeframe: str = "M15", count: int = 100):
    if count < 1 or count > 5000:
        raise HTTPException(status_code=400, detail="count must be between 1 and 5000")
    try:
        df = await ctrader_client.copy_rates_from_pos(
            normalize_symbol(instrument), timeframe, 0, count
        )
        return {
            "status": "success",
            "instrument": normalize_symbol(instrument),
            "timeframe": timeframe,
            "count": len(df),
            "data": df.reset_index().to_dict("records"),
        }
    except Exception as exc:
        raise HTTPException(status_code=503, detail=str(exc))


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host=settings.API_HOST, port=settings.API_PORT)
