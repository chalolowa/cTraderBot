"""Live account and trade-performance dashboard routes."""

import html
from datetime import UTC, datetime, timedelta
from zoneinfo import ZoneInfo

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import HTMLResponse

from config.ctrader_client import ctrader_client
from config.settings import settings
from src.risk_manager import risk_manager
from src.strategy_engine import strategy_engine

router = APIRouter(prefix="/dashboard", tags=["dashboard"])
EASTERN = ZoneInfo("America/New_York")
ACCOUNT_AMOUNT_FORMATTER = (
    "Number(value || 0).toLocaleString(undefined, "
    "{ minimumFractionDigits: 2, maximumFractionDigits: 2 })"
)


@router.get("/", response_class=HTMLResponse)
async def dashboard_home():
    """Render the account dashboard; data is refreshed through the API."""
    strategy = strategy_engine.config
    strategy_name = html.escape(str(strategy["name"]))
    timeframe = html.escape(str(strategy.get("timeframe", "M15")))
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <meta name="color-scheme" content="dark">
  <title>cTraderBot | Account</title>
  <style>
    :root {{ font-family: "Segoe UI", Arial, sans-serif; color: #e5edf8; background: #09111f; }}
    * {{ box-sizing: border-box; }}
    body {{ margin: 0; }}
    main {{ width: min(1440px, 100% - 36px); margin: 0 auto; padding: 30px 0 48px; }}
    header {{ display: flex; justify-content: space-between; align-items: center; gap: 20px; margin-bottom: 24px; }}
    h1 {{ margin: 0; font-size: 26px; }} h2 {{ margin: 0; font-size: 17px; }}
    .muted {{ color: #8da0b8; }} .status {{ color: #fbbf24; font-size: 13px; }}
    .grid {{ display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 14px; }}
    .card, section {{ background: #111d2d; border: 1px solid #26364a; border-radius: 12px; }}
    .card {{ padding: 18px; min-height: 108px; }}
    .label {{ color: #91a3b9; font-size: 13px; }} .value {{ margin-top: 10px; font-size: 26px; font-weight: 650; }}
    .positive {{ color: #4ade80; }} .negative {{ color: #fb7185; }}
    section {{ padding: 18px; margin-top: 18px; overflow: hidden; }}
    .section-head {{ display: flex; justify-content: space-between; align-items: center; margin-bottom: 14px; gap: 12px; }}
    .table-wrap {{ overflow-x: auto; }}
    table {{ border-collapse: collapse; width: 100%; min-width: 760px; }}
    th, td {{ padding: 11px 10px; border-bottom: 1px solid #233247; text-align: left; white-space: nowrap; font-size: 13px; }}
    th {{ color: #93a6be; font-weight: 600; }} tbody tr:last-child td {{ border-bottom: 0; }}
    .empty {{ color: #91a3b9; padding: 20px 10px; text-align: center; }}
    .strategy {{ display: flex; gap: 28px; flex-wrap: wrap; color: #aab9cb; font-size: 13px; }}
    .error {{ color: #fda4af; }} button {{ border: 1px solid #334960; background: #18283b; color: #dbeafe; border-radius: 7px; padding: 8px 12px; cursor: pointer; }}
    @media (max-width: 820px) {{ .grid {{ grid-template-columns: repeat(2, minmax(0, 1fr)); }} header {{ align-items: flex-start; }} }}
    @media (max-width: 480px) {{ main {{ width: calc(100% - 22px); padding-top: 20px; }} .grid {{ gap: 8px; }} .card {{ padding: 13px; min-height: 90px; }} .value {{ font-size: 21px; }} }}
  </style>
</head>
<body>
<main>
  <header>
    <div><h1>cTraderBot account</h1><div class="muted">{strategy_name} · {timeframe}</div></div>
    <div><span id="connection" class="status">Connecting to cTrader…</span><br><button id="refresh">Refresh</button></div>
  </header>
  <div class="grid">
    <div class="card"><div class="label">Account balance · all account activity</div><div id="balance" class="value">—</div></div>
    <div class="card"><div class="label">Estimated balance + bot open P&amp;L</div><div id="equity" class="value">—</div></div>
    <div class="card"><div class="label">Bot open P&amp;L</div><div id="open-pnl" class="value">—</div></div>
    <div class="card"><div class="label">Realized P&amp;L · last <span id="period-label">30</span> days</div><div id="realized-pnl" class="value">—</div></div>
    <div class="card"><div class="label">Open positions</div><div id="open-count" class="value">—</div></div>
    <div class="card"><div class="label">Closed positions · period</div><div id="closed-count" class="value">—</div></div>
    <div class="card"><div class="label">Today realized P&amp;L · ET</div><div id="today-pnl" class="value">—</div></div>
    <div class="card"><div class="label">Bot win rate · 30 days</div><div id="win-rate" class="value">—</div></div>
  </div>
  <section>
    <div class="section-head"><h2>Open bot positions · live from cTrader</h2><span id="updated" class="muted">—</span></div>
    <div class="table-wrap"><table><thead><tr><th>Position</th><th>Symbol</th><th>Side</th><th>Lots</th><th>Entry</th><th>Stop loss</th><th>Take profit</th><th>Net unrealized P&amp;L</th></tr></thead><tbody id="open-rows"><tr><td class="empty" colspan="8">Loading positions…</td></tr></tbody></table></div>
  </section>
  <section>
  <div class="section-head"><h2>Closed bot positions</h2><span class="muted">Trades recorded by this bot</span></div>
    <div class="table-wrap"><table><thead><tr><th>Closed · ET</th><th>Position</th><th>Symbol</th><th>Side</th><th>Lots</th><th>Entry</th><th>Exit</th><th>Net P&amp;L</th><th>Bot R</th></tr></thead><tbody id="closed-rows"><tr><td class="empty" colspan="9">Loading history…</td></tr></tbody></table></div>
  </section>
  <section>
    <div class="section-head"><h2>Strategy review</h2><span class="muted">Bot-journal results · last 30 days</span></div>
    <div class="strategy">
      <span>Trades: <b id="bot-trades">—</b></span><span>Net P&amp;L: <b id="bot-pnl">—</b></span>
      <span>Expectancy: <b id="expectancy">—</b></span><span>Profit factor: <b id="profit-factor">—</b></span>
      <span>Max drawdown: <b id="drawdown">—</b></span>
    </div>
  </section>
</main>
<script>
  const currency = value => {ACCOUNT_AMOUNT_FORMATTER};
  const number = (value, digits = 5) => value == null ? "—" : Number(value).toFixed(digits);
  const easternTime = value => new Intl.DateTimeFormat(undefined, {{
    dateStyle: "short", timeStyle: "medium", timeZone: "America/New_York"
  }}).format(new Date(value));
  function setValue(id, value, pnl = false) {{
    const element = document.getElementById(id);
    element.textContent = value;
    if (pnl) element.className = "value " + (Number(value.replace(/[^0-9.-]/g, "")) >= 0 ? "positive" : "negative");
  }}
  function appendRow(body, values, pnlIndex = -1) {{
    const row = document.createElement("tr");
    values.forEach((value, index) => {{
      const cell = document.createElement("td");
      cell.textContent = value == null ? "—" : value;
      if (index === pnlIndex) cell.className = Number(String(value).replace(/[^0-9.-]/g, "")) >= 0 ? "positive" : "negative";
      row.appendChild(cell);
    }});
    body.appendChild(row);
  }}
  async function loadLive() {{
    const status = document.getElementById("connection");
    try {{
      const response = await fetch("/dashboard/api/live", {{ cache: "no-store" }});
      const data = await response.json();
      if (!response.ok) throw new Error(data.detail || "Could not load account data");
      status.textContent = "Connected · " + data.environment;
      status.className = "status positive";
      setValue("balance", currency(data.balance));
      setValue("equity", currency(data.equity));
      setValue("open-pnl", currency(data.unrealized_pnl), true);
      setValue("open-count", String(data.positions.length));
      document.getElementById("updated").textContent = "Updated " + new Date(data.updated_at).toLocaleTimeString();
      const body = document.getElementById("open-rows");
      body.replaceChildren();
      if (!data.positions.length) {{
        body.innerHTML = '<tr><td class="empty" colspan="8">No open positions on this account.</td></tr>';
      }} else {{
        data.positions.forEach(position => appendRow(body, [
          position.position_id, position.symbol, position.side, number(position.lots, 2),
          number(position.entry_price), number(position.stop_loss), number(position.take_profit),
          currency(position.unrealized_pnl)
        ], 7));
      }}
    }} catch (error) {{
      status.textContent = "cTrader unavailable · " + error.message;
      status.className = "status error";
    }}
  }}
  async function loadHistory() {{
    const days = 30;
    try {{
      const response = await fetch("/dashboard/api/history?days=" + days, {{ cache: "no-store" }});
      const data = await response.json();
      if (!response.ok) throw new Error(data.detail || "Could not load trade history");
      document.getElementById("period-label").textContent = days;
      setValue("realized-pnl", currency(data.realized_pnl), true);
      setValue("today-pnl", currency(data.today_realized_pnl), true);
      setValue("closed-count", String(data.closed_positions.length));
      setValue("win-rate", data.bot_stats.win_rate + "%");
      document.getElementById("bot-trades").textContent = data.bot_stats.total_trades;
      document.getElementById("bot-pnl").textContent = currency(data.bot_stats.realized_pnl);
      document.getElementById("bot-pnl").className = data.bot_stats.realized_pnl >= 0 ? "positive" : "negative";
      document.getElementById("expectancy").textContent = data.bot_stats.expectancy + "R";
      document.getElementById("profit-factor").textContent = data.bot_stats.profit_factor;
      document.getElementById("drawdown").textContent = data.bot_stats.max_r_drawdown + "R";
      const body = document.getElementById("closed-rows");
      body.replaceChildren();
      if (!data.closed_positions.length) {{
        body.innerHTML = '<tr><td class="empty" colspan="9">No closed positions in this period.</td></tr>';
      }} else {{
        data.closed_positions.slice(0, 100).forEach(position => appendRow(body, [
          easternTime(position.closed_at), position.position_id, position.symbol,
          position.side, number(position.lots, 2), number(position.entry_price),
          number(position.exit_price), currency(position.pnl),
          position.r_multiple == null ? "—" : number(position.r_multiple, 2) + "R"
        ], 7));
      }}
    }} catch (error) {{
      const body = document.getElementById("closed-rows");
      body.innerHTML = "";
      const row = document.createElement("tr");
      const cell = document.createElement("td");
      cell.colSpan = 9;
      cell.className = "empty error";
      cell.textContent = error.message;
      row.appendChild(cell);
      body.appendChild(row);
    }}
  }}
  document.getElementById("refresh").addEventListener("click", () => {{ loadLive(); loadHistory(); }});
  loadLive();
  loadHistory();
  setInterval(loadLive, 5000);
  setInterval(loadHistory, 60000);
</script>
</body>
</html>"""


@router.get("/api/live")
async def api_live_account():
    """Return current account state and open positions placed by this bot."""
    try:
        account = await ctrader_client.get_account_info()
        positions = await ctrader_client.get_bot_positions()
        pnl_by_position = await ctrader_client.get_unrealized_pnl()
        serialized = []
        for position in positions:
            symbol = ctrader_client.symbol_name(position.symbol_id)
            symbol_info = await ctrader_client.get_symbol_info(symbol)
            serialized.append(
                {
                    "position_id": position.position_id,
                    "symbol": symbol,
                    "side": position.side.value.lower(),
                    "lots": float(symbol_info.volume_to_lots(position.volume)),
                    "entry_price": float(position.entry_price),
                    "stop_loss": (
                        float(position.stop_loss)
                        if position.stop_loss is not None
                        else None
                    ),
                    "take_profit": (
                        float(position.take_profit)
                        if position.take_profit is not None
                        else None
                    ),
                    "unrealized_pnl": pnl_by_position.get(position.position_id, 0.0),
                }
            )
        unrealized_pnl = sum(
            float(position["unrealized_pnl"]) for position in serialized
        )
        return {
            "environment": "Live"
            if settings.CTRADER_ENVIRONMENT.lower() == "live"
            else "Demo",
            "balance": float(account.balance),
            "unrealized_pnl": unrealized_pnl,
            "equity": float(account.balance) + unrealized_pnl,
            "positions": serialized,
            "updated_at": datetime.now(UTC).isoformat(),
        }
    except Exception as exc:
        raise HTTPException(
            status_code=503, detail=f"cTrader account data unavailable: {exc}"
        ) from exc


@router.get("/api/history")
async def api_trade_history(days: int = Query(default=30, ge=1, le=90)):
    """Return closed bot deals aggregated into positions and bot statistics."""
    now = datetime.now(UTC)
    start = now - timedelta(days=days)
    try:
        deals = await ctrader_client.get_deals(start, now)
        bot_rows = risk_manager.get_trade_history(limit=1000)
        bot_trades = {str(row["trade_id"]): row for _, row in bot_rows.iterrows()}
        bot_position_ids = {int(trade_id) for trade_id in bot_trades}

        grouped: dict[int, dict] = {}
        for deal in deals:
            if int(deal.position_id) not in bot_position_ids:
                continue
            detail = deal.close_detail
            if detail is None:
                continue
            trade = grouped.setdefault(
                deal.position_id,
                {
                    "position_id": deal.position_id,
                    "symbol_id": deal.symbol_id,
                    "side": "sell" if deal.side.value.lower() == "buy" else "buy",
                    "closed_volume": 0,
                    "entry_value": 0.0,
                    "exit_value": 0.0,
                    "pnl": 0.0,
                    "closed_at": deal.execution_timestamp,
                },
            )
            closed_volume = detail.closed_volume
            trade["closed_volume"] += closed_volume
            trade["entry_value"] += float(detail.entry_price) * closed_volume
            trade["exit_value"] += float(deal.execution_price) * closed_volume
            trade["pnl"] += float(
                detail.gross_profit
                + detail.swap
                + detail.commission
                - detail.pnl_conversion_fee
            )
            trade["closed_at"] = max(trade["closed_at"], deal.execution_timestamp)

        closed_positions = []
        today = datetime.now(EASTERN).date()
        realized_pnl = 0.0
        today_realized_pnl = 0.0
        for position_id, trade in grouped.items():
            symbol = ctrader_client.symbol_name(trade["symbol_id"])
            symbol_info = await ctrader_client.get_symbol_info(symbol)
            volume = trade["closed_volume"]
            pnl = trade["pnl"]
            realized_pnl += pnl
            if trade["closed_at"].astimezone(EASTERN).date() == today:
                today_realized_pnl += pnl
            bot_trade = bot_trades.get(str(position_id))
            closed_positions.append(
                {
                    "position_id": position_id,
                    "symbol": symbol,
                    "side": trade["side"],
                    "lots": float(symbol_info.volume_to_lots(volume)),
                    "entry_price": trade["entry_value"] / volume if volume else 0.0,
                    "exit_price": trade["exit_value"] / volume if volume else 0.0,
                    "pnl": pnl,
                    "closed_at": trade["closed_at"].isoformat(),
                    "r_multiple": (
                        float(bot_trade["r_multiple"])
                        if bot_trade is not None
                        else None
                    ),
                }
            )

        closed_positions.sort(key=lambda item: item["closed_at"], reverse=True)
        stats = risk_manager.get_r_multiple_stats(days)
        stats["realized_pnl"] = float(bot_rows["pnl"].fillna(0).sum())
        return {
            "days": days,
            "realized_pnl": realized_pnl,
            "today_realized_pnl": today_realized_pnl,
            "closed_positions": closed_positions,
            "bot_stats": stats,
        }
    except Exception as exc:
        raise HTTPException(
            status_code=503, detail=f"cTrader trade history unavailable: {exc}"
        ) from exc


@router.get("/api/stats")
async def api_stats(days: int = Query(default=30, ge=1, le=90)):
    """Backward-compatible bot-journal statistics endpoint."""
    return {
        "r_multiple_stats": risk_manager.get_r_multiple_stats(days),
        "daily_summary": risk_manager.get_daily_summary(),
        "open_trades": risk_manager.get_open_trades(),
    }
