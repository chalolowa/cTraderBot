"""FastAPI dashboard routes for R-multiple tracking."""

import json
from typing import Any

import plotly.graph_objects as go
from fastapi import APIRouter, HTTPException
from fastapi.responses import HTMLResponse
from plotly.utils import PlotlyJSONEncoder

from src.risk_manager import risk_manager

router = APIRouter(prefix="/dashboard", tags=["dashboard"])


def generate_r_multiple_chart(days: int = 30) -> dict[str, Any]:
    """Generate Plotly chart of R-multiples over time."""
    closed = risk_manager.get_closed_trades(days)

    if not closed:
        return {}

    dates = []
    r_values = []
    cumulative_r = []
    running_total = 0.0

    for trade in sorted(
        closed, key=lambda x: x.exit_time.isoformat() if x.exit_time else ""
    ):
        if trade.r_multiple is not None:
            dates.append(trade.exit_time.date().isoformat() if trade.exit_time else "")
            r_values.append(trade.r_multiple)
            running_total += trade.r_multiple
            cumulative_r.append(running_total)

    fig = go.Figure()

    # Individual R-multiples
    colors = ["green" if r > 0 else "red" for r in r_values]
    fig.add_trace(
        go.Bar(
            x=dates,
            y=r_values,
            marker_color=colors,
            name="Trade R-Multiple",
            hovertemplate="Date: %{x}<br>R: %{y:.2f}<extra></extra>",
        )
    )

    # Cumulative R curve
    fig.add_trace(
        go.Scatter(
            x=dates,
            y=cumulative_r,
            mode="lines+markers",
            name="Cumulative R",
            line={"color": "blue", "width": 2},
            yaxis="y2",
        )
    )

    fig.update_layout(
        title="R-Multiple Performance Tracker",
        xaxis_title="Date",
        yaxis_title="R-Multiple per Trade",
        yaxis2={"title": "Cumulative R", "overlaying": "y", "side": "right"},
        height=500,
        template="plotly_dark",
        hovermode="x unified",
    )

    return json.loads(json.dumps(fig, cls=PlotlyJSONEncoder))


@router.get("/", response_class=HTMLResponse)
async def dashboard_home():
    """Main dashboard page."""
    try:
        stats = risk_manager.get_r_multiple_stats(days=30)
        daily = risk_manager.get_daily_summary()

        chart_json = generate_r_multiple_chart(days=30)
        chart_div = ""
        if chart_json:
            chart_data_js = json.dumps(chart_json)
            chart_div = f"""
            <div id="r-chart"></div>
            <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
            <script>
                const chartData = {chart_data_js};
                Plotly.newPlot('r-chart', chartData.data, chartData.layout);
            </script>
            """

        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
        <title>cTraderBot Dashboard</title>
        <style>
            body {{ font-family: 'Segoe UI', sans-serif; background: #0f172a; color: #e2e8f0; margin: 0; padding: 20px; }}
            .container {{ max-width: 1200px; margin: 0 auto; }}
            .header {{ text-align: center; margin-bottom: 30px; }}
            .stats-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 15px; margin-bottom: 30px; }}
            .stat-card {{ background: #1e293b; padding: 20px; border-radius: 10px; border: 1px solid #334155; }}
            .stat-value {{ font-size: 28px; font-weight: bold; color: #38bdf8; margin-top: 5px; }}
            .positive {{ color: #4ade80; }}
            .negative {{ color: #f87171; }}
            .chart-container {{ background: #1e293b; padding: 20px; border-radius: 10px; border: 1px solid #334155; }}
            table {{ width: 100%; border-collapse: collapse; margin-top: 20px; }}
            th, td {{ padding: 12px; text-align: left; border-bottom: 1px solid #334155; }}
            th {{ color: #94a3b8; font-weight: 600; }}
        </style>
        </head>
        <body>
        <div class="container">
            <div class="header">
                <h1>🤖 cTraderBot Dashboard</h1>
                <p>Real-time R-Multiple & Performance Tracking</p>
            </div>

            <div class="stats-grid">
                <div class="stat-card">
                    <div>Total Trades (30D)</div>
                    <div class="stat-value">{stats["total_trades"]}</div>
                </div>
                <div class="stat-card">
                    <div>Win Rate</div>
                    <div class="stat-value {"positive" if stats["win_rate"] > 50 else "negative"}">{stats["win_rate"]}%</div>
                </div>
                <div class="stat-card">
                    <div>Avg R</div>
                    <div class="stat-value {"positive" if stats["avg_r"] > 0 else "negative"}">{stats["avg_r"]}R</div>
                </div>
                <div class="stat-card">
                    <div>Total R</div>
                    <div class="stat-value {"positive" if stats["total_r"] > 0 else "negative"}">{stats["total_r"]}R</div>
                </div>
                <div class="stat-card">
                    <div>Expectancy</div>
                    <div class="stat-value {"positive" if stats["expectancy"] > 0 else "negative"}">{stats["expectancy"]}R</div>
                </div>
                <div class="stat-card">
                    <div>Profit Factor</div>
                    <div class="stat-value {"positive" if stats["profit_factor"] > 1 else "negative"}">{stats["profit_factor"]}</div>
                </div>
                <div class="stat-card">
                    <div>Today's PnL</div>
                    <div class="stat-value {"positive" if daily["daily_pnl"] > 0 else "negative"}">${daily["daily_pnl"]:.2f}</div>
                </div>
                <div class="stat-card">
                    <div>Open Trades</div>
                    <div class="stat-value">{daily["open_trades"]}</div>
                </div>
            </div>

            <div class="chart-container">
                <h3>R-Multiple Equity Curve</h3>
                {chart_div}
            </div>

            <div class="chart-container" style="margin-top: 20px;">
                <h3>Recent Closed Trades</h3>
                <table>
                    <tr><th>Trade ID</th><th>Instrument</th><th>Dir</th><th>Entry</th><th>Exit</th><th>PnL</th><th>R-Multiple</th></tr>
        """

        recent = sorted(
            risk_manager.get_closed_trades(days=7),
            key=lambda x: x.exit_time.isoformat() if x.exit_time else "",
            reverse=True,
        )[:20]

        for t in recent:
            pnl_class = "positive" if (t.pnl or 0) > 0 else "negative"
            html += f"""
                    <tr>
                        <td>{t.trade_id}</td>
                        <td>{t.instrument}</td>
                        <td>{t.direction}</td>
                        <td>{t.entry_price:.5f}</td>
                        <td>{f"{t.exit_price:.5f}" if t.exit_price is not None else "N/A"}</td>
                        <td class="{pnl_class}">${t.pnl:.2f}</td>
                        <td class="{pnl_class}">{t.r_multiple:.2f}R</td>
                    </tr>
            """

        html += """
                </table>
            </div>
        </div>
        </body>
        </html>
        """
        return html
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Dashboard error: {exc}")


@router.get("/api/stats")
async def api_stats(days: int = 30):
    """API endpoint for raw stats."""
    try:
        return {
            "r_multiple_stats": risk_manager.get_r_multiple_stats(days),
            "daily_summary": risk_manager.get_daily_summary(),
            "open_trades": risk_manager.get_open_trades(),
        }
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Stats error: {exc}")
