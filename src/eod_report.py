"""
End-of-Day Report (multi-session)

Generates a summary of the day's trading activity across all 3 sessions
and sends it via Telegram.

- Fetches positions, holdings, and fund limits from Dhan (single auth)
- Reports P&L for each session's positions separately
- Reports overall holdings P&L
- Sends a formatted summary via Telegram

Runs at 3:35 PM IST via cron, Mon-Fri.
"""

import os
import sys
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
from auth import get_dhan, get_access_token
from utils import setup_logger, send_telegram, now_iso, load_json

load_dotenv()
logger = setup_logger(__name__, "eod.log")

ALL_SESSIONS = ["A", "B", "C", "T"]
SESSION_LABELS = {"A": "Conservative", "B": "Balanced", "C": "Aggressive", "T": "Tuned"}


def run():
    """Generate and send the EOD report across all sessions."""
    logger.info("=" * 60)
    logger.info("EOD Report (multi-session) — starting")
    logger.info("=" * 60)

    try:
        dhan = get_dhan()
    except Exception as e:
        logger.error(f"Failed to authenticate: {e}")
        send_telegram(f"EOD Report: Failed to authenticate — {e}")
        return

    # --- Fetch data from Dhan (single auth) ---
    funds = dhan.get_fund_limits()
    holdings = dhan.get_holdings()

    fund_data = funds.get("data", funds) if isinstance(funds, dict) else {}
    hold_data = holdings.get("data", []) if isinstance(holdings, dict) else []

    available_balance = fund_data.get("availabelBalance", 0)
    utilized = fund_data.get("utilizedAmount", 0)
    collateral = fund_data.get("collateralAmount", 0)

    # --- Holdings P&L ---
    total_holdings_value = 0
    total_invested = 0
    holdings_pnl = 0

    for h in hold_data:
        qty = int(h.get("totalQty", 0))
        avg_cost = float(h.get("avgCostPrice", 0))
        ltp = float(h.get("lastTradedPrice", 0))
        total_holdings_value += qty * ltp
        total_invested += qty * avg_cost
        holdings_pnl += qty * (ltp - avg_cost)

    # --- Per-session positions ---
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    report_lines = [
        f"EOD Report — {now_str}",
        f"",
        f"Funds:",
        f"  Available: Rs.{available_balance:,.0f}",
        f"  Utilized: Rs.{utilized:,.0f}",
        f"  Collateral: Rs.{collateral:,.0f}",
        f"",
        f"Holdings ({len(hold_data)} stocks):",
        f"  Invested: Rs.{total_invested:,.0f}",
        f"  Current: Rs.{total_holdings_value:,.0f}",
        f"  P&L: Rs.{holdings_pnl:+,.0f}",
        f"",
        f"--- Session Positions ---",
    ]

    total_session_pnl = 0
    for session in ALL_SESSIONS:
        positions = load_json(f"positions_{session}.json")
        label = SESSION_LABELS[session]

        if not positions:
            report_lines.append(f"Session {session} ({label}): no open positions")
            continue

        session_pnl = 0
        report_lines.append(f"Session {session} ({label}): {len(positions)} positions")

        for pos in positions:
            symbol = pos["symbol"]
            entry = pos["entry_price"]
            qty = pos["qty"]
            sec_id = pos["security_id"]

            # Try to get LTP
            try:
                quote = dhan.market_quote("NSE", sec_id)
                data = quote.get("data", {}).get("NSE", {}).get(sec_id, {})
                ltp = float(data.get("last_price", 0))
            except:
                ltp = entry  # fallback

            pnl = (ltp - entry) * qty
            pnl_pct = ((ltp - entry) / entry * 100) if entry else 0
            session_pnl += pnl

            report_lines.append(
                f"  {symbol:15s} x{qty:4d} @ Rs.{entry:.2f} "
                f"LTP:Rs.{ltp:.2f} P&L:Rs.{pnl:+,.0f} ({pnl_pct:+.1f}%)"
            )

        total_session_pnl += session_pnl
        report_lines.append(f"  Subtotal: Rs.{session_pnl:+,.0f}")
        report_lines.append(f"")

    report_lines.extend([
        f"Total session P&L: Rs.{total_session_pnl:+,.0f}",
        f"",
        f"Trading mode: {os.getenv('TRADING_MODE', 'UNKNOWN')}",
    ])

    report = "\n".join(report_lines)

    # --- Log it ---
    logger.info("EOD Report generated:")
    for line in report_lines:
        logger.info(f"  {line}")

    # --- Send via Telegram ---
    try:
        send_telegram(report)
        logger.info("Report sent via Telegram")
    except Exception as e:
        logger.error(f"Failed to send Telegram alert: {e}")

    print(report)

    logger.info("=" * 60)
    logger.info("EOD Report — complete")
    logger.info("=" * 60)

    return report


if __name__ == "__main__":
    run()
