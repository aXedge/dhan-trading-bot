"""
End-of-Day Report
=================

Generates a summary of the day's trading activity and sends it via Telegram.

- Fetches positions, holdings, and fund limits from Dhan
- Calculates unrealized P&L on open positions
- Calculates P&L on holdings (current price vs avg cost)
- Sends a formatted summary via Telegram
- Logs everything to eod.log

Runs at 3:35 PM IST via cron, Mon-Fri.
"""

import os
import sys
from datetime import datetime

# Add project root to path so we can import src.utils, src.auth
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
from auth import get_dhan, get_access_token
from utils import setup_logger, send_telegram, now_iso

load_dotenv()
logger = setup_logger(__name__, "eod.log")


def run():
    """Generate and send the end-of-day report."""
    logger.info("=" * 60)
    logger.info("EOD Report — starting")
    logger.info("=" * 60)

    try:
        dhan = get_dhan()
    except Exception as e:
        logger.error(f"Failed to authenticate: {e}")
        send_telegram(f"❌ EOD Report: Failed to authenticate — {e}")
        return

    # --- Fetch data ---
    funds = dhan.get_fund_limits()
    positions = dhan.get_positions()
    holdings = dhan.get_holdings()

    fund_data = funds.get("data", funds) if isinstance(funds, dict) else {}
    pos_data = positions.get("data", []) if isinstance(positions, dict) else []
    hold_data = holdings.get("data", []) if isinstance(holdings, dict) else []

    available_balance = fund_data.get("availabelBalance", 0)
    utilized = fund_data.get("utilizedAmount", 0)
    collateral = fund_data.get("collateralAmount", 0)

    # --- Positions P&L ---
    total_unrealized = 0
    total_realized = 0
    open_positions = []

    for p in pos_data:
        net_qty = int(p.get("netQty", 0))
        unrealized = float(p.get("unrealizedProfit", 0))
        realized = float(p.get("realizedProfit", 0))
        total_unrealized += unrealized
        total_realized += realized

        if net_qty != 0:
            symbol = p.get("tradingSymbol", "unknown")
            product = p.get("productType", "")
            buy_avg = float(p.get("buyAvg", 0))
            sell_avg = float(p.get("sellAvg", 0))
            open_positions.append({
                "symbol": symbol,
                "netQty": net_qty,
                "product": product,
                "unrealized": unrealized,
            })

    # --- Holdings P&L ---
    total_holdings_value = 0
    total_invested = 0
    holdings_pnl = 0
    holdings_count = len(hold_data)

    for h in hold_data:
        qty = int(h.get("totalQty", 0))
        avg_cost = float(h.get("avgCostPrice", 0))
        ltp = float(h.get("lastTradedPrice", 0))
        invested = qty * avg_cost
        current_value = qty * ltp
        total_holdings_value += current_value
        total_invested += invested
        holdings_pnl += current_value - invested

    # --- Net worth ---
    net_worth = available_balance + total_holdings_value + collateral

    # --- Build report ---
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    report_lines = [
        f"📊 EOD Report — {now_str}",
        f"",
        f"💰 Funds:",
        f"  Available: ₹{available_balance:,.0f}",
        f"  Utilized: ₹{utilized:,.0f}",
        f"  Collateral: ₹{collateral:,.0f}",
        f"",
        f"📈 Positions ({len(open_positions)} open):",
        f"  Realized P&L: ₹{total_realized:,.0f}",
        f"  Unrealized P&L: ₹{total_unrealized:,.0f}",
    ]

    if open_positions:
        report_lines.append(f"")
        report_lines.append(f"  Open positions:")
        for p in open_positions[:10]:
            pnl_str = f"₹{p['unrealized']:+,.0f}"
            report_lines.append(f"    {p['symbol'][:25]:<25} qty:{p['netQty']:>4} {pnl_str}")
    else:
        report_lines.append(f"  (no open positions)")

    report_lines.extend([
        f"",
        f"🏦 Holdings ({holdings_count} stocks):",
        f"  Invested: ₹{total_invested:,.0f}",
        f"  Current: ₹{total_holdings_value:,.0f}",
        f"  P&L: ₹{holdings_pnl:+,.0f} ({(holdings_pnl/total_invested*100) if total_invested else 0:+.1f}%)",
        f"",
        f"💎 Net worth: ₹{net_worth:,.0f}",
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

    # --- Print to stdout (for cron log) ---
    print(report)

    logger.info("=" * 60)
    logger.info("EOD Report — complete")
    logger.info("=" * 60)

    return report


if __name__ == "__main__":
    run()
