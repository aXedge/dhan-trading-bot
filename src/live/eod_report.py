#!/usr/bin/env python3
"""
EOD Report — Daily Summary
============================

Generates end-of-day summary of paper trading performance.
Sends a Telegram message with portfolio status, open positions, and closed trades.

Single-pass script — designed for cron (runs at 3:45 PM after market close).

Usage:
    python src/live/eod_report.py
"""

import json
import os
import sys
from datetime import date, datetime

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, SRC_DIR)
sys.path.insert(0, os.path.dirname(SRC_DIR))  # repo root

import yfinance as yf

PAPER_POSITIONS_FILE = os.path.join(SRC_DIR, "..", "data", "paper_positions.json")


def load_positions():
    if os.path.exists(PAPER_POSITIONS_FILE):
        with open(PAPER_POSITIONS_FILE) as f:
            return json.load(f)
    return {"positions": [], "closed_trades": []}


def fetch_price(symbol):
    try:
        ticker = symbol + ".NS" if not symbol.endswith(".NS") else symbol
        df = yf.download(ticker, period="5d", auto_adjust=True, progress=False)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        if len(df) > 0:
            return float(df["Close"].iloc[-1])
    except:
        pass
    return None


def send_telegram(message):
    try:
        import requests
        token = os.environ.get("TELEGRAM_BOT_TOKEN")
        chat_id = os.environ.get("TELEGRAM_CHAT_ID")
        if token and chat_id:
            requests.post(
                f"https://api.telegram.org/bot{token}/sendMessage",
                json={"chat_id": chat_id, "text": message, "parse_mode": "Markdown"},
                timeout=10
            )
            return True
    except:
        pass
    return False


def main():
    print(f"\n{'='*60}")
    print(f"EOD Report — {date.today()}")
    print(f"{'='*60}")

    data = load_positions()
    open_pos = data.get("positions", [])
    closed = data.get("closed_trades", [])

    # Calculate open position P&L
    total_unrealized = 0
    open_summary = []
    for p in open_pos:
        price = fetch_price(p["symbol"])
        if price:
            pnl_pct = (price - p["entry_price"]) / p["entry_price"] * 100
            pnl_rs = pnl_pct / 100 * p["quantity"] * p["entry_price"]
            total_unrealized += pnl_rs
            open_summary.append({
                "symbol": p["symbol"], "strategy": p["strategy"],
                "entry": p["entry_price"], "current": price,
                "pnl_pct": pnl_pct, "pnl_rs": pnl_rs,
                "sl": p["stop_loss"], "target": p["target"],
            })

    # Calculate closed trade stats
    total_realized = sum(c.get("pnl_rs", 0) for c in closed)
    wins = [c for c in closed if c.get("pnl_pct", 0) > 0]
    losses = [c for c in closed if c.get("pnl_pct", 0) <= 0]
    win_rate = len(wins) / len(closed) * 100 if closed else 0

    # Today's closed trades
    today = str(date.today())
    today_closed = [c for c in closed if c.get("exit_date") == today]
    today_pnl = sum(c.get("pnl_rs", 0) for c in today_closed)

    # Print summary
    print(f"\n📊 Portfolio Summary:")
    print(f"  Open positions: {len(open_pos)}")
    print(f"  Unrealized P&L: ₹{total_unrealized:+,.0f}")
    print(f"  Closed trades: {len(closed)} (Wins: {len(wins)}, Losses: {len(losses)})")
    print(f"  Win rate: {win_rate:.1f}%")
    print(f"  Realized P&L: ₹{total_realized:+,.0f}")
    print(f"  Total P&L: ₹{total_realized + total_unrealized:+,.0f}")
    print(f"\n  Today's exits: {len(today_closed)} | Today's P&L: ₹{today_pnl:+,.0f}")

    if open_summary:
        print(f"\n  Open positions:")
        for p in open_summary:
            print(f"    [{p['strategy'].upper()}] {p['symbol']}: Entry=₹{p['entry']} "
                  f"Now=₹{p['current']:.2f} P&L={p['pnl_pct']:+.1f}% (₹{p['pnl_rs']:+,.0f})")

    # Telegram message
    msg = f"📊 *EOD Report — {date.today()}*\n\n"
    msg += f"*Open Positions:* {len(open_pos)}\n"
    msg += f"*Unrealized P&L:* ₹{total_unrealized:+,.0f}\n\n"
    msg += f"*Closed Trades:* {len(closed)} (W:{len(wins)} L:{len(losses)})\n"
    msg += f"*Win Rate:* {win_rate:.1f}%\n"
    msg += f"*Realized P&L:* ₹{total_realized:+,.0f}\n"
    msg += f"*Total P&L:* ₹{total_realized + total_unrealized:+,.0f}\n\n"

    if today_closed:
        msg += f"*Today's Exits:* {len(today_closed)}\n"
        for c in today_closed:
            msg += f"  [{c['strategy'].upper()}] {c['symbol']}: {c['exit_reason']} "
            msg += f"P&L: {c['pnl_pct']:+.1f}% (₹{c['pnl_rs']:+,.0f})\n"

    if open_summary:
        msg += f"\n*Open Positions:*\n"
        for p in open_summary:
            msg += f"  [{p['strategy'].upper()}] {p['symbol']}: ₹{p['entry']}→₹{p['current']:.2f} "
            msg += f"({p['pnl_pct']:+.1f}%)\n"

    sent = send_telegram(msg)
    print(f"\n{'='*60}")
    print(f"Telegram alert: {'sent' if sent else 'not sent (no token configured)'}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
