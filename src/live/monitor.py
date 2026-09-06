#!/usr/bin/env python3
"""
SL Monitor & Exit Manager — Paper Trading
==========================================

Checks all open paper positions against current prices. Exits positions when:
  1. Stop loss is hit
  2. Target is hit
  3. Strategy signal exit triggers (RSI threshold)

Single-pass script — designed for cron (runs every 15 min during market hours).

Usage:
    python src/live/monitor.py
"""

import json
import os
import sys
from datetime import date, datetime

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, SRC_DIR)

import pandas as pd
import numpy as np
import yfinance as yf

PAPER_POSITIONS_FILE = os.path.join(SRC_DIR, "..", "data", "paper_positions.json")

# Import strategies for exit logic
import strategies.reversal as reversal
import strategies.positional_pullback as pullback

STRATEGY_MAP = {
    "reversal": reversal,
    "pullback": pullback,
}


def load_positions():
    if os.path.exists(PAPER_POSITIONS_FILE):
        with open(PAPER_POSITIONS_FILE) as f:
            return json.load(f)
    return {"positions": [], "closed_trades": []}


def save_positions(data):
    os.makedirs(os.path.dirname(PAPER_POSITIONS_FILE), exist_ok=True)
    with open(PAPER_POSITIONS_FILE, "w") as f:
        json.dump(data, f, indent=2, default=str)


def fetch_current_price(symbol):
    """Fetch current/closing price for a symbol."""
    try:
        ticker = symbol + ".NS" if not symbol.endswith(".NS") else symbol
        df = yf.download(ticker, period="5d", auto_adjust=True, progress=False)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        if len(df) > 0:
            return float(df["Close"].iloc[-1]), df
    except Exception as e:
        print(f"  {symbol}: price fetch error - {e}")
    return None, None


def check_exit(position, current_price, df=None):
    """Check if position should be exited. Returns (should_exit, reason, exit_price)."""
    # 1. Stop loss
    if current_price <= position["stop_loss"]:
        return True, "SL hit", position["stop_loss"]

    # 2. Target
    if current_price >= position["target"]:
        return True, "Target hit", position["target"]

    # 3. Signal exit (strategy-specific)
    strategy_name = position.get("strategy", "")
    module = STRATEGY_MAP.get(strategy_name)
    if module and df is not None and len(df) > 5:
        try:
            config = module.DEFAULT_CONFIG.copy()
            prepared = module.prepare(df, config)
            last = prepared.iloc[-1]
            if module.should_exit(last, config):
                return True, "Signal exit", current_price
        except Exception as e:
            print(f"  {position['symbol']}: exit check error - {e}")

    return False, None, None


def send_telegram_alert(message):
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
    except Exception:
        pass


def main():
    print(f"\n{'='*60}")
    print(f"SL Monitor & Exit Manager — {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print(f"{'='*60}")

    data = load_positions()
    positions = data["positions"]
    closed = data.get("closed_trades", [])

    open_positions = [p for p in positions if p.get("status") == "OPEN"]
    if not open_positions:
        print("No open positions to monitor.")
        return

    print(f"Monitoring {len(open_positions)} open positions\n")

    exits = []
    for pos in open_positions:
        sym = pos["symbol"]
        print(f"  {sym}: ", end="", flush=True)

        price, df = fetch_current_price(sym)
        if price is None:
            print("price fetch failed")
            continue

        pnl_pct = (price - pos["entry_price"]) / pos["entry_price"] * 100
        should_exit, reason, exit_price = check_exit(pos, price, df)

        if should_exit:
            pos["status"] = "CLOSED"
            pos["exit_date"] = str(date.today())
            pos["exit_price"] = exit_price
            pos["exit_reason"] = reason
            pos["pnl_pct"] = (exit_price - pos["entry_price"]) / pos["entry_price"] * 100
            pos["pnl_rs"] = pos["pnl_pct"] / 100 * pos["quantity"] * pos["entry_price"]
            closed.append(pos)
            exits.append(pos)
            print(f"EXIT — {reason} @ ₹{exit_price} (P&L: {pos['pnl_pct']:+.1f}% = ₹{pos['pnl_rs']:+,.0f})")
        else:
            print(f"holding @ ₹{price:.2f} (P&L: {pnl_pct:+.1f}%)")

    # Save
    data["positions"] = [p for p in positions if p.get("status") == "OPEN"]
    data["closed_trades"] = closed
    save_positions(data)

    # Telegram alert for exits
    if exits:
        alert = f"📊 *Exit Alert*\n\n"
        for e in exits:
            alert += f"*[{e['strategy'].upper()}] {e['symbol']}*\n"
            alert += f"Exit: ₹{e['exit_price']} ({e['exit_reason']})\n"
            alert += f"P&L: {e['pnl_pct']:+.1f}% (₹{e['pnl_rs']:+,.0f})\n\n"
        send_telegram_alert(alert)

    # Summary
    total_pnl = sum(c.get("pnl_rs", 0) for c in closed)
    win_count = sum(1 for c in closed if c.get("pnl_pct", 0) > 0)
    print(f"\n{'='*60}")
    print(f"Monitor complete: {len(exits)} exits today")
    print(f"Open positions: {len(data['positions'])}")
    print(f"Closed trades: {len(closed)} | Wins: {win_count} | Total P&L: ₹{total_pnl:+,.0f}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
