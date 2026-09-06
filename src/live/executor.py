#!/usr/bin/env python3
"""
Live Executor — Paper Trading
================================

Reads signals from scanner output and executes paper trades via Dhan API.
In PAPER mode, logs trades to a local JSON file without placing real orders.
In LIVE mode, places real market orders via Dhan.

Single-pass script — designed for cron. Runs after scanner.py.

Usage:
    python src/live/executor.py

Environment variables:
    TRADING_MODE=PAPER  (or LIVE)
    DHAN_CLIENT_ID=1000191656
"""

import json
import os
import sys
from datetime import date, datetime

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, SRC_DIR)

# Paper trading state file
PAPER_POSITIONS_FILE = os.path.join(SRC_DIR, "..", "data", "paper_positions.json")
SIGNALS_FILE = os.path.join(SRC_DIR, "..", "data", "signals_today.json")

# Capital management
CAPITAL_PER_POSITION = 25000
MAX_POSITIONS_PER_STRATEGY = 4
MAX_TOTAL_POSITIONS = 8


def load_positions():
    """Load current open positions."""
    if os.path.exists(PAPER_POSITIONS_FILE):
        with open(PAPER_POSITIONS_FILE) as f:
            return json.load(f)
    return {"positions": [], "closed_trades": []}


def save_positions(data):
    """Save positions to file."""
    os.makedirs(os.path.dirname(PAPER_POSITIONS_FILE), exist_ok=True)
    with open(PAPER_POSITIONS_FILE, "w") as f:
        json.dump(data, f, indent=2, default=str)


def load_signals():
    """Load today's signals from scanner."""
    if os.path.exists(SIGNALS_FILE):
        with open(SIGNALS_FILE) as f:
            return json.load(f).get("signals", [])
    return []


def count_positions_by_strategy(positions, strategy):
    """Count open positions for a given strategy."""
    return sum(1 for p in positions if p.get("strategy") == strategy and p.get("status") == "OPEN")


def get_open_symbols(positions):
    """Get set of symbols that already have open positions."""
    return {p["symbol"] for p in positions if p.get("status") == "OPEN"}


def execute_paper_trade(signal):
    """Execute a paper trade — log it without real order."""
    return {
        "strategy": signal["strategy"],
        "symbol": signal["symbol"],
        "entry_date": str(date.today()),
        "entry_price": signal["entry_price"],
        "stop_loss": signal["stop_loss"],
        "target": signal["target"],
        "stop_loss_pct": signal["stop_loss_pct"],
        "target_pct": signal["target_pct"],
        "quantity": int(CAPITAL_PER_POSITION / signal["entry_price"]),
        "status": "OPEN",
        "entry_time": datetime.now().isoformat(),
    }


def send_telegram_alert(message):
    """Send Telegram alert (if configured)."""
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
        pass  # Non-critical


def main():
    mode = os.environ.get("TRADING_MODE", "PAPER")
    print(f"\n{'='*60}")
    print(f"Live Executor — {mode} mode — {date.today()}")
    print(f"{'='*60}")

    signals = load_signals()
    if not signals:
        print("No signals found. Run scanner.py first.")
        return

    data = load_positions()
    positions = data["positions"]
    open_symbols = get_open_symbols(positions)

    print(f"Current open positions: {len([p for p in positions if p.get('status') == 'OPEN'])}")
    print(f"Signals to process: {len(signals)}\n")

    new_trades = []
    for signal in signals:
        sym = signal["symbol"]
        strat = signal["strategy"]

        # Skip if already in position for this symbol
        if sym in open_symbols:
            print(f"  SKIP {sym} — already in open position")
            continue

        # Check position limits
        total_open = sum(1 for p in positions if p.get("status") == "OPEN")
        strat_open = count_positions_by_strategy(positions, strat)

        if total_open >= MAX_TOTAL_POSITIONS:
            print(f"  SKIP {sym} — max total positions ({MAX_TOTAL_POSITIONS}) reached")
            continue

        if strat_open >= MAX_POSITIONS_PER_STRATEGY:
            print(f"  SKIP {sym} — max {strat} positions ({MAX_POSITIONS_PER_STRATEGY}) reached")
            continue

        # Execute trade
        if mode == "PAPER":
            trade = execute_paper_trade(signal)
            positions.append(trade)
            open_symbols.add(sym)
            new_trades.append(trade)
            print(f"  PAPER BUY [{strat.upper()}] {sym} @ ₹{trade['entry_price']} | "
                  f"Qty: {trade['quantity']} | SL: ₹{trade['stop_loss']} | Target: ₹{trade['target']}")
        elif mode == "LIVE":
            # TODO: Implement live Dhan order placement
            print(f"  LIVE BUY [{strat.upper()}] {sym} — LIVE MODE NOT YET IMPLEMENTED")
            # Placeholder: use dhan.dhan_api.place_order(...)
        else:
            print(f"  UNKNOWN MODE: {mode}")

    # Save updated positions
    data["positions"] = positions
    save_positions(data)

    # Telegram alert
    if new_trades:
        alert = f"📊 *Paper Trading Alert*\n\n"
        for t in new_trades:
            alert += f"*[{t['strategy'].upper()}] {t['symbol']}*\n"
            alert += f"Entry: ₹{t['entry_price']} | SL: ₹{t['stop_loss']} | Target: ₹{t['target']}\n"
            alert += f"Qty: {t['quantity']} | Value: ₹{t['quantity'] * t['entry_price']:,}\n\n"
        send_telegram_alert(alert)

    print(f"\n{'='*60}")
    print(f"Executor complete: {len(new_trades)} new trades")
    print(f"Total open positions: {sum(1 for p in positions if p.get('status') == 'OPEN')}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
