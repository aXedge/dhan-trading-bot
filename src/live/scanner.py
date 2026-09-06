#!/usr/bin/env python3
"""
Live Scanner — Dual Strategy Signal Generator
================================================

Runs daily (cron) to scan the curated basket for entry signals from both
reversal and pullback strategies. Saves signals to JSON for the executor.

Single-pass script — no loops, no infinite waits. Designed for cron.

Usage:
    python src/live/scanner.py

Output:
    data/signals_today.json — list of entry signals with strategy, symbol, entry price, SL, target
"""

import json
import os
import sys
from datetime import datetime, date

# Add src to path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, SRC_DIR)
sys.path.insert(0, os.path.dirname(SRC_DIR))  # repo root for strategies/

import pandas as pd
import numpy as np
import yfinance as yf

# Import strategies
import strategies.reversal as reversal
import strategies.positional_pullback as pullback

# Curated basket (optimized in parameter sweep)
CURATED_BASKET = [
    "EICHERMOT", "BHARTIARTL", "INFY", "ADANIPORTS", "HDFCBANK",
    "HCLTECH", "CIPLA", "WIPRO", "TECHM", "TITAN", "ULTRACEMCO", "SUNPHARMA",
]

# Extended basket (add more NIFTY 50 stocks for more signals)
EXTENDED_BASKET = CURATED_BASKET + [
    "ICICIBANK", "SBIN", "LT", "BAJFINANCE", "KOTAKBANK", "AXISBANK",
    "TATACONSUM", "GRASIM", "ADANIENT", "JSWSTEEL", "BRITANNIA", "DRREDDY",
    "HDFCLIFE", "BAJAJFINSV", "DIVISLAB", "TATASTEEL", "BPCL", "COALINDIA",
    "HEROMOTOCO", "SHRIRAMFIN", "DMART", "BAJAJ-AUTO", "NESTLEIND", "ONGC",
    "NTPC", "POWERGRID", "LICI", "SBILIFE", "INDUSINDBK", "HINDUNILVR",
    "ITC", "TCS", "MARUTI", "ASIANPAINT", "M&M", "TATAMOTORS",
]


def fetch_stock_data(symbol, period="1y"):
    """Fetch recent OHLCV data for a single stock."""
    try:
        ticker = symbol + ".NS" if not symbol.endswith(".NS") else symbol
        df = yf.download(ticker, period=period, auto_adjust=True, progress=False)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        if len(df) > 210:
            return df
    except Exception as e:
        print(f"  {symbol}: fetch error - {e}")
    return None


def scan_strategy(symbol, df, strategy_module, strategy_name):
    """Run strategy entry check on the latest bar."""
    config = strategy_module.DEFAULT_CONFIG.copy()
    prepared_df = strategy_module.prepare(df, config)

    # Check entry on the last bar
    last = prepared_df.iloc[-1]
    prev = prepared_df.iloc[-2] if len(prepared_df) > 1 else None

    if strategy_module.should_enter(last, config, prev):
        close = float(last["Close"])
        sl_price = close * (1 - config["stop_loss_pct"])
        target_price = close * (1 + config["target_pct"])
        return {
            "strategy": strategy_name,
            "symbol": symbol,
            "date": str(last.name.date()),
            "entry_price": round(close, 2),
            "stop_loss": round(sl_price, 2),
            "target": round(target_price, 2),
            "stop_loss_pct": config["stop_loss_pct"],
            "target_pct": config["target_pct"],
            "rsi": round(float(last.get("rsi", 0)), 1) if not pd.isna(last.get("rsi")) else None,
        }
    return None


def main():
    print(f"\n{'='*60}")
    print(f"Live Scanner — {date.today()}")
    print(f"{'='*60}")

    basket = EXTENDED_BASKET
    signals = []

    for i, symbol in enumerate(basket):
        print(f"  Scanning {i+1}/{len(basket)}: {symbol}...", end=" ", flush=True)

        df = fetch_stock_data(symbol)
        if df is None:
            print("SKIP (no data)")
            continue

        # Check reversal
        rev_signal = scan_strategy(symbol, df, reversal, "reversal")
        if rev_signal:
            signals.append(rev_signal)
            print(f"REVERSAL SIGNAL! Entry={rev_signal['entry_price']} SL={rev_signal['stop_loss']} Target={rev_signal['target']}")
            continue

        # Check pullback
        pb_signal = scan_strategy(symbol, df, pullback, "pullback")
        if pb_signal:
            signals.append(pb_signal)
            print(f"PULLBACK SIGNAL! Entry={pb_signal['entry_price']} SL={pb_signal['stop_loss']} Target={pb_signal['target']}")
            continue

        print("no signal")

    # Save signals
    signals_file = os.path.join(SRC_DIR, "..", "data", "signals_today.json")
    os.makedirs(os.path.dirname(signals_file), exist_ok=True)
    with open(signals_file, "w") as f:
        json.dump({"date": str(date.today()), "signals": signals}, f, indent=2)

    print(f"\n{'='*60}")
    print(f"Scan complete: {len(signals)} signals found")
    print(f"Signals saved to {signals_file}")
    if signals:
        print(f"\nSignals for today:")
        for s in signals:
            print(f"  [{s['strategy'].upper()}] {s['symbol']} @ {s['entry_price']} | SL={s['stop_loss']} | Target={s['target']} | RSI={s.get('rsi')}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
