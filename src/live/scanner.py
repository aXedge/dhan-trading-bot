#!/usr/bin/env python3
"""
Live Scanner — Daily Signal Detection
=======================================

Scans the curated basket for reversal and pullback entry signals.
Outputs signals to data/signals_today.json for the executor to process.

Single-pass script — designed for cron. Runs at 3:00 PM IST.

Usage:
    python src/live/scanner.py
"""

import json
import os
import sys
import tempfile
from datetime import date, datetime

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.dirname(SCRIPT_DIR)
REPO_ROOT = os.path.dirname(SRC_DIR)
sys.path.insert(0, SRC_DIR)
sys.path.insert(0, REPO_ROOT)

import pandas as pd
import yfinance as yf
import warnings
warnings.filterwarnings('ignore')

import strategies.reversal as reversal
import strategies.positional_pullback as pullback

# ========================================
# CONFIG
# ========================================
SIGNALS_FILE = os.path.join(REPO_ROOT, "data", "signals_today.json")
BASKET_FILE = os.path.join(REPO_ROOT, "data", "basket_curated.json")

# Fallback basket (used if basket_curated.json doesn't exist)
FALLBACK_BASKET = [
    "EICHERMOT", "BHARTIARTL", "INFY", "ADANIPORTS", "HDFCBANK", "HCLTECH",
    "CIPLA", "WIPRO", "TECHM", "TITAN", "ULTRACEMCO", "SUNPHARMA",
    "ICICIBANK", "SBIN", "LT", "BAJFINANCE", "KOTAKBANK", "AXISBANK",
    "TATACONSUM", "GRASIM", "ADANIENT", "JSWSTEEL", "BRITANNIA", "DRREDDY",
    "HDFCLIFE", "BAJAJFINSV", "DIVISLAB", "HEROMOTOCO", "TATASTEEL", "ITC",
    "HINDUNILVR", "ASIANPAINT", "BPCL", "COALINDIA", "NTPC", "POWERGRID",
]


def load_basket():
    """Load curated basket from JSON, fall back to hardcoded list."""
    if os.path.exists(BASKET_FILE):
        try:
            with open(BASKET_FILE) as f:
                data = json.load(f)
            stocks = data.get("stocks", [])
            if stocks:
                return stocks
        except Exception:
            pass
    return FALLBACK_BASKET


def fetch_yfinance(symbol, period="6mo"):
    """Fetch daily price data from yfinance."""
    ticker = symbol + ".NS" if not symbol.endswith(".NS") else symbol
    df = yf.download(ticker, period=period, auto_adjust=True, progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df


def scan_stock(symbol, reversal_config, pullback_config):
    """Scan a single stock for both reversal and pullback signals."""
    df = fetch_yfinance(symbol)
    if len(df) < 60:
        return None

    # Check reversal signal
    try:
        rev_df = reversal.prepare(df, reversal_config)
        last = rev_df.iloc[-1]
        if reversal.should_enter(last, reversal_config):
            entry_price = float(last["Close"])
            sl_pct = reversal_config["sl_pct"]
            target_pct = reversal_config["target_pct"]
            rsi = float(last.get("RSI", 0))
            return {
                "strategy": "reversal",
                "symbol": symbol,
                "entry_price": round(entry_price, 2),
                "stop_loss": round(entry_price * (1 - sl_pct / 100), 2),
                "target": round(entry_price * (1 + target_pct / 100), 2),
                "stop_loss_pct": sl_pct,
                "target_pct": target_pct,
                "rsi": round(rsi, 1),
                "date": str(date.today()),
            }
    except Exception:
        pass

    # Check pullback signal
    try:
        pb_df = pullback.prepare(df, pullback_config)
        last = pb_df.iloc[-1]
        if pullback.should_enter(last, pullback_config):
            entry_price = float(last["Close"])
            sl_pct = pullback_config["sl_pct"]
            target_pct = pullback_config["target_pct"]
            rsi = float(last.get("RSI", 0))
            return {
                "strategy": "pullback",
                "symbol": symbol,
                "entry_price": round(entry_price, 2),
                "stop_loss": round(entry_price * (1 - sl_pct / 100), 2),
                "target": round(entry_price * (1 + target_pct / 100), 2),
                "stop_loss_pct": sl_pct,
                "target_pct": target_pct,
                "rsi": round(rsi, 1),
                "date": str(date.today()),
            }
    except Exception:
        pass

    return None


def main():
    print(f"\n{'='*60}")
    print(f"Live Scanner — {date.today()}")
    print(f"{'='*60}")

    basket = load_basket()
    print(f"Scanning {len(basket)} stocks...\n")

    reversal_config = reversal.DEFAULT_CONFIG.copy()
    pullback_config = pullback.DEFAULT_CONFIG.copy()

    signals = []
    import time

    for i, sym in enumerate(basket):
        print(f"  Scanning {i+1}/{len(basket)}: {sym}...", end="", flush=True)
        signal = scan_stock(sym, reversal_config, pullback_config)

        if signal:
            signals.append(signal)
            print(f" {signal['strategy'].upper()} SIGNAL! "
                  f"Entry={signal['entry_price']} SL={signal['stop_loss']} "
                  f"Target={signal['target']}")
        else:
            print(" no signal")

        # Rate limit yfinance
        time.sleep(0.5)

    # Save signals (atomic write)
    output = {
        "date": str(date.today()),
        "scan_time": datetime.now().isoformat(),
        "basket_size": len(basket),
        "signals_count": len(signals),
        "signals": signals,
    }

    os.makedirs(os.path.dirname(SIGNALS_FILE), exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(dir=os.path.dirname(SIGNALS_FILE), suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(output, f, indent=2, default=str)
        os.replace(tmp_path, SIGNALS_FILE)
    except:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        raise

    print(f"\n{'='*60}")
    print(f"Scan complete: {len(signals)} signals found")
    print(f"Signals saved to {SIGNALS_FILE}")

    if signals:
        print(f"\n# Signals for today:")
        for s in signals:
            print(f"  [{s['strategy'].upper()}] {s['symbol']} @ {s['entry_price']} "
                  f"| SL={s['stop_loss']} | Target={s['target']} | RSI={s['rsi']}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
