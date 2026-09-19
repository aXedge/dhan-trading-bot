#!/usr/bin/env python3
"""
Paper Trading Performance Report
=================================

Single-pass script that reads paper_positions.json and reports:
  - Closed trade performance (win rate, PF, total P&L, per-strategy breakdown)
  - Open positions with live unrealized P&L
  - Today's signals

Usage:
    python src/live/performance_report.py
"""

import json
import os
import sys
import time
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

PAPER_POSITIONS_FILE = os.path.join(REPO_ROOT, "data", "paper_positions.json")
SIGNALS_FILE = os.path.join(REPO_ROOT, "data", "signals_today.json")


def fetch_current_price(symbol):
    """Fetch last close with retries; returns None on failure (never NaN)."""
    ticker = symbol + ".NS" if not symbol.endswith(".NS") else symbol
    for attempt in range(3):
        try:
            df = yf.download(ticker, period="5d", auto_adjust=True, progress=False)
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            closes = df["Close"].dropna()  # skip NaN bars (yfinance glitch)
            if len(closes) > 0:
                return float(closes.iloc[-1])
        except Exception:
            pass
        time.sleep(2)
    return None


def main():
    print(f"\n{'='*70}")
    print(f"PAPER TRADING PERFORMANCE REPORT — {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print(f"{'='*70}")

    # Load data
    if not os.path.exists(PAPER_POSITIONS_FILE):
        print("No paper_positions.json found — no trading has happened yet.")
        return

    with open(PAPER_POSITIONS_FILE) as f:
        data = json.load(f)

    positions = data.get("positions", [])
    closed = data.get("closed_trades", [])

    # ========================================
    # 1. CLOSED TRADES
    # ========================================
    print(f"\n1. CLOSED TRADES")
    print(f"{'-'*70}")

    if not closed:
        print("   No closed trades yet.")
    else:
        pnls = [c.get("pnl_rs", 0) for c in closed]
        wins = [p for p in pnls if p > 0]
        losses = [p for p in pnls if p <= 0]
        total = sum(pnls)

        print(f"   Total closed trades: {len(closed)}")
        print(f"   Wins: {len(wins)} | Losses: {len(losses)}")
        print(f"   Win rate: {len(wins)/len(closed)*100:.1f}%")
        if wins:
            print(f"   Avg win:  Rs.{sum(wins)/len(wins):,.0f}")
        if losses:
            print(f"   Avg loss: Rs.{sum(losses)/len(losses):,.0f}")
        if wins and losses and sum(losses) != 0:
            print(f"   Profit factor: {abs(sum(wins)/sum(losses)):.2f}")
        print(f"   Total realized P&L: Rs.{total:,.0f}")

        # Per-strategy breakdown
        by_strategy = {}
        for c in closed:
            strat = c.get("strategy", "unknown")
            by_strategy.setdefault(strat, []).append(c.get("pnl_rs", 0))

        print(f"\n   By strategy:")
        for strat, pnl_list in by_strategy.items():
            s_wins = [p for p in pnl_list if p > 0]
            s_losses = [p for p in pnl_list if p <= 0]
            pf = abs(sum(s_wins)/sum(s_losses)) if s_wins and s_losses and sum(s_losses) != 0 else float('inf')
            print(f"     {strat:12s}: {len(pnl_list):3d} trades | "
                  f"win rate {len(s_wins)/len(pnl_list)*100:5.1f}% | "
                  f"PF {pf:5.2f} | P&L Rs.{sum(pnl_list):,.0f}")

        # Trade log
        print(f"\n   Trade log:")
        print(f"   {'Symbol':14s} {'Strat':10s} {'Entry':>9s} {'Exit':>9s} {'Reason':10s} {'P&L%':>7s} {'P&L Rs':>10s}")
        print(f"   {'-'*75}")
        for c in sorted(closed, key=lambda x: x.get("exit_date", "")):
            print(f"   {c.get('symbol','?'):14s} {c.get('strategy','?'):10s} "
                  f"{c.get('entry_price',0):9.2f} {c.get('exit_price',0):9.2f} "
                  f"{c.get('exit_reason','?'):10s} {c.get('pnl_pct',0):7.1f}% "
                  f"{c.get('pnl_rs',0):10,.0f}")

    # ========================================
    # 2. OPEN POSITIONS (with live P&L)
    # ========================================
    print(f"\n2. OPEN POSITIONS")
    print(f"{'-'*70}")

    if not positions:
        print("   No open positions.")
    else:
        print(f"   {len(positions)} open position(s), fetching live prices...\n")
        total_unrealized = 0
        print(f"   {'Symbol':14s} {'Strat':10s} {'Entry':>9s} {'SL':>9s} {'Target':>9s} {'Now':>9s} {'P&L%':>7s} {'P&L Rs':>10s}")
        print(f"   {'-'*85}")
        for p in positions:
            sym = p.get("symbol", "?")
            price = fetch_current_price(sym)
            entry = p.get("entry_price", 0)
            qty = p.get("quantity", 0)
            if price is not None and price == price:  # present and not NaN
                pnl_pct = (price - entry) / entry * 100
                pnl_rs = (price - entry) * qty
                total_unrealized += pnl_rs
                print(f"   {sym:14s} {p.get('strategy','?'):10s} {entry:9.2f} "
                      f"{p.get('stop_loss',0):9.2f} {p.get('target',0):9.2f} "
                      f"{price:9.2f} {pnl_pct:7.1f}% {pnl_rs:10,.0f}")
            else:
                print(f"   {sym:14s} {p.get('strategy','?'):10s} {entry:9.2f} "
                      f"{p.get('stop_loss',0):9.2f} {p.get('target',0):9.2f} "
                      f"{'N/A':>9s} {'?':>7s} {'?':>10s} (price fetch failed)")
            time.sleep(0.3)
        print(f"   {'-'*85}")
        print(f"   Total unrealized P&L: Rs.{total_unrealized:,.0f}")

    # ========================================
    # 3. TODAY'S SIGNALS
    # ========================================
    print(f"\n3. TODAY'S SIGNALS")
    print(f"{'-'*70}")

    if os.path.exists(SIGNALS_FILE):
        with open(SIGNALS_FILE) as f:
            sig = json.load(f)
        print(f"   Signal file date: {sig.get('date', '?')} | "
              f"scanned: {sig.get('basket_size', '?')} stocks | "
              f"signals: {sig.get('signals_count', 0)}")
        for s in sig.get("signals", []):
            print(f"   [{s.get('strategy','?').upper()}] {s.get('symbol','?')} "
                  f"@ {s.get('entry_price','?')} | SL={s.get('stop_loss','?')} "
                  f"| Target={s.get('target','?')} | RSI={s.get('rsi','?')}")
    else:
        print("   No signals file found.")

    # ========================================
    # 4. SUMMARY
    # ========================================
    print(f"\n4. SUMMARY")
    print(f"{'-'*70}")
    realized = sum(c.get("pnl_rs", 0) for c in closed)
    print(f"   Realized P&L:   Rs.{realized:,.0f} ({len(closed)} closed trades)")
    if positions:
        print(f"   Unrealized P&L: Rs.{total_unrealized:,.0f} ({len(positions)} open)")
        print(f"   Total P&L:      Rs.{realized + total_unrealized:,.0f}")
    else:
        print(f"   Total P&L:      Rs.{realized:,.0f}")

    print(f"\n{'='*70}")
    print(f"Backtest expectation for reference:")
    print(f"   Reversal: PF 1.28, 67% win | Pullback: PF 1.05, 46% win")
    print(f"   (~1-2 entries/week across 30 stocks — patience required)")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
