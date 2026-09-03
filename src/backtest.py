"""
Backtest Engine — Multi-Session Historical Replay
==================================================

Replays the 3 trading session profiles (A=conservative, B=balanced,
C=aggressive) over historical price data to validate strategies before
going live.

Fetches 2 years of daily OHLCV data for each basket stock via yfinance,
then walks forward day-by-day, generating signals and simulating trades
using each session's parameters.

Output: data/backtest_results.json + a printed comparison table.

Usage:
    # Full backtest (default: 2 years, all sessions, all basket stocks)
    python -m src.backtest

    # Shorter period
    python -m src.backtest --days 365

    # Single session only
    python -m src.backtest --session B

    # Export to CSV
    python -m src.backtest --csv backtest_report.csv

    python -m src.backtest --help

No Dhan API access needed — runs entirely on yfinance data.
"""

import argparse
import os
import sys
import json
import csv
from datetime import datetime, timedelta
from copy import deepcopy

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils import load_json, save_json, setup_logger, now_iso
import yaml

logger = setup_logger(__name__, "backtest.log")

SESSION_CONFIGS = {
    "A": "config/settings_conservative.yaml",
    "B": "config/settings_balanced.yaml",
    "C": "config/settings_aggressive.yaml",
}

SESSION_LABELS = {"A": "Conservative", "B": "Balanced", "C": "Aggressive"}
ALL_SESSIONS = ["A", "B", "C"]


def _load_session_config(session: str) -> dict:
    """Load config for a specific session."""
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    config_path = os.path.join(project_root, SESSION_CONFIGS[session])
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------------------
# Price data fetching
# ---------------------------------------------------------------------------

def fetch_price_history(symbol: str, days: int):
    """Fetch daily OHLCV from yfinance, return as list of dicts."""
    try:
        import yfinance as yf
        import pandas as pd

        ticker = yf.Ticker(f"{symbol}.NS")
        df = ticker.history(period=f"{days}d")

        if len(df) < 200:
            logger.warning(f"  {symbol}: only {len(df)} days of data (need 200+)")
            return []

        df = df.reset_index()
        df.columns = [c.lower().replace(" ", "_") for c in df.columns]
        if "date" in df.columns:
            df["date"] = df["date"].astype(str)
        elif "datetime" in df.columns:
            df["date"] = df["datetime"].astype(str)
            df = df.drop(columns=["datetime"])

        return df.to_dict(orient="records")

    except Exception as e:
        logger.error(f"  {symbol}: fetch failed: {e}")
        return []


# ---------------------------------------------------------------------------
# Indicator computation (same as technicals.py)
# ---------------------------------------------------------------------------

def compute_indicators_at_index(candles: list, i: int, config: dict) -> dict:
    """
    Compute indicators up to candle index i (inclusive).
    Returns a dict with indicator values for candle i.
    """
    if i < 50:
        return {}

    closes = [float(c["close"]) for c in candles[:i + 1]]
    highs = [float(c["high"]) for c in candles[:i + 1]]
    volumes = [float(c["volume"]) for c in candles[:i + 1]]

    swing = config["swing"]
    rsi_period = config["rsi_period"]
    vol_period = config["volume_avg_period"]

    # EMAs
    ema10 = _ema(closes, 10)
    ema20 = _ema(closes, 20)
    ema50 = _ema(closes, config["positional"]["ema_fast"])
    ema200 = _ema(closes, config["positional"]["ema_slow"]) if len(closes) >= 200 else None

    # RSI
    rsi = _rsi(closes, rsi_period)

    # Volume average
    vol_avg = sum(volumes[-vol_period:]) / vol_period if len(volumes) >= vol_period else 0

    # N-day high (excluding today)
    lookback = swing["lookback_days"]
    if i >= lookback:
        high_lookback = max(highs[i - lookback:i])  # exclude today
    else:
        high_lookback = 0

    high_50d = max(highs[max(0, i - 50):i]) if i >= 50 else 0

    return {
        "close": closes[-1],
        "ema10": ema10,
        "ema20": ema20,
        "ema50": ema50,
        "ema200": ema200,
        "rsi": rsi,
        "vol_avg": vol_avg,
        "volume": volumes[-1],
        "high_lookback": high_lookback,
        "high_50d": high_50d,
    }


def _ema(values: list, period: int) -> float:
    """Compute EMA for the latest value in the series."""
    if len(values) < period:
        return values[-1] if values else 0
    multiplier = 2 / (period + 1)
    ema = sum(values[:period]) / period
    for v in values[period:]:
        ema = (v - ema) * multiplier + ema
    return ema


def _rsi(closes: list, period: int) -> float:
    """Compute RSI for the latest value."""
    if len(closes) < period + 1:
        return 50.0

    gains = []
    losses = []
    for i in range(1, len(closes)):
        diff = closes[i] - closes[i - 1]
        gains.append(max(diff, 0))
        losses.append(max(-diff, 0))

    # Use last `period` values
    avg_gain = sum(gains[-period:]) / period
    avg_loss = sum(losses[-period:]) / period

    if avg_loss == 0:
        return 100.0
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


# ---------------------------------------------------------------------------
# Signal generation (per session config)
# ---------------------------------------------------------------------------

def check_entry(ind: dict, config: dict) -> tuple:
    """
    Check entry conditions. Returns (strategy, sl_price, target_price) or (None, 0, 0).
    """
    swing = config["swing"]

    # Swing entry
    breakout = ind["close"] > ind["high_lookback"]
    vol_confirm = ind["volume"] > ind["vol_avg"] * swing["volume_multiplier"]
    not_overbought = swing["rsi_min"] < ind["rsi"] < swing["rsi_max"]

    if breakout and vol_confirm and not_overbought:
        sl = ind["close"] * (1 - swing["stop_loss_pct"])
        target = ind["close"] * (1 + swing["target_pct"])
        return "swing", sl, target

    # Positional entry
    if ind["ema200"] and ind["ema50"] and ind["ema200"]:
        golden_cross = ind["ema50"] > ind["ema200"]
        trend_cont = ind["close"] > ind["high_50d"]
        vol_ok = ind["volume"] > ind["vol_avg"]

        if (golden_cross or (trend_cont and vol_ok)) and ind["rsi"] < 70:
            pos = config["positional"]
            sl = ind["close"] * (1 - pos["stop_loss_pct"])
            target = ind["close"] * (1 + pos["target_pct"])
            return "positional", sl, target

    return None, 0, 0


def check_exit(ind: dict, strategy: str, config: dict) -> str | None:
    """Check exit conditions. Returns reason string or None."""
    if strategy == "swing":
        if ind["close"] < ind["ema10"]:
            return "Close < EMA10"
        if ind["rsi"] > 75:
            return "RSI > 75 (overbought)"
    else:  # positional
        if ind["ema200"] and ind["ema50"] < ind["ema200"]:
            return "Death cross"
        if ind["ema200"] and ind["close"] < ind["ema200"]:
            return "Close < EMA200"
    return None


# ---------------------------------------------------------------------------
# Backtest engine
# ---------------------------------------------------------------------------

def run_backtest_session(session: str, prices: dict, days: int) -> dict:
    """
    Run backtest for a single session profile.

    Returns dict with trades, stats, and equity curve.
    """
    config = _load_session_config(session)["technical"]
    exec_config = _load_session_config(session)["execution"]
    label = SESSION_LABELS[session]

    max_positions = exec_config["max_positions"]
    capital_per_stock = exec_config["capital_per_stock"]

    positions = []  # Open positions
    closed_trades = []  # Completed round-trips
    equity_curve = []
    peak_equity = 0
    max_drawdown = 0

    logger.info(f"  Session {session} ({label}): backtesting...")

    for symbol, stock_data in prices.items():
        candles = stock_data.get("candles", [])
        if len(candles) < 200:
            continue

        held = None  # Only one position per stock at a time

        for i in range(200, len(candles)):
            ind = compute_indicators_at_index(candles, i, config)
            if not ind:
                continue

            date = candles[i].get("date", "")

            if held:
                # Check exit
                reason = check_exit(ind, held["strategy"], config)
                sl_hit = ind["close"] <= held["sl"]
                target_hit = ind["close"] >= held["target"]

                if reason or sl_hit or target_hit:
                    exit_price = ind["close"]
                    pnl = (exit_price - held["entry_price"]) * held["qty"]
                    pnl_pct = (exit_price - held["entry_price"]) / held["entry_price"] * 100

                    exit_reason = "SL hit" if sl_hit else ("Target hit" if target_hit else reason)

                    closed_trades.append({
                        "symbol": symbol,
                        "session": session,
                        "strategy": held["strategy"],
                        "entry_date": held["entry_date"],
                        "exit_date": date,
                        "entry_price": held["entry_price"],
                        "exit_price": round(exit_price, 2),
                        "qty": held["qty"],
                        "pnl": round(pnl, 2),
                        "pnl_pct": round(pnl_pct, 2),
                        "exit_reason": exit_reason,
                        "hold_days": _date_diff(held["entry_date"], date),
                    })
                    held = None
            else:
                # Check entry
                if len([p for p in positions if p["symbol"] != symbol]) >= max_positions:
                    continue

                strategy, sl, target = check_entry(ind, config)
                if strategy:
                    entry_price = ind["close"]
                    qty = int(capital_per_stock / entry_price)
                    if qty > 0:
                        held = {
                            "symbol": symbol,
                            "strategy": strategy,
                            "entry_price": entry_price,
                            "entry_date": date,
                            "qty": qty,
                            "sl": sl,
                            "target": target,
                        }

    # Compute stats
    if closed_trades:
        total_pnl = sum(t["pnl"] for t in closed_trades)
        winners = [t for t in closed_trades if t["pnl"] > 0]
        losers = [t for t in closed_trades if t["pnl"] < 0]
        win_rate = len(winners) / len(closed_trades) * 100
        avg_profit = sum(t["pnl"] for t in winners) / len(winners) if winners else 0
        avg_loss = sum(t["pnl"] for t in losers) / len(losers) if losers else 0
        avg_hold = sum(t["hold_days"] for t in closed_trades) / len(closed_trades)
        max_pnl = max(t["pnl"] for t in closed_trades)
        min_pnl = min(t["pnl"] for t in closed_trades)

        # Profit factor
        gross_profit = sum(t["pnl"] for t in winners)
        gross_loss = abs(sum(t["pnl"] for t in losers))
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")
    else:
        total_pnl = 0
        winners = []
        losers = []
        win_rate = 0
        avg_profit = 0
        avg_loss = 0
        avg_hold = 0
        max_pnl = 0
        min_pnl = 0
        profit_factor = 0

    # Capital deployed
    total_capital = max_positions * capital_per_stock

    results = {
        "session": session,
        "label": label,
        "total_trades": len(closed_trades),
        "winners": len(winners),
        "losers": len(losers),
        "win_rate": round(win_rate, 1),
        "total_pnl": round(total_pnl, 2),
        "avg_profit": round(avg_profit, 2),
        "avg_loss": round(avg_loss, 2),
        "avg_hold_days": round(avg_hold, 1),
        "max_profit": round(max_pnl, 2),
        "max_loss": round(min_pnl, 2),
        "profit_factor": round(profit_factor, 2),
        "return_pct": round(total_pnl / total_capital * 100, 2) if total_capital else 0,
        "total_capital": total_capital,
        "trades": closed_trades,
    }

    logger.info(
        f"  Session {session} ({label}): {len(closed_trades)} trades, "
        f"Win: {win_rate:.1f}%, P&L: Rs.{total_pnl:,.0f}, "
        f"PF: {profit_factor:.2f}, Return: {results['return_pct']:.1f}%"
    )

    return results


def _date_diff(date1: str, date2: str) -> int:
    """Compute difference in days between two date strings."""
    try:
        d1 = datetime.fromisoformat(date1.split(" ")[0])
        d2 = datetime.fromisoformat(date2.split(" ")[0])
        return (d2 - d1).days
    except (ValueError, TypeError):
        return 0


# ---------------------------------------------------------------------------
# Report printing
# ---------------------------------------------------------------------------

def print_comparison(all_results: dict, days: int):
    """Print a formatted comparison table of all sessions."""
    print("\n" + "=" * 90)
    print(f"  BACKTEST RESULTS — {days} days historical")
    print("=" * 90)
    print(f"  {'Metric':<25} {'Conservative (A)':>18} {'Balanced (B)':>18} {'Aggressive (C)':>18}")
    print("  " + "-" * 88)

    sessions = ["A", "B", "C"]
    r = {s: all_results.get(s, {}) for s in sessions}

    metrics = [
        ("Total Trades", "total_trades", "d"),
        ("Winners", "winners", "d"),
        ("Losers", "losers", "d"),
        ("Win Rate (%)", "win_rate", ".1f"),
        ("Total P&L (Rs.)", "total_pnl", ",.0f"),
        ("Avg Profit (Rs.)", "avg_profit", ",.0f"),
        ("Avg Loss (Rs.)", "avg_loss", ",.0f"),
        ("Max Profit (Rs.)", "max_profit", ",.0f"),
        ("Max Loss (Rs.)", "max_loss", ",.0f"),
        ("Profit Factor", "profit_factor", ".2f"),
        ("Avg Hold (days)", "avg_hold_days", ".1f"),
        ("Return (%)", "return_pct", ".1f"),
    ]

    for label, key, fmt in metrics:
        vals = []
        for s in sessions:
            v = r[s].get(key, 0)
            if fmt == "d":
                vals.append(f"{v}")
            elif fmt == ",.0f":
                vals.append(f"Rs.{v:,.0f}")
            else:
                vals.append(f"{v:{fmt}}")
        print(f"  {label:<25} {vals[0]:>18} {vals[1]:>18} {vals[2]:>18}")

    print("=" * 90)

    # Best session
    best = max(sessions, key=lambda s: r[s].get("total_pnl", 0))
    best_return = r[best].get("return_pct", 0)
    print(f"\n  Best session: {best} ({SESSION_LABELS[best]}) "
          f"— Return: {best_return:.1f}%, Win Rate: {r[best].get('win_rate', 0):.1f}%")

    # Best by profit factor
    best_pf = max(sessions, key=lambda s: r[s].get("profit_factor", 0))
    print(f"  Best profit factor: {best_pf} ({SESSION_LABELS[best_pf]}) "
          f"— PF: {r[best_pf].get('profit_factor', 0):.2f}")

    print()


# ---------------------------------------------------------------------------
# CSV export
# ---------------------------------------------------------------------------

def export_csv(all_results: dict, filepath: str):
    """Export all trades to CSV."""
    fields = [
        "session", "symbol", "strategy", "entry_date", "exit_date",
        "entry_price", "exit_price", "qty", "pnl", "pnl_pct",
        "exit_reason", "hold_days",
    ]

    with open(filepath, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for session in ["A", "B", "C"]:
            for trade in all_results.get(session, {}).get("trades", []):
                writer.writerow({k: trade.get(k, "") for k in fields})

    logger.info(f"Exported trades to {filepath}")
    print(f"\n[CSV] Exported to {filepath}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run(days: int = 730, sessions: list = None, csv_path: str = None):
    """Main entry point for the backtester."""
    if sessions is None:
        sessions = ALL_SESSIONS

    logger.info("=" * 60)
    logger.info(f"Backtest Engine — {days} days, sessions: {', '.join(sessions)}")
    logger.info("=" * 60)

    # Load basket
    basket = load_json("basket.json")
    if not basket:
        logger.error("No basket found. Run screener.py first.")
        return

    logger.info(f"Basket: {len(basket)} stocks")

    # Fetch price history (once, shared across sessions)
    prices = {}
    for i, stock in enumerate(basket, 1):
        symbol = stock["symbol"]
        logger.info(f"[{i}/{len(basket)}] Fetching {symbol} ({days} days)...")
        candles = fetch_price_history(symbol, days)
        if candles:
            prices[symbol] = {"symbol": symbol, "candles": candles}
            logger.info(f"  {symbol}: {len(candles)} candles")

    logger.info(f"Price data fetched for {len(prices)}/{len(basket)} stocks")

    # Run backtest for each session
    all_results = {}
    for session in sessions:
        results = run_backtest_session(session, prices, days)
        all_results[session] = results

    # Save results
    save_json(all_results, "backtest_results.json")
    logger.info("Results saved to data/backtest_results.json")

    # Print comparison
    print_comparison(all_results, days)

    # CSV export
    if csv_path:
        export_csv(all_results, csv_path)

    logger.info("=" * 60)
    logger.info("Backtest complete")
    logger.info("=" * 60)

    return all_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Backtest Engine — replays trading strategies over historical data.\n\n"
            "Fetches price history via yfinance, walks forward day-by-day,\n"
            "simulates trades for each session profile, and compares results.\n\n"
            "Examples:\n"
            "  python -m src.backtest                     # 2 years, all sessions\n"
            "  python -m src.backtest --days 365          # 1 year\n"
            "  python -m src.backtest --session B         # balanced only\n"
            "  python -m src.backtest --csv report.csv    # export trades to CSV"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--days", type=int, default=730,
        help="Number of days of historical data to backtest (default: 730 = 2 years).",
    )
    parser.add_argument(
        "--session", choices=["A", "B", "C"],
        help="Backtest only this session (default: all 3).",
    )
    parser.add_argument(
        "--csv", dest="csv_path",
        help="Export all trades to this CSV file.",
    )
    args = parser.parse_args()

    sessions = [args.session] if args.session else None
    run(days=args.days, sessions=sessions, csv_path=args.csv_path)
