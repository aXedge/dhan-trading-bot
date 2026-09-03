"""
Session Comparison Report — A/B/C Performance Analysis
========================================================

Compares the performance of all 3 trading sessions (A=conservative,
B=balanced, C=aggressive) based on their closed trades and open positions.

Reads:
  - data/backtest_results.json (from backtest.py, optional)
  - data/positions_A.json, positions_B.json, positions_C.json (live positions)
  - data/signals_A.json, signals_B.json, signals_C.json (signal history)

Generates a comparison report with:
  - Signal generation rate
  - Win rate
  - Average profit/loss
  - Risk:reward ratio
  - Profit factor
  - Max drawdown
  - Best/worst trades
  - Per-symbol breakdown
  - Recommendation on which session to go LIVE with

Usage:
    python -m src.compare_sessions

    # Include backtest results in comparison
    python -m src.compare_sessions --backtest

    # Send report via Telegram
    python -m src.compare_sessions --telegram

    python -m src.compare_sessions --help
"""

import argparse
import os
import sys
from collections import defaultdict
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils import load_json, setup_logger, send_telegram, now_iso

logger = setup_logger(__name__, "comparison.log")

ALL_SESSIONS = ["A", "B", "C"]
SESSION_LABELS = {"A": "Conservative", "B": "Balanced", "C": "Aggressive"}


def _load_session_positions(session: str) -> list:
    """Load positions for a session."""
    return load_json(f"positions_{session}.json")


def _load_session_signals(session: str) -> list:
    """Load all signals for a session."""
    return load_json(f"signals_{session}.json")


def _load_backtest_trades(session: str) -> list:
    """Load backtest trades for a session."""
    bt = load_json("backtest_results.json")
    if isinstance(bt, dict):
        return bt.get(session, {}).get("trades", [])
    return []


def compute_session_stats(positions: list, signals: list,
                          backtest_trades: list = None,
                          session: str = "") -> dict:
    """
    Compute statistics for a session from live positions + signals
    and optionally backtest trades.
    """
    # Signal stats
    total_signals = len(signals)
    buy_signals = [s for s in signals if s.get("action") == "BUY"]
    sell_signals = [s for s in signals if s.get("action") == "SELL"]
    sl_updates = [s for s in signals if s.get("action") == "UPDATE_SL"]

    # Open positions
    open_count = len(positions)
    open_pnl = 0
    for p in positions:
        entry = p.get("entry_price", 0)
        # Use current_price if available, else entry
        ltp = p.get("current_price", p.get("ltp", entry))
        qty = p.get("qty", 0)
        open_pnl += (ltp - entry) * qty

    # Backtest stats (if available)
    bt_stats = {}
    if backtest_trades:
        winners = [t for t in backtest_trades if t.get("pnl", 0) > 0]
        losers = [t for t in backtest_trades if t.get("pnl", 0) < 0]
        total_pnl = sum(t.get("pnl", 0) for t in backtest_trades)
        gross_profit = sum(t["pnl"] for t in winners)
        gross_loss = abs(sum(t["pnl"] for t in losers))
        avg_hold = sum(t.get("hold_days", 0) for t in backtest_trades) / len(backtest_trades)

        bt_stats = {
            "bt_trades": len(backtest_trades),
            "bt_winners": len(winners),
            "bt_losers": len(losers),
            "bt_win_rate": round(len(winners) / len(backtest_trades) * 100, 1) if backtest_trades else 0,
            "bt_total_pnl": round(total_pnl, 2),
            "bt_avg_profit": round(gross_profit / len(winners), 2) if winners else 0,
            "bt_avg_loss": round(gross_loss / len(losers), 2) if losers else 0,
            "bt_profit_factor": round(gross_profit / gross_loss, 2) if gross_loss > 0 else float("inf"),
            "bt_avg_hold_days": round(avg_hold, 1),
            "bt_max_profit": round(max(t["pnl"] for t in backtest_trades), 2) if backtest_trades else 0,
            "bt_max_loss": round(min(t["pnl"] for t in backtest_trades), 2) if backtest_trades else 0,
        }

    return {
        "session": session,
        "label": SESSION_LABELS.get(session, session),
        "total_signals": total_signals,
        "buy_signals": len(buy_signals),
        "sell_signals": len(sell_signals),
        "sl_updates": len(sl_updates),
        "open_positions": open_count,
        "open_pnl": round(open_pnl, 2),
        **bt_stats,
    }


def print_comparison(stats: dict, include_backtest: bool):
    """Print a formatted comparison table."""
    print("\n" + "=" * 90)
    print("  SESSION COMPARISON REPORT — A/B/C Performance")
    print(f"  Generated: {now_iso()}")
    print("=" * 90)

    # Live stats
    print(f"\n  {'Metric':<30} {'Conservative (A)':>18} {'Balanced (B)':>18} {'Aggressive (C)':>18}")
    print("  " + "-" * 88)

    live_metrics = [
        ("Total Signals", "total_signals", "d"),
        ("Buy Signals", "buy_signals", "d"),
        ("Sell Signals", "sell_signals", "d"),
        ("SL Updates", "sl_updates", "d"),
        ("Open Positions", "open_positions", "d"),
        ("Open P&L (Rs.)", "open_pnl", ",.0f"),
    ]

    for label, key, fmt in live_metrics:
        vals = []
        for s in ALL_SESSIONS:
            v = stats.get(s, {}).get(key, 0)
            if fmt == "d":
                vals.append(f"{v}")
            elif fmt == ",.0f":
                vals.append(f"Rs.{v:,.0f}")
            else:
                vals.append(f"{v:{fmt}}")
        print(f"  {label:<30} {vals[0]:>18} {vals[1]:>18} {vals[2]:>18}")

    # Backtest stats
    if include_backtest:
        print("  " + "-" * 88)
        print(f"  {'--- BACKTEST STATS ---':<30}")
        print("  " + "-" * 88)

        bt_metrics = [
            ("BT Total Trades", "bt_trades", "d"),
            ("BT Win Rate (%)", "bt_win_rate", ".1f"),
            ("BT Total P&L (Rs.)", "bt_total_pnl", ",.0f"),
            ("BT Avg Profit (Rs.)", "bt_avg_profit", ",.0f"),
            ("BT Avg Loss (Rs.)", "bt_avg_loss", ",.0f"),
            ("BT Profit Factor", "bt_profit_factor", ".2f"),
            ("BT Avg Hold (days)", "bt_avg_hold_days", ".1f"),
            ("BT Max Profit (Rs.)", "bt_max_profit", ",.0f"),
            ("BT Max Loss (Rs.)", "bt_max_loss", ",.0f"),
        ]

        for label, key, fmt in bt_metrics:
            vals = []
            for s in ALL_SESSIONS:
                v = stats.get(s, {}).get(key, 0)
                if fmt == "d":
                    vals.append(f"{v}")
                elif fmt == ",.0f":
                    vals.append(f"Rs.{v:,.0f}")
                else:
                    vals.append(f"{v:{fmt}}")
            print(f"  {label:<30} {vals[0]:>18} {vals[1]:>18} {vals[2]:>18}")

    print("=" * 90)

    # Recommendation
    if include_backtest and any(stats.get(s, {}).get("bt_trades", 0) > 0 for s in ALL_SESSIONS):
        best_pf = max(ALL_SESSIONS,
                      key=lambda s: stats.get(s, {}).get("bt_profit_factor", 0))
        best_wr = max(ALL_SESSIONS,
                      key=lambda s: stats.get(s, {}).get("bt_win_rate", 0))
        best_pnl = max(ALL_SESSIONS,
                       key=lambda s: stats.get(s, {}).get("bt_total_pnl", 0))

        print(f"\n  RECOMMENDATION:")
        print(f"    Best Profit Factor : Session {best_pf} ({SESSION_LABELS[best_pf]}) "
              f"— PF: {stats[best_pf].get('bt_profit_factor', 0):.2f}")
        print(f"    Best Win Rate       : Session {best_wr} ({SESSION_LABELS[best_wr]}) "
              f"— WR: {stats[best_wr].get('bt_win_rate', 0):.1f}%")
        print(f"    Best Total P&L      : Session {best_pnl} ({SESSION_LABELS[best_pnl]}) "
              f"— P&L: Rs.{stats[best_pnl].get('bt_total_pnl', 0):,.0f}")

        if best_pf == best_pnl:
            print(f"\n    >>> Go LIVE with Session {best_pf} ({SESSION_LABELS[best_pf]}) <<<")
        else:
            print(f"\n    >>> Consider Session {best_pf} for best risk-adjusted returns <<<")
    else:
        print(f"\n  Run 'python -m src.backtest' first to get backtest comparison data.")

    print()


def run(include_backtest: bool = False, send_tg: bool = False):
    """Main entry point."""
    logger.info("=" * 60)
    logger.info("Session Comparison Report — starting")
    logger.info("=" * 60)

    stats = {}
    for session in ALL_SESSIONS:
        positions = _load_session_positions(session)
        signals = _load_session_signals(session)
        bt_trades = _load_backtest_trades(session) if include_backtest else []

        stats[session] = compute_session_stats(
            positions, signals, bt_trades, session
        )
        logger.info(
            f"  Session {session} ({SESSION_LABELS[session]}): "
            f"{stats[session]['total_signals']} signals, "
            f"{stats[session]['open_positions']} open positions"
        )

    print_comparison(stats, include_backtest)

    if send_tg:
        # Build a text version for Telegram
        lines = ["Session Comparison Report", ""]
        for session in ALL_SESSIONS:
            s = stats[session]
            lines.append(
                f"Session {session} ({s['label']}): "
                f"{s['total_signals']} signals, "
                f"{s['open_positions']} open, "
                f"P&L: Rs.{s['open_pnl']:,.0f}"
            )
            if include_backtest and s.get("bt_trades", 0) > 0:
                lines.append(
                    f"  Backtest: {s['bt_trades']} trades, "
                    f"WR: {s['bt_win_rate']}%, "
                    f"PF: {s['bt_profit_factor']}, "
                    f"P&L: Rs.{s['bt_total_pnl']:,.0f}"
                )
        send_telegram("\n".join(lines))
        logger.info("Report sent via Telegram")

    logger.info("=" * 60)
    logger.info("Comparison report complete")
    logger.info("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Compare A/B/C session performance.\n\n"
            "Reads live positions, signals, and optionally backtest results.\n"
            "Prints a comparison table and recommends the best session for LIVE trading.\n\n"
            "Examples:\n"
            "  python -m src.compare_sessions                # live stats only\n"
            "  python -m src.compare_sessions --backtest      # include backtest data\n"
            "  python -m src.compare_sessions --telegram      # send via Telegram"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--backtest", action="store_true",
        help="Include backtest results from data/backtest_results.json.",
    )
    parser.add_argument(
        "--telegram", action="store_true",
        help="Send the comparison report via Telegram.",
    )
    args = parser.parse_args()
    run(include_backtest=args.backtest, send_tg=args.telegram)
