"""
Reporting — metrics calculation, comparison tables, per-stock breakdowns.

Shared by all backtests so results are always formatted consistently.
"""

import numpy as np
from collections import defaultdict


def calc_metrics(trades: list) -> dict:
    """
    Calculate key metrics from a list of trades.

    Returns dict with: trades, wins, losses, win_rate, profit_factor,
    pnl, avg_win, avg_loss
    """
    if not trades:
        return {
            "trades": 0, "wins": 0, "losses": 0,
            "win_rate": 0, "pf": 0, "pnl": 0,
            "avg_win": 0, "avg_loss": 0,
        }

    wins = [t for t in trades if t["pnl_pct"] > 0]
    losses = [t for t in trades if t["pnl_pct"] <= 0]
    gp = sum(t["pnl_pct"] for t in wins)
    gl = abs(sum(t["pnl_pct"] for t in losses))

    return {
        "trades": len(trades),
        "wins": len(wins),
        "losses": len(losses),
        "win_rate": len(wins) / len(trades) * 100 if trades else 0,
        "pf": gp / gl if gl > 0 else float("inf"),
        "pnl": sum(t["pnl_pct"] for t in trades),
        "avg_win": np.mean([t["pnl_pct"] for t in wins]) if wins else 0,
        "avg_loss": np.mean([t["pnl_pct"] for t in losses]) if losses else 0,
    }


def print_exit_reasons(trades: list, indent: str = "  "):
    """Print exit reason breakdown."""
    if not trades:
        return
    reasons = defaultdict(int)
    for t in trades:
        reasons[t["exit_reason"]] += 1
    for reason, count in sorted(reasons.items(), key=lambda x: -x[1]):
        pct = count / len(trades) * 100
        print(f"{indent}{reason:25s} {count:3d} ({pct:.0f}%)")


def print_per_stock(trades: list, indent: str = "  "):
    """Print per-stock breakdown."""
    if not trades:
        return
    per_stock = defaultdict(lambda: {"trades": 0, "wins": 0, "pnl": 0})
    for t in trades:
        per_stock[t["symbol"]]["trades"] += 1
        if t["pnl_pct"] > 0:
            per_stock[t["symbol"]]["wins"] += 1
        per_stock[t["symbol"]]["pnl"] += t["pnl_pct"]

    print(f"{indent}{'Symbol':15s} {'Trades':>6s} {'Wins':>5s} {'Win%':>6s} {'P&L%':>8s}")
    print(f"{indent}{'-'*15} {'-'*6} {'-'*5} {'-'*6} {'-'*8}")
    for symbol in sorted(per_stock.keys()):
        s = per_stock[symbol]
        wr = s["wins"] / s["trades"] * 100 if s["trades"] > 0 else 0
        print(f"{indent}{symbol:15s} {s['trades']:6d} {s['wins']:5d} {wr:5.1f}% {s['pnl']*100:7.2f}%")


def print_summary(title: str, trades: list, config: dict = None):
    """Print a full summary for a set of trades."""
    m = calc_metrics(trades)
    print(f"\n{'='*70}")
    print(f"{title}")
    print(f"{'='*70}")
    if config:
        sl = config.get("stop_loss_pct", 0.03)
        target = config.get("target_pct", 0.08)
        print(f"Stop loss:           {sl*100:.1f}%")
        print(f"Target:              {target*100:.1f}%")
    print(f"Total trades:       {m['trades']}")
    print(f"  Wins:              {m['wins']}")
    print(f"  Losses:            {m['losses']}")
    print(f"  Win rate:          {m['win_rate']:.1f}%")
    print(f"Total P&L (sum %):  {m['pnl']*100:.2f}%")
    print(f"Avg win:            {m['avg_win']*100:.2f}%")
    print(f"Avg loss:           {m['avg_loss']*100:.2f}%")
    print(f"Profit factor:      {m['pf']:.2f}")
    if m["trades"] > 0:
        print(f"Avg P&L per trade:  {m['pnl']/m['trades']*100:.2f}%")
    print()
    print_exit_reasons(trades)
    print()
    print_per_stock(trades)
    print(f"{'='*70}")


def print_comparison_table(results: dict, title: str = "COMPARISON"):
    """
    Print a comparison table across multiple configs/strategies.

    Args:
        results: Dict mapping {name: {"metrics": {...}, "trades": [...]}}
        title: Table title
    """
    print(f"\n{'='*70}")
    print(f"{title}")
    print(f"{'='*70}")
    print(f"  {'Config':25s} {'Trades':>7s} {'Win%':>6s} {'PF':>7s} {'P&L%':>8s}")
    print(f"  {'-'*25} {'-'*7} {'-'*6} {'-'*7} {'-'*8}")
    for name, data in results.items():
        m = data["metrics"] if "metrics" in data else data
        print(f"  {name:25s} {m['trades']:7d} {m['win_rate']:5.1f}% "
              f"{m['pf']:7.2f} {m['pnl']*100:7.1f}%")
    print(f"{'='*70}")


def print_verdict(pf: float, trades: int, win_rate: float):
    """Print a verdict based on metrics."""
    print(f"\n{'='*70}")
    print("VERDICT")
    print(f"{'='*70}")
    if pf >= 1.5 and trades >= 30:
        print(f"  PF: {pf:.2f} | Win rate: {win_rate:.1f}% | Trades: {trades}")
        print(f"  STRATEGY VALIDATED — PF >= 1.5 with sufficient trades.")
        print(f"  Safe to paper trade with these parameters.")
    elif pf >= 1.0 and trades >= 30:
        print(f"  PF: {pf:.2f} | Win rate: {win_rate:.1f}% | Trades: {trades}")
        print(f"  MARGINAL — PF >= 1.0 with 30+ trades. Worth paper trading.")
    elif trades < 30:
        print(f"  PF: {pf:.2f} | Win rate: {win_rate:.1f}% | Trades: {trades}")
        print(f"  INSUFFICIENT TRADES — strategy is too selective.")
        print(f"  Try relaxing parameters.")
    else:
        print(f"  PF: {pf:.2f} | Win rate: {win_rate:.1f}% | Trades: {trades}")
        print(f"  STRATEGY FAILED — PF < 1.0.")
    print(f"{'='*70}")
