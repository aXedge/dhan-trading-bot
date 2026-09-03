"""
Layer 3b: Stop-Loss Monitor (multi-session, single-pass)

Checks ALL sessions' open positions in a single pass — fetches LTP once
per stock, then checks against all 3 sessions' positions for SL/target hits.

Single-pass: runs and exits. No infinite loop. Perfect for cron.

Usage:
    # Check all sessions (default — for cron)
    python -m src.sl_monitor

    # Check specific session only
    python -m src.sl_monitor --session A

    # Loop mode (interactive, Ctrl+C to stop)
    python -m src.sl_monitor --loop --interval 60

    python -m src.sl_monitor --help
"""

import argparse
import os
import sys
import time
from datetime import datetime

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils import (
    load_config, load_json, save_json, setup_logger,
    send_telegram, is_market_open, now_iso,
)

logger = setup_logger(__name__, "sl_monitor.log")

ALL_SESSIONS = ["A", "B", "C", "T"]
SESSION_CONFIGS = {
    "A": "config/settings_conservative.yaml",
    "B": "config/settings_balanced.yaml",
    "C": "config/settings_aggressive.yaml",
}


def _load_session_config(session: str) -> dict:
    """Load config for a specific session without changing global env vars."""
    import yaml
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    config_path = os.path.join(project_root, SESSION_CONFIGS[session])
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def run_once(sessions: list = None):
    """
    Check all sessions' positions once. Fetches LTP per stock only once,
    then checks against all sessions that hold that stock.

    Args:
        sessions: List of session IDs to check (default: all 3)
    """
    if sessions is None:
        sessions = ALL_SESSIONS

    from auth import get_dhan
    from executor import get_ltp, execute_sell, _init_constants

    # Collect all positions across sessions
    all_positions = {}  # {session: [positions]}
    total_open = 0

    for session in sessions:
        positions = load_json(f"positions_{session}.json")
        all_positions[session] = positions
        total_open += len(positions)

    if total_open == 0:
        logger.info(f"No open positions across {len(sessions)} sessions")
        return

    logger.info(f"Checking {total_open} positions across {len(sessions)} sessions...")

    # Get Dhan client (single auth for all sessions)
    dhan = get_dhan()
    _init_constants(dhan)

    # Build a map of unique security_ids to fetch LTP only once per stock
    sec_id_to_sessions = {}  # {sec_id: [(session, position), ...]}
    for session, positions in all_positions.items():
        for pos in positions:
            sid = pos["security_id"]
            sec_id_to_sessions.setdefault(sid, []).append((session, pos))

    logger.info(f"Fetching LTP for {len(sec_id_to_sessions)} unique stocks...")

    # Fetch LTP once per unique stock
    ltp_cache = {}  # {sec_id: ltp}
    for sec_id in sec_id_to_sessions:
        ltp = get_ltp(dhan, sec_id)
        ltp_cache[sec_id] = ltp
        if not ltp:
            symbol = sec_id_to_sessions[sec_id][0][1]["symbol"]
            logger.warning(f"Cannot fetch LTP for {symbol} — skipping")

    # Check each position against its SL/target
    for session in sessions:
        positions = all_positions[session]
        if not positions:
            continue

        session_config = _load_session_config(session)
        session_label = {"A": "Conservative", "B": "Balanced", "C": "Aggressive"}[session]

        updated = False
        for pos in positions:
            symbol = pos["symbol"]
            sec_id = pos["security_id"]
            entry = pos["entry_price"]
            sl = pos.get("sl")
            target = pos.get("target")
            strategy = pos.get("strategy", "swing")

            # Get SL/target from config if not in position
            if sl is None:
                sl_pct = session_config["technical"][strategy]["stop_loss_pct"]
                sl = entry * (1 - sl_pct)
            if target is None:
                target_pct = session_config["technical"][strategy]["target_pct"]
                target = entry * (1 + target_pct)

            ltp = ltp_cache.get(sec_id)
            if not ltp:
                continue

            pnl = (ltp - entry) * pos["qty"]
            pnl_pct = (ltp - entry) / entry * 100

            # Check stop-loss hit
            if ltp <= sl:
                logger.warning(
                    f"  [{session}] SL HIT: {symbol} @ Rs.{ltp:.2f} "
                    f"(SL: Rs.{sl:.2f}, P&L: Rs.{pnl:+.2f})"
                )
                send_telegram(
                    f"SL HIT: {symbol} [Session {session}]\n"
                    f"LTP: Rs.{ltp:.2f} | SL: Rs.{sl:.2f}\n"
                    f"P&L: Rs.{pnl:+.2f} ({pnl_pct:+.1f}%)\n"
                    f"Selling {pos['qty']} shares"
                )
                success = execute_sell(dhan, symbol, sec_id, pos["qty"],
                                        f"SL hit @ {ltp}", session)
                if success:
                    pos["_closed"] = True
                    updated = True

            # Check target hit
            elif ltp >= target:
                logger.info(
                    f"  [{session}] TARGET HIT: {symbol} @ Rs.{ltp:.2f} "
                    f"(Target: Rs.{target:.2f}, P&L: Rs.{pnl:+.2f})"
                )
                send_telegram(
                    f"TARGET HIT: {symbol} [Session {session}]\n"
                    f"LTP: Rs.{ltp:.2f} | Target: Rs.{target:.2f}\n"
                    f"P&L: Rs.{pnl:+.2f} ({pnl_pct:+.1f}%)\n"
                    f"Selling {pos['qty']} shares"
                )
                success = execute_sell(dhan, symbol, sec_id, pos["qty"],
                                        f"Target hit @ {ltp}", session)
                if success:
                    pos["_closed"] = True
                    updated = True

            else:
                logger.info(
                    f"  [{session}] {symbol:15s} LTP:Rs.{ltp:.2f} "
                    f"Entry:Rs.{entry:.2f} SL:Rs.{sl:.2f} "
                    f"P&L:Rs.{pnl:+.2f} ({pnl_pct:+.1f}%)"
                )

        # Remove closed positions and save
        if updated:
            positions = [p for p in positions if not p.get("_closed")]
            save_json(positions, f"positions_{session}.json")
            logger.info(f"  [{session}] Updated positions: {len(positions)} active")


def run_loop(interval_sec: int = 600, sessions: list = None):
    """Run the SL monitor in a continuous loop (interactive mode only)."""
    logger.info("=" * 60)
    logger.info(f"SL Monitor (loop mode) — checking every {interval_sec}s")
    if sessions:
        logger.info(f"Sessions: {', '.join(sessions)}")
    logger.info("Press Ctrl+C to stop")
    logger.info("=" * 60)

    try:
        while True:
            if is_market_open():
                try:
                    run_once(sessions)
                except Exception as e:
                    logger.error(f"SL check failed: {e}")
            else:
                logger.debug("Market closed — sleeping")
            time.sleep(interval_sec)
    except KeyboardInterrupt:
        logger.info("SL Monitor stopped by user")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Stop-Loss Monitor — checks all sessions' positions in a single pass.\n\n"
            "Default: checks all 3 sessions (A, B, C) and exits (for cron).\n"
            "Loop mode: continuous monitoring (interactive, Ctrl+C to stop).\n\n"
            "Examples:\n"
            "  python -m src.sl_monitor                     # check all, single-pass (cron)\n"
            "  python -m src.sl_monitor --session A         # check session A only\n"
            "  python -m src.sl_monitor --loop --interval 60  # interactive, every 60s"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--session", choices=["A", "B", "C"],
        help="Check only this session (default: all sessions)",
    )
    parser.add_argument(
        "--loop", action="store_true",
        help="Run in continuous loop mode (interactive, NOT for cron)",
    )
    parser.add_argument(
        "--interval", type=int, default=600,
        help="Check interval in seconds for loop mode (default: 600)",
    )
    args = parser.parse_args()

    sessions = [args.session] if args.session else None

    if args.loop:
        run_loop(interval_sec=args.interval, sessions=sessions)
    else:
        logger.info("=" * 60)
        logger.info("SL Monitor — single-pass mode")
        if sessions:
            logger.info(f"Sessions: {', '.join(sessions)}")
        else:
            logger.info("Sessions: all (A, B, C)")
        logger.info("=" * 60)

        if is_market_open():
            run_once(sessions)
        else:
            logger.info("Market closed — nothing to do")

        logger.info("SL Monitor — done, exiting")
        logger.info("=" * 60)
