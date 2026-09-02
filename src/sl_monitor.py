"""
Layer 3b: Stop-Loss Monitor
===========================

Single-pass script — checks all open positions once for stop-loss or target hits,
then exits immediately. Designed to be called by cron every 10 minutes during
market hours.

Does NOT run an infinite loop — each invocation is a single check.
This prevents memory accumulation from overlapping processes on the e2-micro VM.

Usage:
    # Default single check
    python -m src.sl_monitor

    # Check every 60 seconds (interactive mode — Ctrl+C to stop)
    python -m src.sl_monitor --loop --interval 60

    # Show help
    python -m src.sl_monitor --help
"""

import argparse
import os
import sys
import time
from datetime import datetime

# Add project root (parent of src/) to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils import load_config, load_json, save_json, setup_logger, send_telegram, is_market_open

logger = setup_logger(__name__, "sl_monitor.log")


def run_once():
    """
    Check all positions once. If SL or target hit, execute sell.

    Call this function from cron every 10 minutes during market hours.
    """
    from auth import get_dhan
    from executor import get_ltp, execute_sell, _init_constants

    config = load_config()["execution"]
    positions = load_json("positions.json")

    if not positions:
        logger.info("No open positions to monitor")
        return

    dhan = get_dhan()
    if not _init_constants.__wrapped__:
        _init_constants(dhan)

    logger.info(f"Checking {len(positions)} positions for SL/target...")

    updated = False
    for pos in positions:
        symbol = pos["symbol"]
        sec_id = pos["security_id"]
        entry = pos["entry_price"]
        sl = pos.get("sl")
        target = pos.get("target")

        ltp = get_ltp(dhan, sec_id)
        if not ltp:
            logger.warning(f"Cannot fetch LTP for {symbol} — skipping")
            continue

        pnl = (ltp - entry) * pos["qty"]
        pnl_pct = (ltp - entry) / entry * 100

        # Check stop-loss hit
        if sl and ltp <= sl:
            logger.warning(f"SL HIT: {symbol} @ Rs.{ltp:.2f} (SL: Rs.{sl:.2f}, P&L: Rs.{pnl:+.2f})")
            send_telegram(
                f"STOP LOSS HIT: {symbol}\n"
                f"LTP: Rs.{ltp:.2f} | SL: Rs.{sl:.2f}\n"
                f"P&L: Rs.{pnl:+.2f} ({pnl_pct:+.1f}%)\n"
                f"Selling {pos['qty']} shares"
            )
            success = execute_sell(dhan, symbol, sec_id, pos["qty"], f"SL hit @ {ltp}")
            if success:
                pos["_closed"] = True
                updated = True

        # Check target hit
        elif target and ltp >= target:
            logger.info(f"TARGET HIT: {symbol} @ Rs.{ltp:.2f} (Target: Rs.{target:.2f}, P&L: Rs.{pnl:+.2f})")
            send_telegram(
                f"TARGET HIT: {symbol}\n"
                f"LTP: Rs.{ltp:.2f} | Target: Rs.{target:.2f}\n"
                f"P&L: Rs.{pnl:+.2f} ({pnl_pct:+.1f}%)\n"
                f"Selling {pos['qty']} shares"
            )
            success = execute_sell(dhan, symbol, sec_id, pos["qty"], f"Target hit @ {ltp}")
            if success:
                pos["_closed"] = True
                updated = True

        else:
            logger.info(
                f"  {symbol:15s} LTP:Rs.{ltp:.2f} Entry:Rs.{entry:.2f} "
                f"SL:Rs.{sl or 'N/A'} P&L:Rs.{pnl:+.2f} ({pnl_pct:+.1f}%)"
            )

    # Remove closed positions
    if updated:
        positions = [p for p in positions if not p.get("_closed")]
        save_json(positions, "positions.json")
        logger.info(f"Updated positions: {len(positions)} active")


def run_loop(interval_sec: int = 600):
    """
    Run the SL monitor in a continuous loop (interactive mode only).

    NOT recommended for cron — use run_once() from cron instead.
    This is for manual monitoring when you want to watch positions live.

    Args:
        interval_sec: Sleep between checks (default 600 = 10 minutes)
    """
    logger.info("=" * 60)
    logger.info(f"SL Monitor started (loop mode) — checking every {interval_sec}s")
    logger.info("Press Ctrl+C to stop")
    logger.info("=" * 60)

    try:
        while True:
            if is_market_open():
                try:
                    run_once()
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
            "Stop-Loss Monitor — checks open positions for SL/target hits.\n\n"
            "Default mode: single check and exit (for cron).\n"
            "Loop mode: continuous monitoring (interactive, Ctrl+C to stop).\n\n"
            "Examples:\n"
            "  python -m src.sl_monitor                  # single check (cron mode)\n"
            "  python -m src.sl_monitor --loop           # continuous monitoring\n"
            "  python -m src.sl_monitor --loop --interval 60  # check every 60s"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--loop", action="store_true",
        help="Run in continuous loop mode (interactive, NOT for cron). "
             "Default is single-check mode for cron.",
    )
    parser.add_argument(
        "--interval", type=int, default=600,
        help="Check interval in seconds for loop mode (default: 600 = 10 min).",
    )
    args = parser.parse_args()

    if args.loop:
        run_loop(interval_sec=args.interval)
    else:
        logger.info("=" * 60)
        logger.info("SL Monitor — single check mode")
        logger.info("=" * 60)

        if is_market_open():
            run_once()
        else:
            logger.info("Market closed — nothing to do")

        logger.info("SL Monitor — done, exiting")
        logger.info("=" * 60)
