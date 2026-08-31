"""
Layer 3b: Stop-Loss Monitor
==============================
Runs every 5 minutes during market hours to check if any held position
has hit its stop-loss or target price. If so, fires a SELL order.

Requires Dhan API access — run on VPS with whitelisted static IP.
"""

import time
from utils import load_config, load_json, save_json, setup_logger, now_iso, send_telegram, is_market_open

logger = setup_logger(__name__, "sl_monitor.log")


def run_once():
    """
    Check all positions once. If SL or target hit, execute sell.

    Call this function from a loop or cron every 5 minutes during market hours.
    """
    from auth import get_dhan
    from executor import get_ltp, execute_sell, _init_constants

    config = load_config()["execution"]
    positions = load_json("positions.json")

    if not positions:
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
            logger.warning(f"🚨 SL HIT: {symbol} @ ₹{ltp:.2f} (SL: ₹{sl:.2f}, P&L: ₹{pnl:+.2f})")
            send_telegram(
                f"🚨 <b>STOP LOSS HIT: {symbol}</b>\n"
                f"LTP: ₹{ltp:.2f} | SL: ₹{sl:.2f}\n"
                f"P&L: ₹{pnl:+.2f} ({pnl_pct:+.1f}%)\n"
                f"Selling {pos['qty']} shares"
            )
            success = execute_sell(dhan, symbol, sec_id, pos["qty"], f"SL hit @ {ltp}")
            if success:
                pos["_closed"] = True
                updated = True

        # Check target hit
        elif target and ltp >= target:
            logger.info(f"✅ TARGET HIT: {symbol} @ ₹{ltp:.2f} (Target: ₹{target:.2f}, P&L: ₹{pnl:+.2f})")
            send_telegram(
                f"✅ <b>TARGET HIT: {symbol}</b>\n"
                f"LTP: ₹{ltp:.2f} | Target: ₹{target:.2f}\n"
                f"P&L: ₹{pnl:+.2f} ({pnl_pct:+.1f}%)\n"
                f"Selling {pos['qty']} shares"
            )
            success = execute_sell(dhan, symbol, sec_id, pos["qty"], f"Target hit @ {ltp}")
            if success:
                pos["_closed"] = True
                updated = True

        else:
            logger.info(
                f"  {symbol:15s} LTP:₹{ltp:.2f} Entry:₹{entry:.2f} "
                f"SL:₹{sl or 'N/A'} P&L:₹{pnl:+.2f} ({pnl_pct:+.1f}%)"
            )

    # Remove closed positions
    if updated:
        positions = [p for p in positions if not p.get("_closed")]
        save_json(positions, "positions.json")
        logger.info(f"Updated positions: {len(positions)} active")


def run_loop(interval_sec: int = 300):
    """
    Run the SL monitor in a continuous loop.

    Args:
        interval_sec: Sleep between checks (default 300 = 5 minutes)
    """
    logger.info("=" * 60)
    logger.info(f"SL Monitor started — checking every {interval_sec}s during market hours")
    logger.info("=" * 60)

    while True:
        if is_market_open():
            try:
                run_once()
            except Exception as e:
                logger.error(f"SL check failed: {e}")
        else:
            logger.debug("Market closed — sleeping")

        time.sleep(interval_sec)


if __name__ == "__main__":
    config = load_config()["execution"]
    interval = config.get("sl_check_interval_sec", 300)
    run_loop(interval)
