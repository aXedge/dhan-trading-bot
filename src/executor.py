"""
Layer 3: Dhan Execution Engine (multi-session)

Reads signals_{session}.json and places buy/sell orders via DhanHQ API.
Updates positions_{session}.json with current state.

Usage:
    python -m src.executor --session A    # conservative
    python -m src.executor --session B    # balanced (default)
    python -m src.executor --session C    # aggressive
    python -m src.executor --help

Requires Dhan API access — run on VPS with whitelisted static IP.
"""

import argparse
import os
import sys

# Session → config file mapping
SESSION_CONFIGS = {
    "A": "config/settings_conservative.yaml",
    "B": "config/settings_balanced.yaml",
    "C": "config/settings_aggressive.yaml",
}


def _setup_session(session: str):
    """Set env vars so utils.py loads the right config and data files."""
    config_file = SESSION_CONFIGS.get(session, SESSION_CONFIGS["B"])
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    os.environ["CONFIG_PATH"] = os.path.join(project_root, config_file)


# Parse --session before importing utils
_pre_parser = argparse.ArgumentParser(add_help=False)
_pre_parser.add_argument("--session", default="B", choices=["A", "B", "C"])
_pre_args, _ = _pre_parser.parse_known_args()
_setup_session(_pre_args.session)

from utils import load_config, load_json, save_json, setup_logger, now_iso, send_telegram

logger = setup_logger(__name__, f"executor_{_pre_args.session}.log")

# Dhan exchange segment constants (loaded lazily)
_NSE = None
_CNC = None
_MARKET = None
_SELL = None
_BUY = None


def _init_constants(dhan):
    """Initialize Dhan SDK constants from the dhanhq instance."""
    global _NSE, _CNC, _MARKET, _SELL, _BUY
    _NSE = dhan.NSE
    _CNC = dhan.CNC
    _MARKET = dhan.MARKET
    _SELL = dhan.SELL
    _BUY = dhan.BUY


def load_instrument_master() -> dict:
    """Load the cached Dhan instrument master (security ID mapping)."""
    import json
    from utils import DATA_DIR

    filepath = DATA_DIR / "instrument_master.json"
    if not filepath.exists():
        logger.warning("Instrument master not found. Run scripts/download_instruments.py first.")
        return {}

    with open(filepath, "r") as f:
        instruments = json.load(f)

    mapping = {}
    for inst in instruments:
        symbol = inst.get("SEM_TRADING_SYMBOL", "")
        if symbol and inst.get("SEM_EXM_EXCH_ID") == "NSE":
            mapping[symbol] = inst.get("SEM_SMST_SECURITY_ID", "")

    return mapping


def get_security_id(symbol: str, instrument_map: dict) -> str | None:
    """Map an NSE ticker symbol to Dhan's internal security_id."""
    sec_id = instrument_map.get(symbol)
    if not sec_id:
        logger.warning(f"No security_id found for {symbol} in instrument master")
    return sec_id


def get_ltp(dhan, security_id: str) -> float | None:
    """Get the Last Traded Price for a security."""
    try:
        quote = dhan.market_quote("NSE", security_id)
        data = quote.get("data", {}).get("NSE", {}).get(security_id, {})
        ltp = float(data.get("last_price", 0))
        return ltp if ltp > 0 else None
    except Exception as e:
        logger.error(f"Failed to fetch LTP for {security_id}: {e}")
        return None


def calculate_quantity(ltp: float, capital_per_stock: int) -> int:
    """Calculate the number of shares to buy based on available capital."""
    if ltp <= 0:
        return 0
    return int(capital_per_stock / ltp)


def execute_buy(dhan, symbol: str, security_id: str, strategy: str,
                config: dict, positions: list, session: str) -> dict | None:
    """Execute a BUY order for a stock."""
    if len(positions) >= config["max_positions"]:
        logger.warning(f"Max positions ({config['max_positions']}) reached — skipping BUY {symbol}")
        return None

    ltp = get_ltp(dhan, security_id)
    if not ltp:
        logger.error(f"Cannot get LTP for {symbol} — skipping")
        return None

    qty = calculate_quantity(ltp, config["capital_per_stock"])
    if qty <= 0:
        logger.warning(f"Quantity 0 for {symbol} (LTP {ltp}, capital {config['capital_per_stock']}) — skipping")
        return None

    tag = f"{strategy.upper()}_ENTRY_S{session}"

    try:
        resp = dhan.place_order(
            security_id=security_id,
            exchange_segment=_NSE,
            transaction_type=_BUY,
            quantity=qty,
            order_type=_MARKET,
            product_type=_CNC,
            price=0,
            tag=tag,
        )
        order_id = resp.get("orderId", "unknown") if isinstance(resp, dict) else "unknown"
        logger.info(f"BUY {symbol} x{qty} @ ~Rs.{ltp:.2f} | Order ID: {order_id}")

        send_telegram(
            f"BUY {symbol} [Session {session}]\n"
            f"Strategy: {strategy}\n"
            f"Qty: {qty} @ ~Rs.{ltp:.2f}\n"
            f"Order ID: {order_id}"
        )

        position = {
            "symbol": symbol,
            "security_id": security_id,
            "qty": qty,
            "entry_price": ltp,
            "strategy": strategy,
            "entry_date": now_iso(),
            "order_id": order_id,
            "session": session,
        }
        return position

    except Exception as e:
        logger.error(f"Failed to place BUY for {symbol}: {e}")
        send_telegram(f"BUY FAILED: {symbol} [Session {session}]\nError: {e}")
        return None


def execute_sell(dhan, symbol: str, security_id: str, qty: int, reason: str,
                 session: str) -> bool:
    """Execute a SELL order to square off a position."""
    try:
        resp = dhan.place_order(
            security_id=security_id,
            exchange_segment=_NSE,
            transaction_type=_SELL,
            quantity=qty,
            order_type=_MARKET,
            product_type=_CNC,
            price=0,
            tag=f"EXIT_S{session}",
        )
        order_id = resp.get("orderId", "unknown") if isinstance(resp, dict) else "unknown"
        logger.info(f"SELL {symbol} x{qty} | Reason: {reason} | Order ID: {order_id}")

        send_telegram(
            f"SELL {symbol} [Session {session}]\n"
            f"Qty: {qty}\n"
            f"Reason: {reason}\n"
            f"Order ID: {order_id}"
        )
        return True

    except Exception as e:
        logger.error(f"Failed to place SELL for {symbol}: {e}")
        send_telegram(f"SELL FAILED: {symbol} [Session {session}]\nError: {e}")
        return False


def run(session: str):
    """Main entry point — reads session signals, places orders, updates positions."""
    from auth import get_dhan

    session_label = {"A": "Conservative", "B": "Balanced", "C": "Aggressive"}[session]
    config = load_config()["execution"]

    logger.info("=" * 60)
    logger.info(f"Layer 3: Dhan Execution — Session {session} ({session_label})")
    logger.info("=" * 60)

    # Load signals for this session
    signals = load_json(f"signals_{session}.json")
    if not signals:
        logger.info(f"No signals for session {session}.")
        return

    positions = load_json(f"positions_{session}.json")

    # Auth and init
    dhan = get_dhan()
    _init_constants(dhan)

    # Load instrument master
    instrument_map = load_instrument_master()
    if not instrument_map:
        logger.error("Instrument master missing. Run scripts/download_instruments.py")
        return

    for signal in signals:
        symbol = signal["symbol"]
        action = signal["action"]
        sec_id = get_security_id(symbol, instrument_map)
        if not sec_id:
            continue

        logger.info(f"Processing {action} for {symbol} [Session {session}]...")

        if action == "BUY":
            position = execute_buy(
                dhan, symbol, sec_id, signal["strategy"], config, positions, session
            )
            if position:
                if "sl" in signal:
                    position["sl"] = signal["sl"]
                if "target" in signal:
                    position["target"] = signal["target"]
                positions.append(position)

        elif action == "SELL":
            held = next((p for p in positions if p["symbol"] == symbol), None)
            if held:
                success = execute_sell(dhan, symbol, sec_id, held["qty"],
                                        signal.get("reason", "Signal"), session)
                if success:
                    positions = [p for p in positions if p["symbol"] != symbol]

        elif action == "UPDATE_SL":
            for p in positions:
                if p["symbol"] == symbol:
                    p["sl"] = signal["new_sl"]
                    logger.info(f"Updated SL for {symbol}: Rs.{signal['new_sl']:.2f}")
                    break

    # Save updated positions
    save_json(positions, f"positions_{session}.json")
    logger.info("=" * 60)
    logger.info(f"Session {session} execution complete. Active positions: {len(positions)}")
    for p in positions:
        pnl = 0
        ltp = get_ltp(dhan, p["security_id"])
        if ltp:
            pnl = (ltp - p["entry_price"]) * p["qty"]
        logger.info(
            f"  {p['symbol']:15s} x{p['qty']:4d} @ Rs.{p['entry_price']:.2f} "
            f"({p['strategy']}) SL:Rs.{p.get('sl', 'N/A')} P&L:Rs.{pnl:+.2f}"
        )
    logger.info("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Dhan Execution Engine — places orders for a specific session.\n\n"
            "Sessions:\n"
            "  A = Conservative\n"
            "  B = Balanced (default)\n"
            "  C = Aggressive\n\n"
            "Examples:\n"
            "  python -m src.executor --session A\n"
            "  python -m src.executor --session B\n"
            "  python -m src.executor --session C"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--session", default="B", choices=["A", "B", "C"],
        help="Trading session profile (default: B = balanced)",
    )
    args = parser.parse_args()
    run(args.session)
