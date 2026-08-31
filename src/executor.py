"""
Layer 3: Dhan Execution Engine
================================
Reads signals.json and places buy/sell orders via DhanHQ API.
Updates positions.json with current state.

Runs at market open (9:15 AM IST).
Requires Dhan API access — run on VPS with whitelisted static IP.
"""

from utils import load_config, load_json, save_json, setup_logger, now_iso, send_telegram

logger = setup_logger(__name__, "executor.log")

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
    """
    Load the cached Dhan instrument master (security ID mapping).

    The master should be downloaded once daily via auth.get_dhan().fetch_security_list()
    and saved to data/instrument_master.json.

    Returns:
        Dict mapping {NSE_SYMBOL: security_id}
    """
    import json
    from utils import DATA_DIR

    filepath = DATA_DIR / "instrument_master.json"
    if not filepath.exists():
        logger.warning(
            "Instrument master not found. Run scripts/download_instruments.py first."
        )
        return {}

    with open(filepath, "r") as f:
        instruments = json.load(f)

    # Build symbol → security_id mapping
    mapping = {}
    for inst in instruments:
        symbol = inst.get("SEM_TRADING_SYMBOL", "")
        if symbol and inst.get("SEM_EXM_EXCH_ID") == "NSE":
            mapping[symbol] = inst.get("SEM_SMST_SECURITY_ID", "")

    return mapping


def get_security_id(symbol: str, instrument_map: dict) -> str | None:
    """
    Map an NSE ticker symbol to Dhan's internal security_id.

    Args:
        symbol: NSE ticker (e.g. "RELIANCE")
        instrument_map: Symbol → security_id dict from load_instrument_master()

    Returns:
        Security ID string, or None if not found
    """
    sec_id = instrument_map.get(symbol)
    if not sec_id:
        logger.warning(f"No security_id found for {symbol} in instrument master")
    return sec_id


def get_ltp(dhan, security_id: str) -> float | None:
    """
    Get the Last Traded Price for a security.

    Args:
        dhan: Authenticated dhanhq instance
        security_id: Dhan security ID

    Returns:
        LTP as float, or None if fetch fails
    """
    try:
        quote = dhan.market_quote("NSE", security_id)
        data = quote.get("data", {}).get("NSE", {}).get(security_id, {})
        ltp = float(data.get("last_price", 0))
        return ltp if ltp > 0 else None
    except Exception as e:
        logger.error(f"Failed to fetch LTP for {security_id}: {e}")
        return None


def calculate_quantity(ltp: float, capital_per_stock: int) -> int:
    """
    Calculate the number of shares to buy based on available capital.

    Args:
        ltp: Current market price
        capital_per_stock: Maximum ₹ to invest per stock

    Returns:
        Quantity (integer), rounded down
    """
    if ltp <= 0:
        return 0
    return int(capital_per_stock / ltp)


def execute_buy(dhan, symbol: str, security_id: str, strategy: str,
                config: dict, positions: list) -> dict | None:
    """
    Execute a BUY order for a stock.

    Args:
        dhan: Authenticated dhanhq instance
        symbol: NSE ticker
        security_id: Dhan security ID
        strategy: "swing" or "positional"
        config: 'execution' section of settings.yaml
        positions: Current positions list (for max_positions check)

    Returns:
        Position dict if order placed, None otherwise
    """
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

    tag = f"{strategy.upper()}_ENTRY"

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
        logger.info(f"BUY {symbol} x{qty} @ ~₹{ltp:.2f} | Order ID: {order_id}")

        send_telegram(
            f"📈 <b>BUY {symbol}</b>\n"
            f"Strategy: {strategy}\n"
            f"Qty: {qty} @ ~₹{ltp:.2f}\n"
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
        }
        return position

    except Exception as e:
        logger.error(f"Failed to place BUY for {symbol}: {e}")
        send_telegram(f"❌ <b>BUY FAILED: {symbol}</b>\nError: {e}")
        return None


def execute_sell(dhan, symbol: str, security_id: str, qty: int, reason: str) -> bool:
    """
    Execute a SELL order to square off a position.

    Args:
        dhan: Authenticated dhanhq instance
        symbol: NSE ticker
        security_id: Dhan security ID
        qty: Number of shares to sell
        reason: Exit reason (for logging)

    Returns:
        True if order placed successfully
    """
    try:
        resp = dhan.place_order(
            security_id=security_id,
            exchange_segment=_NSE,
            transaction_type=_SELL,
            quantity=qty,
            order_type=_MARKET,
            product_type=_CNC,
            price=0,
            tag="EXIT",
        )
        order_id = resp.get("orderId", "unknown") if isinstance(resp, dict) else "unknown"
        logger.info(f"SELL {symbol} x{qty} | Reason: {reason} | Order ID: {order_id}")

        send_telegram(
            f"📉 <b>SELL {symbol}</b>\n"
            f"Qty: {qty}\n"
            f"Reason: {reason}\n"
            f"Order ID: {order_id}"
        )
        return True

    except Exception as e:
        logger.error(f"Failed to place SELL for {symbol}: {e}")
        send_telegram(f"❌ <b>SELL FAILED: {symbol}</b>\nError: {e}")
        return False


def run():
    """
    Main entry point for the execution engine.

    Reads signals.json, places orders, updates positions.json.
    """
    from auth import get_dhan

    config = load_config()["execution"]
    logger.info("=" * 60)
    logger.info("Layer 3: Dhan Execution Engine — starting")
    logger.info("=" * 60)

    # Load signals and positions
    signals = load_json("signals.json")
    if not signals:
        logger.info("No signals to execute.")
        return

    positions = load_json("positions.json")

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

        logger.info(f"Processing {action} for {symbol}...")

        if action == "BUY":
            position = execute_buy(
                dhan, symbol, sec_id, signal["strategy"], config, positions
            )
            if position:
                # Load signal SL if provided
                if "sl" in signal:
                    position["sl"] = signal["sl"]
                if "target" in signal:
                    position["target"] = signal["target"]
                positions.append(position)

        elif action == "SELL":
            held = next((p for p in positions if p["symbol"] == symbol), None)
            if held:
                success = execute_sell(dhan, symbol, sec_id, held["qty"], signal.get("reason", "Signal"))
                if success:
                    positions = [p for p in positions if p["symbol"] != symbol]

        elif action == "UPDATE_SL":
            for p in positions:
                if p["symbol"] == symbol:
                    p["sl"] = signal["new_sl"]
                    logger.info(f"Updated SL for {symbol}: ₹{signal['new_sl']:.2f}")
                    break

    # Save updated positions
    save_json(positions, "positions.json")
    logger.info("=" * 60)
    logger.info(f"Execution complete. Active positions: {len(positions)}")
    for p in positions:
        pnl = 0
        ltp = get_ltp(dhan, p["security_id"])
        if ltp:
            pnl = (ltp - p["entry_price"]) * p["qty"]
        logger.info(
            f"  {p['symbol']:15s} x{p['qty']:4d} @ ₹{p['entry_price']:.2f} "
            f"({p['strategy']}) SL:₹{p.get('sl', 'N/A')} P&L:₹{pnl:+.2f}"
        )
    logger.info("=" * 60)


if __name__ == "__main__":
    run()
