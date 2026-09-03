"""
Layer 2: Technical Signal Generator (multi-session)

Reads cached prices.json (fetched once by price_fetcher.py), applies
the session's config parameters, and generates entry/exit signals.

Usage:
    python -m src.technicals --session A    # conservative
    python -m src.technicals --session B    # balanced (default)
    python -m src.technicals --session C    # aggressive
    python -m src.technicals --help

Cron runs B first, then A, then C — all serial on the same cached prices.
"""

import argparse
import os
import sys
import pandas as pd

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

from utils import load_config, load_json, save_json, setup_logger, now_iso

logger = setup_logger(__name__, f"technicals_{_pre_args.session}.log")


def candles_to_dataframe(candles: list) -> pd.DataFrame:
    """Convert cached candle records back to a DataFrame."""
    df = pd.DataFrame(candles)
    # Map to yfinance-style column names
    col_map = {
        "open": "Open", "high": "High", "low": "Low",
        "close": "Close", "volume": "Volume",
    }
    df = df.rename(columns=col_map)
    for col in ["Open", "High", "Low", "Close", "Volume"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def compute_indicators(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """Add technical indicators to a price DataFrame."""
    rsi_period = config["rsi_period"]
    vol_period = config["volume_avg_period"]
    swing = config["swing"]

    df = df.copy()

    # EMAs
    df["ema10"] = df["Close"].ewm(span=10, adjust=False).mean()
    df["ema20"] = df["Close"].ewm(span=20, adjust=False).mean()
    df["ema50"] = df["Close"].ewm(span=config["positional"]["ema_fast"], adjust=False).mean()

    if len(df) >= 200:
        df["ema200"] = df["Close"].ewm(span=config["positional"]["ema_slow"], adjust=False).mean()
    else:
        df["ema200"] = df["Close"].rolling(200).mean()

    # RSI
    delta = df["Close"].diff()
    gain = delta.clip(lower=0).rolling(rsi_period).mean()
    loss = (-delta.clip(upper=0)).rolling(rsi_period).mean()
    rs = gain / loss
    df["rsi"] = 100 - (100 / (1 + rs))

    # Volume average
    df["vol_avg"] = df["Volume"].rolling(vol_period).mean()

    # N-day high (shifted — today's high doesn't count as breakout reference)
    df["high_lookback"] = df["High"].rolling(swing["lookback_days"]).max().shift(1)
    df["high_50d"] = df["High"].rolling(50).max().shift(1)

    return df


def check_swing_entry(df: pd.DataFrame, config: dict) -> bool:
    """Check if swing entry conditions are met on the latest candle."""
    swing = config["swing"]
    if len(df) < swing["lookback_days"] + 5:
        return False

    last = df.iloc[-1]

    breakout = last["Close"] > last["high_lookback"]
    vol_confirm = last["Volume"] > last["vol_avg"] * swing["volume_multiplier"]
    not_overbought = swing["rsi_min"] < last["rsi"] < swing["rsi_max"]

    if breakout and vol_confirm and not_overbought:
        logger.info(
            f"  Swing entry: Close {last['Close']:.2f} > High {last['high_lookback']:.2f}, "
            f"Vol {last['Volume']:.0f} > {last['vol_avg'] * swing['volume_multiplier']:.0f}, "
            f"RSI {last['rsi']:.1f}"
        )
        return True
    return False


def check_positional_entry(df: pd.DataFrame, config: dict) -> bool:
    """Check if positional entry conditions are met."""
    pos = config["positional"]
    if len(df) < 201:
        return False

    last = df.iloc[-1]
    prev = df.iloc[-2]

    golden_cross = prev["ema50"] <= prev["ema200"] and last["ema50"] > last["ema200"]
    trend_cont = last["Close"] > last.get("high_50d", 0)
    vol_confirm = last["Volume"] > last["vol_avg"]

    if (golden_cross or (trend_cont and vol_confirm)) and last["rsi"] < 70:
        logger.info(
            f"  Positional entry: {'Golden cross' if golden_cross else 'Trend continuation'}, "
            f"RSI {last['rsi']:.1f}"
        )
        return True
    return False


def check_exit(df: pd.DataFrame, strategy: str, config: dict) -> bool:
    """Check if exit conditions are met for a held position."""
    last = df.iloc[-1]

    if strategy == "swing":
        if last["Close"] < last["ema10"]:
            logger.info(f"  Swing exit: Close {last['Close']:.2f} < EMA10 {last['ema10']:.2f}")
            return True
        if last["rsi"] > 75:
            logger.info(f"  Swing exit: RSI {last['rsi']:.1f} > 75 (overbought)")
            return True
    else:  # positional
        if last["ema50"] < last["ema200"]:
            logger.info("  Positional exit: Death cross (EMA50 < EMA200)")
            return True
        if last["Close"] < last["ema200"]:
            logger.info(f"  Positional exit: Close {last['Close']:.2f} < EMA200 {last['ema200']:.2f}")
            return True

    return False


def run(session: str):
    """Main entry point — reads cached prices, applies session config, generates signals."""
    config = load_config()["technical"]
    session_label = {"A": "Conservative", "B": "Balanced", "C": "Aggressive"}[session]

    logger.info("=" * 60)
    logger.info(f"Layer 2: Technical Signal Generator — Session {session} ({session_label})")
    logger.info("=" * 60)

    # Load cached prices (shared across all sessions)
    prices = load_json("prices.json")
    if not prices:
        logger.error("No cached prices found. Run price_fetcher.py first.")
        return []

    # Load current positions for this session
    positions_file = f"positions_{session}.json"
    positions = load_json(positions_file)

    signals = []
    held_symbols = {p["symbol"] for p in positions}

    for symbol, stock_data in prices.items():
        logger.info(f"[{session}/{symbol}] Analyzing...")

        candles = stock_data.get("candles", [])
        if len(candles) < 50:
            logger.warning(f"  {symbol}: insufficient candle data ({len(candles)})")
            continue

        df = candles_to_dataframe(candles)
        df = compute_indicators(df, config)
        last_close = float(df["Close"].iloc[-1])

        held = next((p for p in positions if p["symbol"] == symbol), None)

        if held:
            # Check exit
            if check_exit(df, held["strategy"], config):
                signals.append({
                    "symbol": symbol,
                    "action": "SELL",
                    "reason": "Technical exit signal triggered",
                    "strategy": held["strategy"],
                    "qty": held["qty"],
                    "current_price": round(last_close, 2),
                    "session": session,
                    "generated_at": now_iso(),
                })
            else:
                # Update trailing stop loss
                strategy_config = config[held["strategy"]]
                sl_pct = strategy_config["stop_loss_pct"]
                new_sl = max(held.get("sl", 0), last_close * (1 - sl_pct))

                if new_sl > held.get("sl", 0):
                    signals.append({
                        "symbol": symbol,
                        "action": "UPDATE_SL",
                        "new_sl": round(new_sl, 2),
                        "old_sl": held.get("sl"),
                        "strategy": held["strategy"],
                        "session": session,
                        "generated_at": now_iso(),
                    })
                    logger.info(f"  Updated SL: {held.get('sl')} -> {new_sl:.2f}")
        else:
            # Check entry — swing first, then positional
            if check_swing_entry(df, config):
                sl_pct = config["swing"]["stop_loss_pct"]
                signals.append({
                    "symbol": symbol,
                    "action": "BUY",
                    "strategy": "swing",
                    "entry": round(last_close, 2),
                    "sl": round(last_close * (1 - sl_pct), 2),
                    "target": round(last_close * (1 + config["swing"]["target_pct"]), 2),
                    "rsi": round(float(df["rsi"].iloc[-1]), 2),
                    "session": session,
                    "generated_at": now_iso(),
                })
            elif check_positional_entry(df, config):
                sl_pct = config["positional"]["stop_loss_pct"]
                signals.append({
                    "symbol": symbol,
                    "action": "BUY",
                    "strategy": "positional",
                    "entry": round(last_close, 2),
                    "sl": round(last_close * (1 - sl_pct), 2),
                    "target": round(last_close * (1 + config["positional"]["target_pct"]), 2),
                    "rsi": round(float(df["rsi"].iloc[-1]), 2),
                    "session": session,
                    "generated_at": now_iso(),
                })

    # Save signals for this session
    signals_file = f"signals_{session}.json"
    save_json(signals, signals_file)

    logger.info("=" * 60)
    logger.info(f"Session {session} ({session_label}): {len(signals)} actions for {len(prices)} stocks")
    buys = [s for s in signals if s["action"] == "BUY"]
    sells = [s for s in signals if s["action"] == "SELL"]
    sl_updates = [s for s in signals if s["action"] == "UPDATE_SL"]
    logger.info(f"  BUY: {len(buys)} | SELL: {len(sells)} | SL updates: {len(sl_updates)}")
    logger.info("=" * 60)

    return signals


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Technical Signal Generator — runs on cached prices.json.\n\n"
            "Sessions:\n"
            "  A = Conservative (tight RSI 45-60, 1.2x vol, 1.5% SL, 4% target)\n"
            "  B = Balanced (RSI 40-65, 1.5x vol, 2% SL, 6% target)\n"
            "  C = Aggressive (RSI 35-70, 2.0x vol, 3% SL, 8% target)\n\n"
            "Examples:\n"
            "  python -m src.technicals --session A    # conservative\n"
            "  python -m src.technicals --session B    # balanced (default)\n"
            "  python -m src.technicals --session C    # aggressive"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--session", default="B", choices=["A", "B", "C"],
        help="Trading session profile (default: B = balanced)",
    )
    args = parser.parse_args()
    run(args.session)
