"""
Layer 2: Technical Signal Generator
====================================
Reads the fundamental basket, fetches price history, computes technical
indicators, and generates entry/exit signals for the next trading day.

Output: data/signals.json — list of BUY/SELL/UPDATE_SL actions.

Runs daily at 3:45 PM IST (after market close).
Can run on your laptop — no Dhan API access needed.
"""

import pandas as pd
from utils import load_config, load_json, save_json, setup_logger, now_iso

logger = setup_logger(__name__, "signals.log")


def compute_indicators(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """
    Add technical indicators to a price DataFrame.

    Adds: EMA (10, 20, 50, 200), RSI (14), volume average, 20-day high.

    Args:
        df: DataFrame with OHLCV columns (from yfinance)
        config: The 'technical' section of settings.yaml

    Returns:
        DataFrame with indicator columns added
    """
    rsi_period = config["rsi_period"]
    vol_period = config["volume_avg_period"]
    swing = config["swing"]

    # EMAs
    df["ema10"] = df["Close"].ewm(span=10, adjust=False).mean()
    df["ema20"] = df["Close"].ewm(span=20, adjust=False).mean()
    df["ema50"] = df["Close"].ewm(span=config["positional"]["ema_fast"], adjust=False).mean()

    if len(df) >= 200:
        df["ema200"] = df["Close"].ewm(span=config["positional"]["ema_slow"], adjust=False).mean()
    else:
        df["ema200"] = df["Close"].rolling(200).mean()  # NaN if insufficient data

    # RSI
    delta = df["Close"].diff()
    gain = delta.clip(lower=0).rolling(rsi_period).mean()
    loss = (-delta.clip(upper=0)).rolling(rsi_period).mean()
    rs = gain / loss
    df["rsi"] = 100 - (100 / (1 + rs))

    # Volume average
    df["vol_avg"] = df["Volume"].rolling(vol_period).mean()

    # 20-day high (shifted — today's high doesn't count as breakout reference)
    df["high_lookback"] = df["High"].rolling(swing["lookback_days"]).max().shift(1)
    df["high_50d"] = df["High"].rolling(50).max().shift(1)

    return df


def check_swing_entry(df: pd.DataFrame, config: dict) -> bool:
    """
    Check if swing entry conditions are met on the latest candle.

    Conditions:
        - Price breaks above N-day high
        - Volume > 1.5x average
        - RSI between 40 and 65 (not overbought)
    """
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
    """
    Check if positional entry conditions are met.

    Conditions:
        - Golden cross (50 EMA crosses above 200 EMA), OR
        - Price breaks above 50-day high on volume (trend continuation)
        - RSI < 70 (not overbought)
    """
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
    """
    Check if exit conditions are met for a held position.

    Swing: Close below 10-day EMA, or RSI > 75 (overbought)
    Positional: 50 EMA below 200 EMA (death cross), or Close below 200 EMA
    """
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


def fetch_price_history(symbol: str, days: int) -> pd.DataFrame | None:
    """
    Fetch daily price history from Yahoo Finance.

    Args:
        symbol: NSE ticker (e.g. "RELIANCE")
        days: Number of days of history

    Returns:
        DataFrame with OHLCV, or None if fetch fails
    """
    try:
        import yfinance as yf
        ticker = yf.Ticker(f"{symbol}.NS")
        df = ticker.history(period=f"{days}d")
        if len(df) < 50:
            logger.warning(f"  {symbol}: insufficient price data ({len(df)} days)")
            return None
        return df
    except Exception as e:
        logger.error(f"  {symbol}: failed to fetch price data: {e}")
        return None


def run():
    """
    Main entry point for the technical signal generator.

    Reads basket.json, fetches prices, computes indicators, generates signals.
    """
    config = load_config()["technical"]
    logger.info("=" * 60)
    logger.info("Layer 2: Technical Signal Generator — starting")
    logger.info("=" * 60)

    # Load fundamental basket
    basket = load_json("basket.json")
    if not basket:
        logger.error("No basket found. Run Layer 1 (screener.py) first.")
        return []

    # Load current positions
    positions = load_json("positions.json")

    signals = []
    held_symbols = {p["symbol"] for p in positions}

    for stock in basket:
        symbol = stock["symbol"]
        logger.info(f"[{symbol}] Fetching price data...")

        df = fetch_price_history(symbol, config["price_history_days"])
        if df is None:
            continue

        df = compute_indicators(df, config)
        last_close = float(df["Close"].iloc[-1])

        # Check if we already hold this stock
        held = next((p for p in positions if p["symbol"] == symbol), None)

        if held:
            # --- Check exit ---
            if check_exit(df, held["strategy"], config):
                signals.append({
                    "symbol": symbol,
                    "action": "SELL",
                    "reason": "Technical exit signal triggered",
                    "strategy": held["strategy"],
                    "qty": held["qty"],
                    "current_price": round(last_close, 2),
                    "generated_at": now_iso(),
                })
            else:
                # --- Update trailing stop loss ---
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
                        "generated_at": now_iso(),
                    })
                    logger.info(f"  Updated SL: {held.get('sl')} → {new_sl:.2f}")
        else:
            # --- Check entry ---
            # Try swing first (shorter term), then positional
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
                    "generated_at": now_iso(),
                })

    # Save signals
    save_json(signals, "signals.json")
    logger.info("=" * 60)
    logger.info(f"Signal generation complete: {len(signals)} actions for {len(basket)} stocks")
    buys = [s for s in signals if s["action"] == "BUY"]
    sells = [s for s in signals if s["action"] == "SELL"]
    sl_updates = [s for s in signals if s["action"] == "UPDATE_SL"]
    logger.info(f"  BUY: {len(buys)} | SELL: {len(sells)} | SL updates: {len(sl_updates)}")
    logger.info("=" * 60)

    return signals


if __name__ == "__main__":
    run()
