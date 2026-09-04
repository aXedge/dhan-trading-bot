"""
Consensus Voting Engine — Multi-Strategy Signal Generator
==========================================================

Runs 5 independent trading strategies on each stock and requires a
super majority (3 of 5) to agree before generating an entry signal.

Strategies:
    1. Breakout       — price above N-day high + volume confirmation
    2. Pullback-to-EMA — uptrend pullback to EMA20 with RSI 40-50
    3. Supertrend      — ATR-based trend following, buy on green flip
    4. MACD            — bullish crossover (MACD line > signal line)
    5. RSI Mean Reversion — oversold bounce in uptrend (RSI < 35 then turns up)

Entry rule:  >= 3 of 5 strategies vote BUY
Exit rule:   SL/target always active. For signal-based exits,
             >= 3 of 5 strategies vote SELL.

Runs as Session V alongside A/B/C/T.

Usage:
    python -m src.consensus_voter

    # Help
    python -m src.consensus_voter --help

No Dhan API access needed for signal generation — reads cached prices.json.
"""

import argparse
import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime

# Session config mapping
SESSION_CONFIGS = {
    "A": "config/settings_conservative.yaml",
    "B": "config/settings_balanced.yaml",
    "C": "config/settings_aggressive.yaml",
    "T": "config/settings_tuned.yaml",
    "V": "config/settings_voting.yaml",
}


def _setup_session(session: str):
    """Set env vars so utils.py loads the right config."""
    config_file = SESSION_CONFIGS.get(session, SESSION_CONFIGS["V"])
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    os.environ["CONFIG_PATH"] = os.path.join(project_root, config_file)


# Parse --session before importing utils
_pre_parser = argparse.ArgumentParser(add_help=False)
_pre_parser.add_argument("--session", default="V", choices=list(SESSION_CONFIGS.keys()))
_pre_args, _ = _pre_parser.parse_known_args()
_setup_session(_pre_args.session)

from utils import load_config, load_json, save_json, setup_logger, now_iso

logger = setup_logger(__name__, f"consensus_{_pre_args.session}.log")


# ---------------------------------------------------------------------------
# Candle deserialization
# ---------------------------------------------------------------------------

def candles_to_dataframe(candles: list) -> pd.DataFrame:
    """Convert cached candle records to a DataFrame."""
    df = pd.DataFrame(candles)
    col_map = {
        "open": "Open", "high": "High", "low": "Low",
        "close": "Close", "volume": "Volume",
    }
    df = df.rename(columns=col_map)
    for col in ["Open", "High", "Low", "Close", "Volume"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


# ---------------------------------------------------------------------------
# Indicator computation
# ---------------------------------------------------------------------------

def compute_all_indicators(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """Compute ALL indicators needed by the 5 strategies."""
    df = df.copy()

    # EMAs
    df["ema10"] = df["Close"].ewm(span=10, adjust=False).mean()
    df["ema20"] = df["Close"].ewm(span=20, adjust=False).mean()
    df["ema50"] = df["Close"].ewm(span=50, adjust=False).mean()
    df["ema200"] = df["Close"].ewm(span=200, adjust=False).mean()

    # RSI (14)
    delta = df["Close"].diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain / loss
    df["rsi"] = 100 - (100 / (1 + rs))

    # Volume average
    df["vol_avg"] = df["Volume"].rolling(20).mean()

    # Lookback highs (excluding today)
    lookback = config.get("lookback_days", 20)
    df["high_lookback"] = df["High"].rolling(lookback).max().shift(1)
    df["high_50d"] = df["High"].rolling(50).max().shift(1)

    # MACD (12, 26, 9)
    ema12 = df["Close"].ewm(span=12, adjust=False).mean()
    ema26 = df["Close"].ewm(span=26, adjust=False).mean()
    df["macd_line"] = ema12 - ema26
    df["macd_signal"] = df["macd_line"].ewm(span=9, adjust=False).mean()
    df["macd_hist"] = df["macd_line"] - df["macd_signal"]

    # Supertrend (ATR-based, period=10, multiplier=3)
    df = compute_supertrend(df, period=10, multiplier=3)

    # ATR (for context)
    df["tr"] = np.maximum(
        df["High"] - df["Low"],
        np.maximum(
            abs(df["High"] - df["Close"].shift(1)),
            abs(df["Low"] - df["Close"].shift(1))
        )
    )
    df["atr"] = df["tr"].rolling(14).mean()

    return df


def compute_supertrend(df: pd.DataFrame, period: int = 10, multiplier: float = 3) -> pd.DataFrame:
    """
    Compute Supertrend indicator.
    Returns DataFrame with 'supertrend' and 'supertrend_dir' columns.
    supertrend_dir: 1 = green (uptrend), -1 = red (downtrend)
    """
    df = df.copy()

    # ATR
    hl2 = (df["High"] + df["Low"]) / 2
    tr = np.maximum(
        df["High"] - df["Low"],
        np.maximum(
            abs(df["High"] - df["Close"].shift(1)),
            abs(df["Low"] - df["Close"].shift(1))
        )
    )
    atr = tr.rolling(period).mean()

    # Basic bands
    upper_basic = hl2 + multiplier * atr
    lower_basic = hl2 - multiplier * atr

    # Final bands
    upper_band = upper_basic.copy()
    lower_band = lower_basic.copy()

    for i in range(1, len(df)):
        # Upper band
        if (upper_basic.iloc[i] < upper_band.iloc[i - 1]) or (df["Close"].iloc[i - 1] > upper_band.iloc[i - 1]):
            upper_band.iloc[i] = upper_basic.iloc[i]
        else:
            upper_band.iloc[i] = upper_band.iloc[i - 1]

        # Lower band
        if (lower_basic.iloc[i] > lower_band.iloc[i - 1]) or (df["Close"].iloc[i - 1] < lower_band.iloc[i - 1]):
            lower_band.iloc[i] = lower_basic.iloc[i]
        else:
            lower_band.iloc[i] = lower_band.iloc[i - 1]

    # Supertrend and direction
    supertrend = pd.Series(index=df.index, dtype=float)
    direction = pd.Series(index=df.index, dtype=int)

    # Initialize
    if len(df) > 0:
        supertrend.iloc[0] = upper_band.iloc[0]
        direction.iloc[0] = -1

    for i in range(1, len(df)):
        close = df["Close"].iloc[i]

        if close <= supertrend.iloc[i - 1]:
            # Downtrend — use upper band
            supertrend.iloc[i] = min(upper_band.iloc[i], supertrend.iloc[i - 1])
            direction.iloc[i] = -1
        else:
            # Uptrend — use lower band
            supertrend.iloc[i] = max(lower_band.iloc[i], supertrend.iloc[i - 1])
            direction.iloc[i] = 1

            # Check if we just flipped from red to green
            if direction.iloc[i - 1] == -1:
                supertrend.iloc[i] = lower_band.iloc[i]

    df["supertrend"] = supertrend
    df["supertrend_dir"] = direction

    return df


# ---------------------------------------------------------------------------
# Strategy 1: Breakout
# ---------------------------------------------------------------------------

def strategy_breakout(df: pd.DataFrame, config: dict) -> str:
    """
    Breakout strategy: price above N-day high + volume confirmation.
    Trend filter: only buy when EMA50 > EMA200.

    Returns: 'BUY', 'SELL', or 'HOLD'
    """
    if len(df) < 200:
        return "HOLD"

    last = df.iloc[-1]
    prev = df.iloc[-2]

    lookback = config.get("lookback_days", 20)
    vol_mult = config.get("volume_multiplier", 2.0)
    rsi_min = config.get("rsi_min", 35)
    rsi_max = config.get("rsi_max", 70)

    # Trend filter
    in_uptrend = last["ema50"] > last["ema200"]

    # Entry: breakout + volume + RSI range + uptrend
    breakout = last["Close"] > last["high_lookback"]
    vol_confirm = last["Volume"] > last["vol_avg"] * vol_mult
    rsi_ok = rsi_min < last["rsi"] < rsi_max

    if breakout and vol_confirm and rsi_ok and in_uptrend:
        return "BUY"

    # Exit: close below EMA10 or RSI > 75
    if last["Close"] < last["ema10"]:
        return "SELL"
    if last["rsi"] > 75:
        return "SELL"

    return "HOLD"


# ---------------------------------------------------------------------------
# Strategy 2: Pullback-to-EMA
# ---------------------------------------------------------------------------

def strategy_pullback(df: pd.DataFrame, config: dict) -> str:
    """
    Pullback-to-EMA strategy: buy when a stock in an uptrend (EMA50 > EMA200)
    pulls back to EMA20 with RSI between 40-50.

    Returns: 'BUY', 'SELL', or 'HOLD'
    """
    if len(df) < 200:
        return "HOLD"

    last = df.iloc[-1]
    prev = df.iloc[-2]

    # Must be in uptrend
    if last["ema50"] <= last["ema200"]:
        return "HOLD"

    # Entry: price near EMA20 (within 1.5%) and RSI 40-50
    dist_to_ema20 = abs(last["Close"] - last["ema20"]) / last["Close"]
    near_ema20 = dist_to_ema20 < 0.015
    rsi_pullback = 40 <= last["rsi"] <= 55

    # Price should be above EMA50 (pullback to EMA20, not a crash)
    above_ema50 = last["Close"] > last["ema50"]

    # Previous candle should have closed below EMA20 or RSI should be rising
    rsi_rising = last["rsi"] > prev["rsi"]

    if near_ema20 and rsi_pullback and above_ema50 and rsi_rising:
        return "BUY"

    # Exit: close below EMA50 or RSI > 70
    if last["Close"] < last["ema50"]:
        return "SELL"
    if last["rsi"] > 70:
        return "SELL"

    return "HOLD"


# ---------------------------------------------------------------------------
# Strategy 3: Supertrend
# ---------------------------------------------------------------------------

def strategy_supertrend(df: pd.DataFrame, config: dict) -> str:
    """
    Supertrend strategy: buy on green flip, sell on red flip.

    Returns: 'BUY', 'SELL', or 'HOLD'
    """
    if len(df) < 15 or "supertrend_dir" not in df.columns:
        return "HOLD"

    last = df.iloc[-1]
    prev = df.iloc[-2]

    # Entry: supertrend just flipped to green (1)
    if last["supertrend_dir"] == 1 and prev["supertrend_dir"] == -1:
        return "BUY"

    # Exit: supertrend flipped to red (-1)
    if last["supertrend_dir"] == -1 and prev["supertrend_dir"] == 1:
        return "SELL"

    return "HOLD"


# ---------------------------------------------------------------------------
# Strategy 4: MACD
# ---------------------------------------------------------------------------

def strategy_macd(df: pd.DataFrame, config: dict) -> str:
    """
    MACD strategy: buy on bullish crossover (MACD line crosses above signal).
    Sell on bearish crossover.

    Returns: 'BUY', 'SELL', or 'HOLD'
    """
    if len(df) < 35 or "macd_line" not in df.columns:
        return "HOLD"

    last = df.iloc[-1]
    prev = df.iloc[-2]

    # Entry: MACD line crosses above signal line
    if prev["macd_line"] <= prev["macd_signal"] and last["macd_line"] > last["macd_signal"]:
        # Only buy if MACD histogram is positive and growing
        if last["macd_hist"] > 0:
            return "BUY"

    # Exit: MACD line crosses below signal line
    if prev["macd_line"] >= prev["macd_signal"] and last["macd_line"] < last["macd_signal"]:
        return "SELL"

    return "HOLD"


# ---------------------------------------------------------------------------
# Strategy 5: RSI Mean Reversion
# ---------------------------------------------------------------------------

def strategy_rsi_mean_reversion(df: pd.DataFrame, config: dict) -> str:
    """
    RSI Mean Reversion: buy when RSI was oversold (< 35) and is now turning up,
    but only in an uptrend (EMA50 > EMA200).

    Returns: 'BUY', 'SELL', or 'HOLD'
    """
    if len(df) < 200:
        return "HOLD"

    last = df.iloc[-1]
    prev = df.iloc[-2]
    prev2 = df.iloc[-3]

    # Must be in uptrend
    if last["ema50"] <= last["ema200"]:
        return "HOLD"

    # Entry: RSI was below 35 (or below 40) within last 3 candles and is now rising
    was_oversold = any(df["rsi"].iloc[i] < 40 for i in range(-3, 0))
    rsi_turning_up = last["rsi"] > prev["rsi"] and prev["rsi"] <= prev2["rsi"]
    rsi_in_range = 35 < last["rsi"] < 55

    if was_oversold and rsi_turning_up and rsi_in_range:
        return "BUY"

    # Exit: RSI > 65 (mean reversion complete)
    if last["rsi"] > 65:
        return "SELL"

    # Exit: close below EMA50 (trend broken)
    if last["Close"] < last["ema50"]:
        return "SELL"

    return "HOLD"


# ---------------------------------------------------------------------------
# Consensus engine
# ---------------------------------------------------------------------------

# Registry of all strategies
STRATEGIES = {
    "breakout": strategy_breakout,
    "pullback": strategy_pullback,
    "supertrend": strategy_supertrend,
    "macd": strategy_macd,
    "rsi_mean_reversion": strategy_rsi_mean_reversion,
}

VOTE_THRESHOLD = 3  # 3 of 5 needed for super majority


def run_consensus(df: pd.DataFrame, config: dict, held: dict = None) -> dict:
    """
    Run all 5 strategies and collect votes.

    Returns:
        {
            "votes": {"breakout": "BUY", "pullback": "HOLD", ...},
            "buy_votes": 3,
            "sell_votes": 1,
            "hold_votes": 1,
            "consensus": "BUY" | "SELL" | "HOLD",
            "sl": float,      # stop loss price
            "target": float,  # target price
            "strategy": "consensus",
        }
    """
    votes = {}
    for name, func in STRATEGIES.items():
        try:
            vote = func(df, config)
            votes[name] = vote
        except Exception as e:
            logger.warning(f"  {name} strategy error: {e}")
            votes[name] = "HOLD"

    buy_votes = sum(1 for v in votes.values() if v == "BUY")
    sell_votes = sum(1 for v in votes.values() if v == "SELL")
    hold_votes = sum(1 for v in votes.values() if v == "HOLD")

    # Determine consensus
    last = df.iloc[-1]
    last_close = float(last["Close"])

    # Entry: need super majority (>= 3) of BUY
    # Exit: need super majority (>= 3) of SELL
    # SL/target are handled separately by the executor/sl_monitor
    if held:
        # If we hold a position, check for exit votes
        if sell_votes >= VOTE_THRESHOLD:
            consensus = "SELL"
        else:
            consensus = "HOLD"
    else:
        # If we don't hold, check for entry votes
        if buy_votes >= VOTE_THRESHOLD:
            consensus = "BUY"
        else:
            consensus = "HOLD"

    # Compute SL and target from the config
    sl_pct = config.get("stop_loss_pct", 0.03)
    target_pct = config.get("target_pct", 0.08)
    sl = round(last_close * (1 - sl_pct), 2)
    target = round(last_close * (1 + target_pct), 2)

    return {
        "votes": votes,
        "buy_votes": buy_votes,
        "sell_votes": sell_votes,
        "hold_votes": hold_votes,
        "consensus": consensus,
        "sl": sl,
        "target": target,
        "strategy": "consensus",
        "entry": round(last_close, 2),
        "rsi": round(float(last["rsi"]), 2),
    }


# ---------------------------------------------------------------------------
# Main signal generation
# ---------------------------------------------------------------------------

def run(session: str = "V"):
    """Main entry point — reads cached prices, runs 5 strategies, generates consensus signals."""
    config = load_config()

    # Use the technical section for swing params
    tech_config = config.get("technical", {})
    swing_config = tech_config.get("swing", {})

    # Merge config for strategy params
    strat_config = {
        "lookback_days": swing_config.get("lookback_days", 10),
        "volume_multiplier": swing_config.get("volume_multiplier", 2.0),
        "rsi_min": swing_config.get("rsi_min", 35),
        "rsi_max": swing_config.get("rsi_max", 70),
        "stop_loss_pct": swing_config.get("stop_loss_pct", 0.03),
        "target_pct": swing_config.get("target_pct", 0.08),
    }

    session_label = "Consensus Voting"

    logger.info("=" * 60)
    logger.info(f"Consensus Voting Engine — Session {session} ({session_label})")
    logger.info(f"  Strategies: {len(STRATEGIES)} ({', '.join(STRATEGIES.keys())})")
    logger.info(f"  Vote threshold: {VOTE_THRESHOLD}/{len(STRATEGIES)} for entry/exit")
    logger.info("=" * 60)

    # Load cached prices
    prices = load_json("prices.json")
    if not prices:
        logger.error("No cached prices found. Run price_fetcher.py first.")
        return []

    # Load current positions for this session
    positions = load_json(f"positions_{session}.json")

    signals = []

    for symbol, stock_data in prices.items():
        logger.info(f"[{session}/{symbol}] Running 5 strategies...")

        candles = stock_data.get("candles", [])
        if len(candles) < 200:
            logger.warning(f"  {symbol}: insufficient candle data ({len(candles)})")
            continue

        df = candles_to_dataframe(candles)
        df = compute_all_indicators(df, strat_config)

        if len(df) < 200:
            continue

        held = next((p for p in positions if p["symbol"] == symbol), None)

        result = run_consensus(df, strat_config, held)

        # Log the votes
        vote_str = " | ".join(f"{k}:{v}" for k, v in result["votes"].items())
        logger.info(f"  Votes: B={result['buy_votes']} S={result['sell_votes']} H={result['hold_votes']}")
        logger.info(f"  [{vote_str}]")
        logger.info(f"  Consensus: {result['consensus']}")

        if result["consensus"] == "BUY":
            signals.append({
                "symbol": symbol,
                "action": "BUY",
                "strategy": "consensus",
                "entry": result["entry"],
                "sl": result["sl"],
                "target": result["target"],
                "rsi": result["rsi"],
                "votes": result["votes"],
                "buy_votes": result["buy_votes"],
                "sell_votes": result["sell_votes"],
                "session": session,
                "generated_at": now_iso(),
            })
            logger.info(f"  >>> BUY signal generated (buy_votes={result['buy_votes']}/{len(STRATEGIES)})")

        elif result["consensus"] == "SELL":
            signals.append({
                "symbol": symbol,
                "action": "SELL",
                "reason": f"Consensus SELL ({result['sell_votes']}/{len(STRATEGIES)} strategies voted SELL)",
                "votes": result["votes"],
                "sell_votes": result["sell_votes"],
                "buy_votes": result["buy_votes"],
                "session": session,
                "generated_at": now_iso(),
            })
            logger.info(f"  >>> SELL signal generated (sell_votes={result['sell_votes']}/{len(STRATEGIES)})")

        # Check for trailing SL update if held
        if held:
            last_close = float(df["Close"].iloc[-1])
            sl_pct = strat_config["stop_loss_pct"]
            new_sl = max(held.get("sl", 0), last_close * (1 - sl_pct))

            if new_sl > held.get("sl", 0):
                signals.append({
                    "symbol": symbol,
                    "action": "UPDATE_SL",
                    "new_sl": round(new_sl, 2),
                    "old_sl": held.get("sl"),
                    "session": session,
                    "generated_at": now_iso(),
                })
                logger.info(f"  Updated SL: {held.get('sl')} -> {new_sl:.2f}")

    # Save signals
    save_json(signals, f"signals_{session}.json")

    buys = [s for s in signals if s["action"] == "BUY"]
    sells = [s for s in signals if s["action"] == "SELL"]
    sl_updates = [s for s in signals if s["action"] == "UPDATE_SL"]

    logger.info("=" * 60)
    logger.info(f"Session {session}: {len(signals)} actions for {len(prices)} stocks")
    logger.info(f"  BUY: {len(buys)} | SELL: {len(sells)} | SL updates: {len(sl_updates)}")
    logger.info("=" * 60)

    return signals


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Consensus Voting Engine — 5 strategies vote on each stock.\n\n"
            "Strategies:\n"
            "  1. Breakout       — N-day high + volume\n"
            "  2. Pullback-to-EMA — uptrend pullback to EMA20\n"
            "  3. Supertrend      — ATR-based trend following\n"
            "  4. MACD            — bullish/bearish crossover\n"
            "  5. RSI Mean Reversion — oversold bounce in uptrend\n\n"
            "Entry: >= 3 of 5 vote BUY\n"
            "Exit:  SL/target always active, plus >= 3 of 5 vote SELL\n\n"
            "Examples:\n"
            "  python -m src.consensus_voter               # default session V\n"
            "  python -m src.consensus_voter --session V"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--session", default="V", choices=list(SESSION_CONFIGS.keys()),
        help="Session ID (default: V = voting)",
    )
    args = parser.parse_args()
    run(args.session)
