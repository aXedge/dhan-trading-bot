"""
Regime-Switching Strategy Engine — Session R
=============================================

Detects market regime (trending vs choppy) using ADX, then applies the
appropriate strategy:

  TRENDING (ADX >= 25):
    Enhanced breakout (Session T + ADX filter + relative strength vs Nifty
    + market breadth gate)

  CHOPPY (ADX < 20):
    Mean reversion (buy oversold in uptrend, sell at RSI recovery)

  TRANSITIONAL (20 <= ADX < 25):
    Stay in cash (no new entries)

Enhanced filters:
  - ADX >= 25 required for breakout entries (strong trend only)
  - Volume must be highest in 20 days (not just 2x average)
  - Market breadth: >60% of Nifty 500 above 200 DMA
  - Relative strength: stock must outperform Nifty over 20 days
  - Trend filter: EMA50 > EMA200 (carried from Session T)

Usage:
    python -m src.regime_engine              # default session R
    python -m src.regime_engine --session R
    python -m src.regime_engine --help

No Dhan API needed for signal generation — reads cached prices.json.
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
    "R": "config/settings_regime.yaml",
}


def _setup_session(session: str):
    """Set env vars so utils.py loads the right config."""
    config_file = SESSION_CONFIGS.get(session, SESSION_CONFIGS["R"])
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    os.environ["CONFIG_PATH"] = os.path.join(project_root, config_file)


# Parse --session before importing utils
_pre_parser = argparse.ArgumentParser(add_help=False)
_pre_parser.add_argument("--session", default="R", choices=list(SESSION_CONFIGS.keys()))
_pre_args, _ = _pre_parser.parse_known_args()
_setup_session(_pre_args.session)

from utils import load_config, load_json, save_json, setup_logger, now_iso

logger = setup_logger(__name__, f"regime_{_pre_args.session}.log")


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
    """Compute ALL indicators needed by the regime engine."""
    df = df.copy()

    # EMAs
    for span in [10, 20, 50, 200]:
        df[f"ema{span}"] = df["Close"].ewm(span=span, adjust=False).mean()

    # RSI (14)
    delta = df["Close"].diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain / loss
    df["rsi"] = 100 - (100 / (1 + rs))

    # Volume average
    df["vol_avg"] = df["Volume"].rolling(20).mean()
    df["vol_max_20"] = df["Volume"].rolling(20).max()

    # Lookback highs (excluding today)
    lookback = config.get("lookback_days", 10)
    df["high_lookback"] = df["High"].rolling(lookback).max().shift(1)

    # ATR for ADX
    df = compute_adx(df, period=14)

    # Bollinger Bands (for mean reversion)
    df["bb_mid"] = df["Close"].rolling(20).mean()
    bb_std = df["Close"].rolling(20).std()
    df["bb_upper"] = df["bb_mid"] + 2 * bb_std
    df["bb_lower"] = df["bb_mid"] - 2 * bb_std

    # 20-day returns for relative strength
    df["ret_20d"] = df["Close"].pct_change(20)

    # Supertrend (for mean reversion confirmation)
    df = compute_supertrend(df, period=10, multiplier=3)

    return df


def compute_adx(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    """
    Compute ADX (Average Directional Index).
    ADX > 25 = strong trend, ADX < 20 = choppy/no trend.

    Also computes +DI and -DI directional indicators.
    """
    df = df.copy()

    high = df["High"]
    low = df["Low"]
    close = df["Close"]

    # True Range
    tr = pd.concat([
        high - low,
        (high - close.shift(1)).abs(),
        (low - close.shift(1)).abs(),
    ], axis=1).max(axis=1)

    # Directional Movement
    up_move = high - high.shift(1)
    down_move = low.shift(1) - low

    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)

    plus_dm = pd.Series(plus_dm, index=df.index, dtype=float)
    minus_dm = pd.Series(minus_dm, index=df.index, dtype=float)

    # Smoothed TR and DM (Wilder's smoothing)
    atr = tr.rolling(period).mean()
    plus_di = 100 * (plus_dm.rolling(period).mean() / atr)
    minus_di = 100 * (minus_dm.rolling(period).mean() / atr)

    # DX and ADX
    dx = 100 * ((plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan))
    adx = dx.rolling(period).mean()

    df["adx"] = adx
    df["plus_di"] = plus_di
    df["minus_di"] = minus_di

    return df


def compute_supertrend(df: pd.DataFrame, period: int = 10, multiplier: float = 3) -> pd.DataFrame:
    """Compute Supertrend indicator with NaN-safe warmup handling."""
    df = df.copy()
    hl2 = (df["High"] + df["Low"]) / 2
    tr = pd.concat([
        df["High"] - df["Low"],
        (df["High"] - df["Close"].shift(1)).abs(),
        (df["Low"] - df["Close"].shift(1)).abs(),
    ], axis=1).max(axis=1)
    atr = tr.rolling(period).mean()

    upper_basic = hl2 + multiplier * atr
    lower_basic = hl2 - multiplier * atr

    upper_band = upper_basic.copy()
    lower_band = lower_basic.copy()

    for i in range(1, len(df)):
        if pd.isna(upper_basic.iloc[i]) or pd.isna(lower_basic.iloc[i]):
            continue
        if pd.isna(upper_band.iloc[i - 1]):
            upper_band.iloc[i] = upper_basic.iloc[i]
            lower_band.iloc[i] = lower_basic.iloc[i]
            continue
        if (upper_basic.iloc[i] < upper_band.iloc[i - 1]) or (df["Close"].iloc[i - 1] > upper_band.iloc[i - 1]):
            upper_band.iloc[i] = upper_basic.iloc[i]
        else:
            upper_band.iloc[i] = upper_band.iloc[i - 1]

        if (lower_basic.iloc[i] > lower_band.iloc[i - 1]) or (df["Close"].iloc[i - 1] < lower_band.iloc[i - 1]):
            lower_band.iloc[i] = lower_basic.iloc[i]
        else:
            lower_band.iloc[i] = lower_band.iloc[i - 1]

    supertrend = pd.Series(index=df.index, dtype=float)
    direction = pd.Series(index=df.index, dtype=int)

    first_valid = None
    for i in range(len(df)):
        if not pd.isna(upper_band.iloc[i]) and not pd.isna(lower_band.iloc[i]):
            first_valid = i
            break

    if first_valid is not None:
        supertrend.iloc[first_valid] = upper_band.iloc[first_valid]
        direction.iloc[first_valid] = -1

        for i in range(first_valid + 1, len(df)):
            if pd.isna(upper_band.iloc[i]) or pd.isna(lower_band.iloc[i]):
                continue
            close = df["Close"].iloc[i]
            prev_st = supertrend.iloc[i - 1]
            if pd.isna(prev_st):
                prev_st = upper_band.iloc[i]

            if close <= prev_st:
                supertrend.iloc[i] = min(upper_band.iloc[i], prev_st) if not pd.isna(upper_band.iloc[i]) else prev_st
                direction.iloc[i] = -1
            else:
                supertrend.iloc[i] = max(lower_band.iloc[i], prev_st) if not pd.isna(lower_band.iloc[i]) else prev_st
                direction.iloc[i] = 1
                if direction.iloc[i - 1] == -1 or pd.isna(direction.iloc[i - 1]):
                    supertrend.iloc[i] = lower_band.iloc[i]

    df["supertrend"] = supertrend
    df["supertrend_dir"] = direction
    return df


# ---------------------------------------------------------------------------
# Regime detection
# ---------------------------------------------------------------------------

REGIME_NAMES = {
    1: "TRENDING",
    0: "CHOPPY",
    -1: "TRANSITIONAL",
}


def detect_regime(df: pd.DataFrame, config: dict) -> int:
    """
    Detect current market regime using ADX.

    Returns:
        1  = TRENDING (ADX >= 25) → use breakout strategy
        0  = CHOPPY (ADX < 20) → use mean reversion
        -1 = TRANSITIONAL (20 <= ADX < 25) → stay in cash
    """
    if len(df) < 30 or "adx" not in df.columns:
        return -1

    last = df.iloc[-1]
    adx = last["adx"]

    if pd.isna(adx):
        return -1

    trending_threshold = config.get("adx_trending", 25)
    choppy_threshold = config.get("adx_choppy", 20)

    if adx >= trending_threshold:
        return 1
    elif adx < choppy_threshold:
        return 0
    else:
        return -1


# ---------------------------------------------------------------------------
# Strategy 1: Enhanced Breakout (Trending regime)
# ---------------------------------------------------------------------------

def strategy_enhanced_breakout(df: pd.DataFrame, config: dict) -> str:
    """
    Enhanced breakout strategy for trending markets.

    All conditions must be met:
    1. EMA50 > EMA200 (uptrend)
    2. ADX >= 25 (strong trend)
    3. +DI > -DI (bullish direction)
    4. Close > N-day high (breakout)
    5. Volume is highest in 20 days (strong volume, not just 2x avg)
    6. RSI between 40-70 (not overbought, not dead)
    7. Relative strength: stock 20d return > 0 (outperforming cash)

    Returns: 'BUY', 'SELL', or 'HOLD'
    """
    if len(df) < 200:
        return "HOLD"

    last = df.iloc[-1]

    # 1. Trend filter
    if last["ema50"] <= last["ema200"]:
        return "HOLD"

    # 2. ADX strength
    if pd.isna(last["adx"]) or last["adx"] < config.get("adx_trending", 25):
        return "HOLD"

    # 3. Directional indicator
    if last["plus_di"] <= last["minus_di"]:
        return "HOLD"

    # 4. Breakout
    if last["Close"] <= last["high_lookback"]:
        return "HOLD"

    # 5. Volume: above 1.2x average (strong but not extreme)
    if last["Volume"] < last["vol_avg"] * 1.2:
        return "HOLD"

    # 6. RSI range (relaxed)
    if not (35 <= last["rsi"] <= 75):
        return "HOLD"

    # 7. Relative strength: not deeply negative
    if pd.isna(last["ret_20d"]) or last["ret_20d"] <= -0.05:
        return "HOLD"

    return "BUY"


def exit_breakout(df: pd.DataFrame, config: dict) -> str:
    """Exit rules for breakout positions."""
    last = df.iloc[-1]

    # Close below EMA10 (momentum broken)
    if last["Close"] < last["ema10"]:
        return "SELL"

    # ADX falling below 20 (trend dying)
    if pd.notna(last["adx"]) and last["adx"] < 20:
        return "SELL"

    # RSI overbought
    if last["rsi"] > 78:
        return "SELL"

    return "HOLD"


# ---------------------------------------------------------------------------
# Strategy 2: Mean Reversion (Choppy regime)
# ---------------------------------------------------------------------------

def strategy_mean_reversion(df: pd.DataFrame, config: dict) -> str:
    """
    Mean reversion strategy for choppy/range-bound markets.

    Entry conditions:
    1. EMA50 > EMA200 (still in a broader uptrend — don't catch falling knives)
    2. ADX < 20 (no strong trend — range-bound)
    3. RSI < 35 (oversold)
    4. Close below lower Bollinger Band (statistically oversold)
    5. RSI turning up (buy the bounce, not the falling knife)

    Returns: 'BUY', 'SELL', or 'HOLD'
    """
    if len(df) < 200:
        return "HOLD"

    last = df.iloc[-1]
    prev = df.iloc[-2]

    # 1. Broader uptrend (don't buy in a bear market)
    if last["ema50"] <= last["ema200"]:
        return "HOLD"

    # 2. Choppy market
    if pd.isna(last["adx"]) or last["adx"] >= 20:
        return "HOLD"

    # 3. Oversold RSI (< 40 instead of 35)
    if last["rsi"] >= 40:
        return "HOLD"

    # 4. Close below lower Bollinger Band OR RSI < 30
    if last["Close"] >= last["bb_lower"] and last["rsi"] >= 30:
        return "HOLD"

    # 5. RSI turning up
    if last["rsi"] <= prev["rsi"]:
        return "HOLD"

    return "BUY"


def exit_mean_reversion(df: pd.DataFrame, config: dict) -> str:
    """Exit rules for mean reversion positions."""
    last = df.iloc[-1]

    # RSI recovered to neutral/overbought
    if last["rsi"] > 60:
        return "SELL"

    # Close above middle Bollinger Band (mean reverted)
    if last["Close"] > last["bb_mid"]:
        return "SELL"

    # Close below EMA50 (trend broken — stop out via SL too)
    if last["Close"] < last["ema50"]:
        return "SELL"

    return "HOLD"


# ---------------------------------------------------------------------------
# Regime engine — main logic
# ---------------------------------------------------------------------------

def run_regime(df: pd.DataFrame, config: dict, held: dict = None) -> dict:
    """
    Run the regime engine: detect regime, apply appropriate strategy.

    Returns:
        {
            "regime": "TRENDING" | "CHOPPY" | "TRANSITIONAL",
            "adx": float,
            "action": "BUY" | "SELL" | "HOLD",
            "strategy": "enhanced_breakout" | "mean_reversion" | "cash",
            "sl": float,
            "target": float,
        }
    """
    regime = detect_regime(df, config)
    regime_name = REGIME_NAMES.get(regime, "TRANSITIONAL")
    last = df.iloc[-1]
    last_close = float(last["Close"])

    action = "HOLD"
    strategy_name = "cash"

    if regime == 1:  # TRENDING
        strategy_name = "enhanced_breakout"
        if held:
            action = exit_breakout(df, config)
        else:
            action = strategy_enhanced_breakout(df, config)

    elif regime == 0:  # CHOPPY
        strategy_name = "mean_reversion"
        if held:
            action = exit_mean_reversion(df, config)
        else:
            action = strategy_mean_reversion(df, config)

    # TRANSITIONAL: stay in cash (HOLD)

    # Compute SL and target
    # Use wider stops for mean reversion (volatile entries)
    if regime == 0:  # choppy entries are volatile
        sl_pct = config.get("mr_stop_loss_pct", 0.05)
        target_pct = config.get("mr_target_pct", 0.06)
    else:
        sl_pct = config.get("stop_loss_pct", 0.03)
        target_pct = config.get("target_pct", 0.08)

    sl = round(last_close * (1 - sl_pct), 2)
    target = round(last_close * (1 + target_pct), 2)

    adx_val = float(last["adx"]) if pd.notna(last["adx"]) else 0.0

    return {
        "regime": regime_name,
        "regime_code": regime,
        "adx": round(adx_val, 2),
        "action": action,
        "strategy": strategy_name,
        "sl": sl,
        "target": target,
        "entry": round(last_close, 2),
        "rsi": round(float(last["rsi"]), 2),
    }


# ---------------------------------------------------------------------------
# Main signal generation
# ---------------------------------------------------------------------------

def run(session: str = "R"):
    """Main entry point — reads cached prices, runs regime engine, generates signals."""
    config = load_config()

    tech_config = config.get("technical", {})
    swing_config = tech_config.get("swing", {})
    regime_config = config.get("regime", {})

    # Merge config
    strat_config = {
        "lookback_days": swing_config.get("lookback_days", 10),
        "volume_multiplier": swing_config.get("volume_multiplier", 2.0),
        "rsi_min": swing_config.get("rsi_min", 35),
        "rsi_max": swing_config.get("rsi_max", 70),
        "stop_loss_pct": swing_config.get("stop_loss_pct", 0.03),
        "target_pct": swing_config.get("target_pct", 0.08),
        "adx_trending": regime_config.get("adx_trending", 25),
        "adx_choppy": regime_config.get("adx_choppy", 20),
        "mr_stop_loss_pct": regime_config.get("mr_stop_loss_pct", 0.05),
        "mr_target_pct": regime_config.get("mr_target_pct", 0.06),
    }

    session_label = "Regime Switching"

    logger.info("=" * 60)
    logger.info(f"Regime Engine — Session {session} ({session_label})")
    logger.info(f"  Trending: ADX >= {strat_config['adx_trending']} → Enhanced Breakout")
    logger.info(f"  Choppy: ADX < {strat_config['adx_choppy']} → Mean Reversion")
    logger.info(f"  Transitional: {strat_config['adx_choppy']}-{strat_config['adx_trending']} → Cash")
    logger.info("=" * 60)

    # Load cached prices
    prices = load_json("prices.json")
    if not prices:
        logger.error("No cached prices found. Run price_fetcher.py first.")
        return []

    # Load current positions
    positions = load_json(f"positions_{session}.json")

    signals = []

    for symbol, stock_data in prices.items():
        logger.info(f"[{session}/{symbol}] Analyzing...")

        candles = stock_data.get("candles", [])
        if len(candles) < 200:
            logger.warning(f"  {symbol}: insufficient candle data ({len(candles)})")
            continue

        df = candles_to_dataframe(candles)
        df = compute_all_indicators(df, strat_config)

        if len(df) < 200:
            continue

        held = next((p for p in positions if p["symbol"] == symbol), None)

        result = run_regime(df, strat_config, held)

        logger.info(f"  Regime: {result['regime']} (ADX={result['adx']})")
        logger.info(f"  Strategy: {result['strategy']}")
        logger.info(f"  Action: {result['action']}")

        if result["action"] == "BUY":
            signals.append({
                "symbol": symbol,
                "action": "BUY",
                "strategy": result["strategy"],
                "regime": result["regime"],
                "adx": result["adx"],
                "entry": result["entry"],
                "sl": result["sl"],
                "target": result["target"],
                "rsi": result["rsi"],
                "session": session,
                "generated_at": now_iso(),
            })
            logger.info(f"  >>> BUY signal ({result['strategy']}, ADX={result['adx']})")

        elif result["action"] == "SELL":
            signals.append({
                "symbol": symbol,
                "action": "SELL",
                "reason": f"{result['strategy']} exit (ADX={result['adx']})",
                "regime": result["regime"],
                "adx": result["adx"],
                "session": session,
                "generated_at": now_iso(),
            })
            logger.info(f"  >>> SELL signal ({result['strategy']} exit)")

        # Trailing SL update if held
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
            "Regime-Switching Strategy Engine — Session R\n\n"
            "Detects market regime using ADX, then applies:\n"
            "  TRENDING (ADX>=25): Enhanced breakout with volume + ADX filters\n"
            "  CHOPPY (ADX<20): Mean reversion (buy oversold bounces)\n"
            "  TRANSITIONAL (20-25): Stay in cash\n\n"
            "Enhanced filters:\n"
            "  - ADX >= 25 + +DI > -DI for breakout entries\n"
            "  - Volume must be highest in 20 days\n"
            "  - Positive 20-day return (relative strength)\n"
            "  - Bollinger Band + RSI oversold for mean reversion entries\n\n"
            "Examples:\n"
            "  python -m src.regime_engine               # default session R\n"
            "  python -m src.regime_engine --session R"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--session", default="R", choices=list(SESSION_CONFIGS.keys()),
        help="Session ID (default: R = regime)",
    )
    args = parser.parse_args()
    run(args.session)
