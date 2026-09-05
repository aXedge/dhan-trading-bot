"""
Momentum Acceleration Strategy — based on user's personal trading methodology.

Entry framework:
  1. MA Gap Acceleration: EMA5 > EMA20 AND the gap between them is widening
     (today's gap > gap N days ago). This catches accelerating momentum.
  2. MACD Confirmation: MACD line > signal line + histogram positive.
  3. Support Proximity: Price is near a support level (EMA20, recent swing low,
     or 50% Fibonacci retracement of the recent swing).
  4. Trend Filter: EMA50 > EMA200 (broader uptrend intact).

Exit framework:
  - MA gap deceleration: gap between EMA5 and EMA20 starts narrowing
  - MACD bearish: MACD line crosses below signal line
  - Stop loss: ATR-based (adapts to volatility) or fixed %
  - Target: Previous resistance or Fibonacci extension (1.272x or 1.618x)

Holding period: 1-3 months (positional)
Risk: ATR-based stop loss (typically 5-8%)
Target: 15-20% (2:1 to 3:1 reward-risk)
"""

import pandas as pd
import numpy as np
from common.indicators import (
    add_emas, add_rsi, add_adx, add_volume_indicators,
    add_atr, add_returns, add_macd
)


# ---------------------------------------------------------------------------
# Indicator computation
# ---------------------------------------------------------------------------

def prepare(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """
    Compute indicators needed by the momentum acceleration strategy.

    Key indicators:
    - EMA5, EMA10, EMA20, EMA50, EMA200
    - MA gap: (EMA5 - EMA20) / EMA20 (as % of price)
    - MA gap slope: change in gap over N days
    - MACD (line, signal, histogram)
    - ATR for volatility-based stops
    - Swing highs/lows for support/resistance
    - Fibonacci retracement levels
    """
    df = df.copy()

    # EMAs — need both short (5, 10) and medium (20, 50) and long (200)
    df["ema5"] = df["Close"].ewm(span=5, adjust=False).mean()
    df["ema10"] = df["Close"].ewm(span=10, adjust=False).mean()
    df["ema20"] = df["Close"].ewm(span=20, adjust=False).mean()
    df["ema50"] = df["Close"].ewm(span=50, adjust=False).mean()
    df["ema200"] = df["Close"].ewm(span=200, adjust=False).mean()

    # MA gap (short vs medium) — the core signal
    df["ma_gap"] = (df["ema5"] - df["ema20"]) / df["Close"]  # as % of price

    # MA gap slope — is the gap widening or narrowing?
    gap_lookback = config.get("gap_lookback", 3)  # compare gap today vs 3 days ago
    df["ma_gap_prev"] = df["ma_gap"].shift(gap_lookback)
    df["ma_gap_slope"] = df["ma_gap"] - df["ma_gap_prev"]  # positive = widening

    # Also track EMA10 vs EMA50 gap for medium-term momentum
    df["ma_gap_med"] = (df["ema10"] - df["ema50"]) / df["Close"]

    # RSI
    rsi_period = config.get("rsi_period", 14)
    delta = df["Close"].diff()
    gain = delta.clip(lower=0).rolling(rsi_period).mean()
    loss = (-delta.clip(upper=0)).rolling(rsi_period).mean()
    rs = gain / loss
    df["rsi"] = 100 - (100 / (1 + rs))

    # MACD
    df = add_macd(df)

    # ADX
    df = add_adx(df, config.get("adx_period", 14))

    # ATR for volatility-based stops
    df = add_atr(df, 14)

    # Volume
    df = add_volume_indicators(df, config.get("volume_avg_period", 20))

    # Returns
    df = add_returns(df)

    # Swing high/low for support/resistance (20-day window)
    df["swing_low_20"] = df["Low"].rolling(20).min()
    df["swing_high_20"] = df["High"].rolling(20).max()

    # Fibonacci retracement of recent swing (20-day)
    swing_range = df["swing_high_20"] - df["swing_low_20"]
    df["fib_382"] = df["swing_high_20"] - 0.382 * swing_range
    df["fib_500"] = df["swing_high_20"] - 0.500 * swing_range
    df["fib_618"] = df["swing_high_20"] - 0.618 * swing_range

    # Distance to support (EMA20 or Fib 50%)
    df["dist_to_ema20"] = (df["Close"] - df["ema20"]) / df["Close"]
    df["dist_to_fib50"] = (df["Close"] - df["fib_500"]) / df["Close"]

    return df


# ---------------------------------------------------------------------------
# Entry logic
# ---------------------------------------------------------------------------

def should_enter(last, config: dict) -> bool:
    """
    Momentum acceleration entry.

    All conditions must be met:
    1. Trend filter: EMA50 > EMA200 (broader uptrend)
    2. Short > Medium: EMA5 > EMA20 (short-term momentum positive)
    3. Gap acceleration: MA gap is widening (gap today > gap N days ago)
    4. MACD confirmation: MACD line > signal line + histogram > 0
    5. Near support: price within tolerance of EMA20 or Fib 50% or swing low
    6. RSI not overbought: RSI < 70 (room to run)
    7. ADX >= 15 (enough trend to sustain)
    """
    # 1. Broader uptrend
    if last["ema50"] <= last["ema200"]:
        return False

    # 2. Short-term above medium-term
    if last["ema5"] <= last["ema20"]:
        return False

    # 3. MA gap acceleration — the core signal
    # Gap must be positive AND widening
    if pd.isna(last["ma_gap"]) or last["ma_gap"] <= 0:
        return False
    if pd.isna(last["ma_gap_slope"]):
        return False
    # Gap must be widening (slope > 0) or at least not narrowing significantly
    min_slope = config.get("min_gap_slope", 0.0)  # default: just not narrowing
    if last["ma_gap_slope"] < min_slope:
        return False

    # 4. MACD confirmation
    if config.get("use_macd", True):
        if pd.isna(last.get("macd_line")) or pd.isna(last.get("macd_signal")):
            return False
        if last["macd_line"] <= last["macd_signal"]:
            return False
        if last["macd_hist"] <= 0:
            return False

    # 5. Near support level — price should be near a support zone
    # Check if close is within tolerance of EMA20, Fib 50%, or recent swing low
    if config.get("use_support", True):
        tolerance = config.get("support_tolerance", 0.04)  # within 4% of a support level

        near_ema20 = abs(last["dist_to_ema20"]) < tolerance
        near_fib50 = abs(last["dist_to_fib50"]) < tolerance if pd.notna(last.get("dist_to_fib50")) else False
        near_swing_low = (last["Close"] - last["swing_low_20"]) / last["Close"] < tolerance if pd.notna(last.get("swing_low_20")) else False

        if not (near_ema20 or near_fib50 or near_swing_low):
            return False

    # 6. RSI not overbought
    rsi_max = config.get("rsi_max", 70)
    if last["rsi"] > rsi_max:
        return False

    # 7. ADX minimum
    adx_min = config.get("adx_min", 15)
    if pd.isna(last["adx"]) or last["adx"] < adx_min:
        return False

    return True


# ---------------------------------------------------------------------------
# Exit logic
# ---------------------------------------------------------------------------

def should_exit(last, config: dict, prev=None) -> bool:
    """
    Momentum acceleration exit.

    Exits when momentum starts decelerating:
    1. MA gap narrowing: gap today < gap N days ago (momentum fading)
    2. MACD bearish: MACD line < signal line
    3. Price below EMA20 (support broken)
    4. RSI overbought (take profits)
    """
    # MA gap deceleration — primary exit signal
    if pd.notna(last.get("ma_gap_slope")):
        if last["ma_gap_slope"] < config.get("exit_gap_slope", -0.002):
            # Gap is narrowing significantly — momentum fading
            return True

    # MACD bearish
    if pd.notna(last.get("macd_line")) and pd.notna(last.get("macd_signal")):
        if last["macd_line"] < last["macd_signal"] and last["macd_hist"] < 0:
            return True

    # Price below EMA20 (support broken)
    if last["Close"] < last["ema20"]:
        return True

    # RSI overbought — take profits
    rsi_exit = config.get("rsi_exit", 75)
    if last["rsi"] > rsi_exit:
        return True

    return False


# ---------------------------------------------------------------------------
# Stop loss and target (ATR-based)
# ---------------------------------------------------------------------------

def get_stop_loss(entry_price: float, config: dict, atr: float = None) -> float:
    """
    ATR-based stop loss. Falls back to fixed % if ATR not available.

    ATR multiplier of 2x gives a stop that adapts to volatility:
    - Low volatility stock: tighter stop (e.g., 4%)
    - High volatility stock: wider stop (e.g., 8%)
    """
    if atr and pd.notna(atr) and atr > 0:
        atr_mult = config.get("atr_multiplier", 2.0)
        return round(entry_price - (atr * atr_mult), 2)
    else:
        sl_pct = config.get("stop_loss_pct", 0.06)
        return round(entry_price * (1 - sl_pct), 2)


def get_target(entry_price: float, config: dict, atr: float = None) -> float:
    """
    ATR-based target. 3x ATR for 1.5:1 reward-risk with 2x ATR stop.
    Falls back to fixed % if ATR not available.
    """
    if atr and pd.notna(atr) and atr > 0:
        atr_mult = config.get("atr_target_multiplier", 3.0)
        return round(entry_price + (atr * atr_mult), 2)
    else:
        target_pct = config.get("target_pct", 0.15)
        return round(entry_price * (1 + target_pct), 2)


# ---------------------------------------------------------------------------
# Default config
# ---------------------------------------------------------------------------

DEFAULT_CONFIG = {
    # MA gap settings
    "gap_lookback": 3,             # compare gap today vs 3 days ago
    "min_gap_slope": 0.0,          # gap must not be narrowing (0 = flat or widening)

    # Support proximity
    "use_support": True,
    "support_tolerance": 0.04,     # within 4% of a support level

    # MACD
    "use_macd": True,

    # RSI
    "rsi_period": 14,
    "rsi_max": 70,                 # don't enter overbought
    "rsi_exit": 75,                # exit when overbought

    # ADX
    "adx_period": 14,
    "adx_min": 15,

    # Volume
    "volume_avg_period": 20,

    # Risk management
    "stop_loss_pct": 0.06,         # 6% fixed SL fallback
    "target_pct": 0.15,            # 15% fixed target fallback
    "atr_multiplier": 2.0,          # SL = 2x ATR
    "atr_target_multiplier": 3.0,  # Target = 3x ATR (1.5:1 RR)

    # Exit
    "exit_gap_slope": -0.002,      # exit if gap narrows by more than 0.2%

    # Cooldown
    "cooldown_days": 0,
}
