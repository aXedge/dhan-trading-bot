"""
Positional Pullback Strategy (Optimized)
==========================================

Based on parameter sweep of 81 combinations on 41 NIFTY 50 stocks (2-year backtest).
Best config: PF 1.13, 47.7% win rate, 195 trades.

Trend-following strategy that buys pullbacks to EMA20 in confirmed uptrends.
The key optimization was lowering ADX from 15 to 10 — the stricter ADX filter
was rejecting valid signals. RSI exit at 70 captures the momentum peak.

Entry:
  1. Close > EMA200 (long-term uptrend)
  2. EMA5 > EMA20 (short-term uptrend)
  3. RSI 40-60 (pullback zone, not overbought/oversold)
  4. ADX >= 10 (has some trend strength — lowered from 15)
  5. Near EMA20 (within 3% — the pullback level)

Exit:
  - RSI > 70 (momentum peak — captures the run)
  - 7% stop loss
  - 15% target

Backtest results (full NIFTY 50, 41 stocks, Sep 2023 - Sep 2025):
  PF: 1.13 | Win rate: 47.7% | Trades: 195 | Avg win: 8.0% | Avg loss: -6.5%

Curated basket (12 stocks): PF 1.31, 52% win rate, 56 trades
"""

import pandas as pd
import numpy as np

DEFAULT_CONFIG = {
    # Risk management
    "stop_loss_pct": 0.07,          # 7% stop loss (optimized)
    "target_pct": 0.15,            # 15% target (optimized)

    # Entry parameters
    "ema_fast": 5,
    "ema_slow": 20,
    "ema_trend": 50,
    "ema_long": 200,
    "rsi_entry_min": 40,
    "rsi_entry_max": 60,
    "adx_min": 10,                  # KEY OPTIMIZATION: lowered from 15 to 10
    "near_ema20_pct": 0.03,         # within 3% of EMA20

    # Exit parameters
    "rsi_exit": 70,                 # KEY OPTIMIZATION: was 72, now 70

    # Indicator parameters
    "rsi_period": 14,
    "atr_period": 14,
    "macd_fast": 12,
    "macd_slow": 26,
    "macd_signal": 9,
}


def prepare(df, config):
    df = df.copy()

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)

    df["ema5"] = df["Close"].ewm(span=config.get("ema_fast", 5)).mean()
    df["ema20"] = df["Close"].ewm(span=config.get("ema_slow", 20)).mean()
    df["ema50"] = df["Close"].ewm(span=config.get("ema_trend", 50)).mean()
    df["ema200"] = df["Close"].ewm(span=config.get("ema_long", 200)).mean()

    macd_fast = df["Close"].ewm(span=config.get("macd_fast", 12)).mean()
    macd_slow = df["Close"].ewm(span=config.get("macd_slow", 26)).mean()
    df["macd"] = macd_fast - macd_slow
    df["macd_signal"] = df["macd"].ewm(span=config.get("macd_signal", 9)).mean()
    df["macd_hist"] = df["macd"] - df["macd_signal"]

    rsi_period = config.get("rsi_period", 14)
    delta = df["Close"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / rsi_period, min_periods=rsi_period).mean()
    avg_loss = loss.ewm(alpha=1 / rsi_period, min_periods=rsi_period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    df["rsi"] = 100 - (100 / (1 + rs))

    atr_period = config.get("atr_period", 14)
    high_low = df["High"] - df["Low"]
    high_close = (df["High"] - df["Close"].shift()).abs()
    low_close = (df["Low"] - df["Close"].shift()).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df["atr"] = tr.ewm(alpha=1 / atr_period, min_periods=atr_period).mean()

    adx_period = config.get("atr_period", 14)
    plus_dm_raw = df["High"].diff()
    minus_dm_raw = -df["Low"].diff()
    plus_dm = plus_dm_raw.where((plus_dm_raw > minus_dm_raw) & (plus_dm_raw > 0), 0.0)
    minus_dm = minus_dm_raw.where((minus_dm_raw > plus_dm_raw) & (minus_dm_raw > 0), 0.0)
    plus_di = 100 * (plus_dm.ewm(alpha=1 / adx_period, min_periods=adx_period).mean() / df["atr"])
    minus_di = 100 * (minus_dm.ewm(alpha=1 / adx_period, min_periods=adx_period).mean() / df["atr"])
    dx = 100 * ((plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan))
    df["adx"] = dx.ewm(alpha=1 / adx_period, min_periods=adx_period).mean()

    return df


def should_enter(last, config, prev=None):
    if pd.isna(last.get("ema200")) or pd.isna(last.get("adx")) or pd.isna(last.get("rsi")):
        return False

    close = last["Close"]

    # 1. Long-term uptrend
    if close < last["ema200"]:
        return False

    # 2. Short-term uptrend
    if last["ema5"] <= last["ema20"]:
        return False

    # 3. RSI in pullback zone (40-60)
    rsi = last.get("rsi", 50)
    if rsi < config.get("rsi_entry_min", 40) or rsi > config.get("rsi_entry_max", 60):
        return False

    # 4. ADX >= 10 (trend strength — lowered from 15)
    if last["adx"] < config.get("adx_min", 10):
        return False

    # 5. Near EMA20 (within 3%)
    if abs(close - last["ema20"]) > close * config.get("near_ema20_pct", 0.03):
        return False

    return True


def should_exit(last, config, prev=None):
    if pd.isna(last.get("rsi")):
        return False

    # RSI > 70 — momentum peak, take profit
    if last["rsi"] > config.get("rsi_exit", 70):
        return True

    return False
