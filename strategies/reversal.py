"""
Short-Term Reversal Strategy (Optimized)
==========================================

Based on parameter sweep of 108 combinations on 41 NIFTY 50 stocks (2-year backtest).
Best config: PF 1.33, 66.7% win rate, 81 trades.

The edge exploits retail overreaction in Indian equities (35-45% retail participation).
Stocks that fall sharply tend to bounce back — the RSI exit at 45 captures the bounce
before it stalls (waiting for RSI 50 gives back gains).

Entry:
  1. 5-day return < -4% (sharp decline)
  2. RSI < 40 (oversold)
  3. Close > EMA200 (long-term uptrend intact — buying the dip, not catching falling knife)
  4. No gap down > 10% today (avoid panic continuation)

Exit:
  - RSI > 45 (mean reversion bounce captured — exits early before stall)
  - 5% stop loss
  - 8% target

Backtest results (full NIFTY 50, 41 stocks, Sep 2023 - Sep 2025):
  PF: 1.33 | Win rate: 66.7% | Trades: 81 | Avg win: 2.8% | Avg loss: -4.1%

Curated basket (12 stocks): PF 2.43, 72% win rate, 32 trades
"""

import pandas as pd
import numpy as np

DEFAULT_CONFIG = {
    # Risk management
    "stop_loss_pct": 0.05,          # 5% stop loss (optimized)
    "target_pct": 0.08,            # 8% target (optimized)

    # Entry parameters (optimized)
    "min_5d_return": -0.04,         # stock down at least 4% in 5 days (was -3%)
    "max_rsi_entry": 40,           # RSI < 40 (oversold)
    "ema_long": 200,               # must be above EMA200
    "max_gap_down": -0.10,         # skip if gap down > 10%

    # Exit parameters (KEY OPTIMIZATION)
    "rsi_exit": 45,                # exit at RSI 45 (was 50 — captures bounce before stall)

    # Indicator parameters
    "rsi_period": 14,
    "atr_period": 14,
}


def prepare(df, config):
    df = df.copy()

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)

    df["ema200"] = df["Close"].ewm(span=config.get("ema_long", 200)).mean()

    rsi_period = config.get("rsi_period", 14)
    delta = df["Close"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / rsi_period, min_periods=rsi_period).mean()
    avg_loss = loss.ewm(alpha=1 / rsi_period, min_periods=rsi_period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    df["rsi"] = 100 - (100 / (1 + rs))

    df["ret_5d"] = df["Close"].pct_change(5)
    df["gap"] = df["Open"] / df["Close"].shift(1) - 1

    return df


def should_enter(last, config, prev=None):
    if pd.isna(last.get("ema200")) or pd.isna(last.get("rsi")) or pd.isna(last.get("ret_5d")):
        return False

    close = last["Close"]

    # 1. Sharp 5-day decline (>= 4% drop)
    if last["ret_5d"] > config.get("min_5d_return", -0.04):
        return False

    # 2. RSI oversold
    if last["rsi"] > config.get("max_rsi_entry", 40):
        return False

    # 3. Long-term trend intact
    if close < last["ema200"]:
        return False

    # 4. No extreme gap down
    if last.get("gap", 0) < config.get("max_gap_down", -0.10):
        return False

    return True


def should_exit(last, config, prev=None):
    if pd.isna(last.get("rsi")):
        return False

    # RSI has bounced to 45 — mean reversion captured
    if last["rsi"] > config.get("rsi_exit", 45):
        return True

    return False
