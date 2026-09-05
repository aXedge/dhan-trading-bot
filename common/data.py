"""
Data fetching and candle serialization.

Provides:
  - fetch_yfinance: download OHLCV from yfinance with local caching
  - candles_to_dataframe: convert cached JSON candles to pandas DataFrame
  - load_basket: load basket.json
  - load_prices: load cached prices.json
  - fetch_and_cache_all: fetch all basket stocks, save to prices.json
"""

import os
import json
import pandas as pd
from pathlib import Path
from typing import List, Dict, Optional

from common.config import DATA_DIR, PROJECT_ROOT


# ---------------------------------------------------------------------------
# Candle serialization
# ---------------------------------------------------------------------------

def candles_to_dataframe(candles: list) -> pd.DataFrame:
    """Convert cached candle records (list of dicts) to a pandas DataFrame."""
    df = pd.DataFrame(candles)
    col_map = {
        "open": "Open", "high": "High", "low": "Low",
        "close": "Close", "volume": "Volume",
        "Open": "Open", "High": "High", "Low": "Low",
        "Close": "Close", "Volume": "Volume",
    }
    df = df.rename(columns={k: v for k, v in col_map.items() if k in df.columns})
    for col in ["Open", "High", "Low", "Close", "Volume"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def dataframe_to_candles(df: pd.DataFrame) -> list:
    """Convert a pandas DataFrame to serializable candle records."""
    df = df.copy()
    df.index = df.index.astype(str)
    records = df.to_dict(orient="records")
    return records


# ---------------------------------------------------------------------------
# Basket loading
# ---------------------------------------------------------------------------

def load_basket() -> list:
    """Load basket.json from data/ directory."""
    basket_path = DATA_DIR / "basket.json"
    if not basket_path.exists():
        return []
    with open(basket_path, "r") as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Price loading (from cache)
# ---------------------------------------------------------------------------

def load_prices() -> dict:
    """Load cached prices.json."""
    prices_path = DATA_DIR / "prices.json"
    if not prices_path.exists():
        return {}
    with open(prices_path, "r") as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# yfinance fetching with caching
# ---------------------------------------------------------------------------

def fetch_yfinance(
    symbols: List[str],
    start: str,
    end: str,
    cache_dir: str = "/tmp/yf_cache",
) -> Dict[str, pd.DataFrame]:
    """
    Fetch OHLCV data from yfinance with local caching.

    Args:
        symbols: List of NSE symbols (without .NS suffix)
        start: Start date (YYYY-MM-DD)
        end: End date (YYYY-MM-DD)
        cache_dir: Directory for caching downloaded data

    Returns:
        Dict mapping {symbol: DataFrame}
    """
    import yfinance as yf

    os.makedirs(cache_dir, exist_ok=True)
    cache_key = f"{start}_{end}"
    cache_file = os.path.join(cache_dir, f"cache_{cache_key}.pkl")

    # Try loading from cache
    cache = {}
    if os.path.exists(cache_file):
        try:
            cache = pd.read_pickle(cache_file)
        except Exception:
            cache = {}

    result = {}
    for symbol in symbols:
        if symbol in cache and not cache[symbol].empty:
            result[symbol] = cache[symbol]
        else:
            try:
                ticker = yf.Ticker(f"{symbol}.NS")
                hist = ticker.history(start=start, end=end, auto_adjust=True)
                if not hist.empty:
                    hist.index = pd.DatetimeIndex(hist.index)
                    cache[symbol] = hist
                    result[symbol] = hist
            except Exception:
                pass

    # Save cache
    try:
        pd.to_pickle(cache, cache_file)
    except Exception:
        pass

    return result


def fetch_and_cache_all(
    basket: list,
    days: int = 365,
    output_file: str = "prices.json",
) -> dict:
    """
    Fetch price history for all basket stocks and save to prices.json.
    Used by the live price_fetcher.py cron job.

    Args:
        basket: List of dicts with 'symbol' keys
        days: Number of days of history
        output_file: Output filename in data/ directory

    Returns:
        Dict mapping {symbol: {candles: [...], ...}}
    """
    import yfinance as yf

    prices = {}
    total = len(basket)

    for i, stock in enumerate(basket, 1):
        symbol = stock["symbol"]
        try:
            ticker = yf.Ticker(f"{symbol}.NS")
            df = ticker.history(period=f"{days}d")

            if len(df) < 50:
                continue

            # Convert to serializable format
            df_reset = df.reset_index()
            candles = []
            for _, row in df_reset.iterrows():
                candles.append({
                    "date": str(row["Date"]),
                    "open": float(row["Open"]),
                    "high": float(row["High"]),
                    "low": float(row["Low"]),
                    "close": float(row["Close"]),
                    "volume": int(row["Volume"]),
                })

            prices[symbol] = {"candles": candles}

        except Exception:
            pass

    # Save to data/
    output_path = DATA_DIR / output_file
    with open(output_path, "w") as f:
        json.dump(prices, f)

    return prices


# ---------------------------------------------------------------------------
# Stock baskets for backtesting
# ---------------------------------------------------------------------------

NIFTY_50 = [
    "RELIANCE", "TCS", "HDFCBANK", "INFY", "ICICIBANK",
    "BHARTIARTL", "SBIN", "LT", "HINDUNILVR", "ITC",
    "AXISBANK", "KOTAKBANK", "BAJFINANCE", "MARUTI", "ASIANPAINT",
    "HCLTECH", "TITAN", "SUNPHARMA", "ULTRACEMCO", "NESTLEIND",
    "POWERGRID", "NTPC", "TATASTEEL", "M&M", "ONGC",
    "COALINDIA", "WIPRO", "HDFCLIFE", "SBILIFE", "TECHM",
    "DIVISLAB", "DRREDDY", "GRASIM", "CIPLA", "BAJAJFINSV",
    "ADANIPORTS", "JSWSTEEL", "SHRIRAMFIN", "ADANIENT",
    "BRITANNIA", "EICHERMOT", "HEROMOTOCO", "BAJAJ-AUTO",
    "TATACONSUM", "BPCL", "INDUSINDBK", "LICI", "DMART",
]

MIDCAP_75 = [
    "TATAPOWER", "PNB", "IOC", "VEDL", "BPCL",
    "SHRIRAMFIN", "BAJAJFINSV", "DMART", "JINDALSTEL", "SBILIFE",
    "PFC", "RECLTD", "NHPC", "TATAINVEST", "CHOLAFIN",
    "LICI", "TVSMOTOR", "MOTHERSON", "INDIGO", "BANKBARODA",
    "IDFCFIRSTB", "HINDPETRO", "TATAELXSI", "BANDHANBNK", "MAXHEALTH",
    "UPL", "GODREJPROP", "AMBUJACEM", "MRF", "PAGEIND",
    "DABUR", "COLPAL", "SIEMENS", "ABB", "BEL",
    "BHEL", "GAIL", "ONGC", "NMDC", "SAIL",
    "TATACHEM", "PIIND", "COROMANDEL", "CHAMBLFERT", "DEEPAKNTR",
    "SRF", "ASTRAL", "POLYCAB", "CROMPTON", "VOLTAS",
    "CONCOR", "IRCTC", "RBLBANK", "YESBANK", "FEDERALBNK",
    "CANBK", "UNIONBANK", "INDIANB", "MAHABANK", "AUBANK",
    "JKCEMENT", "DALBHARAT", "ACC", "RAMCOCEM", "BERGEPAINT",
    "EICHERMOT", "HEROMOTOCO", "BAJAJ-AUTO", "MARICO",
    "BIOCON", "LUPIN", "HAVELLS", "TRENT",
    "BATAINDIA", "ABFRL", "SUPREMEIND",
]
