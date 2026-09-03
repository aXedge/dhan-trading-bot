"""
Price Fetcher — fetches OHLCV data for all basket stocks once per cycle.

Saves to data/prices.json so that technicals.py --session A/B/C can all
run on the same cached price data without re-fetching from yfinance.

Runs at 9:20 AM IST via cron, before any session's technicals scan.
No Dhan API access needed — uses yfinance only.
"""

import json
import sys
import os
from datetime import datetime

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils import load_config, load_json, save_json, setup_logger, now_iso

logger = setup_logger(__name__, "price_fetcher.log")


def fetch_all_prices(basket: list, days: int) -> dict:
    """
    Fetch price history for all stocks in the basket.
    
    Returns:
        Dict mapping {symbol: {ohlcv: [...], indicators_computed: False}}
    """
    import yfinance as yf
    import pandas as pd
    
    prices = {}
    total = len(basket)
    
    for i, stock in enumerate(basket, 1):
        symbol = stock["symbol"]
        logger.info(f"[{i}/{total}] Fetching {symbol}...")
        
        try:
            ticker = yf.Ticker(f"{symbol}.NS")
            df = ticker.history(period=f"{days}d")
            
            if len(df) < 50:
                logger.warning(f"  {symbol}: insufficient data ({len(df)} days)")
                continue
            
            # Convert to serializable format
            # Reset index to get Date as a column, then convert to records
            df = df.reset_index()
            # Rename columns to lowercase for consistency
            df.columns = [c.lower().replace(' ', '_') for c in df.columns]
            # Convert date to string
            if 'date' in df.columns:
                df['date'] = df['date'].astype(str)
            elif 'datetime' in df.columns:
                df['date'] = df['datetime'].astype(str)
                df = df.drop(columns=['datetime'])
            
            records = df.to_dict(orient="records")
            
            prices[symbol] = {
                "symbol": symbol,
                "name": stock.get("name", symbol),
                "candles": records,
                "fetched_at": now_iso(),
            }
            
            last_close = float(df["close"].iloc[-1])
            logger.info(f"  {symbol}: {len(records)} candles, last close Rs.{last_close:.2f}")
            
        except Exception as e:
            logger.error(f"  {symbol}: failed to fetch: {e}")
    
    return prices


def run():
    """Main entry point — fetch prices for all basket stocks."""
    config = load_config()
    days = config["technical"]["price_history_days"]
    
    logger.info("=" * 60)
    logger.info("Price Fetcher — starting")
    logger.info(f"History: {days} days")
    logger.info("=" * 60)
    
    # Load basket (shared across all sessions)
    basket = load_json("basket.json")
    if not basket:
        logger.error("No basket found. Run Layer 1 (screener.py) first.")
        return
    
    logger.info(f"Fetching prices for {len(basket)} stocks...")
    
    prices = fetch_all_prices(basket, days)
    
    # Save to shared prices.json
    save_json(prices, "prices.json")
    
    logger.info("=" * 60)
    logger.info(f"Price fetch complete: {len(prices)}/{len(basket)} stocks cached to data/prices.json")
    logger.info("=" * 60)
    
    return prices


if __name__ == "__main__":
    run()
