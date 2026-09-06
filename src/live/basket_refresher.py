#!/usr/bin/env python3
"""
Basket Refresher — Periodic Stock Discovery
=============================================

Downloads the latest NIFTY 50 + NIFTY Midcap 100 constituent lists from NSE,
fetches 2 years of price history for each, runs both strategies (reversal + pullback)
in backtest mode, and ranks stocks by blended profit factor.

Outputs a curated basket of top-performing stocks to data/basket_curated.json.
Designed to run monthly (first of each month) via cron.

Usage:
    python src/live/basket_refresher.py
"""

import json
import os
import sys
import time
import io
import csv
import tempfile
from datetime import date

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.dirname(SCRIPT_DIR)
REPO_ROOT = os.path.dirname(SRC_DIR)
sys.path.insert(0, SRC_DIR)
sys.path.insert(0, REPO_ROOT)

import pandas as pd
import numpy as np
import yfinance as yf
import warnings
warnings.filterwarnings('ignore')

# ========================================
# CONFIG
# ========================================
BASKET_OUTPUT = os.path.join(REPO_ROOT, "data", "basket_curated.json")
PRICE_CACHE_DIR = os.path.join(REPO_ROOT, "data", "price_cache")

NIFTY50_CSV_URL = "https://archives.nseindia.com/content/indices/ind_nifty50list.csv"
MIDCAP100_CSV_URL = "https://archives.nseindia.com/content/indices/ind_niftymidcap100list.csv"

# Statistical filters — prevent inflated PF from small samples
MIN_TOTAL_TRADES = 10        # Need enough total trades for significance
MIN_TRADES_PER_STRATEGY = 3  # At least 3 trades in each strategy
MIN_BLENDED_PF = 1.2         # Must be meaningfully profitable
MAX_PF_CAP = 10.0            # Cap individual strategy PF at 10 (prevents infinity artifacts)
MIN_PRICE_HISTORY_DAYS = 250 # ~1 year of trading days — exclude recent IPOs
MAX_BASKET_SIZE = 30

REVERSAL_CONFIG = {
    'sl_pct': 5, 'target_pct': 8,
    'rsi_entry': 40, 'rsi_exit': 45,
    'ret_5d_threshold': -4,
}

PULLBACK_CONFIG = {
    'sl_pct': 7, 'target_pct': 15,
    'rsi_exit': 70, 'adx_min': 10,
    'rsi_entry_low': 45, 'rsi_entry_high': 60,
}

# Fallback constituent lists (used if NSE website is unreachable)
NIFTY50_FALLBACK = [
    "ADANIENT","ADANIPORTS","APOLLOHOSP","ASIANPAINT","AXISBANK","BAJAJ-AUTO",
    "BAJFINANCE","BAJAJFINSV","BPCL","BHARTIARTL","BRITANNIA","CIPLA","COALINDIA",
    "DIVISLAB","DRREDDY","EICHERMOT","GRASIM","HCLTECH","HDFCBANK","HDFCLIFE",
    "HEROMOTOCO","HINDALCO","HINDUNILVR","ICICIBANK","INDUSINDBK","INFY","ITC",
    "JSWSTEEL","KOTAKBANK","LT","M&M","MARUTI","NESTLEIND","NTPC","ONGC","POWERGRID",
    "RELIANCE","SBILIFE","SBIN","SUNPHARMA","TCS","TATACONSUM","TATASTEEL","TITAN",
    "TMPV","ULTRACEMCO","WIPRO",
]

MIDCAP100_FALLBACK = [
    "360ONE","APLAPOLLO","AUBANK","ATGL","ABCAPITAL","ALKEM","ASHOKLEY","ASTRAL",
    "AUROPHARMA","BSE","BANKINDIA","BDL","BHARATFORG","BHEL","GROWW","BIOCON",
    "BLUESTARCO","COCHINSHIP","COFORGE","COLPAL","CONCOR","COROMANDEL","DABUR",
    "DIXON","EXIDEIND","NYKAA","FEDERALBNK","FORTIS","GVT&D","GMRAIRPORT",
    "GLENMARK","GODFRYPHLP","GODREJPROP","HAVELLS","HEROMOTOCO","HINDPETRO",
    "POWERINDIA","HUDCO","ICICIGI","ICICIAMC","IDFCFIRSTB","INDIANB","IRCTC",
    "IREDA","INDUSTOWER","INDUSINDBK","NAUKRI","JSWENERGY","JUBLFOOD","KEI",
    "KPITTECH","KALYANKJIL","LTF","LGEINDIA","LICHSGFIN","LAURUSLABS","LENSKART",
    "LUPIN","MRF","M&MFIN","MANKIND","MARICO","MFSL","MOTILALOFS","MPHASIS",
    "MCX","NHPC","NMDC","NATIONALUM","OBEROIRLTY","OIL","PAYTM","OFSS",
    "POLICYBZR","PIIND","PAGEIND","PATANJALI","PERSISTENT","PHOENIXLTD","POLYCAB",
    "PREMIERENE","PRESTIGE","RADICO","RVNL","SBICARD","SRF","SAIL","SUPREMEIND",
    "SUZLON","SWIGGY","TATACOMM","TATAELXSI","TATAINVEST","TIINDIA","UPL",
    "VMM","IDEA","VOLTAS","WAAREEENER","YESBANK",
]


# ========================================
# INDICATORS
# ========================================
def compute_rsi(prices, period=14):
    delta = prices.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window=period, min_periods=period).mean()
    avg_loss = loss.rolling(window=period, min_periods=period).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


def compute_adx(df, period=14):
    high, low, close = df['High'], df['Low'], df['Close']
    plus_dm = high.diff()
    minus_dm = low.diff()
    plus_dm[plus_dm < 0] = 0
    minus_dm[minus_dm > 0] = 0
    minus_dm = -minus_dm
    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=period).mean()
    plus_di = 100 * (plus_dm.rolling(window=period).mean() / atr)
    minus_di = 100 * (minus_dm.rolling(window=period).mean() / atr)
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di)
    return dx.rolling(window=period).mean()


# ========================================
# STRATEGY PREPARE/ENTER/EXIT
# ========================================
def prepare_reversal(df, config):
    df = df.copy()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df['RSI'] = compute_rsi(df['Close'], 14)
    df['SMA20'] = df['Close'].rolling(20).mean()
    df['ret_5d'] = df['Close'].pct_change(5) * 100
    df['prev_ret_5d'] = df['ret_5d'].shift(1)
    df['prev_RSI'] = df['RSI'].shift(1)
    return df

def should_enter_reversal(row, config):
    rsi = row.get('RSI', 50)
    prev_rsi = row.get('prev_RSI', 50)
    ret_5d = row.get('prev_ret_5d', 0)
    close = row.get('Close', 0)
    sma20 = row.get('SMA20', 0)
    if pd.isna(rsi) or pd.isna(prev_rsi) or pd.isna(ret_5d) or pd.isna(sma20):
        return False
    if rsi >= config['rsi_entry']:
        return False
    if prev_rsi >= config['rsi_entry'] + 5:
        return False
    if ret_5d > config['ret_5d_threshold']:
        return False
    if close >= sma20:
        return False
    return True

def should_exit_reversal(row, config):
    rsi = row.get('RSI', 50)
    if pd.isna(rsi):
        return False
    return rsi >= config['rsi_exit']

def prepare_pullback(df, config):
    df = df.copy()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df['RSI'] = compute_rsi(df['Close'], 14)
    df['ADX'] = compute_adx(df, 14)
    df['SMA50'] = df['Close'].rolling(50).mean()
    df['SMA20'] = df['Close'].rolling(20).mean()
    df['prev_RSI'] = df['RSI'].shift(1)
    df['ret_3d'] = df['Close'].pct_change(3) * 100
    df['prev_ret_3d'] = df['ret_3d'].shift(1)
    return df

def should_enter_pullback(row, config):
    rsi = row.get('RSI', 50)
    prev_rsi = row.get('prev_RSI', 50)
    adx = row.get('ADX', 0)
    close = row.get('Close', 0)
    sma50 = row.get('SMA50', 0)
    ret_3d = row.get('prev_ret_3d', 0)
    if pd.isna(rsi) or pd.isna(prev_rsi) or pd.isna(adx) or pd.isna(sma50):
        return False
    if close < sma50:
        return False
    if adx < config['adx_min']:
        return False
    if rsi < config['rsi_entry_low'] or rsi > config['rsi_entry_high']:
        return False
    if ret_3d > 0:
        return False
    return True

def should_exit_pullback(row, config):
    rsi = row.get('RSI', 50)
    if pd.isna(rsi):
        return False
    return rsi >= config['rsi_exit']


# ========================================
# BACKTEST ENGINE
# ========================================
def backtest(df, config, prepare_fn, enter_fn, exit_fn):
    df = prepare_fn(df, config)
    trades = []
    in_position = False
    entry_idx = 0
    entry_price = 0
    stop_loss = 0
    target = 0
    for i in range(len(df)):
        row = df.iloc[i]
        if not in_position:
            if enter_fn(row, config):
                entry_price = row['Close']
                stop_loss = entry_price * (1 - config['sl_pct'] / 100)
                target = entry_price * (1 + config['target_pct'] / 100)
                entry_idx = i
                in_position = True
        else:
            if row['Close'] <= stop_loss:
                pnl_pct = (stop_loss - entry_price) / entry_price * 100
                trades.append({'pnl_pct': pnl_pct, 'reason': 'SL'})
                in_position = False
            elif row['Close'] >= target:
                pnl_pct = (target - entry_price) / entry_price * 100
                trades.append({'pnl_pct': pnl_pct, 'reason': 'Target'})
                in_position = False
            elif exit_fn(row, config):
                pnl_pct = (row['Close'] - entry_price) / entry_price * 100
                trades.append({'pnl_pct': pnl_pct, 'reason': 'Signal'})
                in_position = False
    return trades


def calc_metrics(trades):
    """Calculate metrics with PF capped at MAX_PF_CAP to prevent infinity artifacts."""
    if not trades:
        return {'trades': 0, 'pf': 0, 'win_rate': 0}
    pnls = [t['pnl_pct'] for t in trades]
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p <= 0]
    gross_profit = sum(wins) if wins else 0
    gross_loss = abs(sum(losses)) if losses else 0
    if gross_loss > 0:
        pf = gross_profit / gross_loss
    elif gross_profit > 0:
        pf = MAX_PF_CAP  # All wins, no losses — cap instead of infinity
    else:
        pf = 0
    pf = min(pf, MAX_PF_CAP)
    return {
        'trades': len(trades),
        'pf': round(pf, 2),
        'win_rate': round(len(wins) / len(trades) * 100, 1),
        'avg_pnl': round(np.mean(pnls), 2),
    }


# ========================================
# NSE CONSTITUENT FETCHER (with SSL fallback)
# ========================================
def fetch_nse_constituents(url):
    """Download NSE index constituent CSV. Falls back gracefully on SSL errors."""
    try:
        import requests
        resp = requests.get(url, timeout=15, verify=False,
                            headers={'User-Agent': 'Mozilla/5.0'})
        if resp.status_code == 200:
            reader = csv.DictReader(io.StringIO(resp.text))
            symbols = [row['Symbol'] for row in reader]
            if symbols:
                return symbols
    except Exception:
        pass

    try:
        import ssl
        import urllib.request
        ctx = ssl.create_default_context()
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE
        req = urllib.request.Request(url, headers={
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)',
            'Accept': 'text/csv,*/*'
        })
        response = urllib.request.urlopen(req, timeout=15, context=ctx)
        data = response.read().decode('utf-8')
        reader = csv.DictReader(io.StringIO(data))
        symbols = [row['Symbol'] for row in reader]
        if symbols:
            return symbols
    except Exception:
        pass

    return []


def get_constituents():
    """Get NIFTY 50 + Midcap 100 constituents, with fallback to hardcoded lists."""
    print("\n1. Fetching NSE constituent lists...")
    nifty50 = fetch_nse_constituents(NIFTY50_CSV_URL)
    print(f"   NIFTY 50: {len(nifty50)} stocks")
    if not nifty50:
        print(f"   Using fallback NIFTY 50 list ({len(NIFTY50_FALLBACK)} stocks)")
        nifty50 = NIFTY50_FALLBACK

    midcap100 = fetch_nse_constituents(MIDCAP100_CSV_URL)
    print(f"   NIFTY Midcap 100: {len(midcap100)} stocks")
    if not midcap100:
        print(f"   Using fallback Midcap 100 list ({len(MIDCAP100_FALLBACK)} stocks)")
        midcap100 = MIDCAP100_FALLBACK

    all_symbols = list(set(nifty50 + midcap100))
    print(f"   Total unique stocks: {len(all_symbols)}")
    return all_symbols


# ========================================
# PRICE DATA FETCHER
# ========================================
def fetch_price_data(symbol, period="2y", cache=True):
    os.makedirs(PRICE_CACHE_DIR, exist_ok=True)
    cache_file = os.path.join(PRICE_CACHE_DIR, f"{symbol}.csv")

    if cache and os.path.exists(cache_file):
        age = (time.time() - os.path.getmtime(cache_file)) / 86400
        if age < 7:
            df = pd.read_csv(cache_file, index_col=0, parse_dates=True)
            if len(df) > 50:
                return df

    try:
        ticker = yf.Ticker(symbol + ".NS")
        df = ticker.history(period=period, auto_adjust=True)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        if len(df) > 50 and cache:
            df.to_csv(cache_file)
        return df
    except Exception as e:
        print(f"  {symbol}: fetch error - {str(e)[:60]}")
        return pd.DataFrame()


# ========================================
# SUPPRESS SSL WARNINGS
# ========================================
import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)


# ========================================
# MAIN
# ========================================
def main():
    print(f"\n{'='*60}")
    print(f"Basket Refresher — {date.today()}")
    print(f"{'='*60}")

    all_symbols = get_constituents()
    if not all_symbols:
        print("ERROR: No symbols available. Exiting.")
        return

    # Step 2: Fetch price data and backtest
    print(f"\n2. Fetching price data & backtesting {len(all_symbols)} stocks...")
    results = []
    skipped_ipo = 0
    skipped_trades = 0

    for i, sym in enumerate(all_symbols):
        df = fetch_price_data(sym)
        if len(df) < 100:
            continue

        # Filter: exclude stocks with < 1 year of price history (IPOs)
        if len(df) < MIN_PRICE_HISTORY_DAYS:
            skipped_ipo += 1
            print(f"   SKIP {sym}: only {len(df)} bars (~{len(df)//252:.1f}y) — likely recent IPO")
            continue

        rev_trades = backtest(df, REVERSAL_CONFIG, prepare_reversal, should_enter_reversal, should_exit_reversal)
        pb_trades = backtest(df, PULLBACK_CONFIG, prepare_pullback, should_enter_pullback, should_exit_pullback)

        rev_m = calc_metrics(rev_trades)
        pb_m = calc_metrics(pb_trades)

        total_trades = rev_m['trades'] + pb_m['trades']

        # Filter: need enough total trades
        if total_trades < MIN_TOTAL_TRADES:
            skipped_trades += 1
            continue

        # Filter: each strategy needs at least 3 trades
        if rev_m['trades'] < MIN_TRADES_PER_STRATEGY and pb_m['trades'] < MIN_TRADES_PER_STRATEGY:
            skipped_trades += 1
            continue

        # Blended PF (both PFs already capped at MAX_PF_CAP by calc_metrics)
        blended_pf = (rev_m['pf'] * rev_m['trades'] + pb_m['pf'] * pb_m['trades']) / max(total_trades, 1)

        results.append({
            'symbol': sym,
            'price_bars': len(df),
            'rev_trades': rev_m['trades'],
            'rev_pf': rev_m['pf'],
            'rev_win': rev_m['win_rate'],
            'pb_trades': pb_m['trades'],
            'pb_pf': pb_m['pf'],
            'pb_win': pb_m['win_rate'],
            'total_trades': total_trades,
            'blended_pf': round(blended_pf, 2),
        })

        if (i + 1) % 10 == 0:
            print(f"   Processed {i+1}/{len(all_symbols)} stocks...")

        time.sleep(1.5)

    print(f"   Skipped {skipped_ipo} IPOs (< {MIN_PRICE_HISTORY_DAYS} bars), "
          f"{skipped_trades} stocks with < {MIN_TOTAL_TRADES} trades")

    # Step 3: Rank and select
    print(f"\n3. Ranking {len(results)} stocks by blended PF...")
    results.sort(key=lambda x: x['blended_pf'], reverse=True)

    qualified = [r for r in results if r['blended_pf'] >= MIN_BLENDED_PF]
    print(f"   Qualified (blended PF >= {MIN_BLENDED_PF}, PF capped at {MAX_PF_CAP}, "
          f"min {MIN_TOTAL_TRADES} trades): {len(qualified)} stocks")

    selected = qualified[:MAX_BASKET_SIZE]

    print(f"\n   Selected {len(selected)} stocks for curated basket:")
    for i, s in enumerate(selected):
        print(f"   {i+1:3d}. {s['symbol']:15s} | Blended PF: {s['blended_pf']:6.2f} "
              f"| Rev: {s['rev_trades']}t PF{s['rev_pf']} | PB: {s['pb_trades']}t PF{s['pb_pf']} "
              f"| {s['price_bars']} bars")

    # Step 4: Save (atomic write)
    basket_data = {
        'date': str(date.today()),
        'total_universe': len(all_symbols),
        'total_tested': len(results),
        'total_qualified': len(qualified),
        'skipped_ipo': skipped_ipo,
        'skipped_low_trades': skipped_trades,
        'basket_size': len(selected),
        'filters': {
            'min_total_trades': MIN_TOTAL_TRADES,
            'min_trades_per_strategy': MIN_TRADES_PER_STRATEGY,
            'min_blended_pf': MIN_BLENDED_PF,
            'max_pf_cap': MAX_PF_CAP,
            'min_price_history_days': MIN_PRICE_HISTORY_DAYS,
        },
        'stocks': [s['symbol'] for s in selected],
        'details': selected,
    }

    os.makedirs(os.path.dirname(BASKET_OUTPUT), exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(dir=os.path.dirname(BASKET_OUTPUT), suffix=".tmp")
    try:
        with os.fdopen(fd, 'w') as f:
            json.dump(basket_data, f, indent=2, default=str)
        os.replace(tmp_path, BASKET_OUTPUT)
    except:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        raise

    print(f"\n   Basket saved to {BASKET_OUTPUT}")

    # Step 5: Telegram alert
    try:
        import requests
        token = os.environ.get("TELEGRAM_BOT_TOKEN")
        chat_id = os.environ.get("TELEGRAM_CHAT_ID")
        if token and chat_id:
            msg = f"\U0001F504 *Basket Refreshed*\n\n"
            msg += f"Universe: {len(all_symbols)} stocks\n"
            msg += f"Tested: {len(results)} (skipped {skipped_ipo} IPOs)\n"
            msg += f"Qualified: {len(qualified)} (PF >= {MIN_BLENDED_PF})\n"
            msg += f"Selected: {len(selected)} stocks\n\n"
            msg += f"Top 5:\n"
            for s in selected[:5]:
                msg += f"  {s['symbol']} (PF {s['blended_pf']})\n"
            requests.post(
                f"https://api.telegram.org/bot{token}/sendMessage",
                json={"chat_id": chat_id, "text": msg, "parse_mode": "Markdown"},
                timeout=10, verify=False
            )
            print("   Telegram alert sent")
    except Exception:
        print("   Telegram alert failed (no token)")

    print(f"\n{'='*60}")
    print(f"Basket refresh complete: {len(selected)} stocks selected")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
