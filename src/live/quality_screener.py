#!/usr/bin/env python3
"""
Quality Screener — Fundamental Stock Scanner (NIFTY 50 + Midcap 150)
====================================================================

Monthly fundamental screen that builds the portfolio's quality universe.

Pipeline:
  1. Fetch NIFTY 50 + NIFTY Midcap 150 constituent lists (NSE archives,
     cached locally as fallback)
  2. Pull fundamentals per stock (yfinance): ROE, margins, D/E, growth,
     FCF, market cap, sector
  3. Quality filters:
       - ROE >= 15%
       - Debt/Equity <= 0.5 (non-financials only)
       - Operating margin >= 10% (non-financials; financials: ROE + growth)
       - Revenue growth >= 0
  4. Composite quality score (0-100, transparent weights)
  5. Technical readiness for qualifiers only: price vs EMA200 (uptrend),
     distance to EMA20 (buy zone)
  6. Output: ranked table + data/quality_screener_results.json
     + data/core_universe.json (top N feed the Core Accumulator)

Usage:      python src/live/quality_screener.py [--top 25] [--min-roe 15]
Cron (VM):  monthly, 1st at 11:00 IST
"""

import argparse
import csv
import io
import json
import os
import ssl
import sys
import time
import urllib.request
from datetime import datetime

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.dirname(SCRIPT_DIR)
REPO_ROOT = os.path.dirname(SRC_DIR)
sys.path.insert(0, SRC_DIR)
sys.path.insert(0, REPO_ROOT)

import pandas as pd
import yfinance as yf
import warnings
warnings.filterwarnings('ignore')

# ========================================
# CONFIG
# ========================================
RESULTS_FILE = os.path.join(REPO_ROOT, "data", "quality_screener_results.json")
UNIVERSE_OUT = os.path.join(REPO_ROOT, "data", "core_universe.json")
CONSTITUENT_CACHE = os.path.join(REPO_ROOT, "data", "constituents_cache.json")

NIFTY50_CSV = "https://archives.nseindia.com/content/indices/ind_nifty50list.csv"
MIDCAP150_CSV = "https://archives.nseindia.com/content/indices/ind_niftymidcap150list.csv"

FINANCIAL_SECTORS = {"Financial Services", "Financial", "Banks", "Insurance",
                     "Capital Markets", "Credit Services"}

# ========================================
# HELPERS
# ========================================
def log(msg):
    line = f"[{datetime.now():%Y-%m-%d %H:%M:%S}] {msg}"
    print(line)
    try:
        os.makedirs(os.path.join(REPO_ROOT, "logs"), exist_ok=True)
        with open(os.path.join(REPO_ROOT, "logs", "quality_screener.log"), "a") as f:
            f.write(line + "\n")
    except Exception as e:
        print(f"  !! log write failed: {e}")


def fetch_constituents(url, name):
    """NSE constituent CSV with SSL fallback. Never raises."""
    try:
        ctx = ssl.create_default_context()
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE
        req = urllib.request.Request(url, headers={
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"})
        data = urllib.request.urlopen(req, timeout=15, context=ctx).read().decode("utf-8")
        symbols = [row["Symbol"] for row in csv.DictReader(io.StringIO(data))]
        if symbols:
            log(f"  fetched {name}: {len(symbols)} stocks")
            return symbols
    except Exception as e:
        log(f"  !! {name} fetch failed ({str(e)[:60]})")
    return None


def get_universe():
    n50 = fetch_constituents(NIFTY50_CSV, "NIFTY 50")
    mc150 = fetch_constituents(MIDCAP150_CSV, "Midcap 150")

    cache = {}
    if os.path.exists(CONSTITUENT_CACHE):
        try:
            with open(CONSTITUENT_CACHE) as f:
                cache = json.load(f)
        except Exception as e:
            log(f"  !! cache unreadable: {e}")

    if n50 is None:
        n50 = cache.get("nifty50")
        log(f"  using cached NIFTY 50 ({len(n50) if n50 else 0} stocks)")
    else:
        cache["nifty50"] = n50
    if mc150 is None:
        mc150 = cache.get("midcap150")
        log(f"  using cached Midcap 150 ({len(mc150) if mc150 else 0} stocks)")
    else:
        cache["midcap150"] = mc150

    if not n50 or not mc150:
        log("  !! no constituent data available at all — aborting")
        return None

    try:
        os.makedirs(os.path.dirname(CONSTITUENT_CACHE), exist_ok=True)
        with open(CONSTITUENT_CACHE, "w") as f:
            json.dump(cache, f, indent=2)
    except Exception as e:
        log(f"  !! cache write failed: {e}")

    all_syms = sorted(set(n50 + mc150))
    large_set = set(n50)
    return [{"symbol": s, "cap": "LARGE" if s in large_set else "MID"} for s in all_syms]


def fetch_fundamentals(sym):
    """Pull the metrics we need. Returns dict or None. Logs failures."""
    last_err = None
    for attempt in range(2):
        try:
            info = yf.Ticker(sym + ".NS").info
            if info and (info.get("marketCap") or info.get("returnOnEquity")):
                de = info.get("debtToEquity")
                if de is not None:
                    de = float(de)
                    de = de / 100.0 if de > 10 else de
                return {
                    "symbol": sym,
                    "name": (info.get("shortName") or sym)[:24],
                    "sector": info.get("sector") or "?",
                    "market_cap_cr": round((info.get("marketCap") or 0) / 1e7),
                    "roe": info.get("returnOnEquity"),
                    "op_margin": info.get("operatingMargins"),
                    "net_margin": info.get("profitMargins"),
                    "debt_equity": de,
                    "rev_growth": info.get("revenueGrowth"),
                    "earn_growth": info.get("earningsGrowth"),
                    "fcf_cr": round((info.get("freeCashflow") or 0) / 1e7),
                    "pe": info.get("trailingPE"),
                }
            last_err = "empty info"
        except Exception as e:
            last_err = str(e)[:60]
        time.sleep(2)
    log(f"  !! {sym}: fundamentals unavailable ({last_err})")
    return None


# ========================================
# QUALITY FILTER + SCORE
# ========================================
def data_flags(f):
    """Flag implausible/missing fundamentals — Yahoo data for NSE midcaps
    is unreliable, so suspicious values get flagged instead of trusted."""
    flags = []
    if f["roe"] is None:
        flags.append("ROE missing")
    elif f["roe"] == 0 and (f["op_margin"] or 0) > 0.05:
        flags.append("ROE=0 despite margins — suspect")
    if f["pe"] is not None and f["pe"] >= 99:
        flags.append("PE placeholder")
    if f["debt_equity"] is not None and f["debt_equity"] > 3 \
            and f["sector"] not in FINANCIAL_SECTORS:
        flags.append("D/E implausible — verify")
    if f["op_margin"] is None and f["sector"] not in FINANCIAL_SECTORS:
        flags.append("OPM missing")
    return flags


def evaluate_quality(f, min_roe, max_de, min_margin):
    """
    Three-bucket evaluation:
      PASS   - passed every check on real data
      VERIFY - no hard failure, but data gaps prevent judgment
               (manual-review candidate, NOT auto-included in universe)
      FAIL   - genuinely failed a threshold on real data
    Financials skip D/E and margin checks by design (policy, not data gap).
    Returns (bucket, reasons).
    """
    reasons = []
    is_fin = f["sector"] in FINANCIAL_SECTORS

    roe = f["roe"]
    if roe is None:
        reasons.append("ROE missing")
    elif roe < min_roe / 100.0:
        reasons.append(f"ROE {roe*100:.1f}% < {min_roe:.0f}%")

    if not is_fin:
        de = f["debt_equity"]
        if de is None:
            reasons.append("D/E missing")
        elif de > max_de:
            reasons.append(f"D/E {de:.2f} > {max_de}")
        om = f["op_margin"]
        if om is None:
            reasons.append("OPM missing")
        elif om < min_margin / 100.0:
            reasons.append(f"OPM {om*100:.1f}% < {min_margin:.0f}%")

    rg = f["rev_growth"]
    if rg is None:
        reasons.append("rev growth missing")
    elif rg < 0:
        reasons.append(f"rev {rg*100:.1f}% < 0")

    hard_fail = any(not r.endswith("missing") for r in reasons)
    if hard_fail:
        return "FAIL", reasons
    if reasons:
        return "VERIFY", reasons
    return "PASS", []


def quality_score(f):
    """Transparent 0-100 composite. Higher = better."""
    score = 0.0
    roe = f["roe"] or 0
    score += min(roe / 0.30, 1.0) * 35            # ROE, capped at 30%
    om = f["op_margin"] or 0
    score += min(om / 0.25, 1.0) * 15             # operating margin
    de = f["debt_equity"]
    if de is not None:
        score += max(0.0, 1.0 - de / 0.5) * 15    # lower debt = more points
    rg = f["rev_growth"] or 0
    score += min(max(rg, 0) / 0.20, 1.0) * 10     # growth
    eg = f["earn_growth"] or 0
    score += min(max(eg, 0) / 0.20, 1.0) * 10     # earnings growth
    if f["fcf_cr"] and f["fcf_cr"] > 0:
        score += 10                                # positive FCF
    eg_pen = f["earn_growth"]
    if eg_pen is not None and eg_pen < 0:
        score -= 10                                # shrinking earnings penalty
    return round(score, 1)


# ========================================
# TECHNICAL READINESS (qualifiers only)
# ========================================
def technical_readiness(sym):
    try:
        df = yf.download(sym + ".NS", period="1y", auto_adjust=True, progress=False)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        if len(df) < 220:
            return None
        close = float(df["Close"].iloc[-1])
        ema200 = df["Close"].ewm(span=200, adjust=False).mean().iloc[-1]
        ema20 = df["Close"].ewm(span=20, adjust=False).mean().iloc[-1]
        return {
            "price": round(close, 1),
            "uptrend": bool(close > ema200 and ema20 > ema200),
            "dist_ema20_pct": round((close / ema20 - 1) * 100, 1),
        }
    except Exception as e:
        log(f"  !! {sym}: technical check failed ({str(e)[:50]})")
        return None


# ========================================
# MAIN
# ========================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--top", type=int, default=25,
                    help="size of core universe output")
    ap.add_argument("--min-roe", type=float, default=15.0)
    ap.add_argument("--max-de", type=float, default=0.5)
    ap.add_argument("--min-margin", type=float, default=10.0)
    args = ap.parse_args()

    log("=" * 60)
    log("Quality Screener run starting")

    universe = get_universe()
    if not universe:
        log("no universe — aborting")
        return
    log(f"universe: {len(universe)} stocks "
        f"({sum(1 for u in universe if u['cap']=='LARGE')} large / "
        f"{sum(1 for u in universe if u['cap']=='MID')} mid)")

    # ---- pass 1: fundamentals ----
    rows, failures = [], 0
    for i, u in enumerate(universe):
        f = fetch_fundamentals(u["symbol"])
        if f is None:
            failures += 1
        else:
            f["cap"] = u["cap"]
            rows.append(f)
        if (i + 1) % 25 == 0:
            log(f"  fundamentals: {i+1}/{len(universe)} done "
                f"({failures} failures)")
        time.sleep(0.6)

    log(f"fundamentals complete: {len(rows)} ok, {failures} failed")

    # ---- pass 2: filter + score ----
    passed, verify, rejected = [], [], []
    for f in rows:
        bucket, reasons = evaluate_quality(f, args.min_roe,
                                            args.max_de, args.min_margin)
        f["quality_score"] = quality_score(f)
        f["data_flags"] = data_flags(f)
        if bucket == "PASS":
            passed.append(f)
        elif bucket == "VERIFY":
            f["verify_reasons"] = reasons
            verify.append(f)
        else:
            f["reject_reasons"] = reasons
            rejected.append(f)

    log(f"quality filter: {len(passed)} passed | {len(verify)} need manual "
        f"verification (data gaps) | {len(rejected)} rejected on numbers")

    # ---- pass 3: technical readiness on qualifiers ----
    for f in passed:
        t = technical_readiness(f["symbol"])
        if t:
            f.update(t)
        else:
            f["uptrend"] = None
        time.sleep(0.4)

    passed.sort(key=lambda x: -x["quality_score"])

    # ---- output ----
    print(f"\n{'='*100}")
    print(f"QUALITY UNIVERSE — {datetime.now():%Y-%m-%d} | "
          f"{len(passed)}/{len(rows)} passed "
          f"(ROE>={args.min_roe:.0f}%, D/E<={args.max_de}, OPM>={args.min_margin:.0f}%)")
    print(f"{'='*100}")
    print(f"{'#':>3} {'Symbol':12s} {'Cap':5s} {'Score':>5} {'ROE%':>5} "
          f"{'OPM%':>5} {'D/E':>5} {'Rev%':>6} {'PE':>6} {'MktCap':>8} "
          f"{'Uptrend':>7} {'vsEMA20':>7}")
    print("-" * 100)
    for i, f in enumerate(passed[:40]):
        print(f"{i+1:3d} {f['symbol']:12s} {f['cap']:5s} {f['quality_score']:5.1f} "
              f"{(f['roe'] or 0)*100:5.1f} {(f['op_margin'] or 0)*100:5.1f} "
              f"{(f['debt_equity'] or 0):5.2f} {(f['rev_growth'] or 0)*100:6.1f} "
              f"{(f['pe'] or 0):6.1f} {f['market_cap_cr']:7,d}cr "
              f"{'YES' if f.get('uptrend') else ('no' if f.get('uptrend') is False else '?'):>7} "
              f"{f.get('dist_ema20_pct', 0):6.1f}%"
              + (f"  [{'; '.join(f['data_flags'])}]" if f.get('data_flags') else ""))

    # manual-verification table: data gaps prevented judgment
    verify.sort(key=lambda x: -x["quality_score"])
    print(f"\nNEEDS MANUAL VERIFICATION — data gaps, could not judge "
          f"({len(verify)} stocks; cross-check on screener.in):")
    print(f"{'#':>3} {'Symbol':12s} {'Cap':5s} {'Score':>5} {'PE':>6} "
          f"{'MktCap':>8} {'Uptrend':>7}  gaps")
    print("-" * 80)
    for i, f in enumerate(verify[:25]):
        print(f"{i+1:3d} {f['symbol']:12s} {f['cap']:5s} {f['quality_score']:5.1f} "
              f"{(f['pe'] or 0):6.1f} {f['market_cap_cr']:7,d}cr "
              f"{'YES' if f.get('uptrend') else ('no' if f.get('uptrend') is False else '?'):>7}  "
              f"{'; '.join(f['verify_reasons'])}")

    # persist full results
    out = {"generated": str(datetime.now()),
           "filters": {"min_roe": args.min_roe, "max_de": args.max_de,
                       "min_margin": args.min_margin},
           "universe_size": len(universe), "fundamentals_ok": len(rows),
           "passed": len(passed), "verify": len(verify),
           "rejected": len(rejected),
           "stocks": passed, "verify_list": verify, "rejected": rejected}
    try:
        os.makedirs(os.path.dirname(RESULTS_FILE), exist_ok=True)
        with open(RESULTS_FILE, "w") as fh:
            json.dump(out, fh, indent=2, default=str)
        log(f"full results saved to {RESULTS_FILE}")
    except Exception as e:
        log(f"!! results save failed: {e}")

    # core universe feed (top N of PASSED only — verified quality)
    top = passed[:args.top]
    if len(top) < args.top:
        log(f"note: only {len(top)} verified-quality stocks available "
            f"(asked for {args.top}); check the VERIFY list manually")
    try:
        os.makedirs(os.path.dirname(UNIVERSE_OUT), exist_ok=True)
        with open(UNIVERSE_OUT, "w") as fh:
            json.dump([f["symbol"] for f in top], fh, indent=2)
        log(f"core universe ({len(top)} stocks) written to {UNIVERSE_OUT}")
    except Exception as e:
        log(f"!! core universe save failed: {e}")

    # telegram summary (optional)
    tok = os.environ.get("TELEGRAM_BOT_TOKEN")
    chat = os.environ.get("TELEGRAM_CHAT_ID")
    if tok and chat:
        try:
            msg = (f"Quality Screener: {len(passed)} passed | "
                   f"{len(verify)} need verification | {len(rejected)} failed "
                   f"({len(universe)} scanned)\nTop 10:\n")
            for f in passed[:10]:
                msg += f"  {f['symbol']} (score {f['quality_score']})\n"
            import urllib.parse
            url = (f"https://api.telegram.org/bot{tok}/sendMessage"
                   f"?chat_id={chat}&text={urllib.parse.quote(msg)}")
            urllib.request.urlopen(
                urllib.request.Request(url), timeout=10)
            log("telegram summary sent")
        except Exception as e:
            log(f"!! telegram failed: {e}")

    log("run complete")
    log("=" * 60)


if __name__ == "__main__":
    main()
