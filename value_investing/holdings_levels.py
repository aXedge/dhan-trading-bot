#!/usr/bin/env python3
"""
holdings_levels.py — evening level-confluence report for your equity holdings.

Answers one question per holding: is the price sitting on, or approaching, a
level where SEVERAL independent methods agree? The more methods that cluster
at the same price, the stronger the level.

Levels computed per name
  1. SMA 20 / 50 / 100 / 200
  2. 52-week high and low
  3. 20-day and 50-day high and low
  4. Fibonacci retracements of the 52-week range (23.6 / 38.2 / 50 / 61.8 / 78.6)
  5. Fibonacci retracements of the last 90-session swing
  6. Bollinger bands (20, 2 sigma): upper / mid / lower
  7. Round-number (psychological) levels
  8. Prior week and prior month high and low

Confluence = levels falling within --tol ATRs of each other. Support is the
nearest cluster BELOW the price, resistance the nearest cluster ABOVE.

Usage
  python holdings_levels.py                       # full report
  python holdings_levels.py --near 1.0            # only names within 1 ATR of a level
  python holdings_levels.py --min-confluence 3    # only clusters with >= 3 levels
  python holdings_levels.py --selftest
  python holdings_levels.py --holdings my.csv --report out.csv

The holdings CSV needs a header: symbol,qty,avg_price  (qty/avg optional).
Symbols are NSE tickers without the .NS suffix.
"""

import argparse
import csv
import os
import sys

try:
    import numpy as np
    import pandas as pd
except ImportError:                      # pragma: no cover
    np = pd = None

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_HOLDINGS = os.path.join(HERE, "holdings.csv")
DEFAULT_REPORT = os.path.join(HERE, "levels_report.csv")
DEFAULT_HISTORY = os.path.join(HERE, "holdings_history.csv")
DHAN_HOLDINGS_URL = "https://api.dhan.co/v2/holdings"

FIB = (0.236, 0.382, 0.5, 0.618, 0.786)

REPORT_COLUMNS = [
    "symbol", "price", "pnl_pct", "rsi", "atr_pct",
    "trend", "adx", "di_plus", "di_minus", "ma_align", "slope50",
    "poc", "val", "vah", "zone_pos", "obv_slope", "ad_slope", "mfi", "accum",
    "support", "sup_dist_pct", "sup_dist_atr", "sup_levels", "sup_types",
    "resistance", "res_dist_pct", "res_dist_atr", "res_levels", "res_types",
    "verdict",
]


def _log(msg):
    print(f"[{__import__('datetime').datetime.now():%H:%M:%S}] {msg}", flush=True)


# --------------------------------------------------------------------------
# indicators
# --------------------------------------------------------------------------

def sma(close, n):
    if len(close) < n:
        return None
    return float(close.tail(n).mean())


def atr(high, low, close, n=14):
    prev = close.shift()
    tr = [max(h - l, abs(h - p) if p == p else h - l, abs(l - p) if p == p else h - l)
          for h, l, p in zip(high, low, prev)]
    if len(tr) < n:
        return None
    return float(sum(tr[-n:]) / n)


def rsi(close, n=14):
    if len(close) < n + 1:
        return None
    gains, losses = [], []
    vals = list(close)
    for i in range(len(vals) - n, len(vals)):
        d = vals[i] - vals[i - 1]
        gains.append(max(d, 0.0))
        losses.append(max(-d, 0.0))
    ag, al = sum(gains) / n, sum(losses) / n
    if al == 0:
        return 100.0
    rs = ag / al
    return 100.0 - 100.0 / (1.0 + rs)


def bollinger(close, n=20, k=2.0):
    if len(close) < n:
        return None
    w = close.tail(n)
    mid = float(w.mean())
    sd = float(w.std(ddof=0))
    return mid - k * sd, mid, mid + k * sd


# --------------------------------------------------------------------------
# trend  (ADX / DI, MA structure, slope)
# --------------------------------------------------------------------------

def adx(high, low, close, n=14):
    """Wilder ADX with +DI / -DI. Returns (adx, +di, -di) or (None, None, None)."""
    if pd is None or len(close) < n * 2:
        return None, None, None
    h = high.astype(float)
    l = low.astype(float)
    c = close.astype(float)
    up, dn = h.diff(), -l.diff()
    plus_dm = pd.Series(np.where((up > dn) & (up > 0), up, 0.0), index=h.index)
    minus_dm = pd.Series(np.where((dn > up) & (dn > 0), dn, 0.0), index=h.index)
    tr = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()],
                   axis=1).max(axis=1)
    a = tr.ewm(alpha=1.0 / n, adjust=False).mean()
    if float(a.iloc[-1]) == 0:
        return None, None, None
    pdi = 100 * plus_dm.ewm(alpha=1.0 / n, adjust=False).mean() / a
    mdi = 100 * minus_dm.ewm(alpha=1.0 / n, adjust=False).mean() / a
    dx = 100 * (pdi - mdi).abs() / (pdi + mdi).replace(0, np.nan)
    ax = dx.ewm(alpha=1.0 / n, adjust=False).mean()
    if not np.isfinite(float(ax.iloc[-1])):
        return None, None, None
    return float(ax.iloc[-1]), float(pdi.iloc[-1]), float(mdi.iloc[-1])


def trend_block(df):
    """Trend direction and strength from MA structure + ADX/DI + slope."""
    close = df["Close"].dropna()
    if len(close) < 210:
        return {}
    s20, s50, s200 = sma(close, 20), sma(close, 50), sma(close, 200)
    ax, pdi, mdi = adx(df["High"].reindex(close.index),
                       df["Low"].reindex(close.index), close)
    ma50 = close.rolling(50).mean()
    slope50 = (float(ma50.iloc[-1]) / float(ma50.iloc[-21]) - 1) * 100 if len(ma50.dropna()) > 21 else 0.0
    if s20 and s50 and s200 and s20 > s50 > s200:
        align = "up"
    elif s20 and s50 and s200 and s20 < s50 < s200:
        align = "down"
    else:
        align = "mixed"

    # a rising price above still-falling averages is a transition, not a trend
    rising, falling = slope50 > 0.5, slope50 < -0.5
    if ax is None:
        verdict = ("UP" if align == "up" and rising
                   else "DOWN" if align == "down" and falling else "MIXED")
    elif ax >= 25 and pdi > mdi and align == "up" and rising:
        verdict = "STRONG-UP"
    elif ax >= 25 and mdi > pdi and align == "down" and falling:
        verdict = "STRONG-DOWN"
    elif pdi > mdi and (align == "up" or (align == "mixed" and rising)):
        verdict = "UP"
    elif mdi > pdi and (align == "down" or (align == "mixed" and falling)):
        verdict = "DOWN"
    else:
        verdict = "MIXED"
    def _f(v):
        # a genuine 0.0 (-DI on a one-way rise) must not read as "missing"
        return round(float(v), 1) if v is not None and np.isfinite(v) else ""

    return {"trend": verdict, "adx": _f(ax), "di_plus": _f(pdi),
            "di_minus": _f(mdi), "ma_align": align,
            "slope50": round(slope50, 1)}


# --------------------------------------------------------------------------
# accumulation / distribution  (volume profile + OBV + A/D + MFI)
# --------------------------------------------------------------------------

def volume_profile(high, low, close, volume, sessions=126, bins=40):
    """Volume-by-price over the last `sessions` bars.

    Each bar's volume is spread evenly across the price bins its high-low
    range covers, which is more faithful than dumping it on one price.
    Returns POC (point of control), VAL/VAH (70% value area) and bar count.
    """
    if pd is None:
        return None
    h = high.tail(sessions).astype(float)
    l = low.tail(sessions).astype(float)
    v = volume.tail(sessions).astype(float)
    if len(h) < 20 or float(v.sum()) <= 0:
        return None
    lo, hi = float(l.min()), float(h.max())
    if hi <= lo:
        return None
    edges = [lo + (hi - lo) * i / bins for i in range(bins + 1)]
    vol = [0.0] * bins
    for hh, ll, vv in zip(h.values, l.values, v.values):
        if not (vv > 0) or not np.isfinite(vv):
            continue
        a, b = max(float(ll), lo), min(float(hh), hi)
        if b <= a:
            i = min(bins - 1, max(0, int((float(ll) - lo) / (hi - lo) * bins)))
            vol[i] += float(vv)
            continue
        i0 = min(bins - 1, max(0, int((a - lo) / (hi - lo) * bins)))
        i1 = min(bins - 1, max(0, int((b - lo) / (hi - lo) * bins)))
        share = float(vv) / (i1 - i0 + 1)
        for i in range(i0, i1 + 1):
            vol[i] += share
    total = sum(vol)
    if total <= 0:
        return None
    poc_i = max(range(bins), key=lambda i: vol[i])
    target = 0.70 * total
    acc, lo_i, hi_i = vol[poc_i], poc_i, poc_i
    while acc < target and (lo_i > 0 or hi_i < bins - 1):
        up = vol[hi_i + 1] if hi_i < bins - 1 else -1.0
        dn = vol[lo_i - 1] if lo_i > 0 else -1.0
        if up >= dn:
            hi_i += 1
            acc += up
        else:
            lo_i -= 1
            acc += dn
    mid = lambda i: lo + (hi - lo) * (i + 0.5) / bins
    return {"poc": round(mid(poc_i), 2), "val": round(edges[lo_i], 2),
            "vah": round(edges[hi_i + 1], 2), "bars": int(len(h))}


def obv(close, volume):
    c, v = close.values, volume.reindex(close.index).fillna(0).values
    out = [0.0]
    for i in range(1, len(c)):
        if c[i] > c[i - 1]:
            out.append(out[-1] + v[i])
        elif c[i] < c[i - 1]:
            out.append(out[-1] - v[i])
        else:
            out.append(out[-1])
    return pd.Series(out, index=close.index)


def ad_line(high, low, close, volume):
    rng = (high - low).reindex(close.index).replace(0, np.nan)
    clv = (((close - low.reindex(close.index)) -
            (high.reindex(close.index) - close)) / rng).fillna(0)
    return (clv * volume.reindex(close.index).fillna(0)).cumsum()


def mfi(high, low, close, volume, n=14):
    if pd is None or len(close) < n + 1:
        return None
    tp = (high.reindex(close.index) + low.reindex(close.index) + close) / 3
    mf = tp * volume.reindex(close.index).fillna(0)
    up = mf.where(tp > tp.shift(), 0.0).tail(n).sum()
    dn = mf.where(tp < tp.shift(), 0.0).tail(n).sum()
    if dn == 0:
        return 100.0
    return float(100 - 100 / (1 + up / dn))


def zone_block(df):
    """Accumulation / distribution read from volume behaviour at price."""
    close = df["Close"].dropna()
    if len(close) < 130:
        return {}
    vol = df["Volume"].reindex(close.index).fillna(0)
    vp = volume_profile(df["High"], df["Low"], close, df["Volume"])
    if not vp:
        return {}
    px = float(close.iloc[-1])
    pos = "ABOVE" if px > vp["vah"] else ("BELOW" if px < vp["val"] else "INSIDE")

    o = obv(close, df["Volume"])
    a = ad_line(df["High"], df["Low"], close, df["Volume"])
    turn20 = float(vol.tail(20).sum())
    obv_slope = ((float(o.iloc[-1]) - float(o.iloc[-21])) / turn20 * 100
                 if turn20 > 0 else 0.0)
    ad_scale = float(a.abs().tail(60).mean()) or 1.0
    ad_slope = (float(a.iloc[-1]) - float(a.iloc[-21])) / ad_scale * 100
    m = mfi(df["High"], df["Low"], close, df["Volume"])

    if pos in ("ABOVE", "INSIDE") and obv_slope > 10 and ad_slope > 0:
        verdict = "ACCUMULATION"
    elif pos in ("BELOW", "INSIDE") and obv_slope < -10 and ad_slope < 0:
        verdict = "DISTRIBUTION"
    elif pos == "ABOVE" and obv_slope < -10:
        verdict = "DISTRIBUTING-AT-HIGHS"
    else:
        verdict = "NEUTRAL"

    return {"poc": vp["poc"], "val": vp["val"], "vah": vp["vah"],
            "zone_pos": pos, "obv_slope": round(obv_slope, 1),
            "ad_slope": round(ad_slope, 1),
            "mfi": round(m, 1) if m is not None else "",
            "accum": verdict}


# --------------------------------------------------------------------------
# levels
# --------------------------------------------------------------------------

def round_steps(px):
    """Two round-number levels either side of px, at a sensible granularity."""
    if px >= 1000:
        step = 100.0
    elif px >= 200:
        step = 50.0
    elif px >= 50:
        step = 10.0
    elif px >= 10:
        step = 5.0
    else:
        step = 1.0
    base = int(px // step) * step
    return [base - step, base, base + step, base + 2 * step]


def build_levels(df):
    """Return a list of (price, label) for one name. df: Date-indexed OHLC."""
    close = df["Close"].dropna()
    high = df["High"].reindex(close.index)
    low = df["Low"].reindex(close.index)
    if len(close) < 60:
        return []
    px = float(close.iloc[-1])
    out = []

    for n in (20, 50, 100, 200):
        v = sma(close, n)
        if v:
            out.append((v, f"SMA{n}"))

    hi52, lo52 = float(close.max()), float(close.min())
    out.append((hi52, "52w-high"))
    out.append((lo52, "52w-low"))

    for n, tag in ((20, "20d"), (50, "50d")):
        if len(close) >= n:
            out.append((float(close.tail(n).max()), f"{tag}-high"))
            out.append((float(close.tail(n).min()), f"{tag}-low"))

    # Fibonacci retracement of the 52-week range (measured down from the high)
    span = hi52 - lo52
    if span > 0:
        for r in FIB:
            out.append((hi52 - span * r, f"fib52-{r:g}"))

    # Fibonacci of the last 90-session swing
    if len(close) >= 90:
        w = close.tail(90)
        h90, l90 = float(w.max()), float(w.min())
        span90 = h90 - l90
        if span90 > 0:
            for r in FIB:
                out.append((h90 - span90 * r, f"fib90-{r:g}"))

    bb = bollinger(close)
    if bb:
        lo_b, mid_b, up_b = bb
        out.append((lo_b, "bb-lower"))
        out.append((mid_b, "bb-mid"))
        out.append((up_b, "bb-upper"))

    for v in round_steps(px):
        if v > 0:
            out.append((v, "round"))

    # prior week / prior month high-low
    try:
        idx = close.index
        if hasattr(idx, "isocalendar"):
            wk = close.groupby([idx.isocalendar().year, idx.isocalendar().week])
            groups = list(wk)
            if len(groups) >= 2:
                _, w0 = groups[-2]
                out.append((float(w0.max()), "pw-high"))
                out.append((float(w0.min()), "pw-low"))
        if hasattr(idx, "month"):
            mo = close.groupby([idx.year, idx.month])
            groups = list(mo)
            if len(groups) >= 2:
                _, m0 = groups[-2]
                out.append((float(m0.max()), "pm-high"))
                out.append((float(m0.min()), "pm-low"))
    except Exception:
        pass

    clean = [(round(p, 2), lbl) for p, lbl in out if p and p == p and p > 0]
    # de-duplicate: two methods landing on the same price is ONE level, not two
    keep = []
    for price, label in sorted(clean, key=lambda x: -x[0]):
        if any(abs(price - q) <= 0.0005 * px for q, _ in keep):
            continue
        keep.append((price, label))
    return keep


def cluster(levels, px, tol):
    """Group levels within tol of each other. Returns list of clusters,
    each {price, count, types, side, dist_pct}."""
    below = sorted([l for l in levels if l[0] < px], key=lambda x: -x[0])
    above = sorted([l for l in levels if l[0] > px], key=lambda x: x[0])

    def group(items):
        clusters = []
        for price, label in items:
            # anchor on the FIRST (nearest) level, not the last, so a chain of
            # levels each just inside tolerance cannot walk away from the price
            if clusters and abs(clusters[-1][0][0] - price) <= tol:
                clusters[-1].append((price, label))
            else:
                clusters.append([(price, label)])
        out = []
        for c in clusters:
            prices = [p for p, _ in c]
            out.append({
                "price": sum(prices) / len(prices),
                "lo": min(prices),
                "hi": max(prices),
                "count": len(c),
                "types": [lbl for lbl, in [(x[1],) for x in c]],
            })
        return out

    return group(below), group(above)


# --------------------------------------------------------------------------
# analysis
# --------------------------------------------------------------------------

def analyse_one(symbol, df, tol_atr=0.25):
    close = df["Close"].dropna()
    if len(close) < 60:
        return {"symbol": symbol, "error": "insufficient history"}

    px = float(close.iloc[-1])
    a = atr(df["High"], df["Low"], df["Close"].dropna())
    if not a:
        return {"symbol": symbol, "error": "no ATR"}
    tol = tol_atr * a

    levels = build_levels(df)
    sup, res = cluster(levels, px, tol)

    def pack(c, side):
        if not c:
            return None
        d = abs(c["price"] - px)
        return {
            "price": round(c["price"], 2),
            "dist_pct": d / px * 100.0,
            "dist_atr": d / a,
            "count": c["count"],
            "types": ",".join(c["types"]),
            "width": round(c["hi"] - c["lo"], 2),
        }

    s = pack(sup[0], "support") if sup else None
    r = pack(res[0], "resistance") if res else None

    verdict = "MID-RANGE"
    if s and s["dist_atr"] <= 0.5:
        verdict = f"AT-SUPPORT ({s['count']} levels)"
    elif r and r["dist_atr"] <= 0.5:
        verdict = f"AT-RESISTANCE ({r['count']} levels)"

    row = {
        "symbol": symbol,
        "price": round(px, 2),
        "rsi": round(rsi(close) or 0, 1),
        "atr_pct": round(a / px * 100.0, 1),
        "support": s["price"] if s else "",
        "sup_dist_pct": round(s["dist_pct"], 1) if s else "",
        "sup_dist_atr": round(s["dist_atr"], 2) if s else "",
        "sup_levels": s["count"] if s else 0,
        "sup_width": s["width"] if s else 0,
        "sup_types": s["types"] if s else "",
        "resistance": r["price"] if r else "",
        "res_dist_pct": round(r["dist_pct"], 1) if r else "",
        "res_dist_atr": round(r["dist_atr"], 2) if r else "",
        "res_levels": r["count"] if r else 0,
        "res_width": r["width"] if r else 0,
        "res_types": r["types"] if r else "",
        "verdict": verdict,
    }
    tb = trend_block(df) or {}
    zb = zone_block(df) or {}
    for k in ("trend", "adx", "di_plus", "di_minus", "ma_align", "slope50"):
        row[k] = tb.get(k, "")
    for k in ("poc", "val", "vah", "zone_pos", "obv_slope", "ad_slope",
              "mfi", "accum"):
        row[k] = zb.get(k, "")
    return row


# --------------------------------------------------------------------------
# Dhan sync  (audit trail)
# --------------------------------------------------------------------------

def _dhan_token():
    """Token from the environment, else the bot's own auth cache."""
    tok = (os.environ.get("DHAN_ACCESS_TOKEN") or "").strip()
    if tok:
        return tok
    repo = os.path.dirname(HERE)
    if repo not in sys.path:
        sys.path.insert(0, repo)
    try:
        from src.auth import get_access_token          # noqa: E402
        return get_access_token()
    except Exception as e:
        raise RuntimeError(
            "no Dhan token: set DHAN_ACCESS_TOKEN, or make sure src/auth.py "
            f"works from {repo} ({str(e)[:60]})")


def dhan_holdings():
    import requests
    token = _dhan_token()
    client_id = os.environ.get("DHAN_CLIENT_ID", "1000191656")
    r = requests.get(
        DHAN_HOLDINGS_URL, timeout=30,
        headers={"access-token": token, "client-id": client_id,
                 "Content-Type": "application/json", "Accept": "application/json"})
    r.raise_for_status()
    payload = r.json()
    data = payload.get("data") if isinstance(payload, dict) else payload
    if isinstance(payload, dict) and payload.get("status") not in (None, "success"):
        raise RuntimeError(f"Dhan returned: {payload.get('remarks') or payload}")
    return data or []


def parse_holdings(data):
    """Dhan holdings payload -> [{symbol, qty, avg_price}], zero-qty dropped."""
    out = []
    for h in data or []:
        sym = str(h.get("tradingSymbol") or "").strip().upper()
        for suf in ("-EQ", "-BE", "-BZ", "-BL", "-IL"):
            if sym.endswith(suf):
                sym = sym[: -len(suf)]
        try:
            qty = int(h.get("totalQty") or 0)
        except (TypeError, ValueError):
            qty = 0
        if qty <= 0 or not sym:
            continue
        try:
            avg = round(float(h.get("avgCostPrice") or 0), 2)
        except (TypeError, ValueError):
            avg = 0.0
        out.append({"symbol": sym, "qty": qty, "avg_price": avg})
    return sorted(out, key=lambda x: x["symbol"])


def write_holdings(path, rows):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["symbol", "qty", "avg_price"])
        for r in rows:
            w.writerow([r["symbol"], r["qty"], r["avg_price"]])


HISTORY_COLUMNS = ["date", "symbol", "qty", "avg_price", "cost_value"]


def read_history(path):
    rows = []
    if not os.path.exists(path):
        return rows
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            rows.append(r)
    return rows


def latest_snapshot(path, before=None):
    """{symbol: (qty, avg)} from the most recent date in the history."""
    rows = read_history(path)
    if before:
        rows = [r for r in rows if r["date"] < before]
    if not rows:
        return {}, None
    last = max(r["date"] for r in rows)
    snap = {r["symbol"]: (int(r["qty"]), float(r["avg_price"]))
            for r in rows if r["date"] == last}
    return snap, last


def write_history(path, day, rows):
    """Replace this day's block, keep every other day untouched."""
    kept = [r for r in read_history(path) if r["date"] != day]
    for r in rows:
        kept.append({"date": day, "symbol": r["symbol"], "qty": r["qty"],
                     "avg_price": r["avg_price"],
                     "cost_value": round(r["qty"] * r["avg_price"], 2)})
    kept.sort(key=lambda r: (r["date"], r["symbol"]))
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=HISTORY_COLUMNS)
        w.writeheader()
        w.writerows(kept)


def diff_snapshots(prev, cur):
    """[(change, symbol, old_qty, new_qty)] across the two snapshots."""
    out = []
    for sym in sorted(set(prev) | set(cur)):
        old, new = prev.get(sym), cur.get(sym)
        if old is None:
            out.append(("ADDED", sym, 0, new[0]))
        elif new is None:
            out.append(("EXITED", sym, old[0], 0))
        elif old[0] != new[0]:
            out.append(("QTY", sym, old[0], new[0]))
    return out


def sync_from_dhan(args):
    import datetime
    try:
        data = dhan_holdings()
    except Exception as e:
        _log(f"Dhan sync failed: {str(e)[:120]}")
        return 1
    rows = parse_holdings(data)
    if not rows:
        _log("Dhan returned no equity holdings")
        return 1

    day = datetime.date.today().isoformat()
    prev, prev_day = latest_snapshot(args.history, before=day)
    write_holdings(args.holdings, rows)
    write_history(args.history, day, rows)

    cur = {r["symbol"]: (r["qty"], r["avg_price"]) for r in rows}
    _log(f"synced {len(rows)} holdings from Dhan -> {os.path.basename(args.holdings)}")
    if prev_day:
        changes = diff_snapshots(prev, cur)
        if changes:
            print(f"  changes since {prev_day}:")
            for kind, sym, old, new in changes:
                print(f"    {kind:7s} {sym:14s} {old:>6d} -> {new:>6d}")
        else:
            print(f"  no change since {prev_day}")
    else:
        print("  first snapshot - this becomes the audit baseline")
    total = sum(r["qty"] * r["avg_price"] for r in rows)
    print(f"  cost value Rs.{total:,.0f} | history: {os.path.basename(args.history)}")
    return 0


def load_holdings(path):
    rows = []
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            sym = (r.get("symbol") or "").strip().upper()
            if not sym or sym.startswith("#"):
                continue
            rows.append({
                "symbol": sym,
                "qty": r.get("qty") or "",
                "avg_price": r.get("avg_price") or "",
            })
    return rows


def fetch(symbol, period="1y"):
    import yfinance as yf
    t = yf.Ticker(symbol + ".NS")
    return t.history(period=period, auto_adjust=False)


# --------------------------------------------------------------------------
# selftest
# --------------------------------------------------------------------------

def selftest():
    import pandas as pd
    import numpy as np

    ok = fail = 0

    def check(name, cond):
        nonlocal ok, fail
        if cond:
            print(f"PASS  {name}")
            ok += 1
        else:
            print(f"FAIL  {name}")
            fail += 1

    # 260 synthetic sessions, price pinned near a known cluster
    idx = pd.date_range("2025-11-01", periods=260, freq="B")
    base = np.linspace(100, 200, 260)
    close = pd.Series(base, index=idx)
    df = pd.DataFrame({
        "Close": close,
        "High": close * 1.01,
        "Low": close * 0.99,
        "Open": close,
        "Volume": 1000,
    })

    lv = build_levels(df)
    check("levels are produced", len(lv) > 10)
    check("all levels are finite and positive",
          all(p > 0 and p == p for p, _ in lv))
    labels = [l for _, l in lv]
    for want in ("SMA20", "SMA50", "SMA200", "52w-high", "fib52-0.5", "bb-upper"):
        check(f"level present: {want}", want in labels)
    check("bollinger family present", any(l.startswith("bb-") for l in labels))
    # bb-mid IS SMA20; counting one price twice would inflate the confluence
    check("bb-mid collapses into SMA20 (no double counting)",
          "bb-mid" not in labels)

    px = float(close.iloc[-1])
    a = atr(df["High"], df["Low"], df["Close"])
    check("ATR is positive", a and a > 0)

    sup, res = cluster(lv, px, 0.5 * a)
    check("support clusters are below the price",
          all(c["price"] < px for c in sup))
    check("resistance clusters are above the price",
          all(c["price"] > px for c in res))
    check("clusters are ordered by proximity",
          all(sup[i]["price"] >= sup[i + 1]["price"] for i in range(len(sup) - 1)))

    # a synthetic cluster: three levels within tolerance must merge
    tight = [(100.0, "a"), (100.2, "b"), (100.4, "c"), (90.0, "d")]
    s2, r2 = cluster(tight, 110.0, 0.5)
    check("nearby levels merge into one cluster", s2[0]["count"] == 3)
    check("a distant level stays separate", len(s2) == 2)

    row = analyse_one("TEST", df)
    check("analysis returns a verdict", "verdict" in row and row["verdict"])
    check("verdict mentions a level count",
          "levels" in row["verdict"] or row["verdict"] == "MID-RANGE")

    # --- Dhan payload parsing & audit diff -------------------------------
    payload = [
        {"tradingSymbol": "MAHABANK-EQ", "totalQty": 470, "avgCostPrice": 83.62},
        {"tradingSymbol": "TITAN", "totalQty": 22, "avgCostPrice": 2776.50},
        {"tradingSymbol": "SOLD-EQ", "totalQty": 0, "avgCostPrice": 100.0},
    ]
    parsed = parse_holdings(payload)
    check("holdings: zero-qty rows dropped", len(parsed) == 2)
    check("holdings: -EQ suffix stripped",
          [r["symbol"] for r in parsed] == ["MAHABANK", "TITAN"])
    check("holdings: qty and avg parsed",
          parsed[0]["qty"] == 470 and abs(parsed[0]["avg_price"] - 83.62) < 0.01)

    d = diff_snapshots({"MAHABANK": (350, 82.23), "OLD": (5, 10.0)},
                       {"MAHABANK": (470, 83.62), "TITAN": (22, 2776.5)})
    kinds = {(k, s): (o, n) for k, s, o, n in d}
    check("diff: added name detected", kinds.get(("ADDED", "TITAN")) == (0, 22))
    check("diff: exited name detected", kinds.get(("EXITED", "OLD")) == (5, 0))
    check("diff: quantity change detected",
          kinds.get(("QTY", "MAHABANK")) == (350, 470))

    import tempfile
    with tempfile.TemporaryDirectory() as td:
        hp = os.path.join(td, "h.csv")
        write_holdings(hp, parsed)
        back = load_holdings(hp)
        check("holdings CSV round-trips", len(back) == 2 and back[0]["symbol"] == "MAHABANK")
        hist = os.path.join(td, "hist.csv")
        write_history(hist, "2026-11-09", parsed)
        write_history(hist, "2026-11-10", parsed[:1])
        check("history keeps both days", len(read_history(hist)) == 3)
        snap, day = latest_snapshot(hist)
        check("latest snapshot is the newest day",
              day == "2026-11-10" and list(snap) == ["MAHABANK"])

    # --- trend ------------------------------------------------------------
    up = pd.DataFrame({"Close": pd.Series(np.linspace(100, 300, 260), index=idx)})
    up["High"] = up["Close"] * 1.01
    up["Low"] = up["Close"] * 0.99
    up["Volume"] = 1000.0
    tb = trend_block(up)
    check("trend: a rising series reads UP", tb.get("trend") in ("UP", "STRONG-UP"))
    check("trend: +DI exceeds -DI on a rise", tb.get("di_plus", 0) > tb.get("di_minus", 0))
    check("trend: MA alignment is up", tb.get("ma_align") == "up")

    # price above falling averages = transition, not a trend
    # 250 down + only 10 up: the 50-DMA is still falling, so it is a bounce
    vshape = pd.Series(np.concatenate([np.linspace(300, 200, 250),
                                       np.linspace(200, 280, 10)]), index=idx)
    vs = pd.DataFrame({"Close": vshape})
    vs["High"] = vs["Close"] * 1.01
    vs["Low"] = vs["Close"] * 0.99
    vs["Volume"] = 1000.0
    check("trend: a V-bounce is not called UP",
          trend_block(vs).get("trend") not in ("UP", "STRONG-UP"))

    dn = pd.DataFrame({"Close": pd.Series(np.linspace(300, 100, 260), index=idx)})
    dn["High"] = dn["Close"] * 1.01
    dn["Low"] = dn["Close"] * 0.99
    dn["Volume"] = 1000.0
    check("trend: a falling series reads DOWN",
          trend_block(dn).get("trend") in ("DOWN", "STRONG-DOWN"))

    flat = pd.Series([100.0] * 200 + [150.0] * 60, index=idx)
    vp = volume_profile(flat * 1.01, flat * 0.99, flat,
                        pd.Series(1000.0, index=idx), sessions=126, bins=20)
    check("profile: POC lands in the heavy-volume band", vp and 95 <= vp["poc"] <= 105)
    check("profile: value area brackets the POC", vp and vp["val"] <= vp["poc"] <= vp["vah"])
    check("profile: value area is inside the price range",
          vp and vp["val"] >= 99 and vp["vah"] <= 152)

    c5 = pd.Series([10, 11, 12, 11, 10], index=idx[:5])
    o5 = obv(c5, pd.Series([100.0] * 5, index=idx[:5]))
    check("obv adds volume on up days", float(o5.iloc[2]) == 200.0)
    check("obv subtracts volume on down days", float(o5.iloc[4]) == 0.0)

    zb = zone_block(up)
    check("zone: returns a position", zb.get("zone_pos") in ("ABOVE", "INSIDE", "BELOW"))
    check("zone: returns a verdict", zb.get("accum") in (
        "ACCUMULATION", "DISTRIBUTION", "DISTRIBUTING-AT-HIGHS", "NEUTRAL"))
    check("zone: OBV slope is positive in an orderly rise", zb.get("obv_slope", 0) > 0)

    print(f"\n{ok} passed, {fail} failed")
    return 0 if fail == 0 else 1


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="Evening level-confluence report for equity holdings.")
    ap.add_argument("--holdings", default=DEFAULT_HOLDINGS)
    ap.add_argument("--report", default=DEFAULT_REPORT)
    ap.add_argument("--near", type=float, default=None,
                    help="only show names within N ATRs of the nearest level")
    ap.add_argument("--min-confluence", type=int, default=1,
                    help="only show clusters with at least N coinciding levels")
    ap.add_argument("--tol", type=float, default=0.25,
                    help="cluster tolerance in ATRs (default 0.25)")
    ap.add_argument("--sync-from-dhan", action="store_true",
                    help="pull holdings from Dhan, rewrite holdings.csv, "
                         "append a dated snapshot to holdings_history.csv")
    ap.add_argument("--history", default=DEFAULT_HISTORY)
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()

    if args.selftest:
        return selftest()

    if args.sync_from_dhan:
        rc = sync_from_dhan(args)
        if rc != 0:
            return rc

    if not os.path.exists(args.holdings):
        _log(f"holdings file not found: {args.holdings}")
        _log("expected header: symbol,qty,avg_price")
        return 2

    holdings = load_holdings(args.holdings)
    _log(f"{len(holdings)} holdings from {os.path.basename(args.holdings)}")

    rows = []
    for h in holdings:
        sym = h["symbol"]
        try:
            df = fetch(sym)
        except Exception as e:
            _log(f"  {sym:14s} fetch failed: {str(e)[:50]}")
            continue
        if df is None or df.empty:
            _log(f"  {sym:14s} no data")
            continue
        r = analyse_one(sym, df, args.tol)
        if "error" in r:
            _log(f"  {sym:14s} {r['error']}")
            continue
        if h["avg_price"]:
            try:
                r["pnl_pct"] = round((r["price"] / float(h["avg_price"]) - 1) * 100, 1)
            except (TypeError, ValueError):
                r["pnl_pct"] = ""
        else:
            r["pnl_pct"] = ""
        r["asof"] = str(df["Close"].dropna().index[-1].date())
        rows.append(r)
        _log(f"  {sym:14s} {r['verdict']}")

    if not rows:
        _log("nothing analysed")
        return 1

    # filters
    shown = rows
    if args.near is not None:
        shown = [r for r in shown
                 if min([x for x in (r["sup_dist_atr"], r["res_dist_atr"])
                         if x != ""] or [99]) <= args.near]
    if args.min_confluence > 1:
        shown = [r for r in shown
                 if max(r["sup_levels"], r["res_levels"]) >= args.min_confluence]

    shown.sort(key=lambda r: -max(r["sup_levels"], r["res_levels"]))

    def va(r):
        if r.get("val") == "" or r.get("vah") == "":
            return "-"
        return f"{r['val']:,.0f}-{r['vah']:,.0f}"

    print()
    print(f"{'sym':12s} {'price':>9s} {'pnl%':>6s} {'rsi':>5s} {'trend':11s} "
          f"{'adx':>4s} {'value area':>15s} {'pos':7s} {'volume read':20s} {'level':24s}")
    print("-" * 128)
    for r in shown:
        print(f"{r['symbol']:12s} {r['price']:>9,.1f} {str(r['pnl_pct']):>6s} "
              f"{r['rsi']:>5.1f} {str(r.get('trend','')):11s} "
              f"{str(r.get('adx','')):>4s} {va(r):>15s} "
              f"{str(r.get('zone_pos','')):7s} {str(r.get('accum','')):20s} "
              f"{r['verdict']:24s}")

    print()
    print("Trend + volume detail:")
    for r in shown:
        if r.get("trend") in ("STRONG-UP", "STRONG-DOWN") or r.get("accum") in (
                "ACCUMULATION", "DISTRIBUTION", "DISTRIBUTING-AT-HIGHS"):
            print(f"  {r['symbol']:12s} trend {str(r.get('trend','')):11s} "
                  f"ADX {str(r.get('adx','')):>4s} (+DI {str(r.get('di_plus','')):>4s} "
                  f"-DI {str(r.get('di_minus','')):>4s}, MA {r.get('ma_align','')}, "
                  f"slope {r.get('slope50','')}%)")
            print(f"  {'':12s} POC {r.get('poc','')} in {va(r)} | price {r.get('zone_pos','')} "
                  f"| OBV {r.get('obv_slope','')}% A/D {r.get('ad_slope','')} "
                  f"MFI {r.get('mfi','')} -> {r.get('accum','')}")

    print()
    print("Confluence detail (levels that agree):")
    for r in shown[:8]:
        if r["sup_levels"] >= 2:
            print(f"  {r['symbol']:12s} SUPPORT {r['support']:>9} "
                  f"({r['sup_levels']}, width {r.get('sup_width',0)}): {r['sup_types']}")
        if r["res_levels"] >= 2:
            print(f"  {r['symbol']:12s} RESIST  {r['resistance']:>9} "
                  f"({r['res_levels']}, width {r.get('res_width',0)}): {r['res_types']}")

    with open(args.report, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=REPORT_COLUMNS + ["asof"],
                           extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"\n[OK] {len(shown)} of {len(rows)} shown | full report: {args.report}")
    print("Levels are structure, not signals. Check the price before acting.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
