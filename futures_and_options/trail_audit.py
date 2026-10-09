#!/usr/bin/env python3
"""
trail_audit.py — daily audit of guard trail exits: was the exit helpful?

WHAT THIS ANSWERS
----------------
Audits EVERY exit — guard flattens (trail/loss/profit) AND structures the
algo closed on its own (typed ALGO/EXPIRY when a structure simply vanishes
from the guard log without a FLATTENING line).

Every time the F&O structure guard flattens a position (trail breach, loss
breach or profit target), the obvious next-day question is: did that exit
save us from a loss, or did it cut a winner short?  This script answers it
mechanically, per exit, and appends the evidence to a CSV so the decision
"keep / loosen / tighten the trailing stop" is made on accumulated data
instead of one-day impressions.

HOW THE AUDIT IS GENERATED (the method)
---------------------------------------
Inputs (no Dhan API, no trading, read-only):
  1. logs/fno_guard.log   — the guard's own cycle prints: every 5-min
     structure MTM sample ("MTM Rs.X | peak Y ... -> STATUS"), the
     "*** ... BREACH — FLATTENING <key> ***" events, and the [EXIT] order
     lines (symbol, side, qty).
  2. NIFTY (or underlying) 5-minute bars for the same day, via yfinance.

Pipeline per exit episode:
  Step 1  PARSE   the log: group MTM samples by structure key, find the
                   flatten events, take the [EXIT] symbols to learn the
                   exact legs (strike / call-put / long-short / qty).
  Step 2  CALIBRATE a pricing model on the guard's REAL samples captured
                   BEFORE the exit — no guesswork about entry prices:
      a) linear: MTM = a + b * NIFTY  (b = effective delta, ~Rs/point)
      b) Black-Scholes: price both legs from the real NIFTY path with
         time-to-expiry, calibrating (IV, entry-cost offset) to fit the
         observed MTM samples.  Needs no scipy: erf-based normal CDF.
  Step 3  PROJECT the counterfactual: walk the SAME model along the real
                   NIFTY bars from the exit moment to market close, and
                   record the best / worst / close marks a HELD position
                   would have shown.
  Step 4  CLASSIFY
                   hold_close < exit_mtm   -> SAVED-FROM-LOSS (HELPED)
                   hold_close > exit_mtm   -> LIMITED-PROFIT (PREMATURE)
                   |hold_close - exit_mtm| < tol -> NEUTRAL
                   and append one CSV row per episode to data/trail_audit.csv
                   (dupes by date+key+exit_time are skipped on re-runs).

HONEST LIMITS (kept visible in the output)
------------------------------------------
  * The counterfactual is a model, not a tape: the linear fit has an RMSE
    and the BS model assumes one flat IV (no smile, no intraday vol moves).
    Both methods are reported when possible; when they disagree in sign,
    the episode is flagged LOW-CONFIDENCE.
  * "Hold to close" ignores what Stratzy itself would have done with its
    own exits — it measures the trail decision, not the algo's.
  * Run AFTER market close (16:00+); mid-day runs only see bars so far.

USAGE
-----
  python futures_and_options/trail_audit.py                    # today
  python futures_and_options/trail_audit.py --date 2026-09-28
  python futures_and_options/trail_audit.py --selftest         # offline test

CRON (optional, after EOD report):
  20 16 * * 1-5  cd /home/kay22_ind/dhan-trading-bot && \
      /usr/bin/python3 futures_and_options/trail_audit.py >> logs/trail_audit.log 2>&1
"""
import argparse
import csv
import math
import os
import re
import sys
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

KOL = "Asia/Kolkata"

TICKER_MAP = [
    ("BANKNIFTY", "^NSEBANK"),
    ("FINNIFTY", "NIFTY_FIN_SERVICE.NS"),
    ("MIDCPNIFTY", "NIFTY_MID_SELECT.NS"),
    ("NIFTY", "^NSEI"),
]


# ---------------------------------------------------------------------------
# Step 1 — parse the guard log
# ---------------------------------------------------------------------------
RE_STARTED = re.compile(r"Started at\s+:\s+(\d{4}-\d{2}-\d{2}) ")
RE_MTM = re.compile(
    r"^\[(\d{2}:\d{2}:\d{2})\]\s+(.+?):\s+(\d+) lot\(s\)\s+\|"
    r"\s+MTM Rs\.(-?[\d,]+(?:\.\d+)?)")
RE_PEAK = re.compile(r"peak Rs\.(-?[\d,]+(?:\.\d+)?)")
RE_FLATTEN = re.compile(r"(\w+) THRESHOLD BREACH.{0,6}FLATTENING\s+(.+?)\s+\*{3}")
RE_EXIT = re.compile(r"\[EXIT\]\s+(\S+):\s+(BUY|SELL)\s+(\d+)\s+MKT")
RE_NOOPEN = re.compile(r"^\[(\d{2}:\d{2}:\d{2})\] No open F&O")
RE_STRUCT = re.compile(r"\[STRUCT\]\s+(.+?)\s+\|\s*(.+)$")
RE_SYM_POS = re.compile(r"^(\S+)\s+(\d+)\s+([CP])E$")   # NIFTY 23300 CE
RE_SYM = re.compile(
    r"^(.+?)-([A-Za-z]{3})(\d{4})-(\d+)-([CP])[EP]$")  # NIFTY-Sep2026-23250-CE


def parse_log(text):
    """Return episodes: [{date, key, samples:[(t, mtm)], exits:[t],
    exit_orders:[(sym, side, qty)]}, ...] keyed by (date, key)."""
    episodes = {}

    def ep(d, k):
        return episodes.setdefault(
            (d, k), {"date": d, "key": k, "samples": [],
                     "exits": [], "exit_orders": []})

    cur_date = None
    cur_exit = None  # (date, key) of last FLATTENING line
    no_open = {}     # date -> last HH:MM:SS with "No open F&O positions"
    struct_legs = {} # key -> [(symbol, signed_qty), ...] from [STRUCT] lines
    for line in text.splitlines():
        m = RE_STARTED.search(line)
        if m:
            cur_date = m.group(1)
            continue
        if cur_date is None:
            continue
        m = RE_MTM.match(line)
        if m:
            t, key, _lots, mtm = m.groups()
            pm = RE_PEAK.search(line)
            ep(cur_date, key)["samples"].append(
                (t, float(mtm.replace(",", "")),
                 float(pm.group(1).replace(",", "")) if pm else None))
            continue
        m = RE_FLATTEN.search(line)
        if m:
            exit_type = m.group(1).upper()
            key = m.group(2).strip()
            ep(cur_date, key)["exits"].append(
                line[1:9])  # [HH:MM:SS] prefix of this cycle
            ep(cur_date, key).setdefault("exit_types", []).append(exit_type)
            cur_exit = (cur_date, key)
            continue
        m = RE_NOOPEN.match(line)
        if m:
            no_open[cur_date] = m.group(1)
            continue
        m = RE_STRUCT.search(line)
        if m:
            key = m.group(1).strip()
            legs = []
            for seg in m.group(2).split("|"):
                parts = seg.strip().rsplit(" q=", 1)
                if len(parts) == 2:
                    try:
                        legs.append((parts[0].strip(), int(parts[1])))
                    except ValueError:
                        pass
            if legs:
                struct_legs[key] = legs
            continue
        m = RE_EXIT.search(line)
        if m and cur_exit:
            episodes[cur_exit]["exit_orders"].append(m.groups())
            continue
    out = []
    for e in episodes.values():
        if not e["samples"]:
            continue
        e["no_open_after"] = no_open.get(e["date"])
        e["struct_legs"] = struct_legs.get(e["key"], [])
        out.append(e)
    return out


def exit_of(episode):
    """
    (exit_time, exit_type) for an episode.
      * guard exits  -> the FLATTENING time and its type (TRAIL/LOSS/PROFIT)
      * vanished     -> the last sample, typed ALGO (algo closed it) or EXPIRY
                        (needs the guard's [STRUCT] line for strikes)
                        (disappeared at its own expiry moment)
    Returns (None, None) when the structure is still open at the end of the
    log — there was no exit, so there is nothing to audit.
    """
    if episode.get("exits"):
        return episode["exits"][0], (episode.get("exit_types") or ["GUARD"])[0]
    last_t = episode["samples"][-1][0]
    nt = episode.get("no_open_after")
    if not nt or last_t >= nt:
        return None, None            # still open — not an exit
    m = re.search(r"\|(\d{4}-\d{2}-\d{2}) (\d{2}:\d{2})", episode["key"])
    if m:
        exp = pd.Timestamp(f"{m.group(1)} {m.group(2)}", tz=KOL)
        got = pd.Timestamp(f"{episode['date']} {last_t}", tz=KOL)
        if abs((exp - got).total_seconds()) <= 1800:
            return last_t, "EXPIRY"
    return last_t, "ALGO"


def _parse_symbol(sym):
    """(underlying, strike, is_call) from any symbol style, else None.
    Handles the broker order format (NIFTY-Sep2026-23250-CE) and the
    positions format (NIFTY 23300 CE / NIFTY 13 OCT 22100 PUT)."""
    m = RE_SYM.match(sym)                      # NIFTY-Sep2026-23250-CE
    if m:
        under, _mon, _yr, strike, cp = m.groups()
        return under, float(strike), cp == "C"
    s = str(sym).upper()
    mt = re.search(r"(CE|PE|CALL|PUT)\s*$", s)
    if not mt:
        return None
    nums = re.findall(r"\d+", s)
    if not nums or not s.split():
        return None
    return s.split()[0].split("-")[0], float(nums[-1]), mt.group(1) in (
        "CE", "CALL")


def episode_legs(episode):
    """(expiry_ts, qty, [(strike, is_call, direction)], underlying) or None.
    Legs come from the guard's [EXIT] order lines when it exited; otherwise
    from the [STRUCT] snapshot the guard logs for every structure it sees."""
    first_side = {}
    for sym, side, qty in episode["exit_orders"]:
        parsed = _parse_symbol(sym)
        if not parsed:
            return None
        under, strike, is_call = parsed
        if sym not in first_side:
            first_side[sym] = (under, strike, is_call,
                               1 if side == "SELL" else -1, abs(int(qty)))
    if not first_side:
        for sym, q in episode.get("struct_legs", []):
            parsed = _parse_symbol(sym)
            if not parsed or q == 0:
                continue
            under, strike, is_call = parsed
            first_side[sym] = (under, strike, is_call,
                               1 if q > 0 else -1, abs(int(q)))
    if not first_side:
        return None
    expiry = None
    legs = []
    qty_common = None
    underlying = None
    for under, strike, is_call, direction, qty in first_side.values():
        underlying = under
        qty_common = qty if qty_common is None else qty_common
        legs.append((strike, is_call, direction))
    # expiry day: take from the structure key (format ...|YYYY-MM-DD HH:MM)
    m = re.search(r"\|(\d{4}-\d{2}-\d{2}) (\d{2}:\d{2})", episode["key"])
    if m:
        expiry = pd.Timestamp(f"{m.group(1)} {m.group(2)}", tz=KOL)
    return expiry, qty_common, legs, underlying


# ---------------------------------------------------------------------------
# Market data
# ---------------------------------------------------------------------------
def fetch_bars(date_str, ticker):
    """5-minute bars for one date, tz-Kolkata, retrying yfinance flakiness."""
    import yfinance as yf
    start = f"{date_str}"
    end = (datetime.strptime(date_str, "%Y-%m-%d") + timedelta(days=1)
           ).strftime("%Y-%m-%d")
    last_err = None
    for _ in range(4):
        try:
            df = yf.download(ticker, interval="5m", start=start, end=end,
                             progress=False, auto_adjust=True)
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = [c[0] for c in df.columns]
            df = df.dropna(subset=["Close"])
            if len(df) >= 20:
                df.index = (df.index.tz_localize(KOL)
                            if df.index.tz is None else df.index.tz_convert(KOL))
                return df["Close"].astype(float)
        except Exception as e:  # network flake — retry
            last_err = e
    if last_err:
        print(f"[WARN] bar fetch failed for {ticker} {date_str}: {last_err}")
    return None


def bar_at(closes, when):
    """Close of the last 5m bar that ENDED at or before `when`."""
    ok = [(t, c) for t, c in closes.items() if t + pd.Timedelta(minutes=5) <= when]
    return ok[-1][1] if ok else None


# ---------------------------------------------------------------------------
# Step 2 — the two models
# ---------------------------------------------------------------------------
def fit_linear(samples_x, mtms):
    if len(samples_x) < 4:
        return None
    b, a = np.polyfit(np.array(samples_x), np.array(mtms), 1)
    y = np.array(mtms)
    yhat = a + b * np.array(samples_x)
    ss = ((y - yhat) ** 2).sum()
    rmse = math.sqrt(ss / len(y))
    var = ((y - y.mean()) ** 2).sum()
    r2 = float(1 - ss / var) if var > 0 else 0.0
    return {"a": float(a), "b": float(b), "rmse": rmse, "r2": r2}


def _ncdf(x):
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _bs_call(S, K, T, sigma, r=0.065):
    if T <= 0:
        return max(S - K, 0.0)
    d1 = (math.log(S / K) + (r + sigma ** 2 / 2) * T) / (sigma * math.sqrt(T))
    return S * _ncdf(d1) - K * math.exp(-r * T) * _ncdf(
        d1 - sigma * math.sqrt(T))


def fit_bs(samples_ts, Ss, mtms, expiry, qty, legs):
    """Calibrate (flat IV, entry offset) on pre-exit samples."""
    if expiry is None or len(samples_ts) < 4:
        return None

    def value(S, dt, sigma):
        T = (expiry - dt).total_seconds() / (365 * 24 * 3600)
        if T <= 0:
            return qty * sum(d * (max(S - k, 0.0) if call else 0.0)
                             for k, call, d in legs)
        return qty * sum(
            d * _bs_call(S, k, T, sigma) for k, call, d in legs)

    best = None
    for i in range(4, 181):  # sigma 10% .. 45.25%
        sigma = i * 0.0025
        vals = np.array([value(S, dt, sigma) for S, dt in zip(Ss, samples_ts)])
        off = float((vals - mtms).mean())
        sse = float(((vals - off - mtms) ** 2).sum())
        if best is None or sse < best[0]:
            best = (sse, sigma, off)
    sse, sigma, off = best
    return {"sigma": sigma, "off": off, "rmse": math.sqrt(sse / len(mtms)),
            "value": value}


# ---------------------------------------------------------------------------
# Step 3/4 — project, classify
# ---------------------------------------------------------------------------
def analyse_episode(episode, closes, min_samples, exit_t=None,
                    exit_type=None):
    date = episode["date"]
    if exit_t is None:
        exit_t = episode["exits"][0]
    if exit_type is None:
        exit_type = (episode.get("exit_types") or ["GUARD"])[0]
    samples = [(t, m, p) for t, m, p in episode["samples"] if t <= exit_t]
    if len(samples) < min_samples:
        return {"verdict": "INSUFFICIENT-SAMPLES", "n": len(samples)}

    def ts_of(t):
        return pd.Timestamp(f"{date} {t}", tz=KOL)

    exit_ts = ts_of(exit_t)
    exit_mtm = samples[-1][1]
    peak = max((p or m) for _, m, p in samples)

    # pair each sample with the NIFTY bar that had just closed
    xs, Ss, dts, ys = [], [], [], []
    for t, m, _p in samples:
        c = bar_at(closes, ts_of(t))
        if c is not None:
            xs.append(c); Ss.append(c); dts.append(ts_of(t)); ys.append(m)
    if len(xs) < min_samples:
        return {"verdict": "NO-BAR-OVERLAP", "n": len(xs)}

    lin = fit_linear(xs, ys)
    bs = None
    leg_info = episode_legs(episode)
    if leg_info:
        expiry, qty, legs, underlying = leg_info
        bs = fit_bs(dts, Ss, np.array(ys), expiry, qty, legs)

    # hold path: bars whose 5m window closed strictly after the exit
    hold = [(t, c) for t, c in closes.items()
            if t + pd.Timedelta(minutes=5) > exit_ts]
    if not hold:
        return {"verdict": "NO-AFTER-BARS", "n": len(xs)}

    rows = []
    for t, c in hold:
        marks = {}
        if lin:
            marks["lin"] = lin["a"] + lin["b"] * c
        if bs:
            marks["bs"] = bs["value"](c, t, bs["sigma"]) - bs["off"]
        rows.append((t, c, marks))

    def agg(method):
        vals = [m[method] for _, _, m in rows if method in m]
        return (min(vals), max(vals), vals[-1]) if vals else None

    lin_stats, bs_stats = agg("lin"), agg("bs")
    final = (bs_stats or lin_stats)[2]
    worst = min((s[0] for s in (lin_stats, bs_stats) if s), default=None)
    best = max((s[2] for s in (lin_stats, bs_stats) if s), default=None)
    method = "bs" if bs_stats and lin_stats else ("bs" if bs_stats else "lin")

    # --- sanity gate: refuse a verdict the data cannot support ---------
    # A short, quiet window cannot constrain the model. Seen on 2026-10-08:
    # a long-put structure fitted a +56/pt delta (wrong sign) and a 1%
    # implied vol, inverting the projection. Rather than record a confident-
    # looking wrong answer, mark the episode UNRELIABLE-FIT so it stays
    # visible but is never treated as evidence.
    def _unreliable(why):
        return {"verdict": "UNRELIABLE-FIT", "why": why, "n": len(xs),
                "exit_type": exit_type, "exit_time": exit_t,
                "exit_mtm": exit_mtm, "peak": peak, "confidence": "LOW",
                "delta": lin["b"] if lin else None,
                "sigma": bs["sigma"] if bs else None}

    if bs is not None:
        if not (0.05 <= bs["sigma"] <= 0.80):
            return _unreliable(f"implied vol {bs['sigma']:.0%} implausible")
    elif lin is None or lin.get("r2", 0) < 0.30:
        return _unreliable("linear fit too weak (no legs to price)")

    conflict = (lin_stats and bs_stats
                and (lin_stats[2] - exit_mtm) * (bs_stats[2] - exit_mtm) < 0)
    tol = 500.0
    if final < exit_mtm - tol:
        verdict = "SAVED-FROM-LOSS"
    elif final > exit_mtm + tol:
        verdict = "LIMITED-PROFIT"
    else:
        verdict = "NEUTRAL"
    # plain-language answer to "did the stop help, or was it premature?"
    verdict_plain = {"SAVED-FROM-LOSS": "HELPED",
                     "LIMITED-PROFIT": "PREMATURE",
                     "NEUTRAL": "NEUTRAL"}[verdict]
    low_n = len(xs) < 5

    return {"verdict": verdict, "verdict_plain": verdict_plain,
            "exit_type": exit_type,
            "confidence": ("LOW" if (conflict or low_n or not bs)
                           else "OK"),
            "exit_mtm": exit_mtm, "peak": peak, "n": len(xs),
            "hold_worst": worst, "hold_best": best, "hold_close": final,
            "delta": lin["b"] if lin else None,
            "sigma": bs["sigma"] if bs else None,
            "exit_time": exit_t, "method": method,
            "lin_close": lin_stats[2] if lin_stats else None,
            "bs_close": bs_stats[2] if bs_stats else None}


def fmt(v):
    return f"{v:,.0f}" if isinstance(v, (int, float)) else "-"


CSV_COLUMNS = ["date", "key", "exit_type", "exit_time", "exit_mtm", "peak",
               "hold_worst", "hold_best", "hold_close", "verdict",
               "verdict_plain", "confidence", "method", "delta_per_pt", "iv",
               "samples"]


def _ensure_csv_schema(path, columns):
    """If the CSV header is stale, rewrite it keeping old rows (blank-padded)."""
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        return
    with open(path, newline="") as f:
        rows = list(csv.reader(f))
    if not rows or rows[0] == columns:
        return
    old = rows[0]
    out = [columns]
    for r in rows[1:]:
        rec = dict(zip(old, r))
        out.append([rec.get(c, "") for c in columns])
    with open(path, "w", newline="") as f:
        csv.writer(f).writerows(out)
    print(f"[migrated] {path}: schema updated, {len(out) - 1} row(s) kept")


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--log", default="logs/fno_guard.log")
    ap.add_argument("--csv", default="data/trail_audit.csv")
    ap.add_argument("--date", default=None,
                    help="audit only this date (default: every date found)")
    ap.add_argument("--min-samples", type=int, default=3,
                    help="min MTM samples before the exit to fit a "
                         "model (default 3; <5 is flagged LOW)")
    ap.add_argument("--guard-only", action="store_true",
                    help="audit only guard exits; skip structures the algo "
                         "closed itself (default: audit both)")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()

    if args.selftest:
        return run_selftest()

    if not os.path.exists(args.log):
        print(f"[ERROR] log not found: {args.log}")
        return 1

    with open(args.log) as f:
        episodes = parse_log(f.read())
    if args.date:
        episodes = [e for e in episodes if e["date"] == args.date]
    if not episodes:
        print("[OK] no structures in log — nothing to audit.")
        return 0

    # dedup against CSV
    done = set()
    if os.path.exists(args.csv):
        with open(args.csv) as f:
            for r in csv.DictReader(f):
                done.add((r["date"], r["key"], r["exit_time"]))

    bar_cache = {}
    results = []
    for e in sorted(episodes, key=lambda x: (x["date"],
                                             exit_of(x)[0] or "99:99")):
        exit_t, exit_type = exit_of(e)
        if exit_t is None:
            print(f"[SKIP] {e['date']} {e['key']} — still open at log end")
            continue
        if args.guard_only and not e.get("exits"):
            continue
        if (e["date"], e["key"], exit_t) in done:
            print(f"[SKIP] {e['date']} {e['key']} @ {exit_t} "
                  f"(already in CSV)")
            continue
        under = "NIFTY"
        m = re.match(r"([A-Z]+)", e["key"])
        if m:
            for name, tick in TICKER_MAP:
                if m.group(1).startswith(name):
                    under, ticker = name, tick
                    break
            else:
                ticker = m.group(1) + ".NS"
        else:
            ticker = "^NSEI"
        if (e["date"], ticker) not in bar_cache:
            bar_cache[(e["date"], ticker)] = fetch_bars(e["date"], ticker)
        closes = bar_cache[(e["date"], ticker)]
        if closes is None:
            print(f"[WARN] no bars for {ticker} {e['date']} — skipping "
                  f"{e['key']}")
            continue
        r = analyse_episode(e, closes, args.min_samples, exit_t, exit_type)
        r.update({"date": e["date"], "key": e["key"]})
        results.append(r)

        print(f"\n=== {e['date']}  {e['key']}  (exit {r.get('exit_time')}) ===")
        if r["verdict"] in ("INSUFFICIENT-SAMPLES", "NO-BAR-OVERLAP",
                            "NO-AFTER-BARS", "UNRELIABLE-FIT"):
            why = f" — {r['why']}" if r.get("why") else ""
            print(f"  verdict: {r['verdict']} (samples {r.get('n')}){why}")
            print("  (not enough signal to judge this exit — left out of "
                  "the evidence)")
            continue
        print(f"  exit locked  : {fmt(r['exit_mtm'])}   "
              f"(day peak {fmt(r['peak'])}, "
              f"delta ~{fmt(r['delta']) if r['delta'] else '-'}/pt"
              + (f", IV {r['sigma']:.0%}" if r.get("sigma") else "") + ")")
        print(f"  hold worst  : {fmt(r['hold_worst'])}")
        print(f"  hold best   : {fmt(r['hold_best'])}")
        print(f"  hold close  : {fmt(r['hold_close'])}"
              + (f"   [lin {fmt(r['lin_close'])} / bs {fmt(r['bs_close'])}]"
                 if r.get("lin_close") and r.get("bs_close") else ""))
        print(f"  exit type   : {r.get('exit_type', '?')}")
        print(f"  PLAIN       : {r['verdict_plain']} — holding would have "
              f"closed at {fmt(r['hold_close'])} vs {fmt(r['exit_mtm'])} "
              f"locked "
              f"({fmt((r['hold_close'] or 0) - r['exit_mtm'])} given up)")
        print(f"  VERDICT     : {r['verdict']}  ({r['confidence']}, "
              f"n={r['n']} samples)")

    if not results:
        return 0
    _ensure_csv_schema(args.csv, CSV_COLUMNS)
    new = (not os.path.exists(args.csv)) or os.path.getsize(args.csv) == 0
    with open(args.csv, "a", newline="") as f:
        w = csv.writer(f)
        if new:
            w.writerow(CSV_COLUMNS)

        def num(v):
            return int(round(v)) if isinstance(v, (int, float)) else ""
        for r in results:
            w.writerow([r["date"], r["key"], r.get("exit_type", ""),
                        r.get("exit_time", ""),
                        num(r.get("exit_mtm")), num(r.get("peak")),
                        num(r.get("hold_worst")), num(r.get("hold_best")),
                        num(r.get("hold_close")), r["verdict"],
                        r.get("verdict_plain", ""),
                        r.get("confidence", ""), r.get("method", ""),
                        num(r.get("delta")),
                        f"{r['sigma']:.3f}" if r.get("sigma") else "",
                        r.get("n", "")])
    print(f"\n[OK] appended {len(results)} episode(s) to {args.csv}")
    return 0


# ---------------------------------------------------------------------------
# self-test (offline; uses today's real guard-log shapes)
# ---------------------------------------------------------------------------
def run_selftest():
    log_text = "\n".join([
        "  Started at  : 2026-09-28 10:20:01 IST",
        "[10:20:01] NIFTY-SEP|2026-09-29 14:30:00: 4 lot(s) | MTM Rs.618 | peak Rs.618 | loss-limit Rs.14,400 (profit-limit Rs.18,000) -> ok",
        "[10:35:02] NIFTY-SEP|2026-09-29 14:30:00: 4 lot(s) | MTM Rs.2,256 | peak Rs.2,256 | floor Rs.1,128 (profit-limit Rs.18,000) -> TRAIL-ARMED",
        "[10:50:01] NIFTY-SEP|2026-09-29 14:30:00: 4 lot(s) | MTM Rs.3,880 | peak Rs.3,880 | floor Rs.1,940 (profit-limit Rs.18,000) -> TRAIL-ARMED",
        "[11:05:02] NIFTY-SEP|2026-09-29 14:30:00: 4 lot(s) | MTM Rs.4,050 | peak Rs.4,050 | floor Rs.2,025 (profit-limit Rs.18,000) -> TRAIL-ARMED",
        "[11:15:01] NIFTY-SEP|2026-09-29 14:30:00: 4 lot(s) | MTM Rs.4,790 | peak Rs.4,790 | floor Rs.2,395 (profit-limit Rs.18,000) -> TRAIL-ARMED",
        "[11:25:01] NIFTY-SEP|2026-09-29 14:30:00: 4 lot(s) | MTM Rs.5,258 | peak Rs.5,258 | floor Rs.2,629 (profit-limit Rs.18,000) -> TRAIL-ARMED",
        "[11:30:02] NIFTY-SEP|2026-09-29 14:30:00: 4 lot(s) | MTM Rs.3,568 | peak Rs.5,258 | floor Rs.2,629 (profit-limit Rs.18,000) -> TRAIL-ARMED",
        "[11:35:01] NIFTY-SEP|2026-09-29 14:30:00: 4 lot(s) | MTM Rs.4,180 | peak Rs.5,258 | floor Rs.2,629 (profit-limit Rs.18,000) -> TRAIL-ARMED",
        "[11:40:01] NIFTY-SEP|2026-09-29 14:30:00: 4 lot(s) | MTM Rs.2,450 | peak Rs.5,258 | floor Rs.2,629 (profit-limit Rs.18,000) -> TRAIL-BREACH",
        "[11:40:01] *** LOSS THRESHOLD BREACH \u2014 FLATTENING NIFTY-SEP|2026-09-29 14:30:00 ***",
        "    [EXIT] NIFTY-Sep2026-23250-CE: SELL 260 MKT (MARGIN)",
        "           -> OK orderId=222260928331531",
        "    [EXIT] NIFTY-Sep2026-22850-CE: BUY 260 MKT (MARGIN)",
        "           -> OK orderId=34226092868131",
        "[DONE] Single-pass check complete, exiting.",
        "  Started at  : 2026-09-28 11:45:01 IST",
        "[11:45:01] NIFTY-SEP|2026-09-29 14:30:00: 4 lot(s) | MTM Rs.260 | peak Rs.5,258 | floor Rs.2,629 (profit-limit Rs.18,000) -> TRAIL-BREACH",
        "[11:45:01] *** LOSS THRESHOLD BREACH \u2014 FLATTENING NIFTY-SEP|2026-09-29 14:30:00 ***",
        "    [EXIT] NIFTY-Sep2026-22850-CE: SELL 260 MKT (MARGIN)",
        "[DONE] Single-pass check complete, exiting.",
    ])
    eps = parse_log(log_text)
    assert len(eps) == 1, f"expected 1 episode, got {len(eps)}"
    e = eps[0]
    assert e["exits"] == ["11:40:01", "11:45:01"]
    assert e["samples"][0][1] == 618, "first sample must parse"
    assert e["samples"][-1][1] == 260, "post-exit cleanup sample retained"
    pre = [s for s in e["samples"] if s[0] <= e["exits"][0]]
    assert pre[-1][1] == 2450, "analysis window must end at first exit"
    legs = episode_legs(e)
    assert legs is not None
    expiry, qty, lgs, under = legs
    assert qty == 260 and under == "NIFTY"
    assert sorted((k, c, d) for k, c, d in lgs) == \
        [(22850.0, True, -1), (23250.0, True, 1)], lgs
    assert expiry == pd.Timestamp("2026-09-29 14:30", tz=KOL)

    def mk_bars(path):  # path: [(hhmm, level), ...] -> 5m closes
        idx, vals = [], []
        for t, lvl in path:
            idx.append(pd.Timestamp(f"2026-09-28 {t}", tz=KOL))
            vals.append(lvl)
        s = pd.Series(vals, index=pd.DatetimeIndex(idx))
        return s

    base = [("09:15", 22840), ("09:20", 22838), ("10:15", 22839),
            ("10:20", 22838), ("10:30", 22842), ("10:35", 22831),
            ("10:50", 22828), ("11:05", 22825), ("11:15", 22818),
            ("11:25", 22830), ("11:35", 22837), ("11:40", 22842)]
    # scenario A: afternoon fades (today-like) -> LIMITED-PROFIT
    fade = base + [("11:45", 22848), ("12:25", 22854), ("13:00", 22830),
                   ("14:00", 22810), ("14:35", 22768), ("15:25", 22780)]
    r = analyse_episode(e, mk_bars(fade), 5)
    assert r["exit_mtm"] == 2450 and r["peak"] == 5258
    assert r["verdict"] == "LIMITED-PROFIT", r
    assert r["verdict_plain"] == "PREMATURE", r
    assert r["exit_type"] == "LOSS", r
    assert r["hold_close"] > r["exit_mtm"] + 1000, r
    assert r["hold_worst"] < r["exit_mtm"] < r["hold_best"], \
        "hold path ordering: worst < exit < best"
    # scenario B: afternoon rallies hard (Friday-like) -> SAVED-FROM-LOSS
    rally = base + [("11:45", 22860), ("12:25", 22920), ("13:00", 22990),
                    ("14:00", 23060), ("14:35", 23120), ("15:25", 23150)]
    r2 = analyse_episode(e, mk_bars(rally), 5)
    assert r2["verdict"] == "SAVED-FROM-LOSS", r2
    assert r2["verdict_plain"] == "HELPED", r2
    assert r2["hold_close"] < r2["exit_mtm"], r2

    # ---- vanished structures (algo-side exits) ----
    log2 = "\n".join([
        "  Started at  : 2026-10-08 10:50:01 IST",
        "[10:50:01] NIFTY-OCT|2026-10-13 14:30:00: 2 lot(s) | MTM Rs.-1404 | peak Rs.-585 | loss-limit Rs.7,200 (profit-limit Rs.9,000) -> ok",
        "[10:55:01] NIFTY-OCT|2026-10-13 14:30:00: 2 lot(s) | MTM Rs.-1573 | peak Rs.-585 | loss-limit Rs.7,200 (profit-limit Rs.9,000) -> ok",
        "[11:00:01] NIFTY-OCT|2026-10-13 14:30:00: 2 lot(s) | MTM Rs.-2574 | peak Rs.-585 | loss-limit Rs.7,200 (profit-limit Rs.9,000) -> ok",
        "[11:05:01] NIFTY-OCT|2026-10-13 14:30:00: 2 lot(s) | MTM Rs.-2086 | peak Rs.-585 | loss-limit Rs.7,200 (profit-limit Rs.9,000) -> ok",
        "[11:25:32] NIFTY-OCT|2026-10-13 14:30:00: 2 lot(s) | MTM Rs.-6149 | peak Rs.-585 | loss-limit Rs.7,200 (profit-limit Rs.9,000) -> WARN75",
        "[11:26:01] No open F&O positions.",
    ])
    eps2 = parse_log(log2)
    assert len(eps2) == 1, eps2
    et, etype = exit_of(eps2[0])
    assert et == "11:25:32" and etype == "ALGO", (et, etype)

    # still open at the end of the log -> not an exit, not audited
    log3 = log2.replace("[11:26:01] No open F&O positions.", "").rstrip("\n")
    et3, etype3 = exit_of(parse_log(log3)[0])
    assert et3 is None and etype3 is None, (et3, etype3)

    # disappearing at its own expiry moment -> EXPIRY, not ALGO
    log4 = "\n".join([
        "  Started at  : 2026-10-13 14:20:01 IST",
        "[14:25:01] NIFTY-OCT|2026-10-13 14:30:00: 2 lot(s) | MTM Rs.600 | peak Rs.600 | loss-limit Rs.7,200 (profit-limit Rs.9,000) -> ok",
        "[14:30:01] NIFTY-OCT|2026-10-13 14:30:00: 2 lot(s) | MTM Rs.650 | peak Rs.650 | loss-limit Rs.7,200 (profit-limit Rs.9,000) -> ok",
        "[14:35:01] No open F&O positions.",
    ])
    assert exit_of(parse_log(log4)[0])[1] == "EXPIRY"

    # a vanished episode gets audited (linear-only, no legs to price)
    bars2 = pd.Series(
        [22838, 22838, 22825, 22818, 22800, 22854, 22810, 22780],
        index=pd.DatetimeIndex([
            pd.Timestamp(f"2026-10-08 {t}", tz=KOL) for t in
            ["10:50", "11:00", "11:05", "11:25", "11:40", "12:25",
             "14:00", "15:25"]]))
    r3 = analyse_episode(eps2[0], bars2, 3, "11:25:32", "ALGO")
    assert r3.get("exit_type") == "ALGO", r3
    assert r3.get("verdict") in ("SAVED-FROM-LOSS", "LIMITED-PROFIT",
                                 "NEUTRAL"), r3
    assert r3.get("method") == "lin", r3

    print("[SELFTEST] parse, legs, calibration, projection, verdicts, "
          "vanished-structure audit: ALL PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
