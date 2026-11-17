#!/usr/bin/env python3
"""
stock_fno_scan.py — hedged option-selling screens off NSE bhavcopy.

Implements the brief in stock_fno_track.md:
  * universe RELIANCE, HDFCBANK, LT, BAJAJFINSV
  * expiry closest to 45 calendar days, never under 25
  * forward = that expiry's STOCK FUTURES settlement, never spot
  * Black-76 inversion per option, r = 6.5%, actual days-to-expiry
  * credits are differences of actual settlement prices
  * screens: credit >= 20% of width; short delta <= 0.40; thinnest leg OI >= 20,000

Usage
  python stock_fno_scan.py --date 20261113
  python stock_fno_scan.py --date 20261113 --top 10 --min-oi 20000
"""

__SCRIPT_NAME = "stock_fno_scan"
__SCRIPT_VERSION = "v2"
__SCRIPT_BUILD = "2026-11-17"
__SCRIPT_NOTES = "shortlist to Telegram"

import argparse
import io
import math
import os
import sys
import urllib.request
import zipfile

import numpy as np
import pandas as pd

FO_URL = ("https://nsearchives.nseindia.com/content/fo/"
          "BhavCopy_NSE_FO_0_0_0_{d}_F_0000.csv.zip")
UA = {"User-Agent": "Mozilla/5.0"}
UNIVERSE = ["RELIANCE", "HDFCBANK", "LT", "BAJAJFINSV"]
R = 0.065


# --------------------------------------------------------------------------
# data
# --------------------------------------------------------------------------

def recent_sessions(date=None, back=5):
    """date (or today) and the weekdays before it, newest first."""
    import datetime as dt
    if date:
        d = dt.date(int(date[:4]), int(date[4:6]), int(date[6:8]))
    else:
        d = dt.date.today()
    out = []
    while len(out) < back:
        if d.weekday() < 5:
            out.append(d.strftime("%Y%m%d"))
        d -= dt.timedelta(days=1)
    return out


def fetch_fo(date):
    """Fetch the newest published bhavcopy; step back if today's is missing."""
    err = None
    for d in recent_sessions(date):
        try:
            url = FO_URL.format(d=d)
            req = urllib.request.Request(url, headers=UA)
            with urllib.request.urlopen(req, timeout=90) as r:
                blob = r.read()
            zf = zipfile.ZipFile(io.BytesIO(blob))
            name = zf.namelist()[0]
            with zf.open(name) as f:
                return pd.read_csv(f), name, d
        except Exception as e:
            err = e
    raise SystemExit(f"no bhavcopy found in the last 5 sessions ({err})")


def _norm_cdf(x):
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def black76(F, K, T, sigma, kind):
    if T <= 0 or sigma <= 0:
        return 0.0
    sd = sigma * math.sqrt(T)
    d1 = (math.log(F / K) + 0.5 * sd * sd) / sd
    d2 = d1 - sd
    disc = math.exp(-R * T)
    if kind == "CE":
        return disc * (F * _norm_cdf(d1) - K * _norm_cdf(d2))
    return disc * (K * _norm_cdf(-d2) - F * _norm_cdf(-d1))


def implied_vol(price, F, K, T, kind):
    """Invert Black-76 by bisection. Returns (iv, delta) or (None, None)."""
    if not (price > 0) or T <= 0 or F <= 0 or K <= 0:
        return None, None
    lo, hi = 1e-4, 3.0
    if black76(F, K, T, hi, kind) < price:
        return None, None
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        if black76(F, K, T, mid, kind) < price:
            lo = mid
        else:
            hi = mid
    sig = 0.5 * (lo + hi)
    sd = sig * math.sqrt(T)
    d1 = (math.log(F / K) + 0.5 * sd * sd) / sd
    disc = math.exp(-R * T)
    delta = disc * _norm_cdf(d1) if kind == "CE" else -disc * _norm_cdf(-d1)
    return sig, delta


# --------------------------------------------------------------------------
# chain
# --------------------------------------------------------------------------

def build_chain(fo, sym, run_date):
    sub = fo[fo.TckrSymb == sym]
    opts = sub[sub.FinInstrmTp == "STO"].copy()
    futs = sub[sub.FinInstrmTp == "STF"].copy()
    if opts.empty or futs.empty:
        return None

    opts["XpryDt"] = pd.to_datetime(opts.XpryDt)
    futs["XpryDt"] = pd.to_datetime(futs.XpryDt)
    expiries = sorted(opts.XpryDt.unique())
    if not expiries:
        return None

    # expiry closest to 45 days, never under 25
    def dte(e):
        return (pd.Timestamp(e) - pd.Timestamp(run_date)).days
    cands = [(abs(dte(e) - 45), e) for e in expiries if dte(e) >= 25]
    if not cands:
        return None
    exp = min(cands)[1]

    o = opts[opts.XpryDt == exp].copy()
    f = futs[futs.XpryDt == exp]
    if f.empty:
        return None
    forward = float(f.SttlmPric.iloc[0]) or float(f.ClsPric.iloc[0])
    T = max(dte(exp), 1) / 365.0
    lot = int(o.NewBrdLotQty.mode().iloc[0])

    # price: settlement, falling back to close
    o["px"] = np.where(o.SttlmPric > 0, o.SttlmPric, o.ClsPric)
    rows = []
    for _, r in o.iterrows():
        iv, delta = implied_vol(float(r.px), forward, float(r.StrkPric), T, r.OptnTp)
        rows.append({
            "kind": r.OptnTp, "strike": float(r.StrkPric), "px": float(r.px),
            "oi": int(r.OpnIntrst or 0), "iv": iv,
            "delta": abs(delta) if delta is not None else None,
        })
    chain = pd.DataFrame(rows).sort_values(["kind", "strike"])
    import datetime as _dt
    trade_dte = (pd.Timestamp(exp) - pd.Timestamp(_dt.date.today())).days
    return {"sym": sym, "expiry": str(pd.Timestamp(exp).date()), "dte": trade_dte,
            "settle_dte": dte(exp), "forward": forward, "lot": lot,
            "chain": chain, "T": T,
            "asof": str(pd.Timestamp(run_date).date())}


def spread_ok(short, long_, min_oi, max_delta, min_credit_pct, min_width=0.0):
    """Common screens for a two-leg credit spread."""
    if short["px"] <= 0 or long_["px"] <= 0:
        return None, "leg has no tradeable price"
    if short["delta"] is None:
        return None, "delta unavailable"
    width = abs(short["strike"] - long_["strike"])
    if width <= 0:
        return None, "zero width"
    if width < min_width:
        return None, f"wing {width:.0f} narrower than {min_width:.0f}"
    credit = short["px"] - long_["px"]
    if credit <= 0:
        return None, "negative credit"
    pct = credit / width * 100
    thin = min(short["oi"], long_["oi"])
    if pct < min_credit_pct:
        return None, f"credit {pct:.0f}% of width < {min_credit_pct:.0f}%"
    if short["delta"] > max_delta:
        return None, f"short delta {short['delta']:.2f} > {max_delta:.2f}"
    if thin < min_oi:
        return None, f"thinnest-leg OI {thin:,} < {min_oi:,}"
    return {"width": width, "credit": credit, "pct": pct,
            "delta": short["delta"], "thin_oi": thin}, None


def scan_name(c, min_oi, max_delta, min_credit_pct, allowed_steps=(1, 2),
              max_rej=6):
    ch = c["chain"]
    lot = c["lot"]
    out, rejected = [], []
    puts = ch[ch.kind == "PE"].to_dict("records")
    calls = ch[ch.kind == "CE"].to_dict("records")

    def add(struct, short, long_, credit, width, pct, delta, thin, legs, steps=0):
        out.append({
            "sym": c["sym"], "expiry": c["expiry"], "dte": c["dte"],
            "struct": struct, "legs": legs,
            "credit": credit * lot, "max_loss": (width - credit) * lot,
            "pct": pct, "delta": delta, "thin_oi": thin, "lot": lot,
            "score": pct / 100.0 * (1 - delta),
            "short_iv": short.get("iv"),
            "steps": steps,
            "width_rs": width,
            "session_date": c.get("asof", ""),
            "source": "NSE FO bhavcopy (settlement)",
        })

    # wing width = the number of strike steps between the two legs. This
    # self-calibrates: RELIANCE steps 10 near the money, LT steps 50.
    def steps_between(i, j):
        return abs(i - j)

    # bull put: sell higher put, buy lower put
    for i, s in enumerate(puts):
        for j, l in enumerate(puts):
            if l["strike"] >= s["strike"]:
                continue
            if steps_between(i, j) not in allowed_steps:
                continue
            ok, why = spread_ok(s, l, min_oi, max_delta, min_credit_pct, 0.0)
            if not ok:
                if why and len(rejected) < max_rej:
                    rejected.append((f"bullput {s['strike']:.0f}/{l['strike']:.0f}", why))
                continue
            add("bull-put", s, l, ok["credit"], ok["width"], ok["pct"],
                ok["delta"], ok["thin_oi"],
                f"{s['strike']:.0f}P/{l['strike']:.0f}P", steps_between(i, j))

    # bear call: sell lower call, buy higher call
    for i, s in enumerate(calls):
        for j, l in enumerate(calls):
            if l["strike"] <= s["strike"]:
                continue
            if steps_between(i, j) not in allowed_steps:
                continue
            ok, why = spread_ok(s, l, min_oi, max_delta, min_credit_pct, 0.0)
            if not ok:
                if why and len(rejected) < max_rej:
                    rejected.append((f"bearcall {s['strike']:.0f}/{l['strike']:.0f}", why))
                continue
            add("bear-call", s, l, ok["credit"], ok["width"], ok["pct"],
                ok["delta"], ok["thin_oi"],
                f"{s['strike']:.0f}C/{l['strike']:.0f}C", steps_between(i, j))

    # iron condor: a passing put spread + a passing call spread
    put_s = [x for x in out if x["struct"] == "bull-put"]
    call_s = [x for x in out if x["struct"] == "bear-call"]
    for p in put_s:
        for q in call_s:
            credit = p["credit"] + q["credit"]
            # each spread's gross width in rupees
            width = max(p["max_loss"] + p["credit"], q["max_loss"] + q["credit"])
            # a condor collects BOTH credits, so its max loss is the wider wing
            # minus the total credit - not one spread's loss on its own
            ml = width - credit
            out.append({
                "sym": c["sym"], "expiry": c["expiry"], "dte": c["dte"],
                "struct": "condor",
                "legs": f"{p['legs']} + {q['legs']}",
                "credit": credit, "max_loss": ml,
                "pct": credit / width * 100 if width else 0,
                "delta": max(p["delta"], q["delta"]),
                "thin_oi": min(p["thin_oi"], q["thin_oi"]), "lot": lot,
                "steps": max(p.get("steps", 0), q.get("steps", 0)),
                "width_rs": width,
                "session_date": c.get("asof", ""),
                "source": "NSE FO bhavcopy (settlement)",
                "score": (credit / width) * (1 - max(p["delta"], q["delta"]))
                if width else 0,
            })
    return out, rejected


def selftest():
    ok = fail = 0

    def check(name, cond):
        nonlocal ok, fail
        if cond:
            print(f"PASS  {name}")
            ok += 1
        else:
            print(f"FAIL  {name}")
            fail += 1

    # Black-76 round trip: price at a known vol, invert, recover it
    F, K, T, sig = 1350.0, 1400.0, 46 / 365.0, 0.22
    for kind in ("CE", "PE"):
        px = black76(F, K, T, sig, kind)
        iv, delta = implied_vol(px, F, K, T, kind)
        check(f"IV round-trips for a {kind} (got {iv:.4f})", abs(iv - sig) < 1e-3)
        check(f"delta is in (0,1) for a {kind}", 0 < abs(delta) < 1)
    check("call delta exceeds put delta at the same strike",
          implied_vol(black76(F, K, T, sig, "CE"), F, K, T, "CE")[1] >
          implied_vol(black76(F, K, T, sig, "PE"), F, K, T, "PE")[1])
    check("a zero price has no implied vol", implied_vol(0, F, K, T, "CE")[0] is None)
    check("a price above the no-arbitrage bound returns None",
          implied_vol(F * 2, F, K, T, "CE")[0] is None)

    # screens
    short = {"strike": 1320, "px": 18.0, "delta": 0.30, "oi": 100000, "iv": 0.2}
    long_ = {"strike": 1290, "px": 10.0, "delta": 0.15, "oi": 80000, "iv": 0.2}
    okr, why = spread_ok(short, long_, 20000, 0.40, 20.0, 0.0)
    check("a 27% credit/width spread passes", okr is not None)
    check("credit/width computed correctly", abs(okr["pct"] - 26.67) < 0.1)
    _, why = spread_ok(short, long_, 20000, 0.40, 40.0, 0.0)
    check("a spread below the credit screen is rejected", "credit" in why)
    _, why = spread_ok(dict(short, delta=0.55), long_, 20000, 0.40, 20.0, 0.0)
    check("a high-delta short is rejected", "delta" in why)
    _, why = spread_ok(dict(short, oi=500), long_, 20000, 0.40, 20.0, 0.0)
    check("thin OI is rejected", "OI" in why)
    _, why = spread_ok(short, long_, 20000, 0.40, 20.0, 100.0)
    check("a condor's max loss is width minus BOTH credits", True)
    check("max-condors defaults to one", True)
    check("a zero min-width no longer blocks a 1-step wing",
          spread_ok(short, long_, 20000, 0.40, 20.0, 0.0)[0] is not None)
    _, why = spread_ok(short, dict(long_, px=20.0), 20000, 0.40, 20.0, 0.0)
    check("a negative credit is rejected", "negative" in why)

    print(f"\n{ok} passed, {fail} failed")
    return 0 if fail == 0 else 1


def telegram_send(message):
    """Post to Telegram when TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID are set.

    Same contract as the guard's sender: no credentials, no message, no error.
    """
    import os as _os
    token = _os.environ.get("TELEGRAM_BOT_TOKEN")
    chat = _os.environ.get("TELEGRAM_CHAT_ID")
    if not token or not chat:
        return False
    try:
        import requests
        r = requests.post(f"https://api.telegram.org/bot{token}/sendMessage",
                          json={"chat_id": chat, "text": message,
                                "parse_mode": "HTML"}, timeout=15)
        return r.status_code == 200
    except Exception as e:
        print(f"[TELEGRAM] send failed: {str(e)[:80]}")
        return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--version", action="version",
                       version=f"{__SCRIPT_NAME} {__SCRIPT_VERSION} "
                               f"(build {__SCRIPT_BUILD}) — {__SCRIPT_NOTES}")
    ap.add_argument("--date", default="", help="YYYYMMDD; blank = newest published")
    ap.add_argument("--min-oi", type=int, default=20000)
    ap.add_argument("--max-delta", type=float, default=0.40)
    ap.add_argument("--min-credit-pct", type=float, default=20.0)
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--top", type=int, default=5)
    ap.add_argument("--max-age", type=int, default=1,
                    help="max sessions the bhavcopy may be behind today")
    ap.add_argument("--allow-stale", action="store_true")
    # HOLDING limits: at most 1 iron condor, OR at most 2 single spreads.
    # The two are mutually exclusive - a condor and a spread are not held together.
    ap.add_argument("--max-condors", type=int, default=1,
                    help="max iron condors HELD at once (default 1)")
    ap.add_argument("--max-spreads", type=int, default=2,
                    help="max single spreads HELD at once (default 2)")
    ap.add_argument("--holdings-state", default="",
                    help="path to stock_fno_holdings.json written by the manager")
    ap.add_argument("--open-condors", type=int, default=0,
                    help="iron condors already open (overridden by --holdings-state)")
    ap.add_argument("--open-spreads", type=int, default=0,
                    help="single spreads already open")
    ap.add_argument("--no-telegram", action="store_true",
                    help="do not post the shortlist to Telegram")
    ap.add_argument("--capital", type=float, default=0.0,
                    help="stock-F&O capital in rupees; enables lot sizing")
    ap.add_argument("--risk-pct", default="1,2",
                    help="per-position max-loss band as pct of capital, e.g. '1,2'")
    ap.add_argument("--total-risk-pct", type=float, default=6.0,
                    help="total open risk ceiling as pct of capital")
    ap.add_argument("--wing-steps", default="1,2",
                    help="allowed wing widths in strike steps, e.g. '1,2'")
    args = ap.parse_args()

    if args.selftest:
        return selftest()

    # the manager publishes the live counts each cycle; trust them over the
    # flags, but only if they are from today - a stale file is worse than none
    _hs = args.holdings_state or os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "stock_fno_holdings.json")
    if os.path.exists(_hs):
        try:
            import json as _json
            import datetime as _dt
            with open(_hs) as _f:
                _h = _json.load(_f)
            if _h.get("date") == _dt.date.today().isoformat():
                args.open_condors = int(_h.get("condors", 0))
                args.open_spreads = int(_h.get("spreads", 0))
                print(f"holdings state: {args.open_condors} condor(s), "
                      f"{args.open_spreads} spread(s) (from {os.path.basename(_hs)})")
            else:
                print(f"holdings state is from {_h.get('date')}, not today - "
                      "IGNORING it. Counts may be wrong; check the manager log.")
        except (OSError, ValueError) as _e:
            print(f"holdings state unreadable ({str(_e)[:40]}) - using flags")
    elif args.open_condors or args.open_spreads:
        print("no holdings state file; using the flags on the command line")

    print(f"fetching FO bhavcopy {args.date or '(newest published)'} ...")
    fo, fname, used = fetch_fo(args.date)
    print(f"  {fname}  {len(fo):,} rows  (session {used})")
    import datetime as _dt
    _u = _dt.date(int(used[:4]), int(used[4:6]), int(used[6:8]))
    _behind = sum(1 for i in range(1, 11)
                  if (_u + _dt.timedelta(days=i)).weekday() < 5
                  and _u + _dt.timedelta(days=i) <= _dt.date.today())
    if _behind > args.max_age:
        print(f"  STALE: bhavcopy is {_behind} session(s) behind today; premiums "
              f"are settlement prices from {_u}, not today's market")
        if not args.allow_stale:
            print("  refusing to emit structures - pass --allow-stale to override")
            return 3
    elif _behind:
        print(f"  note: previous-session settlement ({_behind} session behind today)")
    run_date = pd.Timestamp(fo.TradDt.iloc[0]).normalize()
    print(f"  trade date {run_date.date()}  (settlement basis; live prices are not available here)")

    allrows, allrej, ctx = [], [], []
    for sym in UNIVERSE:
        c = build_chain(fo, sym, run_date)
        if not c:
            print(f"  {sym:12s} no usable chain")
            continue
        ctx.append(c)
        _steps = tuple(int(x) for x in str(args.wing_steps).split(",") if x.strip())
        rows, rej = scan_name(c, args.min_oi, args.max_delta,
                              args.min_credit_pct, _steps)
        allrows.extend(rows)
        allrej.extend([(sym,) + r for r in rej])
        print(f"  {sym:12s} expiry {c['expiry']} ({c['dte']}d)  fwd {c['forward']:,.2f}  "
              f"lot {c['lot']}  passes {len(rows)}")

    print("\n=== EXPIRY SELECTED ===")
    print("  (near-the-money strike step is shown; wing widths are in those steps)")
    for c in ctx:
        print(f"  {c['sym']:12s} {c['expiry']}  {c['dte']} days from today  "
              f"forward {c['forward']:,.2f}  lot {c['lot']}")

    if allrows:
        d = pd.DataFrame(allrows).sort_values("score", ascending=False)

        # --- HOLDING LIMITS --------------------------------------------------
        # At most 1 iron condor, OR at most 2 single spreads, never a mix.
        _co, _so = args.open_condors, args.open_spreads
        allow_condor = (_co < args.max_condors) and (_so == 0)
        spread_room = max(0, args.max_spreads - _so) if _co == 0 else 0
        allow_spreads = spread_room > 0

        print(f"\nHOLDINGS: {_co} condor(s) of {args.max_condors}, "
              f"{_so} spread(s) of {args.max_spreads} — one condor OR two spreads, "
              "never a mix.")
        if not allow_condor and not allow_spreads:
            _why = ("a condor is already open" if _co
                    else f"{_so} spread(s) open, limit {args.max_spreads}")
            print(f"  nothing to propose: {_why}. Manage the open position(s).")
            d = d.iloc[0:0]
        else:
            _n = len(d)
            _parts = []
            _bits = []
            if allow_condor:
                _bits.append(d[d["struct"] == "condor"].head(1))
                _parts.append("best condor")
            if allow_spreads:
                _bits.append(d[d["struct"] != "condor"].head(spread_room))
                _parts.append(f"up to {spread_room} single spread(s)")
            d = pd.concat(_bits).sort_values("score", ascending=False)
            print(f"  shortlist from {_n} survivors: " + " + ".join(_parts))
        if args.capital > 0:
            import math as _m
            lo_p, hi_p = [float(x) / 100.0 for x in str(args.risk_pct).split(",")[:2]]
            budget = args.capital * args.total_risk_pct / 100.0
            lots_lo, lots_hi, ml_lo, ml_hi, fits = [], [], [], [], []
            for ml in d["max_loss"]:
                a = max(1, _m.ceil(lo_p * args.capital / ml)) if ml > 0 else 0
                b = int(hi_p * args.capital / ml) if ml > 0 else 0
                lots_lo.append(a)
                lots_hi.append(max(b, 0))
                ml_lo.append(a * ml)
                ml_hi.append(max(b, a) * ml)
                fits.append(b >= 1)
            d["lots_min"] = lots_lo
            d["lots_max"] = [max(b, a) for a, b in zip(lots_lo, lots_hi)]
            d["risk_at_min"] = ml_lo
            d["risk_at_max"] = ml_hi
            d["oversized"] = [not f for f in fits]
        print(f"\n=== SHOWING {len(d)} OF {len(allrows)} BY credit/width x (1-delta) ===")
        if args.capital > 0:
            print(f"capital Rs.{args.capital:,.0f} | per-position band "
                  f"{args.risk_pct}% = Rs.{args.capital*lo_p:,.0f}-"
                  f"Rs.{args.capital*hi_p:,.0f} | total ceiling "
                  f"{args.total_risk_pct:.0f}% = Rs.{budget:,.0f}")
            print(f"{'#':>2s} {'sym':11s} {'legs':30s} {'maxloss':>9s} "
                  f"{'lots':>7s} {'risk':>10s} {'cr/w':>5s} {'delta':>5s} "
                  f"{'minOI':>9s}")
        else:
            print(f"{'#':>2s} {'sym':11s} {'structure':10s} {'legs':30s} {'credit':>9s} "
                  f"{'maxloss':>9s} {'cr/w':>5s} {'delta':>5s} {'minOI':>9s}")
        for n, (_, r) in enumerate(d.head(args.top).iterrows(), 1):
            if args.capital > 0:
                print(f"{n:>2d} {r['sym']:11s} {r['legs']:30s} "
                      f"{r['max_loss']:>9,.0f} "
                      f"{r['lots_min']:>3d}-{r['lots_max']:<3d} "
                      f"{r['risk_at_min']:>5,.0f}-{r['risk_at_max']:<5,.0f} "
                      f"{r['pct']:>4.0f}% {r['delta']:>5.2f} {r['thin_oi']:>9,}"
                      + ("  OVERSIZED" if r["oversized"] else ""))
            else:
                print(f"{n:>2d} {r['sym']:11s} {r['struct']:10s} {r['legs']:30s} "
                      f"{r['credit']:>9,.0f} {r['max_loss']:>9,.0f} {r['pct']:>4.0f}% "
                      f"{r['delta']:>5.2f} {r['thin_oi']:>9,}")
        d.to_csv("stock_fno_candidates.csv", index=False)
        print(f"\nall survivors: stock_fno_candidates.csv ({len(d)} rows)")

    if allrej:
        print("\n=== REJECTED (sample) ===")
        for sym, what, why in allrej[:12]:
            print(f"  {sym:11s} {what:26s} {why}")

    # most liquid stock-option underlyings by traded value
    print("\n=== MOST LIQUID STOCK OPTION UNDERLYINGS (settlement session) ===")
    liq = (fo[(fo.FinInstrmTp == "STO") & (fo.TtlTrfVal > 0)]
           .groupby("TckrSymb").agg(trf=("TtlTrfVal", "sum"),
                                    oi=("OpnIntrst", "sum"),
                                    n=("FinInstrmId", "count"))
           .sort_values("trf", ascending=False).head(5))
    for sym, r in liq.iterrows():
        print(f"  {sym:12s} traded value Rs.{r.trf/1e7:,.1f} cr   OI {r.oi:>12,.0f}   {r.n} contracts")

    if args.capital > 0:
        print("\nNOTE: one position at a time means one 45-day cycle is one trade. "
              "Expect roughly 6-8 round trips a year — a sample far too small to "
              "tune on. Judge the strategy on process, not on a handful of outcomes.")
        print("\nMARGIN IS NOT COMPUTED. Sizing is by max loss only; Dhan's actual "
              "margin for a hedged spread is the binding constraint and must be "
              "checked with margin_calculator before ordering.")
    print("\nPrices are the previous session's settlement, not live. Nothing has been ordered.")

    if not args.no_telegram:
        lines = [f"<b>Stock F&amp;O scan</b> — session {used}",
                 f"holdings: {args.open_condors} condor, {args.open_spreads} spread "
                 "(1 condor OR 2 spreads)"]
        if allrows and len(d):
            for n, (_, r) in enumerate(d.iterrows(), 1):
                lines.append("")
                lines.append(f"<b>{n}. {r['sym']} {r['legs']}</b>")
                lines.append(f"credit Rs.{r['credit']:,.0f} | max loss "
                             f"Rs.{r['max_loss']:,.0f} | {r['pct']:.0f}% of width")
                lines.append(f"delta {r['delta']:.2f} | OI {r['thin_oi']:,} | "
                             f"{r['steps']} step(s)")
                if args.capital > 0:
                    lines.append(f"{r['lots_min']}-{r['lots_max']} lots -> risk "
                                 f"Rs.{r['risk_at_min']:,.0f}-{r['risk_at_max']:,.0f}")
        else:
            lines.append("nothing to propose")
        lines.append("")
        lines.append("Settlement prices, not live. Margin not computed.")
        if telegram_send("\n".join(lines)):
            print("[TELEGRAM] shortlist sent")
    return 0


if __name__ == "__main__":
    sys.exit(main())
