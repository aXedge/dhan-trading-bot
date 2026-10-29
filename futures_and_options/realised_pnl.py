#!/usr/bin/env python3
"""
realised_pnl.py — the authoritative P&L, straight from the broker's fills.

Why this exists
---------------
Two instruments disagree about what the week earned:
  * the guard's audit reads its OWN MTM, which we measured understates
    losses by ~36% (8 Oct episode);
  * fno_trade_analyzer reconstructs "round trips" with a heuristic that
    mis-pairs multi-leg spreads — on 23 Oct it reported -Rs.3,113 for a
    structure whose own legs imply roughly +Rs.6,825.
Neither is arithmetic on the fills. This is: it groups the broker's trade
book by LEG (expiry + strike + option type + security id), FIFO-pairs the
buys against the sells within each leg, and sums the realised P&L. No MTM,
no pairing guesses, nothing to get wrong but the arithmetic.

Read-only. Places no orders. Its own output; touches no bot state.

Usage
-----
  python futures_and_options/realised_pnl.py --from 2026-10-23 --to 2026-10-29
  python futures_and_options/realised_pnl.py --days 7
  python futures_and_options/realised_pnl.py --selftest
"""
import argparse
import os
import sys
import time as _time
from collections import defaultdict, deque
from datetime import date, timedelta

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try:
    from src.auth import get_access_token
except Exception:                                    # offline / selftest
    get_access_token = None

API_BASE = "https://api.dhan.co/v2"
FNO_SEGMENTS = {"NSE_FNO", "BSE_FNO", "MCX_COMM"}


def leg_key(t):
    """One leg = one tradable contract. Two fills of the same leg pair up."""
    return "|".join(str(t.get(k) or "") for k in (
        "drvExpiryDate", "drvStrikePrice", "drvOptionType", "securityId"))


def realised(fills):
    """
    FIFO-pair buys against sells within each leg.
    fills: [{leg, side, qty, price, time}]
    -> (total_pnl, per_leg {leg: pnl}, unmatched_qty)
    """
    lots = defaultdict(deque)          # leg -> deque of [side, qty, price]
    total = 0.0
    per_leg = defaultdict(float)
    for f in sorted(fills, key=lambda x: x["time"]):
        q, px, side = f["qty"], f["price"], f["side"]
        dq = lots[f["leg"]]
        while q > 0 and dq and dq[0][0] != side:
            o_side, o_qty, o_px = dq[0]
            m = min(q, o_qty)
            # resting BUY + new SELL -> (sell - buy); resting SELL + new BUY -> (sell - buy)
            pnl = (px - o_px) * m if o_side == "BUY" else (o_px - px) * m
            total += pnl
            per_leg[f["leg"]] += pnl
            q -= m
            if m == o_qty:
                dq.popleft()
            else:
                dq[0][1] -= m
        if q > 0:
            dq.append([side, q, px])
    unmatched = sum(l[1] for dq in lots.values() for l in dq)
    return total, per_leg, unmatched


def _get_client():
    cid = os.getenv("DHAN_CLIENT_ID")
    if not cid:
        print("ERROR: DHAN_CLIENT_ID not set (.env)")
        sys.exit(1)
    if not get_access_token:
        print("ERROR: src.auth.get_access_token unavailable")
        sys.exit(1)
    token = None
    for attempt in range(3):
        try:
            token = get_access_token()
            break
        except RuntimeError as e:
            if "once every 2 minutes" in str(e) and attempt < 2:
                _time.sleep(120)
            else:
                raise
    import requests
    h = {"access-token": token, "client-id": cid,
         "Content-Type": "application/json", "Accept": "application/json"}
    return cid, h


def fetch_trades(h, frm, to):
    import requests
    out, page = [], 0
    while True:
        r = requests.get(f"{API_BASE}/trades/{frm}/{to}/{page}",
                         headers=h, timeout=30)
        if r.status_code != 200:
            print(f"  API {r.status_code}: {r.text[:160]}")
            break
        resp = r.json()
        if not isinstance(resp, list) or not resp:
            break
        out.extend(resp)
        if len(resp) < 100:
            break
        page += 1
    return out


def main():
    ap = argparse.ArgumentParser(description="Realised F&O P&L from the broker's fills")
    ap.add_argument("--from", dest="frm", default="")
    ap.add_argument("--to", dest="to", default="")
    ap.add_argument("--days", type=int, default=7)
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        sys.exit(selftest())

    to = a.to or date.today().isoformat()
    frm = a.frm or (date.fromisoformat(to) - timedelta(days=a.days)).isoformat()

    cid, h = _get_client()
    raw = fetch_trades(h, frm, to)
    fno = [t for t in raw if t.get("exchangeSegment") in FNO_SEGMENTS]
    print(f"Trades {frm} -> {to}: {len(raw)} total, {len(fno)} F&O")

    fills = []
    for t in fno:
        side = str(t.get("transactionType", "")).upper()
        if side not in ("BUY", "SELL"):
            continue
        fills.append({
            "leg": leg_key(t),
            "side": side,
            "qty": int(t.get("tradedQuantity") or t.get("quantity") or 0),
            "price": float(t.get("tradedPrice") or t.get("price") or 0),
            "time": str(t.get("createTime") or t.get("exchangeTime") or ""),
            "sym": str(t.get("tradingSymbol") or ""),
        })

    total, per_leg, unmatched = realised(fills)
    print()
    print("  leg                                         realised P&L")
    print("  " + "-" * 58)
    for leg, pnl in sorted(per_leg.items(), key=lambda kv: kv[1]):
        label = leg.replace("|", " ")[:44]
        print(f"  {label:44s} Rs{pnl:+12,.2f}")
    print("  " + "-" * 58)
    print(f"  {'TOTAL REALISED':44s} Rs{total:+12,.2f}")
    if unmatched:
        print(f"  (open/unmatched qty: {unmatched} — still-held legs, excluded)")
    print()
    print("  This is arithmetic on the fills. Compare it to the guard's")
    print("  exit_mtm in data/trail_audit.csv — the difference is the MTM error.")
    return 0


def selftest():
    ok = True

    def chk(c, m):
        nonlocal ok
        print(("  PASS  " if c else "  FAIL  ") + m)
        ok = ok and c

    # 23 Oct 12:47 structure, hand-computed: short 22500 +Rs.9,880,
    # long 22900 -Rs.3,055, net +Rs.6,825
    fills = [
        {"leg": "A", "side": "SELL", "qty": 260, "price": 174.375, "time": "1"},
        {"leg": "B", "side": "BUY",  "qty": 260, "price": 40.225,  "time": "1"},
        {"leg": "A", "side": "BUY",  "qty": 260, "price": 136.28,  "time": "2"},
        {"leg": "B", "side": "SELL", "qty": 260, "price": 28.25,   "time": "2"},
    ]
    total, per_leg, unmatched = realised(fills)
    chk(abs(per_leg["A"] - (174.375 - 136.28) * 260) < 0.01, "short leg pairs FIFO (profit)")
    chk(abs(per_leg["B"] - (28.25 - 40.225) * 260) < 0.01, "long leg pairs FIFO (loss)")
    chk(unmatched == 0, "no unmatched quantity when fully closed")
    chk(abs(total - (per_leg["A"] + per_leg["B"])) < 0.01, "total = sum of legs")

    # a partial exit must leave the remainder open, not invent a pairing
    t2, _, un2 = realised([
        {"leg": "C", "side": "BUY", "qty": 130, "price": 100.0, "time": "1"},
        {"leg": "C", "side": "SELL", "qty": 65, "price": 120.0, "time": "2"},
    ])
    chk(abs(t2 - (120.0 - 100.0) * 65) < 0.01, "partial exit realises only the matched qty")
    chk(un2 == 65, "unmatched qty stays open")

    # a multi-leg spread must NOT be cross-paired between legs
    t3, pl3, _ = realised([
        {"leg": "X", "side": "SELL", "qty": 130, "price": 200.0, "time": "1"},
        {"leg": "Y", "side": "BUY",  "qty": 130, "price": 50.0,  "time": "1"},
    ])
    chk(t3 == 0 and len(pl3) == 0, "a leg never pairs against a different leg")

    print()
    print("[SELFTEST] " + ("ALL PASS" if ok else "FAILURES PRESENT"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
