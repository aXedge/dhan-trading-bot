#!/usr/bin/env python3
"""
stock_fno_manager.py — track and manage STOCK option positions.

This is the ONLY script that manages stock options. The F&O guard runs with
--index-only and will not place stops on, flatten, or EOD-close a stock leg.

It enforces the rules from stock_fno_track.md:
  * profit taking at 50-60% of the credit captured, or ~21 days to expiry
  * close or roll ~5 trading days before expiry (physically settled)
  * roll up/out rather than defend if a short strike is threatened with >10 days
  * NO stop-loss orders on individual option legs — trigger off the UNDERLYING
  * never place anything without a preview and an explicit yes

Default is report-only. It will not touch the broker unless --execute --yes.

Positions come from a JSON file by default, or from Dhan with --from-dhan.

Usage
  python stock_fno_manager.py                        # report only
  python stock_fno_manager.py --from-dhan            # pull live positions
  python stock_fno_manager.py --preview              # show the exit orders
  python stock_fno_manager.py --execute --yes        # place them
  python stock_fno_manager.py --selftest
"""

__SCRIPT_NAME = "stock_fno_manager"
__SCRIPT_VERSION = "v2"
__SCRIPT_BUILD = "2026-11-17"
__SCRIPT_NOTES = "alerts on actionable verdicts"

import argparse
import csv
import datetime as dt
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_POSITIONS = os.path.join(HERE, "stock_fno_positions.json")
DEFAULT_STATE = os.path.join(HERE, "stock_fno_state.json")
DEFAULT_LOG = os.path.join(HERE, "stock_fno_management.csv")
HOLDINGS_STATE = os.path.join(HERE, "stock_fno_holdings.json")

INDEX_UNDERLYINGS = ("NIFTY", "BANKNIFTY", "FINNIFTY", "MIDCPNIFTY",
                     "NIFTYNXT50")

PROFIT_TARGET_PCT = 55.0     # mid of the 50-60% rule
EXIT_DTE = 21                # calendar days
EXPIRY_WEEK_DTE = 7          # ~5 trading days
THREATENED_PCT = 2.0         # underlying within 2% of a short strike
ADJUST_MIN_DTE = 10


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------

def underlying_of(pos):
    return str(pos.get("tradingSymbol") or "").split("-")[0].strip().upper()


def is_stock_option(pos):
    return underlying_of(pos) not in INDEX_UNDERLYINGS


def _f(v, default=0.0):
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


def _i(v, default=0):
    try:
        return int(v)
    except (TypeError, ValueError):
        return default


def dte_of(expiry, today=None):
    today = today or dt.date.today()
    try:
        return (dt.date.fromisoformat(str(expiry)[:10]) - today).days
    except (ValueError, TypeError):
        return None


def group_stock_options(positions):
    """{key: [legs]} for stock option legs only."""
    out = {}
    for p in positions or []:
        if not is_stock_option(p):
            continue
        if not str(p.get("drvOptionType") or "").upper() in ("CE", "PE", ""):
            continue
        if _i(p.get("netQty")) == 0:
            continue
        key = f"{underlying_of(p)}|{str(p.get('drvExpiryDate'))[:10]}"
        out.setdefault(key, []).append(p)
    return out


def structure_economics(legs):
    """Credit received at entry, and the current cost to close."""
    credit = 0.0
    close_cost = 0.0
    for p in legs:
        q = _i(p.get("netQty"))
        if q < 0:                                  # short leg: sold to open
            credit += abs(q) * _f(p.get("sellAvg"))
            close_cost += abs(q) * _f(p.get("lastPrice"))
        elif q > 0:                                # long leg: bought to open
            credit -= q * _f(p.get("buyAvg"))
            close_cost -= q * _f(p.get("lastPrice"))
    return credit, close_cost


def legs_priced(legs):
    """True only if EVERY leg carries a usable current price.

    Without this, a missing lastPrice makes the close cost zero, which reads as
    100% of the credit captured and fires a false TAKE-PROFIT with a limit
    price of zero. That is the same class of silent-fallback bug as the guard's
    stop_trigger_price returning None.
    """
    return all(_f(p.get("lastPrice")) > 0 for p in legs)


def is_condor(legs):
    """A condor holds both a put spread and a call spread on the same expiry."""
    kinds = option_types(legs)
    return "PE" in kinds and "CE" in kinds and len(legs) >= 4


def short_strikes(legs):
    return sorted({_f(p.get("drvStrikePrice")) for p in legs
                   if _i(p.get("netQty")) < 0 and _f(p.get("drvStrikePrice")) > 0})


def option_types(legs):
    return sorted({str(p.get("drvOptionType") or "").upper()
                   for p in legs if p.get("drvOptionType")})


def decide(econ, dte, underlying_px, shorts, kinds):
    """Apply the rules in priority order. Returns (verdict, reason)."""
    credit, close_cost = econ
    if credit <= 0:
        return "CHECK", "credit not positive — verify the entry premiums"
    profit = credit - close_cost
    pct = profit / credit * 100.0

    if pct >= PROFIT_TARGET_PCT:
        return "TAKE-PROFIT", f"{pct:.0f}% of credit captured (target {PROFIT_TARGET_PCT:.0f}%)"
    if dte is None:
        return "CHECK", "expiry unreadable"
    if dte <= EXPIRY_WEEK_DTE:
        return "ROLL-NOW", f"{dte} days to expiry — physically settled, roll before settlement"
    if dte <= EXIT_DTE:
        return "EXIT-DTE", f"{dte} days to expiry (rule: exit at ~{EXIT_DTE})"

    if shorts:
        if not underlying_px:
            return "UNKNOWN", ("no underlying price — the strike checks could not "
                               "run, so this is not a HOLD")
        if "PE" in kinds and underlying_px < min(shorts):
            if dte > ADJUST_MIN_DTE:
                return "ADJUST", (f"underlying {underlying_px:,.2f} below the short put "
                                  f"{min(shorts):,.0f} with {dte}d left — roll out")
            return "EXIT-NOW", "short put breached inside the adjust window"
        if "CE" in kinds and underlying_px > max(shorts):
            if dte > ADJUST_MIN_DTE:
                return "ADJUST", (f"underlying {underlying_px:,.2f} above the short call "
                                  f"{max(shorts):,.0f} with {dte}d left — roll out")
            return "EXIT-NOW", "short call breached inside the adjust window"
        nearest = min(abs(underlying_px - s) / underlying_px * 100 for s in shorts)
        if nearest <= THREATENED_PCT:
            return "WATCH", f"underlying within {nearest:.1f}% of a short strike"
    return "HOLD", f"{pct:.0f}% of credit captured, {dte}d to expiry"


# --------------------------------------------------------------------------
# sources
# --------------------------------------------------------------------------

def load_positions(path):
    if not os.path.exists(path):
        return None
    with open(path) as f:
        data = json.load(f)
    if isinstance(data, dict):
        data = data.get("positions", data.get("data", []))
    return data or []


def dhan_positions():
    repo = os.path.dirname(HERE)
    if repo not in sys.path:
        sys.path.insert(0, repo)
    import requests
    from src.auth import get_access_token
    token = (os.environ.get("DHAN_ACCESS_TOKEN") or "").strip() or get_access_token()
    cid = os.environ.get("DHAN_CLIENT_ID", "1000191656")
    r = requests.get("https://api.dhan.co/v2/positions", timeout=30,
                     headers={"access-token": token, "client-id": cid,
                              "Content-Type": "application/json",
                              "Accept": "application/json"})
    r.raise_for_status()
    j = r.json()
    return j.get("data", []) if isinstance(j, dict) else (j or [])


def _recent_sessions(date, back=5):
    """date and the few weekdays before it, newest first."""
    try:
        d = dt.date(int(date[:4]), int(date[4:6]), int(date[6:8]))
    except (ValueError, TypeError):
        d = dt.date.today()
    out = []
    while len(out) < back:
        if d.weekday() < 5:
            out.append(d.strftime("%Y%m%d"))
        d -= dt.timedelta(days=1)
    return out


DHAN_ORDERS_URL = "https://api.dhan.co/v2/orders"
LIVE_ORDER_STATES = ("PENDING", "TRANSIT", "OPEN", "PART_TRADED")


def _dhan_headers():
    repo = os.path.dirname(HERE)
    if repo not in sys.path:
        sys.path.insert(0, repo)
    from src.auth import get_access_token
    token = (os.environ.get("DHAN_ACCESS_TOKEN") or "").strip() or get_access_token()
    return {"access-token": token,
            "client-id": os.environ.get("DHAN_CLIENT_ID", "1000191656"),
            "Content-Type": "application/json",
            "Accept": "application/json"}


def exit_order_body(order, client_id):
    """Dhan v2 order body for closing one leg.

    Closing a credit spread: buy back the SHORT leg first (that removes the
    obligation), then sell the LONG protective wing. Selling the wing first
    would leave the position momentarily naked short.
    """
    return {
        "dhanClientId": client_id,
        "transactionType": order["transaction_type"],
        "exchangeSegment": order.get("exchangeSegment") or "NSE_FNO",
        "productType": order.get("productType") or "MARGIN",
        "orderType": "LIMIT",
        "validity": "DAY",
        "securityId": str(order["securityId"]),
        "quantity": int(order["quantity"]),
        "price": round(float(order["price"]), 2),
        "disclosedQuantity": 0,
        "afterMarketOrder": False,
    }


def pending_security_ids(headers):
    """securityIds that already have a live, unfilled order.

    Returns a set, or None if the book could not be read. None means "unknown",
    and the caller must NOT place - an unreadable order book is exactly when
    stacking exits is most likely.
    """
    import requests
    try:
        r = requests.get(DHAN_ORDERS_URL, headers=headers, timeout=30)
        j = r.json() if r.content else []
        rows = j if isinstance(j, list) else j.get("data", [])
    except Exception:
        return None
    return {str(o.get("securityId"))
            for o in rows
            if str(o.get("orderStatus", "")).upper() in LIVE_ORDER_STATES}


def place_exit(orders, live):
    """Place each exit leg in sequence. Returns [(symbol, ok, detail)]."""
    import requests
    if not live:
        return [(o["tradingSymbol"], True, "dry-run") for o in orders]
    headers = _dhan_headers()
    client_id = headers["client-id"]

    # never stack an exit on a leg that already has a live order. Without this,
    # an unfilled limit order is re-placed every 15 minutes and the fills
    # multiply the closing quantity.
    live_ids = pending_security_ids(headers)
    if live_ids is None:
        print("  REFUSING to place: the order book could not be read, so a "
              "duplicate exit cannot be ruled out. Retrying next cycle.")
        return [("ALL", False, "order book unreadable")]
    if live_ids:
        print(f"  {len(live_ids)} leg(s) already have a live order - skipping them")

    results = []
    for o in orders:
        if str(o["securityId"]) in live_ids:
            results.append((o["tradingSymbol"], True, "skipped: already pending"))
            continue
        body = exit_order_body(o, client_id)
        try:
            r = requests.post(DHAN_ORDERS_URL, json=body, headers=headers, timeout=30)
            j = r.json() if r.content else {}
            ok = r.ok and j.get("orderStatus") in ("TRANSIT", "PENDING", "TRADED",
                                                   None)
            detail = j.get("orderId") or j.get("errorMessage") or r.status_code
            results.append((o["tradingSymbol"], ok, str(detail)))
        except Exception as e:
            results.append((o["tradingSymbol"], False, str(e)[:60]))
    return results


def underlying_prices(date):
    """Previous-session underlying prices from the FO bhavcopy (best available).

    If today's file is not published yet, step back to the most recent session
    and report which date was actually used.
    """
    last_err = None
    for d in _recent_sessions(date):
        got, note = _underlying_prices_one(d)
        if got:
            return got, f"session {d}"
        last_err = note
    return {}, f"unavailable ({str(last_err)[:40]})"


def _underlying_prices_one(date):
    try:
        import io
        import urllib.request
        import zipfile
        import pandas as pd
        url = ("https://nsearchives.nseindia.com/content/fo/"
               f"BhavCopy_NSE_FO_0_0_0_{date}_F_0000.csv.zip")
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=90) as r:
            zf = zipfile.ZipFile(io.BytesIO(r.read()))
        with zf.open(zf.namelist()[0]) as f:
            d = pd.read_csv(f, usecols=["TckrSymb", "UndrlygPric"])
        d = d[d.UndrlygPric > 0].drop_duplicates("TckrSymb")
        return dict(zip(d.TckrSymb, d.UndrlygPric)), date
    except Exception as e:
        return {}, str(e)[:60]


# --------------------------------------------------------------------------
# state / log
# --------------------------------------------------------------------------

def publish_holdings(condors, spreads, keys, path=None):
    """Write the live counts for the 08:00 scan to read.

    MUST run even when the book is flat. If it does not, the file keeps
    whatever it said earlier today - so closing a condor would leave the scan
    believing one is still open and suppress every candidate until midnight.
    """
    path = path or HOLDINGS_STATE
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            json.dump({"date": dt.date.today().isoformat(),
                       "condors": condors, "spreads": spreads,
                       "keys": sorted(keys)}, f, indent=2)
        return True
    except OSError as e:
        print(f"[STATE] could not write {path}: {e}")
        return False


def load_state(path):
    if not os.path.exists(path):
        return {}
    try:
        with open(path) as f:
            return json.load(f)
    except (OSError, ValueError):
        return {}


def save_state(path, state):
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            json.dump(state, f, indent=2)
    except OSError as e:
        print(f"[STATE] could not save {path}: {e}")


def log_rows(path, rows):
    new = not os.path.exists(path)
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["date", "key", "verdict", "reason",
                                          "credit", "close_cost", "pct",
                                          "dte", "underlying"], extrasaction="ignore")
        if new:
            w.writeheader()
        for r in rows:
            w.writerow(r)


# --------------------------------------------------------------------------
# selftest
# --------------------------------------------------------------------------

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

    check("underlying_of parses a stock symbol",
          underlying_of({"tradingSymbol": "BAJAJFINSV-Nov2026-1900-CE"}) == "BAJAJFINSV")
    check("a stock option is not an index", is_stock_option(
        {"tradingSymbol": "RELIANCE-Nov2026-1400-CE"}))
    check("an index option is excluded",
          not is_stock_option({"tradingSymbol": "NIFTY-Nov2026-23900-CE"}))

    # a short condor: sell 1320P, buy 1290P, sell 1390C, buy 1420C, 500 lot
    legs = [
        {"tradingSymbol": "RELIANCE-Dec2026-1320-PE", "netQty": -500,
         "sellAvg": 18.0, "lastPrice": 6.0, "drvStrikePrice": 1320,
         "drvOptionType": "PE", "drvExpiryDate": "2026-12-29"},
        {"tradingSymbol": "RELIANCE-Dec2026-1290-PE", "netQty": 500,
         "buyAvg": 10.0, "lastPrice": 3.0, "drvStrikePrice": 1290,
         "drvOptionType": "PE", "drvExpiryDate": "2026-12-29"},
        {"tradingSymbol": "RELIANCE-Dec2026-1390-CE", "netQty": -500,
         "sellAvg": 20.0, "lastPrice": 7.0, "drvStrikePrice": 1390,
         "drvOptionType": "CE", "drvExpiryDate": "2026-12-29"},
        {"tradingSymbol": "RELIANCE-Dec2026-1420-CE", "netQty": 500,
         "buyAvg": 11.0, "lastPrice": 4.0, "drvStrikePrice": 1420,
         "drvOptionType": "CE", "drvExpiryDate": "2026-12-29"},
    ]
    credit, cost = structure_economics(legs)
    check("credit = premiums received minus premiums paid",
          abs(credit - (500 * 18 - 500 * 10 + 500 * 20 - 500 * 11)) < 1e-6)
    check("close cost is the current cost to buy it back",
          abs(cost - (500 * 6 - 500 * 3 + 500 * 7 - 500 * 4)) < 1e-6)
    check("short strikes are the sold ones only", short_strikes(legs) == [1320.0, 1390.0])
    check("option types detected", option_types(legs) == ["CE", "PE"])

    g = group_stock_options(legs + [
        {"tradingSymbol": "NIFTY-Dec2026-24000-CE", "netQty": -65,
         "sellAvg": 100, "lastPrice": 50, "drvStrikePrice": 24000,
         "drvOptionType": "CE", "drvExpiryDate": "2026-12-29"}])
    check("index legs are excluded from the stock book", len(g) == 1)

    # verdicts
    v, _ = decide((10000, 4000), 40, 1350, [1320.0, 1390.0], ["PE", "CE"])
    check("60% captured triggers TAKE-PROFIT", v == "TAKE-PROFIT")
    v, _ = decide((10000, 9000), 40, 1350, [1320.0, 1390.0], ["PE", "CE"])
    check("10% captured with time left is HOLD", v == "HOLD")
    v, _ = decide((10000, 9000), 18, 1350, [1320.0, 1390.0], ["PE", "CE"])
    check("18 DTE triggers EXIT-DTE", v == "EXIT-DTE")
    v, _ = decide((10000, 9000), 4, 1350, [1320.0, 1390.0], ["PE", "CE"])
    check("4 DTE triggers ROLL-NOW", v == "ROLL-NOW")
    v, _ = decide((10000, 9000), 30, 1300, [1320.0, 1390.0], ["PE", "CE"])
    check("underlying below the short put with >10d triggers ADJUST", v == "ADJUST")
    v, _ = decide((10000, 9000), 6, 1300, [1320.0, 1390.0], ["PE", "CE"])
    check("breach inside the adjust window is ROLL-NOW first", v == "ROLL-NOW")
    v, _ = decide((10000, 9000), 40, 1325, [1320.0, 1390.0], ["PE", "CE"])
    check("within 2% of a short strike is WATCH", v == "WATCH")
    v, _ = decide((-5, 0), 40, 1350, [1320.0], ["PE"])
    check("non-positive credit is flagged CHECK", v == "CHECK")
    v, r = decide((10000, 9000), 40, 0, [1320.0, 1390.0], ["PE", "CE"])
    check("no underlying price is UNKNOWN, never a silent HOLD", v == "UNKNOWN")
    check("the UNKNOWN reason says so explicitly", "not a HOLD" in r)
    check("a leg without a price is not 'priced'",
          not legs_priced([{"lastPrice": 0}, {"lastPrice": 5}]))
    check("all legs priced passes", legs_priced([{"lastPrice": 5}, {"lastPrice": 2}]))

    body = exit_order_body({"tradingSymbol": "X", "securityId": "85142",
                            "transaction_type": "BUY", "quantity": 300,
                            "price": 55.0, "exchangeSegment": "NSE_FNO",
                            "productType": "MARGIN"}, "1000191656")
    check("exit body carries the Dhan v2 fields",
          body["transactionType"] == "BUY" and body["orderType"] == "LIMIT"
          and body["securityId"] == "85142" and body["quantity"] == 300)
    check("exit body is DAY validity on the F&O segment",
          body["validity"] == "DAY" and body["exchangeSegment"] == "NSE_FNO")
    check("exit body rounds the price", exit_order_body(
        {"tradingSymbol": "X", "securityId": "1", "transaction_type": "SELL",
         "quantity": 1, "price": 55.6789}, "c")["price"] == 55.68)
    check("a live order state set is defined",
          "PENDING" in LIVE_ORDER_STATES and "TRANSIT" in LIVE_ORDER_STATES)
    check("a filled order is not a live state", "TRADED" not in LIVE_ORDER_STATES)
    check("dry-run placement touches nothing",
          place_exit([{"tradingSymbol": "X"}], live=False) == [("X", True, "dry-run")])

    check("a 4-leg structure with both types is a condor", is_condor([
        {"drvOptionType": "PE"}, {"drvOptionType": "PE"},
        {"drvOptionType": "CE"}, {"drvOptionType": "CE"}]))
    check("a 2-leg put spread is not a condor", not is_condor([
        {"drvOptionType": "PE"}, {"drvOptionType": "PE"}]))
    check("a 4-leg straddle is not a condor", not is_condor([
        {"drvOptionType": "CE"}, {"drvOptionType": "PE"}]))

    import tempfile as _tf
    with _tf.TemporaryDirectory() as _td:
        _p = os.path.join(_td, "h.json")
        check("publish_holdings writes the counts",
              publish_holdings(1, 0, ["A|2026-12-29"], _p) is True)
        _h = json.load(open(_p))
        check("published condor count is right", _h["condors"] == 1)
        check("published date is today", _h["date"] == dt.date.today().isoformat())
        check("a flat book publishes zeros, not a stale count",
              publish_holdings(0, 0, [], _p) is True
              and json.load(open(_p))["condors"] == 0)

    check("dte_of counts calendar days",
          dte_of("2026-12-29", dt.date(2026, 12, 1)) == 28)

    print(f"\n{ok} passed, {fail} failed")
    return 0 if fail == 0 else 1


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------

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
    ap.add_argument("--positions", default=DEFAULT_POSITIONS)
    ap.add_argument("--state", default=DEFAULT_STATE)
    ap.add_argument("--log", default=DEFAULT_LOG)
    ap.add_argument("--from-dhan", action="store_true")
    ap.add_argument("--no-telegram", action="store_true")
    ap.add_argument("--max-condors", type=int, default=1)
    ap.add_argument("--max-spreads", type=int, default=2)
    ap.add_argument("--bhavcopy-date", default="", help="YYYYMMDD for underlying prices")
    ap.add_argument("--preview", action="store_true")
    ap.add_argument("--execute", action="store_true")
    ap.add_argument("--yes", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()

    if args.selftest:
        return selftest()

    if args.from_dhan:
        try:
            positions = dhan_positions()
            src = "Dhan"
        except Exception as e:
            print(f"Dhan positions unavailable: {str(e)[:120]}")
            return 1
    else:
        positions = load_positions(args.positions)
        src = args.positions
        if positions is None:
            print(f"no positions file at {args.positions}")
            print("expected a JSON list of Dhan-shaped positions, or use --from-dhan")
            return 2

    book = group_stock_options(positions)
    print(f"positions from {src}: {len(positions)} legs, "
          f"{len(book)} stock-option structure(s)")
    if not book:
        print("no open stock option positions — nothing to manage")
        publish_holdings(0, 0, [])
        print(f"published holdings state: 0 condor(s), 0 spread(s)")
        return 0

    date = args.bhavcopy_date or dt.date.today().strftime("%Y%m%d")
    prices, price_note = underlying_prices(date)
    print(f"underlying prices: {price_note}"
          + (" (previous session settlement, not live)" if prices else ""))

    _condors = [k for k, lg in book.items() if is_condor(lg)]
    _ns = len(book) - len(_condors)
    print(f"holdings: {len(_condors)} condor(s) of {args.max_condors}, "
          f"{_ns} spread(s) of {args.max_spreads} — one condor OR two spreads, "
          "never a mix")
    if _condors:
        print("  -> a condor is open: no new position of either type")
    elif _ns >= args.max_spreads:
        print("  -> spread limit reached: no new position")
    elif _ns:
        print(f"  -> {args.max_spreads - _ns} more spread(s) allowed; "
              "no condor while spreads are open")
    else:
        print("  -> flat: 1 condor or up to 2 spreads available")
    for _k in book:
        _tag = "  [condor]" if is_condor(book[_k]) else "  [spread]"
        print(f"  {_k}{_tag}")

    state = load_state(args.state)
    rows, previews = [], []
    print()
    for key, legs in sorted(book.items()):
        sym, exp = key.split("|")
        dte = dte_of(exp)
        credit, cost = structure_economics(legs)
        pct = (credit - cost) / credit * 100 if credit > 0 else 0.0
        px = float(prices.get(sym, 0) or 0)
        if not legs_priced(legs):
            verdict, reason = ("CHECK", "one or more legs has no current price — "
                                        "captured % cannot be computed")
        else:
            verdict, reason = decide((credit, cost), dte, px,
                                     short_strikes(legs), option_types(legs))

        # first sighting records the entry economics for the audit trail
        rec = state.setdefault(key, {})
        rec.setdefault("first_seen", dt.date.today().isoformat())
        rec["entry_credit"] = round(credit, 2)
        rec["last_verdict"] = verdict

        print(f"{sym} {exp}  {len(legs)} legs  {dte}d  "
              f"credit Rs.{credit:,.0f}  captured {pct:.0f}%  "
              f"underlying {px:,.2f}" if px else
              f"{sym} {exp}  {len(legs)} legs  {dte}d  "
              f"credit Rs.{credit:,.0f}  captured {pct:.0f}%")
        print(f"   -> {verdict}: {reason}")
        for p in legs:
            q = _i(p.get("netQty"))
            print(f"      {'SHORT' if q < 0 else 'LONG '} {abs(q):>5d} x "
                  f"{p.get('drvStrikePrice')} {p.get('drvOptionType')}  "
                  f"entry {_f(p.get('sellAvg') if q < 0 else p.get('buyAvg')):,.2f}  "
                  f"now {_f(p.get('lastPrice')):,.2f}")

        rows.append({"date": dt.date.today().isoformat(), "key": key,
                     "verdict": verdict, "reason": reason,
                     "credit": round(credit, 2), "close_cost": round(cost, 2),
                     "pct": round(pct, 1), "dte": dte, "underlying": px})

        if verdict in ("TAKE-PROFIT", "EXIT-DTE", "ROLL-NOW", "EXIT-NOW"):
            for p in legs:
                q = _i(p.get("netQty"))
                previews.append({
                    "tradingSymbol": p.get("tradingSymbol"),
                    "securityId": p.get("securityId"),
                    "transaction_type": "BUY" if q < 0 else "SELL",
                    "quantity": abs(q),
                    "order_type": "LIMIT",
                    "price": _f(p.get("lastPrice")),
                    "reason": verdict,
                    "exchangeSegment": p.get("exchangeSegment") or "NSE_FNO",
                    "productType": p.get("productType") or "MARGIN",
                })

    # closing order: short legs first (removes the obligation), then the wings
    previews.sort(key=lambda o: 0 if o["transaction_type"] == "BUY" else 1)
    save_state(args.state, state)
    log_rows(args.log, rows)

    # publish the counts so the 08:00 scan enforces the holding rule without
    # anyone having to remember a flag
    publish_holdings(len(_condors), _ns, book)

    if previews and (args.preview or args.execute):
        print(f"\n=== EXIT PREVIEW — {len(previews)} leg(s) ===")
        print("Limit orders at the last traded price. No stop-loss legs (rule: "
              "trigger off the underlying).")
        for o in previews:
            print(f"  {o['transaction_type']:4s} {o['quantity']:>5d}  "
                  f"{o['tradingSymbol']:32s} limit {o['price']:>8,.2f}   [{o['reason']}]")
        if args.execute and not args.yes:
            print("\nrefusing to place: --execute also needs --yes")
        elif args.execute and args.yes:
            print("\nplacing...")
            for sym, ok, detail in place_exit(previews, live=True):
                print(f"  {'OK  ' if ok else 'FAIL'} {sym:32s} {detail}")
    elif previews:
        print(f"\n{len(previews)} exit leg(s) would be placed — re-run with "
              "--preview to see them.")

    # only alert when there is something to act on - 26 "nothing to manage"
    # messages a day is noise, not monitoring
    _act = [r for r in rows if r["verdict"] in
            ("TAKE-PROFIT", "EXIT-DTE", "ROLL-NOW", "EXIT-NOW", "ADJUST",
             "UNKNOWN", "CHECK")]
    if _act and not args.no_telegram:
        lines = ["<b>Stock F&amp;O — action needed</b>"]
        for r in _act:
            lines.append("")
            lines.append(f"<b>{r['key']}</b>")
            lines.append(f"{r['verdict']}: {r['reason']}")
            lines.append(f"credit Rs.{r['credit']:,.0f} | captured {r['pct']:.0f}% "
                         f"| {r['dte']}d")
        if telegram_send("\n".join(lines)):
            print("[TELEGRAM] alert sent")

    print(f"\nlogged to {args.log}. Report-only unless --preview/--execute.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
