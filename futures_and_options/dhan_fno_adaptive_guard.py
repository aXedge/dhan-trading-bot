#!/usr/bin/env python3
"""
Dhan F&O Adaptive P&L Guard — Self-Learning Edition
====================================================
A self-learning system that progressively identifies the right daily
profit and loss limits for your F&O trading, based on your actual
historical performance.

HOW IT WORKS
------------
1. Fetches your F&O trade history over the last N days (default 90).
2. Groups trades into completed round-trips (FIFO matching).
3. Computes exponentially-weighted moving statistics (EWMA) over the
   most recent ~50 round-trips — recent trades count more, but the
   full history provides stability.
4. Derives profit and loss limits using the combined formula:

      profitLimit = N_trades_per_day * winRate * (avg_profit + 1*std_profit)
      lossLimit   = min(N * (1-W) * (avg_loss + 1*std_loss),  profitLimit / R)

5. Persists the computed statistics and limits to a JSON state file
   so each daily run builds on the previous one and you get a visible
   audit trail of how the limits evolved.
6. Optionally starts the P&L guard polling loop with the computed limits.

USAGE
-----
    # Daily run: compute limits, show recommendation, start guard
    python dhan_fno_adaptive_guard.py

    # Compute only, don't start the guard (review first)
    python dhan_fno_adaptive_guard.py --dry-run

    # Custom parameters
    python dhan_fno_adaptive_guard.py --trades-per-day 3 --half-life 50

    # Override the computed limits manually
    python dhan_fno_adaptive_guard.py --force-profit 13000 --force-loss 8000

    # Show history of computed limits
    python dhan_fno_adaptive_guard.py --history

    # Verbose debug logging
    python dhan_fno_adaptive_guard.py --debug

REQUIREMENTS
------------
    pip install requests pyotp python-dotenv dhanhq

AUTHENTICATION
--------------
    Uses src/auth.py from the dhan-trading-bot repo for automatic token
    generation via PIN + TOTP. No manual DHAN_ACCESS_TOKEN needed.
    Requires DHAN_CLIENT_ID, DHAN_PIN, DHAN_TOTP_SECRET in .env.
    When using --from-file + --dry-run, no authentication is needed.
"""

import argparse
import json
import logging
import math
import os
import statistics
import sys
import time
from collections import defaultdict
from datetime import datetime, timedelta

# Add project root (parent of futures_and_options/) to path so we can import src.auth
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    import requests
except ImportError:
    print("ERROR: requests package not installed. Run: pip install requests")
    sys.exit(1)

try:
    from src.auth import get_access_token
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    # Fallback: allow standalone use with DHAN_ACCESS_TOKEN env var
    get_access_token = None

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
API_BASE = "https://api.dhan.co/v2"

FNO_SEGMENTS = {"NSE_FNO", "BSE_FNO", "MCX_COMM", "NSE_CURRENCY"}
MAX_PAGES = 50

# Default state file — persists between runs, stored alongside the fno scripts
_STATE_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".dhan_adaptive_guard_state.json")
STATE_FILE = os.environ.get("DHAN_STATE_FILE", _STATE_FILE)

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logger = logging.getLogger("dhan_adaptive_guard")


def setup_logging(debug=False):
    level = logging.DEBUG if debug else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    if debug:
        logging.getLogger("urllib3").setLevel(logging.WARNING)
        logging.getLogger("requests").setLevel(logging.WARNING)
    logger.debug("Logging initialised at %s level", logging.getLevelName(level))


# ===========================================================================
# PART 1: DHAN API CLIENT
# ===========================================================================
class DhanClient:
    """Thin wrapper around the DhanHQ v2 REST API."""

    def __init__(self, access_token, client_id):
        self.access_token = access_token
        self.client_id = client_id
        self.base_url = API_BASE
        self.headers = {
            "access-token": access_token,
            "client-id": client_id,
            "Accept": "application/json",
            "Content-Type": "application/json",
        }
        logger.debug("DhanClient initialised for client_id=%s", client_id)

    def _request(self, method, endpoint, params=None, json_body=None):
        url = f"{self.base_url}{endpoint}"
        logger.debug("HTTP %s %s params=%s", method, url, params)
        try:
            r = requests.request(method, url, headers=self.headers,
                                 params=params, json=json_body, timeout=30)
            logger.debug("Response: HTTP %s, %d bytes", r.status_code, len(r.content))
            if r.status_code in (401, 403):
                logger.error("Auth error: HTTP %s", r.status_code)
                print(f"[AUTH ERROR] HTTP {r.status_code}: {r.text.strip()}")
                print("Your access token may have expired. Regenerate it from web.dhan.co.")
                sys.exit(1)
            if r.status_code == 200:
                data = r.json()
                logger.debug("Response body (first 300 chars): %s", json.dumps(data)[:300])
                return data
            else:
                logger.warning("Non-200: HTTP %s — %s", r.status_code, r.text[:200])
                return {"errorCode": f"HTTP-{r.status_code}", "errorMessage": r.text.strip()}
        except requests.exceptions.RequestException as e:
            logger.error("Network error: %s", e)
            return {"errorCode": "NETWORK_ERROR", "errorMessage": str(e)}

    def get_trade_history(self, from_date, to_date, page=0):
        endpoint = f"/trades/{from_date}/{to_date}/{page}"
        return self._request("GET", endpoint)

    def get_fund_limits(self):
        return self._request("GET", "/fundlimit")

    def get_order_list(self):
        return self._request("GET", "/orders")

    def set_pnl_exit(self, profit_value, loss_value, product_types, enable_kill_switch=False):
        body = {
            "dhanClientId": self.client_id,
            "profitValue": str(profit_value),
            "lossValue": str(loss_value),
            "productType": product_types,
            "enableKillSwitch": enable_kill_switch,
        }
        return self._request("POST", "/pnlExit", json_body=body)

    def get_pnl_exit(self):
        return self._request("GET", "/pnlExit")

    def stop_pnl_exit(self):
        return self._request("DELETE", "/pnlExit")


# ===========================================================================
# PART 2: TRADE HISTORY FETCHER
# ===========================================================================
def fetch_fno_trades(client, from_date, to_date):
    """Fetch all F&O trades across all pages for the date range."""
    all_trades = []
    page = 0

    print(f"\n[FETCH] Retrieving F&O trades from {from_date} to {to_date}...")
    logger.info("Fetching F&O trades from %s to %s", from_date, to_date)

    while page < MAX_PAGES:
        resp = client.get_trade_history(from_date, to_date, page)

        if isinstance(resp, list):
            if len(resp) == 0:
                break
            all_trades.extend(resp)
            logger.debug("Page %d: %d trades (total: %d)", page, len(resp), len(all_trades))
            print(f"  Page {page}: {len(resp)} trades (total: {len(all_trades)})")
            page += 1
        elif isinstance(resp, dict) and "errorCode" in resp:
            logger.error("Page %d error: %s", page, resp.get("errorMessage", resp))
            print(f"  [ERROR] Page {page}: {resp.get('errorMessage', resp)}")
            break
        else:
            logger.warning("Page %d: unexpected response", page)
            break

    fno_trades = [t for t in all_trades if t.get("exchangeSegment", "") in FNO_SEGMENTS]
    logger.info("Total trades: %d | F&O trades: %d", len(all_trades), len(fno_trades))
    print(f"[FETCH] Total trades fetched: {len(all_trades)} | F&O trades: {len(fno_trades)}")

    return fno_trades


# ===========================================================================
# PART 3: ROUND-TRIP MATCHING (FIFO)
# ===========================================================================
def compute_round_trips(trades):
    """Match BUY and SELL trades into FIFO round-trips and compute P&L."""
    logger.info("Computing round-trips from %d trades (FIFO)", len(trades))

    by_security = defaultdict(list)
    for t in trades:
        sid = t.get("securityId", "") or t.get("tradingSymbol", "") or t.get("customSymbol", "")
        by_security[sid].append(t)

    round_trips = []

    for sid, sec_trades in by_security.items():
        sec_trades.sort(key=lambda t: t.get("exchangeTime", "") or t.get("updateTime", ""))

        buy_queue = []
        sell_queue = []

        for t in sec_trades:
            txn = t.get("transactionType", "").upper()
            qty = int(t.get("tradedQuantity", 0))
            price = float(t.get("tradedPrice", 0))
            charges = sum(float(t.get(k, 0) or 0) for k in
                          ["brokerageCharges", "stt", "exchangeTransactionCharges",
                           "sebiTax", "serviceTax", "stampDuty"])
            symbol = t.get("tradingSymbol") or t.get("customSymbol") or sid
            time_str = t.get("exchangeTime", "") or t.get("updateTime", "")

            entry = {"qty": qty, "price": price, "charges": charges,
                     "time": time_str, "symbol": symbol}
            if txn == "BUY":
                buy_queue.append(entry)
            elif txn == "SELL":
                sell_queue.append(entry)

        bi, si = 0, 0
        while si < len(sell_queue) and bi < len(buy_queue):
            buy, sell = buy_queue[bi], sell_queue[si]
            matched_qty = min(buy["qty"], sell["qty"])
            if matched_qty == 0:
                bi += (buy["qty"] == 0)
                si += (sell["qty"] == 0)
                continue

            buy_value = matched_qty * buy["price"]
            sell_value = matched_qty * sell["price"]
            gross_pnl = sell_value - buy_value
            buy_charge_share = (matched_qty / buy["qty"]) * buy["charges"] if buy["qty"] else 0
            sell_charge_share = (matched_qty / sell["qty"]) * sell["charges"] if sell["qty"] else 0
            total_charges = buy_charge_share + sell_charge_share
            net_pnl = gross_pnl - total_charges

            round_trips.append({
                "symbol": buy["symbol"],
                "securityId": sid,
                "buyQty": matched_qty,
                "sellQty": matched_qty,
                "buyValue": round(buy_value, 2),
                "sellValue": round(sell_value, 2),
                "grossPnl": round(gross_pnl, 2),
                "totalCharges": round(total_charges, 2),
                "netPnl": round(net_pnl, 2),
                "buyTime": buy["time"],
                "sellTime": sell["time"],
            })

            buy["qty"] -= matched_qty
            sell["qty"] -= matched_qty
            bi += (buy["qty"] == 0)
            si += (sell["qty"] == 0)

    logger.info("Round-trips: %d from %d securities", len(round_trips), len(by_security))
    return round_trips


# ===========================================================================
# PART 4: EWMA-WEIGHTED STATISTICS (THE LEARNING ENGINE)
# ===========================================================================
def ewma_weighted_stats(round_trips, half_life=50):
    """
    Compute exponentially-weighted moving statistics over round-trips.

    Recent trades carry more weight; trades older than ~3 half-lives
    contribute negligibly. This lets the system adapt to regime changes
    in your strategy without being whipsawed by a single bad day.

    Returns a dict with all the stats needed for limit computation.
    """
    if not round_trips:
        return None

    # Sort by sell time (earliest first) so the most recent trades are last
    sorted_trips = sorted(round_trips, key=lambda rt: rt.get("sellTime", "") or rt.get("buyTime", ""))

    # Only use the most recent (half_life * 3) trades for the EWMA window
    # This keeps computation focused while allowing gradual adaptation
    window = min(len(sorted_trips), half_life * 3)
    recent_trips = sorted_trips[-window:]
    n = len(recent_trips)

    # Compute EWMA decay weights: w_i = 0.5^((n-1-i)/half_life)
    # Most recent trade (i=n-1) has weight 1.0, trade half_life ago has weight 0.5
    alpha = math.log(2) / half_life
    weights = [math.exp(-alpha * (n - 1 - i)) for i in range(n)]
    total_weight = sum(weights)
    norm_weights = [w / total_weight for w in weights]

    logger.debug("EWMA window: %d trades, half_life=%d, alpha=%.6f", n, half_life, alpha)
    logger.debug("Weight of most recent trade: %.6f", norm_weights[-1])
    logger.debug("Weight of oldest trade in window: %.6f", norm_weights[0])

    # Separate winners and losers
    winners = [(rt, norm_weights[i]) for i, rt in enumerate(recent_trips) if rt["netPnl"] > 0]
    losers = [(rt, norm_weights[i]) for i, rt in enumerate(recent_trips) if rt["netPnl"] < 0]

    if not winners or not losers:
        logger.warning("Insufficient data: need at least 1 winner and 1 loser")
        return None

    # Weighted win rate
    win_weight = sum(w for _, w in winners)
    loss_weight = sum(w for _, w in losers)
    win_rate = win_weight / (win_weight + loss_weight)

    # Weighted mean profit (winners)
    mu_p = sum(rt["netPnl"] * w for rt, w in winners) / win_weight

    # Weighted std dev of profit (winners)
    var_p = sum(w * (rt["netPnl"] - mu_p) ** 2 for rt, w in winners) / win_weight
    sigma_p = math.sqrt(var_p)

    # Weighted mean loss (losers, as positive number)
    mu_l = sum(abs(rt["netPnl"]) * w for rt, w in losers) / loss_weight

    # Weighted std dev of loss (losers)
    var_l = sum(w * (abs(rt["netPnl"]) - mu_l) ** 2 for rt, w in losers) / loss_weight
    sigma_l = math.sqrt(var_l)

    # Risk:Reward
    risk_reward = mu_p / mu_l if mu_l > 0 else 0

    # Weighted total net P&L (for context)
    total_net_pnl = sum(rt["netPnl"] * w for rt, w in zip(recent_trips, norm_weights))

    stats = {
        "num_trades": n,
        "num_winners": len(winners),
        "num_losers": len(losers),
        "win_rate": round(win_rate, 4),
        "avg_profit": round(mu_p, 2),
        "std_profit": round(sigma_p, 2),
        "avg_loss": round(mu_l, 2),
        "std_loss": round(sigma_l, 2),
        "risk_reward": round(risk_reward, 4),
        "total_net_pnl": round(total_net_pnl, 2),
        "half_life": half_life,
        "window_size": n,
        "computed_at": datetime.now().isoformat(),
    }

    logger.info("EWMA stats: W=%.2f%%, μ_p=₹%.2f, σ_p=₹%.2f, μ_l=₹%.2f, σ_l=₹%.2f, R=%.4f",
                win_rate * 100, mu_p, sigma_p, mu_l, sigma_l, risk_reward)
    logger.debug("Full stats: %s", json.dumps(stats, indent=2))

    return stats


# ===========================================================================
# PART 5: LIMIT COMPUTATION (COMBINED FORMULA)
# ===========================================================================
def compute_limits(stats, n_trades_per_day, sigma_multiplier=1.0):
    """
    Compute profit and loss limits using the combined formula:

      profitLimit = N * W * (μ_p + σ_mult * σ_p)
      lossLimit   = min(N * (1-W) * (μ_l + σ_mult * σ_l),  profitLimit / R)

    The sigma_multiplier defaults to 1.0 (1st std dev) but can be
    adjusted for more/less conservative limits.
    """
    W = stats["win_rate"]
    mu_p = stats["avg_profit"]
    sigma_p = stats["std_profit"]
    mu_l = stats["avg_loss"]
    sigma_l = stats["std_loss"]
    R = stats["risk_reward"]

    N = n_trades_per_day

    # Profit limit: average profit + 1σ, scaled by expected winning trades/day
    profit_per_trade = mu_p + sigma_multiplier * sigma_p
    profit_limit = N * W * profit_per_trade

    # Loss Option A: average loss + 1σ, scaled by expected losing trades/day
    loss_per_trade_A = mu_l + sigma_multiplier * sigma_l
    loss_A = N * (1 - W) * loss_per_trade_A

    # Loss Option B: profit limit divided by risk:reward ratio
    loss_B = profit_limit / R if R > 0 else loss_A

    # Blended: take the tighter (smaller) loss limit
    loss_limit = min(loss_A, loss_B)

    # Which option was chosen
    loss_source = "R:R-derived" if loss_B <= loss_A else "σ-derived"

    # Daily R:R
    daily_rr = profit_limit / loss_limit if loss_limit > 0 else 0

    limits = {
        "profit_limit": round(profit_limit, 2),
        "loss_limit": round(loss_limit, 2),
        "loss_source": loss_source,
        "daily_rr": round(daily_rr, 4),
        "n_trades_per_day": N,
        "sigma_multiplier": sigma_multiplier,
        "profit_per_trade_target": round(profit_per_trade, 2),
        "loss_per_trade_tolerance": round(min(loss_per_trade_A, loss_limit / (N * (1 - W))), 2),
    }

    logger.info("Computed limits: profit=₹%.2f, loss=₹%.2f (%s), daily R:R=%.2f",
                profit_limit, loss_limit, loss_source, daily_rr)
    logger.debug("Limit details: %s", json.dumps(limits, indent=2))

    return limits


# ===========================================================================
# PART 6: STATE PERSISTENCE (THE MEMORY)
# ===========================================================================
def load_state(filepath):
    """Load the persisted state from previous runs."""
    if not os.path.exists(filepath):
        logger.debug("No state file found at %s", filepath)
        return {"history": []}
    try:
        with open(filepath, "r") as f:
            state = json.load(f)
        logger.debug("Loaded state from %s: %d history entries",
                     filepath, len(state.get("history", [])))
        return state
    except Exception as e:
        logger.warning("Could not load state file: %s", e)
        return {"history": []}


def save_state(filepath, state):
    """Persist state to JSON file for the next run."""
    try:
        os.makedirs(os.path.dirname(filepath), exist_ok=True) if os.path.dirname(filepath) else None
        with open(filepath, "w") as f:
            json.dump(state, f, indent=2)
        logger.info("State saved to %s", filepath)
    except Exception as e:
        logger.error("Could not save state file: %s", e)


def append_history(state, stats, limits):
    """Append the current run's stats and limits to the history."""
    entry = {
        "date": datetime.now().strftime("%Y-%m-%d"),
        "timestamp": datetime.now().isoformat(),
        "stats": stats,
        "limits": limits,
    }
    state.setdefault("history", []).append(entry)
    # Keep last 365 entries
    if len(state["history"]) > 365:
        state["history"] = state["history"][-365:]


def print_history(state, n=20):
    """Print the history of computed limits."""
    history = state.get("history", [])
    if not history:
        print("\n[HISTORY] No previous runs found.")
        return

    print(f"\n{'=' * 90}")
    print(f"  ADAPTIVE LIMIT HISTORY (last {min(n, len(history))} runs)")
    print(f"{'=' * 90}")
    print(f"  {'Date':<12} {'Win%':>5} {'μ_p':>8} {'σ_p':>7} {'μ_l':>8} {'σ_l':>7} "
          f"{'R:R':>5} {'Profit':>10} {'Loss':>10} {'Source':<14}")
    print(f"  {'-'*12} {'-'*5} {'-'*8} {'-'*7} {'-'*8} {'-'*7} {'-'*5} {'-'*10} {'-'*10} {'-'*14}")

    for entry in history[-n:]:
        s = entry["stats"]
        l = entry["limits"]
        print(f"  {entry['date']:<12} {s['win_rate']*100:>4.1f}% "
              f"₹{s['avg_profit']:>6,.0f} ₹{s['std_profit']:>5,.0f} "
              f"₹{s['avg_loss']:>6,.0f} ₹{s['std_loss']:>5,.0f} "
              f"{s['risk_reward']:>4.2f} "
              f"₹{l['profit_limit']:>8,.0f} ₹{l['loss_limit']:>8,.0f} "
              f"{l['loss_source']:<14}")

    print(f"{'=' * 90}")

    # Show trend
    if len(history) >= 3:
        recent = history[-3:]
        old_profit = recent[0]["limits"]["profit_limit"]
        new_profit = recent[-1]["limits"]["profit_limit"]
        old_loss = recent[0]["limits"]["loss_limit"]
        new_loss = recent[-1]["limits"]["loss_limit"]

        profit_trend = "↑" if new_profit > old_profit else ("↓" if new_profit < old_profit else "→")
        loss_trend = "↑" if new_loss > old_loss else ("↓" if new_loss < old_loss else "→")

        print(f"\n  TREND (last 3 runs):")
        print(f"    Profit limit: ₹{old_profit:,.0f} → ₹{new_profit:,.0f} {profit_trend}")
        print(f"    Loss limit:   ₹{old_loss:,.0f} → ₹{new_loss:,.0f} {loss_trend}")


# ===========================================================================
# PART 7: P&L GUARD (POLLING LOOP)
# ===========================================================================
_running = True


def _signal_handler(sig, frame):
    global _running
    print("\n\n[STOP] Received interrupt. Shutting down...")
    _running = False


import signal
signal.signal(signal.SIGINT, _signal_handler)
signal.signal(signal.SIGTERM, _signal_handler)


def run_guard(client, profit_limit, loss_limit, poll_interval=15):
    """Poll for F&O orders and set P&L exit when detected."""
    pnl_configured = False
    poll_count = 0

    print(f"\n[POLL] Watching for F&O orders every {poll_interval}s...")
    print("       Press Ctrl+C to stop.\n")

    while _running:
        poll_count += 1
        ts = datetime.now().strftime("%H:%M:%S")

        try:
            order_resp = client.get_order_list()
        except Exception as e:
            logger.error("Error fetching orders: %s", e)
            time.sleep(poll_interval)
            continue

        if isinstance(order_resp, dict) and "errorCode" in order_resp:
            print(f"[{ts}] Order book fetch failed: "
                  f"{order_resp.get('errorMessage', 'unknown')}")
            time.sleep(poll_interval)
            continue

        orders = order_resp if isinstance(order_resp, list) else []
        found = False
        order_info = None

        for order in orders:
            seg = order.get("exchangeSegment", "")
            if seg in FNO_SEGMENTS:
                found = True
                order_info = {
                    "symbol": order.get("tradingSymbol", order.get("customSymbol", "unknown")),
                    "status": order.get("orderStatus", "UNKNOWN"),
                    "transactionType": order.get("transactionType", ""),
                    "exchangeSegment": seg,
                }
                break

        if found and not pnl_configured:
            print(f"[{ts}] *** F&O ORDER DETECTED ***")
            print(f"       Symbol   : {order_info['symbol']}")
            print(f"       Segment  : {order_info['exchangeSegment']}")
            print(f"       Type     : {order_info['transactionType']}")
            print(f"       Status   : {order_info['status']}")

            print(f"\n  {'='*60}")
            print(f"  CONFIGURING P&L-BASED AUTO-EXIT (ADAPTIVE)")
            print(f"  {'='*60}")
            print(f"  Max Profit : ₹{profit_limit:,.2f}")
            print(f"  Max Loss   : ₹{loss_limit:,.2f}")
            print(f"  Kill Switch: DISABLED")
            print(f"  {'='*60}\n")

            resp = client.set_pnl_exit(
                profit_value=profit_limit,
                loss_value=loss_limit,
                product_types=["INTRADAY"],
                enable_kill_switch=False,
            )

            if isinstance(resp, dict) and "errorCode" not in resp:
                pnl_status = resp.get("pnlExitStatus", resp)
                print(f"[SUCCESS] P&L exit configured. Status: {pnl_status}")
                pnl_configured = True
            else:
                print(f"[FAILED] Could not configure P&L exit.")
                print(f"  Response: {json.dumps(resp, indent=2)}")
                print(f"  Will retry on next poll.")
        elif found and pnl_configured:
            print(f"[{ts}] F&O order present ({order_info['symbol']}, "
                  f"{order_info['status']}). P&L exit already configured.")
        else:
            print(f"[{ts}] Poll #{poll_count}: No F&O orders found.")

        slept = 0
        while _running and slept < poll_interval:
            time.sleep(1)
            slept += 1

    print("\n[DONE] Adaptive P&L Guard stopped.")


# ===========================================================================
# PART 8: REPORT DISPLAY
# ===========================================================================
def print_report(stats, limits, previous_limits=None):
    """Print a detailed report of the computed statistics and limits."""
    print(f"\n{'=' * 70}")
    print(f"  ADAPTIVE P&L LIMIT RECOMMENDATION")
    print(f"{'=' * 70}")
    print(f"  Computed at: {stats['computed_at'][:19]}")
    print(f"  EWMA window: {stats['window_size']} trades (half-life: {stats['half_life']})")
    print(f"{'-' * 70}")
    print(f"  LEARNED STATISTICS (EWMA-weighted, last ~{stats['window_size']} trades)")
    print(f"{'-' * 70}")
    print(f"  Win rate              : {stats['win_rate']*100:.1f}%  "
          f"({stats['num_winners']}W / {stats['num_losers']}L)")
    print(f"  Avg profit (winners)  : ₹{stats['avg_profit']:,.2f}")
    print(f"  Std dev (winners)     : ₹{stats['std_profit']:,.2f}")
    print(f"  Avg loss (losers)     : ₹{stats['avg_loss']:,.2f}")
    print(f"  Std dev (losers)      : ₹{stats['std_loss']:,.2f}")
    print(f"  Risk : Reward         : 1 : {stats['risk_reward']:.2f}")
    print(f"  Weighted net P&L      : ₹{stats['total_net_pnl']:,.2f}")
    print(f"{'-' * 70}")
    print(f"  COMPUTED LIMITS (N={limits['n_trades_per_day']} trades/day, "
          f"σ×{limits['sigma_multiplier']})")
    print(f"{'-' * 70}")
    print(f"  Profit target/trade   : ₹{limits['profit_per_trade_target']:,.2f}  "
          f"(μ_p + {limits['sigma_multiplier']}σ_p)")
    print(f"  >>> PROFIT LIMIT      : ₹{limits['profit_limit']:,.2f}")
    print(f"  >>> LOSS LIMIT        : ₹{limits['loss_limit']:,.2f}  "
          f"({limits['loss_source']})")
    print(f"  Daily R:R             : 1 : {limits['daily_rr']:.2f}")
    print(f"{'=' * 70}")

    if previous_limits:
        p = previous_limits["limits"]
        profit_delta = limits["profit_limit"] - p["profit_limit"]
        loss_delta = limits["loss_limit"] - p["loss_limit"]
        print(f"\n  CHANGE FROM LAST RUN ({previous_limits['date']}):")
        print(f"    Profit: ₹{p['profit_limit']:,.2f} → ₹{limits['profit_limit']:,.2f} "
              f"({'+' if profit_delta >= 0 else ''}₹{profit_delta:,.2f})")
        print(f"    Loss:   ₹{p['loss_limit']:,.2f} → ₹{limits['loss_limit']:,.2f} "
              f"({'+' if loss_delta >= 0 else ''}₹{loss_delta:,.2f})")

    print()


# ===========================================================================
# PART 9: MAIN
# ===========================================================================
def main():
    parser = argparse.ArgumentParser(
        description=(
            "Dhan F&O Adaptive P&L Guard — a self-learning system that computes "
            "daily profit/loss limits from your historical trade performance and "
            "optionally starts the P&L guard polling loop.\n\n"
            "The system uses exponentially-weighted moving statistics (EWMA) over "
            "your most recent trades to adapt limits as your strategy evolves.\n\n"
            "Examples:\n"
            "  python dhan_fno_adaptive_guard.py                  # full daily run\n"
            "  python dhan_fno_adaptive_guard.py --dry-run         # compute only\n"
            "  python dhan_fno_adaptive_guard.py --history          # show limit evolution\n"
            "  python dhan_fno_adaptive_guard.py --debug           # verbose logging\n"
            "  python dhan_fno_adaptive_guard.py --force-profit 15000 --force-loss 9000"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--days", type=int, default=90,
        help="Fetch trade history for this many trailing days (default: 90).",
    )
    parser.add_argument(
        "--trades-per-day", type=int, default=3,
        help="Expected number of F&O trades per day for limit scaling (default: 3).",
    )
    parser.add_argument(
        "--half-life", type=int, default=50,
        help="EWMA half-life in trades — recent trades weighted more (default: 50). "
        "Lower = faster adaptation, higher = more stable.",
    )
    parser.add_argument(
        "--sigma", type=float, default=1.0,
        help="Std dev multiplier for limit computation (default: 1.0). "
        "Use 0.5 for tighter limits, 2.0 for wider.",
    )
    parser.add_argument(
        "--state-file", default=STATE_FILE,
        help=f"Path to the state JSON file (default: {STATE_FILE})",
    )
    parser.add_argument(
        "--poll-interval", type=int, default=15,
        help="Guard polling interval in seconds (default: 15).",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Compute and display limits but don't start the guard.",
    )
    parser.add_argument(
        "--from-file", dest="from_file",
        help="Load raw F&O trades from a CSV file saved by dhan_fno_avg_pnl.py "
             "(--data-file) instead of fetching from the API. Use this to avoid "
             "re-fetching the same trade history.",
    )
    parser.add_argument(
        "--history", action="store_true",
        help="Show the history of computed limits and exit.",
    )
    parser.add_argument(
        "--force-profit", type=float,
        help="Override the computed profit limit with this value.",
    )
    parser.add_argument(
        "--force-loss", type=float,
        help="Override the computed loss limit with this value.",
    )
    parser.add_argument(
        "--debug", action="store_true",
        help="Enable DEBUG-level logging.",
    )
    args = parser.parse_args()

    setup_logging(debug=args.debug)
    logger.debug("Arguments: %s", vars(args))

    # --- Load state ---
    state = load_state(args.state_file)

    # --- History mode ---
    if args.history:
        print_history(state)
        sys.exit(0)

    # --- Authenticate (token needed for API calls and guard; not for --from-file + --dry-run) ---
    token = None
    client_id = os.environ.get("DHAN_CLIENT_ID")

    needs_token = not args.from_file or (args.from_file and not args.dry_run)

    if needs_token:
        if not client_id:
            print("ERROR: DHAN_CLIENT_ID not set in .env or environment.")
            sys.exit(1)

        if get_access_token:
            # Use auto token generation from src/auth.py (PIN + TOTP)
            try:
                token = get_access_token()
                logger.info("Access token generated via PIN + TOTP")
            except Exception as e:
                print(f"[ERROR] Failed to generate access token: {e}")
                sys.exit(1)
        else:
            # Fallback: manual token from environment
            token = os.environ.get("DHAN_ACCESS_TOKEN")
            if not token:
                print("ERROR: DHAN_ACCESS_TOKEN not set and src/auth.py not available.")
                print("       Either install with src/auth.py or export DHAN_ACCESS_TOKEN manually.")
                sys.exit(1)

    today = datetime.now().strftime("%Y-%m-%d")
    from_date = (datetime.now() - timedelta(days=args.days)).strftime("%Y-%m-%d")

    print(f"{'=' * 70}")
    print(f"  DHAN F&O ADAPTIVE P&L GUARD")
    print(f"{'=' * 70}")
    if args.from_file:
        print(f"  Mode            : OFFLINE (loading from {args.from_file})")
    else:
        print(f"  Client ID       : {client_id}")
        print(f"  History range   : {from_date} to {today} ({args.days} days)")
    print(f"  EWMA half-life  : {args.half_life} trades")
    print(f"  Trades/day (N)  : {args.trades_per_day}")
    print(f"  Sigma multiplier: {args.sigma}σ")
    print(f"  State file      : {args.state_file}")
    print(f"{'=' * 70}")

    # --- Fetch trades (from API or file) ---
    client = None
    if args.from_file:
        import csv
        if not os.path.exists(args.from_file):
            print(f"\n[ERROR] File not found: {args.from_file}")
            sys.exit(1)
        trades = []
        with open(args.from_file, "r", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                trade = dict(row)
                for nf in ["tradedQuantity"]:
                    if trade.get(nf):
                        try:
                            trade[nf] = int(trade[nf])
                        except (ValueError, TypeError):
                            pass
                for nf in ["tradedPrice", "drvStrikePrice", "sebiTax", "stt",
                           "brokerageCharges", "serviceTax",
                           "exchangeTransactionCharges", "stampDuty"]:
                    if trade.get(nf) and trade[nf] != "NA":
                        try:
                            trade[nf] = float(trade[nf])
                        except (ValueError, TypeError):
                            pass
                trades.append(trade)
        logger.info("Loaded %d trades from %s", len(trades), args.from_file)
        print(f"\n[DATA] Loaded {len(trades)} F&O trades from {args.from_file}")

        # Guard still needs an API client to poll the live order book.
        # Create it here using the environment credentials.
        if not args.dry_run:
            if not token:
                print("\n[ERROR] DHAN_ACCESS_TOKEN is required to run the guard,")
                print("        even when using --from-file for trade history.")
                print("        The guard needs a live API connection to poll orders.")
                sys.exit(1)
            client = DhanClient(token, client_id)
            print("[GUARD] API client created for live order polling.")
    else:
        client = DhanClient(token, client_id)

        # --- Validate token ---
        logger.info("Validating access token")
        resp = client.get_fund_limits()
        if isinstance(resp, dict) and ("dhanClientId" in resp or "availabelBalance" in resp):
            print(f"[OK] Token valid. Available balance: ₹{resp.get('availabelBalance', 'N/A')}")
        else:
            print(f"[ERROR] Token validation failed: {resp}")
            sys.exit(1)

        # --- Fetch trades ---
        trades = fetch_fno_trades(client, from_date, today)

    if not trades:
        print("\n[INFO] No F&O trades found. Cannot compute adaptive limits.")
        sys.exit(0)

    # --- Compute round-trips ---
    print("\n[MATCH] Computing round-trip P&L (FIFO matching)...")
    round_trips = compute_round_trips(trades)
    print(f"[MATCH] {len(round_trips)} completed round-trips identified.")

    if len(round_trips) < 10:
        print(f"\n[WARN] Only {len(round_trips)} round-trips found. "
              "Need at least 10 for meaningful statistics.")
        print("       Limits may be unreliable. Consider increasing --days.")

    # --- Compute EWMA statistics ---
    print(f"\n[LEARN] Computing EWMA-weighted statistics (half-life={args.half_life})...")
    stats = ewma_weighted_stats(round_trips, half_life=args.half_life)

    if not stats:
        print("\n[ERROR] Could not compute statistics. Need both winners and losers.")
        sys.exit(1)

    # --- Compute limits ---
    limits = compute_limits(stats, args.trades_per_day, sigma_multiplier=args.sigma)

    # --- Apply manual overrides ---
    if args.force_profit is not None:
        limits["profit_limit"] = args.force_profit
        limits["profit_per_trade_target"] = args.force_profit / (args.trades_per_day * stats["win_rate"])
        logger.info("Profit limit overridden to ₹%.2f", args.force_profit)

    if args.force_loss is not None:
        limits["loss_limit"] = args.force_loss
        limits["loss_source"] = "manual override"
        logger.info("Loss limit overridden to ₹%.2f", args.force_loss)

    # --- Get previous limits for comparison ---
    previous = state.get("history", [])[-1] if state.get("history") else None

    # --- Print report ---
    print_report(stats, limits, previous)

    # --- Persist state ---
    append_history(state, stats, limits)
    save_state(args.state_file, state)

    # --- Start guard or exit ---
    if args.dry_run:
        print("[DRY RUN] Limits computed and saved. Not starting guard.")
        print(f"  To start: python dhan_fno_adaptive_guard.py")
        print(f"  Or use:  python dhan_fno_pnl_guard.py "
              f"--profit {limits['profit_limit']:.0f} --loss {limits['loss_limit']:.0f}")
    else:
        print(f"\n[GUARD] Starting P&L guard with adaptive limits...")
        print(f"  Profit limit: ₹{limits['profit_limit']:,.2f}")
        print(f"  Loss limit:   ₹{limits['loss_limit']:,.2f}")
        run_guard(
            client,
            profit_limit=limits["profit_limit"],
            loss_limit=limits["loss_limit"],
            poll_interval=args.poll_interval,
        )


if __name__ == "__main__":
    main()
