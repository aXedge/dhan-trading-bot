# Dhan F&O P&L Guard Suite

A set of Python scripts for managing F&O trade risk on Dhan. Fetch trade history, analyse performance, compute adaptive profit/loss limits using statistical models, and automatically set daily P&L exit triggers.

## Scripts

| Script | Path | Purpose |
|---|---|---|
| `dhan_fno_avg_pnl.py` | `futures_and_options/` | Fetch F&O trade history, compute round-trip P&L, report averages |
| `dhan_fno_adaptive_guard.py` | `futures_and_options/` | Self-learning system: compute EWMA-weighted limits and start the guard |
| `dhan_fno_pnl_guard.py` | `futures_and_options/` | Lightweight polling guard: set P&L exit when an F&O order is detected |

## Prerequisites

```bash
pip install -r requirements.txt
```

### Authentication

The scripts use `src/auth.py` from the dhan-trading-bot repo for automatic token generation via Dhan's PIN + TOTP flow. No manual token refresh needed.

Configure your credentials in a `.env` file (copy from `.env.example`):

```bash
DHAN_CLIENT_ID=your_client_id
DHAN_PIN=your_dhan_pin
DHAN_TOTP_SECRET=your_totp_base32_secret
```

The access token is auto-generated at startup via `DhanLogin.generate_token(pin, totp)` and cached in memory for the session. It expires around midnight IST and is refreshed automatically on the next run.

**Fallback**: If `src/auth.py` is not available (e.g. running standalone), the scripts fall back to reading `DHAN_ACCESS_TOKEN` from the environment. Export it manually:
```bash
export DHAN_ACCESS_TOKEN="your-access-token"
```

**IP Whitelisting**: The P&L exit API (`/v2/pnlExit`) and order APIs require your public IP to be whitelisted in Dhan's API settings. The trade history API (`/v2/trades`) and fund limits API (`/v2/fundlimit`) do not require IP whitelisting.

---

## 1. dhan_fno_avg_pnl.py — Trade History Analyser

Fetches your F&O trade history, matches buys and sells into completed round-trips using FIFO, and computes per-trade P&L after all charges (brokerage, STT, exchange charges, SEBI tax, GST, stamp duty).

### Usage

```bash
# Default: last 90 days, fetch from API
python dhan_fno_avg_pnl.py

# Last 30 days
python dhan_fno_avg_pnl.py --days 30

# Save raw trades to CSV for reuse by other scripts
python dhan_fno_avg_pnl.py --data-file fno_trades.csv

# Load from previously saved CSV (no API call, no token needed)
python dhan_fno_avg_pnl.py --from-file fno_trades.csv

# Export computed round-trip results to CSV
python dhan_fno_avg_pnl.py --csv fno_pnl_report.csv

# Verbose debug logging
python dhan_fno_avg_pnl.py --debug
```

### Flags

| Flag | Description |
|---|---|
| `--days N` | Trailing days of history to fetch (default: 90) |
| `--data-file PATH` | Save raw F&O trades to this CSV file for reuse |
| `--from-file PATH` | Load trades from a saved CSV instead of calling the API |
| `--csv PATH` | Export computed round-trip P&L results to CSV |
| `--debug` | Enable DEBUG-level logging |

### Output

The script prints a performance summary including total trades, win rate, average profit per winning trade, average loss per losing trade, risk:reward ratio, total gross/net P&L, and a per-symbol breakdown. With `--debug`, it also logs every API request, trade matching step, and charge breakdown.

### CSV Format (`--data-file`)

The raw trades CSV contains all 28 fields returned by Dhan's trade history API:

```
dhanClientId, orderId, exchangeOrderId, exchangeTradeId, transactionType,
exchangeSegment, productType, orderType, tradingSymbol, customSymbol,
securityId, tradedQuantity, tradedPrice, isin, instrument, sebiTax, stt,
brokerageCharges, serviceTax, exchangeTransactionCharges, stampDuty,
createTime, updateTime, exchangeTime, drvExpiryDate, drvOptionType, drvStrikePrice
```

This file can be loaded by `dhan_fno_adaptive_guard.py --from-file` to avoid re-fetching.

---

## 2. dhan_fno_adaptive_guard.py — Self-Learning P&L Guard

The core of the system. Fetches trade history, computes exponentially-weighted moving statistics (EWMA) over your most recent trades, derives profit and loss limits using a combined statistical formula, persists state between runs, and optionally starts the P&L guard polling loop.

### Usage

```bash
# Daily run: compute limits + start guard
python dhan_fno_adaptive_guard.py

# Compute only, review before starting
python dhan_fno_adaptive_guard.py --dry-run

# Load from saved CSV instead of API
python dhan_fno_adaptive_guard.py --from-file fno_trades.csv

# Show history of computed limits over time
python dhan_fno_adaptive_guard.py --history

# Override computed limits manually
python dhan_fno_adaptive_guard.py --force-profit 15000 --force-loss 9000

# Custom parameters
python dhan_fno_adaptive_guard.py --trades-per-day 4 --half-life 50 --sigma 1.0
```

### Flags

| Flag | Default | Description |
|---|---|---|
| `--days N` | 90 | Trailing days of history to fetch |
| `--trades-per-day N` | 3 | Expected F&O trades per day for limit scaling |
| `--half-life N` | 50 | EWMA half-life in trades. Lower = faster adaptation |
| `--sigma X` | 1.0 | Std dev multiplier (0.5 = tighter, 2.0 = wider) |
| `--state-file PATH` | `~/.dhan_adaptive_guard_state.json` | Persistence file |
| `--poll-interval S` | 15 | Guard polling interval in seconds |
| `--from-file PATH` | — | Load trades from saved CSV |
| `--dry-run` | — | Compute limits but don't start guard |
| `--history` | — | Show limit evolution over time and exit |
| `--force-profit ₹` | — | Override computed profit limit |
| `--force-loss ₹` | — | Override computed loss limit |
| `--debug` | — | Enable DEBUG-level logging |

### The Learning Engine (EWMA)

The script uses exponentially-weighted moving statistics with a configurable half-life (default: 50 trades). Recent trades carry more weight than older ones:

- A trade from 50 trades ago has 50% the weight of the most recent trade
- A trade from 100 trades ago has 25% the weight
- A trade from 150 trades ago has 12.5% the weight (negligible)

This allows the limits to adapt as your strategy performance changes, without being whipsawed by a single bad day.

### The Limit Formula

**Profit limit** — average profit + 1 standard deviation, scaled by expected winning trades per day:

```
profitLimit = N × W × (μ_p + σ × σ_p)
```

**Loss limit** — the tighter of two approaches:

```
Option A (σ-derived):  N × (1-W) × (μ_l + σ × σ_l)
Option B (R:R-derived): profitLimit / R

lossLimit = min(Option A, Option B)
```

Where:
- `N` = trades per day (`--trades-per-day`)
- `W` = EWMA-weighted win rate
- `μ_p` = EWMA-weighted mean profit of winning trades
- `σ_p` = EWMA-weighted std dev of winning trades
- `μ_l` = EWMA-weighted mean loss of losing trades (positive)
- `σ_l` = EWMA-weighted std dev of losing trades
- `R` = risk:reward ratio = μ_p / μ_l
- `σ` = sigma multiplier (`--sigma`)

### State Persistence

Every run saves to `~/.dhan_adaptive_guard_state.json`, recording:
- The computed statistics (win rate, averages, std devs, R:R)
- The derived limits (profit, loss, source, daily R:R)
- A timestamp

View the full history with `--history` to see how limits have evolved over days and weeks.

### How Limits Auto-Tune

| Scenario | Effect on profit limit | Effect on loss limit |
|---|---|---|
| Win rate improves | Increases (more winners expected) | Decreases (fewer losers expected) |
| Win rate drops | Decreases | Increases |
| Average profit rises | Increases | Increases (via R:R) |
| Loss variance rises | Unchanged | Option A widens, may switch to Option B |
| Strategy degrades (R drops) | Decreases | R:R-derived option tightens |

---

## 3. dhan_fno_pnl_guard.py — Simple Polling Guard

A lightweight script that polls the Dhan order book and sets a P&L exit when the first F&O order is detected. Use this when you already know your limits and just want the guard running.

### Usage

```bash
# Set ₹12,000 profit / ₹8,000 loss, poll every 10s
python dhan_fno_pnl_guard.py --profit 12000 --loss 8000 --interval 10

# Cover both INTRADAY and DELIVERY products
python dhan_fno_pnl_guard.py --profit 15000 --loss 9000 --products INTRADAY DELIVERY
```

### Flags

| Flag | Default | Description |
|---|---|---|
| `--profit ₹` | required | Max profit at which positions auto-exit |
| `--loss ₹` | required | Max loss at which positions auto-exit |
| `--interval S` | 15 | Polling interval in seconds |
| `--products` | INTRADAY | Product types: INTRADAY, DELIVERY, or both |

The kill switch is never activated — only the P&L-based auto-exit.

---

## Typical Daily Workflow

```bash
# 1. Set up environment (once)
cp .env.example .env
# Edit .env with DHAN_CLIENT_ID, DHAN_PIN, DHAN_TOTP_SECRET
pip install -r requirements.txt

# 2. Fetch trade history and save for reuse
python futures_and_options/dhan_fno_avg_pnl.py --days 90 --data-file futures_and_options/fno_trades.csv

# 3. Compute adaptive limits (reads from saved CSV, no API call)
python futures_and_options/dhan_fno_adaptive_guard.py --from-file futures_and_options/fno_trades.csv --dry-run

# 4. Review the output, then start the guard
python futures_and_options/dhan_fno_adaptive_guard.py --from-file futures_and_options/fno_trades.csv

# Or use the simple guard with manually chosen limits
python futures_and_options/dhan_fno_pnl_guard.py --profit 12000 --loss 8000
```

## File Overview

```
dhan-trading-bot/
├── src/
│   ├── auth.py                # Dhan OAuth + TOTP token generation (shared)
│   └── utils.py               # Config loading, logging, file I/O
├── futures_and_options/                       # F&O P&L guard suite
│   ├── dhan_fno_avg_pnl.py          # Trade history fetcher + analyser
│   ├── dhan_fno_adaptive_guard.py   # Self-learning limit computer + guard
│   ├── dhan_fno_pnl_guard.py        # Simple polling guard
│   ├── dhan_daily_runner.sh          # Daily automation wrapper for VM
│   ├── fno_trades.csv               # Saved raw trades (when using --data-file)
│   └── .dhan_adaptive_guard_state.json  # Persisted state (auto-created)
├── dhan-trading.service       # systemd service file
├── requirements.txt           # pip install -r requirements.txt
├── README.md
├── AGENT.md
└── .env                       # Credentials (gitignored)
```

## Important Notes

- The Dhan access token expires every 24 hours. Regenerate it from web.dhan.co.
- The P&L exit API requires IP whitelisting. If you get `DH-905: Invalid IP`, add your public IP to Dhan's API settings.
- The trade history API does not require IP whitelisting — it works from any network.
- P&L exit limits are absolute rupee amounts, not percentages.
- If your current P&L is already beyond the set limit when configured, the exit triggers immediately.
- The P&L exit resets at the end of each trading day.
- Lot sizes and freeze quantities for F&O change — always validate before placing orders.
