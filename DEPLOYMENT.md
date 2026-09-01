# Dhan Swing & Positional Equity Trading Bot — Full Build & Deploy

Build a complete automated swing/positional equity trading bot for the Indian stock market using the Dhan broker API, and deploy it on Google Cloud Platform.

## Architecture: 3-Layer System

Layer 1 (Fundamental Screener) — runs on laptop:
- Scans Nifty Midcap 150 + Nifty 100 universe (~250 stocks)
- Computes Piotroski F-score (0-9) for each stock using openscreener library
- Applies hard filters: min F-score 7, min ROCE 18%, max debt/equity 0.5, min promoter holding 50%, max pledged 5%
- Computes composite score (weighted: Piotroski 25%, ROCE 20%, D/E 15%, revenue growth 20%, promoter 10%, pledge 10%)
- Outputs top 15 stocks to data/basket.json
- Uses openscreener library (which scrapes screener.in via Playwright)
- Runs weekly (Saturday 10 AM)

Layer 2 (Technical Signal Generator) — runs on VPS:
- Reads basket.json for stock universe
- Fetches 1-year price history via yfinance
- Computes EMA (10, 20, 50, 200), RSI (14), volume average (20-day)
- Swing entry: price above EMA10, volume > 1.5x avg, RSI 40-65, breakout above 20-day high
- Swing exit: price below EMA10, or RSI > 70, or stop-loss 2%, or target 6%
- Positional entry: EMA50 crosses above EMA200 (golden cross), volume confirmation
- Positional exit: EMA50 crosses below EMA200, or stop-loss 5%, or target 15%
- Outputs signals to data/signals.json
- Runs daily at 9:20 AM IST (Mon-Fri)

Layer 3 (Dhan Execution) — runs on VPS:
- Reads signals.json from Layer 2
- Places CNC (delivery) orders via DhanHQ Python API
- MARKET order type for entries
- Max 8 concurrent positions, ₹25,000 per position
- PAPER mode (logs without placing real orders) or LIVE mode
- Tracks positions in data/positions.json
- Runs daily at 9:25 AM IST

Layer 3b (Stop-Loss Monitor) — runs on VPS:
- Checks open positions every 5 minutes during market hours (9:15-15:30)
- Exits if stop-loss or target is hit
- Sends Telegram alert on every action
- Runs every 5 min during market hours

## Authentication

- Uses DhanLogin with PIN + TOTP flow (NOT OAuth — the library API changed)
- `DhanLogin(client_id)` then `login.generate_token(pin, totp)`
- Token is valid for ~24 hours, refreshed daily by cron at 8:45 AM
- TOTP generated via `pyotp` library
- Rate limited: token can only be generated once every 2 minutes

## openscreener Field Names (CRITICAL — these are the actual field names)

- **P&L:** `net_profit`, `sales`, `operating_profit`, `operating_margin_percent`, `interest`, `depreciation`, `profit_before_tax`, `tax_percent`, `eps`, `dividend_payout`, `other_income`, `expenses`
- **Balance Sheet:** `equity_capital`, `reserves`, `borrowings`, `other_liabilities`, `total_liabilities`, `fixed_assets`, `capital_work_in_progress`, `investments`, `other_assets`, `total_assets`
- **Cash Flow:** `operating_cash_flow`, `investing_cash_flow`, `financing_cash_flow`, `net_cash_flow`, `free_cash_flow`, `cfo_op`
- **Shareholding:** `date`, `promoters`, `fiis`, `diis`, `government`, `public`, `number_of_shareholders`
- **Ratios:** `roce_percent`, `year`, `debtor_days`, `inventory_days`, `days_payable`, `cash_conversion_cycle`, `working_capital_days`
- **Summary:** `company_name`, `current_price`, `nse_symbol`, `bse_code`, `ratios` (dict containing `market_cap`)
- **Index.constituents()** returns a dict with key `companies` (list of dicts with `symbol` key)

## Piotroski F-score Adaptations

Because openscreener doesn't provide all traditional Piotroski fields:

1. **Positive net income** → use `pl[-1]['net_profit']`
2. **Positive CFO** → use `cf[-1]['operating_cash_flow']`
3. **CFO > Net Income** → earnings quality check
4. **ROCE improving** → use `ratios['roce_percent']` for current; compute previous from `pl[-2]` operating_profit / (equity + reserves + borrowings)
5. **D/E decreasing** → `borrowings / (equity_capital + reserves)`, compare current vs previous
6. **Current ratio > 1** → REPLACED with positive free cash flow (openscreener has no current assets/liabilities)
7. **No share dilution** → `equity_capital` same or lower vs previous year
8. **OPM expanding** → `pl[-1]['operating_margin_percent']` vs `pl[-2]`
9. **Higher revenue** → `pl[-1]['sales']` vs `pl[-2]['sales']`
- **Pledged %** not available via openscreener → set to 0 (best case)

## Dhan API (dhanhq library v2)

- `dhanhq.dhanhq(dhan_context)` — takes DhanContext, not individual credentials
- `DhanContext(client_id, access_token)`
- Key methods: `get_fund_limits()`, `get_positions()`, `get_holdings()`, `place_order()`, `cancel_order()`, `modify_order()`
- Constants: `dhanhq.CNC`, `dhanhq.MARKET`, `dhanhq.NSE`, `dhanhq.BUY`, `dhanhq.SELL`, `dhanhq.DAY`
- IP whitelisting required for order APIs — set via web.dhan.co or API (`POST /v2/ip/setIP`)
- IP locked for 7 days once set — cannot modify for a week
- Error `DH-905` = Invalid IP (not whitelisted or mismatched)

## Infrastructure: GCP

- **Provider:** Google Cloud Platform (chosen over AWS, Utho, E2E Networks, CtrlS)
- **Region:** asia-south1 (Mumbai) — lowest latency to Dhan's Mumbai servers
- **VM:** e2-micro (2 shared vCPU, 1 GB RAM), ~$12/month total
- **OS:** Ubuntu 24.04 LTS (NOT 26.04 — Python 3.14 breaks numba/pandas-ta)
- **Boot disk:** 30 GB standard persistent disk
- **Static external IP:** reserved and attached (required by SEBI for algo trading)
- **Free tier:** $300 credit for 90 days (new accounts)
- **Security:** UFW firewall (SSH in, HTTPS/HTTP out), fail2ban, key-based SSH only
- **Timezone:** Asia/Kolkata (IST)
- **VM name:** `dhan-bot-prod`

## Project Structure

```
dhan-trading-bot/
├── src/
│   ├── __init__.py
│   ├── auth.py          # Dhan PIN+TOTP auth, token caching, get_dhan()
│   ├── screener.py      # Layer 1: Piotroski F-score, openscreener, basket.json
│   ├── technicals.py    # Layer 2: EMA/RSI/volume, swing+positional signals
│   ├── executor.py      # Layer 3: Dhan order placement, position management
│   ├── sl_monitor.py     # Layer 3b: stop-loss monitoring every 5 min
│   ├── eod_report.py     # End-of-day summary report
│   └── utils.py         # Config loading, logging, Telegram, file I/O
├── config/
│   └── settings.yaml     # All thresholds, weights, schedules
├── tests/
│   ├── test_screener.py  # Piotroski, CAGR, safe_div tests
│   └── test_technicals.py # Indicators, RSI range, entry/exit tests
├── data/
│   ├── basket.json       # Screener output (14 stocks)
│   ├── signals.json      # Technical signals
│   └── positions.json    # Open positions tracker
├── logs/                 # Cron log files
├── .env                  # Credentials (NEVER commit to git)
├── .gitignore
├── requirements.txt
├── README.md
└── LICENSE (MIT)
```

## .env File

```env
DHAN_CLIENT_ID=your_numeric_client_id
DHAN_API_KEY=your_api_key
DHAN_API_SECRET=your_api_secret
DHAN_TOTP_SECRET=your_base32_totp_secret
DHAN_PIN=your_4_or_6_digit_pin
DHAN_REDIRECT_URL=http://127.0.0.1:5000/dhan/callback

TELEGRAM_BOT_TOKEN=your_bot_token
TELEGRAM_CHAT_ID=your_chat_id

TRADING_MODE=PAPER
MAX_POSITIONS=8
CAPITAL_PER_POSITION=25000
```

## requirements.txt

```
dhanhq>=2.0.0
pyyaml>=6.0
python-dotenv>=1.0.0
requests>=2.31.0
yfinance>=0.2.0
pandas>=2.0.0
numpy>=1.24.0
pyotp>=2.9.0
python-telegram-bot>=20.0
pandas-ta>=0.3.14b
openscreener>=0.1.0
playwright>=1.40.0
pytest>=8.0.0
pytest-asyncio>=0.23.0
```

## config/settings.yaml

- **fundamental:** universe_indices (CNXMIDCAP, CNX100), min_f_score 7, min_roce 18, max_debt_equity 0.5, min_promoter_holding 50, max_pledged 5, weights, basket_size 15, request_delay 2
- **technical:** swing (lookback 20, volume 1.5x, RSI 40-65, SL 2%, target 6%), positional (EMA 50/200, SL 5%, target 15%), RSI period 14, volume avg 20, price history 365 days
- **execution:** max_positions 8, capital_per_stock 25000, CNC, MARKET, SL check every 300s, market hours 09:15-15:30
- **dhan:** base_url https://api.dhan.co/v2, exchange_segment NSE_EQ

## Cron Schedule (IST, Mon-Fri)

| Time | Job | Command |
|------|-----|---------|
| 8:45 AM | Auth token refresh | `python -m src.auth` |
| 9:20 AM | Layer 2 technical scan | `python -m src.technicals` |
| 9:25 AM | Layer 3 order execution | `python -m src.executor` |
| Every 5 min (9:15-15:30) | SL monitor | `python -m src.sl_monitor` |
| 3:35 PM | EOD report | `python -m src.eod_report` |
| Saturday 10 AM | Layer 1 screener | `python -m src.screener` (on laptop) |

### Crontab Entries

```bash
# Set PYTHONPATH in each cron entry — cron doesn't read .bashrc

# Daily auth token refresh at 8:45 AM IST
45 8 * * 1-5 cd /home/$USER/dhan-trading-bot && source venv/bin/activate && export PYTHONPATH=/home/$USER/dhan-trading-bot/src:$PYTHONPATH && python -m src.auth >> logs/auth.log 2>&1

# Layer 2: Technical scan at 9:20 AM IST
20 9 * * 1-5 cd /home/$USER/dhan-trading-bot && source venv/bin/activate && export PYTHONPATH=/home/$USER/dhan-trading-bot/src:$PYTHONPATH && python -m src.technicals >> logs/technicals.log 2>&1

# Layer 3: Order execution at 9:25 AM IST
25 9 * * 1-5 cd /home/$USER/dhan-trading-bot && source venv/bin/activate && export PYTHONPATH=/home/$USER/dhan-trading-bot/src:$PYTHONPATH && python -m src.executor >> logs/executor.log 2>&1

# Layer 3b: Stop-loss monitor every 5 min during market hours
*/5 9-15 * * 1-5 cd /home/$USER/dhan-trading-bot && source venv/bin/activate && export PYTHONPATH=/home/$USER/dhan-trading-bot/src:$PYTHONPATH && python -m src.sl_monitor >> logs/sl_monitor.log 2>&1

# EOD report at 3:35 PM IST
35 15 * * 1-5 cd /home/$USER/dhan-trading-bot && source venv/bin/activate && export PYTHONPATH=/home/$USER/dhan-trading-bot/src:$PYTHONPATH && python -m src.eod_report >> logs/eod.log 2>&1
```

## Important Gotchas

1. **PYTHONPATH** must include `src/` directory: `export PYTHONPATH=~/dhan-trading-bot/src:$PYTHONPATH`. Add to `.bashrc` AND to every crontab entry — cron does not read `.bashrc`.
2. **Ubuntu 26.04** ships Python 3.14 which breaks `numba` (used by `pandas-ta`) — use **Ubuntu 24.04** with Python 3.12.
3. `python3-venv` package must be installed separately: `sudo apt install python3-venv` (or `python3.12-venv`).
4. Dhan token generation **rate-limited to once every 2 minutes**.
5. IP whitelisting on Dhan **locked for 7 days** — get it right the first time.
6. `.env` must never be committed to git — ensure `.gitignore` contains `.env`.
7. `openscreener` requires Playwright: `playwright install chromium`.
8. Screener takes 10-15 minutes to run (250 stocks, 2s delay each).
9. Telegram bot can only message users who have **started a conversation** with it first — send `/start` to the bot before testing alerts.
10. GCP sudo may not work for browser SSH — use `gcloud compute ssh` or add startup script with `usermod -aG sudo`.
11. Static IP on GCP costs ~$3/month even when attached — but is required for Dhan whitelist.
12. `git clone` fails if target directory is not empty (e.g., already has `venv/`) — clone to temp dir and `cp -a` contents.

## Deployment Steps

1. Create GCP account, project, enable Compute Engine API
2. Provision e2-micro VM in `asia-south1-c`, Ubuntu 24.04 LTS, 30GB disk
3. Reserve static external IP and attach to VM
4. SSH into VM, fix sudo if needed, update system
5. Configure UFW firewall (SSH in, HTTPS/HTTP out), set timezone to IST
6. Install Python 3.12, create venv
7. Clone GitHub repo, install `requirements.txt`
8. Create `.env` with Dhan credentials
9. Whitelist VM static IP on Dhan (web.dhan.co → Profile → Get Trading & Data APIs → Add Static IP)
10. Test auth: `get_access_token()` via PIN+TOTP
11. Test API: `get_fund_limits()`, `get_positions()`, `get_holdings()`
12. Run test suite: `pytest tests/ -v`
13. Set up Telegram bot via @BotFather, add tokens to `.env`
14. Test Telegram alert
15. Run screener on laptop, generate `basket.json`, transfer to VM
16. Set up crontab with all 5 cron jobs
17. Keep `TRADING_MODE=PAPER` for 2-4 weeks
18. Monitor logs daily, verify signals make sense
19. Switch to `TRADING_MODE=LIVE` when confident

## Updating the Bot

```bash
# On VM — pull latest code from GitHub
cd ~/dhan-trading-bot
git pull origin main
source venv/bin/activate
pip install -r requirements.txt

# Run tests to verify nothing broke
python -m pytest tests/ -v
```

## SEBI Compliance (Feb 2025 Circular)

- Static IP mandatory for all algo order placement
- Per-client API keys (no shared keys)
- OAuth+TOTP authentication (no simple API keys)
- IP can only be changed once per 7 days
- Reference: https://www.sebi.gov.in/legal/circulars/feb-2025/safer-participation-of-retail-investors-in-algorithmic-trading_91614.html
