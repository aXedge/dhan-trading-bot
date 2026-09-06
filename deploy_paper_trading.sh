#!/bin/bash
# ============================================================
# Paper Trading Deployment Script
# ============================================================
# Run this on the VM to set up paper trading.
#
# Usage: bash deploy_paper_trading.sh
# ============================================================

set -e

REPO_DIR="/home/kay22_ind/dhan-trading-bot"
cd "$REPO_DIR"

echo "=================================================="
echo "Paper Trading Deployment"
echo "=================================================="

# Create directories
mkdir -p src/live data logs

# Activate venv
source venv/bin/activate

# Install yfinance if not present
pip install yfinance --quiet 2>/dev/null || true

# Set environment variables
export TRADING_MODE=PAPER
export PYTHONPATH="$REPO_DIR/src"
export DHAN_CLIENT_ID="1000191656"

# Verify strategy files exist
echo ""
echo "Checking strategy files..."
for f in strategies/reversal.py strategies/positional_pullback.py; do
    if [ -f "$f" ]; then
        echo "  ✓ $f"
    else
        echo "  ✗ $f MISSING — downloading from GitHub"
        git pull origin main
    fi
done

# Verify live scripts exist
echo ""
echo "Checking live scripts..."
for f in src/live/scanner.py src/live/executor.py src/live/monitor.py src/live/eod_report.py; do
    if [ -f "$f" ]; then
        echo "  ✓ $f"
    else
        echo "  ✗ $f MISSING"
    fi
done

# Initialize paper positions file if not exists
if [ ! -f data/paper_positions.json ]; then
    echo '{"positions": [], "closed_trades": []}' > data/paper_positions.json
    echo "  ✓ Initialized data/paper_positions.json"
fi

# Test scanner (dry run)
echo ""
echo "Testing scanner (first 3 stocks only)..."
python -c "
import sys; sys.path.insert(0, 'src')
from live.scanner import fetch_stock_data, scan_strategy
import strategies.reversal as rev
import strategies.positional_pullback as pb

for sym in ['HDFCBANK', 'INFY', 'EICHERMOT']:
    df = fetch_stock_data(sym)
    if df:
        for mod, name in [(rev, 'reversal'), (pb, 'pullback')]:
            sig = scan_strategy(sym, df, mod, name)
            if sig:
                print(f'  {sym}: {name} SIGNAL found!')
                break
        else:
            print(f'  {sym}: no signal (OK)')
    else:
        print(f'  {sym}: no data')
"

echo ""
echo "=================================================="
echo "Deployment complete!"
echo ""
echo "Next steps:"
echo "  1. Copy crontab_config.txt contents to crontab:"
echo "     crontab -e"
echo "  2. Set Telegram env vars in crontab or .bashrc:"
echo "     export TELEGRAM_BOT_TOKEN=your_token"
echo "     export TELEGRAM_CHAT_ID=your_chat_id"
echo "  3. Test manually:"
echo "     python src/live/scanner.py"
echo "     python src/live/executor.py"
echo "     python src/live/monitor.py"
echo "     python src/live/eod_report.py"
echo "  4. Monitor logs in logs/ directory"
echo "=================================================="
