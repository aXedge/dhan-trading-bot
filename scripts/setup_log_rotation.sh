#!/bin/bash
# ============================================================
# Log Rotation for Dhan Trading Bot
# ============================================================
# Installs a logrotate config that keeps 14 days of logs,
# compresses old logs, and rotates daily.
# ============================================================

CONFIG_PATH="/etc/logrotate.d/dhan-bot"

# Create the logrotate config
sudo tee "$CONFIG_PATH" << 'EOF'
/home/*/dhan-trading-bot/logs/*.log {
    daily
    rotate 14
    compress
    delaycompress
    missingok
    notifempty
    create 0644
    dateext
    dateformat -%Y%m%d
    sharedscripts
    postrotate
        # No restart needed — cron creates new log files on next run
        :
    endscript
}
EOF

echo "Log rotation installed at $CONFIG_PATH"
echo ""
echo "Settings:"
echo "  - Rotates daily"
echo "  - Keeps 14 days of compressed logs"
echo "  - Compresses old logs (gzip)"
echo "  - Uses date-based filenames (e.g. executor_B.log-20260903.gz)"
echo ""
echo "To test rotation manually:"
echo "  sudo logrotate -f $CONFIG_PATH"
echo ""
echo "To verify current disk usage of logs:"
du -sh ~/dhan-trading-bot/logs/
ls -la ~/dhan-trading-bot/logs/
