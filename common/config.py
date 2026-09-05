"""
Common configuration, logging, file I/O, and Telegram alerts.

This module is strategy-agnostic — every strategy and backtest imports
from here. It provides:
  - YAML config loading with session-based override
  - JSON load/save to data/ directory
  - Logger setup (console + file)
  - Telegram message sending
  - Market hours check
  - Timestamp helpers
"""

import os
import json
import logging
import yaml
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv

# Load .env once at import time
load_dotenv()

# Project root (parent of src/)
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = PROJECT_ROOT / os.getenv("DATA_DIR", "data")
LOG_DIR = PROJECT_ROOT / os.getenv("LOG_DIR", "logs")
CONFIG_PATH = PROJECT_ROOT / os.getenv("CONFIG_PATH", "config/settings.yaml")

# Ensure directories exist
DATA_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------

def set_config_path(config_file: str):
    """Override the config path (used by --session flag)."""
    global CONFIG_PATH
    CONFIG_PATH = PROJECT_ROOT / config_file
    os.environ["CONFIG_PATH"] = str(CONFIG_PATH)


def load_config() -> dict:
    """Load the YAML configuration file."""
    with open(CONFIG_PATH, "r") as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def setup_logger(name: str, log_file: str = None) -> logging.Logger:
    """Set up a logger with console + optional file output."""
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    if logger.handlers:
        return logger

    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    console.setFormatter(
        logging.Formatter("%(asctime)s [%(name)s] %(levelname)s: %(message)s", "%H:%M:%S")
    )
    logger.addHandler(console)

    if log_file:
        file_handler = logging.FileHandler(LOG_DIR / log_file)
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(
            logging.Formatter("%(asctime)s [%(name)s] %(levelname)s: %(message)s")
        )
        logger.addHandler(file_handler)

    return logger


# ---------------------------------------------------------------------------
# File I/O
# ---------------------------------------------------------------------------

def load_json(filename: str) -> list | dict:
    """Load a JSON file from the data/ directory."""
    filepath = DATA_DIR / filename
    if not filepath.exists():
        return [] if filename.endswith((".json",)) else {}
    with open(filepath, "r") as f:
        return json.load(f)


def save_json(data: list | dict, filename: str) -> None:
    """Save data as JSON to the data/ directory."""
    filepath = DATA_DIR / filename
    with open(filepath, "w") as f:
        json.dump(data, f, indent=2, default=str)


# ---------------------------------------------------------------------------
# Telegram
# ---------------------------------------------------------------------------

def send_telegram(message: str) -> bool:
    """Send a message to Telegram. Returns True on success."""
    token = os.getenv("TG_BOT_TOKEN", "")
    chat_id = os.getenv("TG_CHAT_ID", "")

    if not token or not chat_id:
        return False

    try:
        import requests
        url = f"https://api.telegram.org/bot{token}/sendMessage"
        resp = requests.post(
            url,
            json={"chat_id": chat_id, "text": message, "parse_mode": "HTML"},
            timeout=10,
        )
        return resp.status_code == 200
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Time helpers
# ---------------------------------------------------------------------------

def now_iso() -> str:
    """Current timestamp in ISO format."""
    return datetime.now().isoformat()


def is_market_open() -> bool:
    """Check if NSE is open (9:15 AM - 3:30 PM IST, Mon-Fri)."""
    now = datetime.now()
    if now.weekday() >= 5:
        return False
    hour_min = now.hour * 100 + now.minute
    return 915 <= hour_min <= 1530
