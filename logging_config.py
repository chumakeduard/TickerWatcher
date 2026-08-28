#!/usr/bin/env python3
"""
Logging configuration for TickerWatcher.

Sets up a single unified daily log file in ./logs folder that captures:
- Data refresh operations (refresh)
- Model calibration/backtest operations (calibration)
- General application events (app startup, etc.)

Each calendar date gets its own file: tickerwatcher-YYYY-MM-DD.log
"""

import logging
import os
from pathlib import Path
from datetime import datetime

# Create logs directory if it doesn't exist
LOGS_DIR = Path(__file__).parent / "logs"
LOGS_DIR.mkdir(exist_ok=True)


class DailyRotatingFileHandler(logging.FileHandler):
    """Custom file handler that rotates based on calendar date.

    Creates a new log file for each calendar day with filename:
    tickerwatcher-YYYY-MM-DD.log
    """

    def __init__(self, log_dir, level=logging.INFO):
        self.log_dir = Path(log_dir)
        self.current_date = None
        self.log_path = None
        super().__init__(self._get_log_path(), mode='a')
        self.setLevel(level)

    def _get_log_path(self):
        """Get log file path for today's date."""
        today = datetime.now().strftime('%Y-%m-%d')
        self.log_path = self.log_dir / f"tickerwatcher-{today}.log"
        return str(self.log_path)

    def emit(self, record):
        """Override to check if date has changed and rotate if needed."""
        today = datetime.now().strftime('%Y-%m-%d')

        # If date changed, close current file and open new one
        if today != self.current_date:
            self.current_date = today
            if self.stream:
                self.flush()
                self.stream.close()
            self.baseFilename = self._get_log_path()
            self.stream = self._open()

        super().emit(record)


def setup_unified_logger(level=logging.INFO):
    """
    Set up a unified logger with date-based file rotation.

    All operations (app, refresh, calibration) write to a single daily log file:
    tickerwatcher-YYYY-MM-DD.log (new file created each calendar day)

    Args:
        level: Logging level (default INFO)

    Returns:
        Configured logger instance
    """
    logger = logging.getLogger('tickerwatcher')
    logger.setLevel(level)

    # Remove existing handlers to avoid duplicates
    logger.handlers = []

    # Custom file handler with date-based rotation
    file_handler = DailyRotatingFileHandler(LOGS_DIR, level=level)

    # Format: timestamp | level | message
    formatter = logging.Formatter(
        '%(asctime)s | %(levelname)-8s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    # Console handler (stderr output)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    return logger


# Create single unified logger used by all modules
_unified_logger = setup_unified_logger()


def get_data_refresh_logger():
    """Get the unified logger for data refresh operations."""
    return _unified_logger


def get_model_calibration_logger():
    """Get the unified logger for model calibration/backtest operations."""
    return _unified_logger


def get_app_logger():
    """Get the unified logger for application events."""
    return _unified_logger
