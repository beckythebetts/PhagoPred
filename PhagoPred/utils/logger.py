import logging
from pathlib import Path

_logger = None
_log_file_path = None


def get_logger(name: str = "PhagoPred") -> logging.Logger:
    """
    Returns the shared logger for this run. Creates it on first call,
    writing to a temporary .txt file that persists for the process lifetime.
    """
    global _logger, _log_file_path

    if _logger is not None:
        return logging.getLogger(name)

    _log_file_path = Path('temp') / 'log.txt'
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[
            logging.FileHandler(_log_file_path, mode='w'),
        ],
    )

    _logger = logging.getLogger(name)
    _logger.info("Log file: %s", _log_file_path)

    return _logger


def get_log_path() -> Path:
    """Returns the path to the current run's log file, or None if not yet initialised."""
    return _log_file_path
