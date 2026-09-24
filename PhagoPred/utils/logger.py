import logging
import sys
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

    _log_file_path = Path('temp') / 'log.log'

    handler = logging.FileHandler(_log_file_path, mode='w')
    handler.setFormatter(
        logging.Formatter(
            "%(asctime)s [%(levelname)s] %(pathname)s: %(message)s"))

    _logger = logging.getLogger(name)
    _logger.setLevel(logging.DEBUG)
    _logger.addHandler(handler)
    _logger.propagate = False  # don't pass records up to the root logger

    _logger.info("Log file: %s", _log_file_path)

    def _log_uncaught_exception(exc_type, exc_value, exc_traceback):
        if issubclass(exc_type, KeyboardInterrupt):
            sys.__excepthook__(exc_type, exc_value, exc_traceback)
            return
        _logger.critical("Uncaught exception",
                         exc_info=(exc_type, exc_value, exc_traceback))
        sys.__excepthook__(exc_type, exc_value, exc_traceback)

    sys.excepthook = _log_uncaught_exception

    return _logger


def get_log_path() -> Path:
    """Returns the path to the current run's log file, or None if not yet initialised."""
    return _log_file_path
