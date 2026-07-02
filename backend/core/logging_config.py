import logging
import os
from logging.handlers import RotatingFileHandler

_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()

_BACKEND_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_LOG_DIR = os.getenv("LOG_DIR", os.path.join(_BACKEND_DIR, "logs"))
_LOG_FILE = os.path.join(_LOG_DIR, "app.log")

_MAX_BYTES = 10 * 1024 * 1024
_BACKUP_COUNT = 5

_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
_DATEFMT = "%Y-%m-%d %H:%M:%S"

_UVICORN_LOGGERS = ("uvicorn", "uvicorn.error", "uvicorn.access")


def setup_logging(level: str | None = None) -> None:
    os.makedirs(_LOG_DIR, exist_ok=True)
    level_val = getattr(logging, (level or _LEVEL), logging.INFO)
    formatter = logging.Formatter(_FORMAT, datefmt=_DATEFMT)

    console = logging.StreamHandler()
    console.setFormatter(formatter)

    file_handler = RotatingFileHandler(
        _LOG_FILE, maxBytes=_MAX_BYTES, backupCount=_BACKUP_COUNT, encoding="utf-8"
    )
    file_handler.setFormatter(formatter)

    root = logging.getLogger()
    root.setLevel(level_val)
    root.handlers.clear()
    root.addHandler(console)
    root.addHandler(file_handler)

    uvicorn_file = RotatingFileHandler(
        _LOG_FILE, maxBytes=_MAX_BYTES, backupCount=_BACKUP_COUNT, encoding="utf-8"
    )
    uvicorn_file.setFormatter(formatter)
    for name in _UVICORN_LOGGERS:
        lg = logging.getLogger(name)
        if not any(isinstance(h, RotatingFileHandler) for h in lg.handlers):
            lg.addHandler(uvicorn_file)
