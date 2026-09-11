"""A thin, consistent logging setup shared by scripts and library code.

Library modules should call `get_logger(__name__)` and log through it rather
than printing directly, so verbosity is controllable from one place and
`scripts/*.py` can additionally attach a file handler per run.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

_CONFIGURED = False
_FORMAT = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"
_DATEFMT = "%Y-%m-%d %H:%M:%S"


def _configure_root(level: int) -> None:
    global _CONFIGURED
    if _CONFIGURED:
        return
    root = logging.getLogger("knee_mri")
    root.setLevel(level)
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter(_FORMAT, datefmt=_DATEFMT))
    root.addHandler(handler)
    root.propagate = False
    _CONFIGURED = True


def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """Return a logger under the shared `knee_mri` namespace. Safe to call
    repeatedly (e.g. once per module at import time) -- the stream handler is
    only attached once."""
    _configure_root(level)
    return logging.getLogger(f"knee_mri.{name}")


def add_file_handler(logger: logging.Logger, path: str | Path) -> None:
    """Attach a file handler to the shared `knee_mri` root logger so every
    module's logs for this run also land in `path`. Intended to be called
    once per run from a `scripts/*.py` entry point."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    handler = logging.FileHandler(path, encoding="utf-8")
    handler.setFormatter(logging.Formatter(_FORMAT, datefmt=_DATEFMT))
    logging.getLogger("knee_mri").addHandler(handler)
