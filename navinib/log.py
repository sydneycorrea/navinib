"""Logging for Navinib.

Every tool logs what it was asked and what it returned, so a run can be
reviewed afterwards (and the planner's trace quoted in reports). The log file
always gets full detail; the terminal shows warnings only, or everything with
--verbose.

    from navinib.log import get_logger
    log = get_logger("navinib.tools.route")   # the module path, not __name__

Entry points (python -m ...) call setup_logging() once.
"""
import logging
import sys
from pathlib import Path

from navinib import config

LOG_PATH = config.ROOT / "logs" / "navinib.log"
FORMAT = "%(asctime)s %(levelname)-7s %(name)-24s %(message)s"


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name if name.startswith("navinib") else f"navinib.{name}")


def setup_logging(verbose: bool | None = None, log_file: Path | str | None = LOG_PATH) -> logging.Logger:
    """Configure the 'navinib' logger. verbose=None reads --verbose / -v from the command line."""
    if verbose is None:
        verbose = any(a in ("--verbose", "-v") for a in sys.argv[1:])
        sys.argv[1:] = [a for a in sys.argv[1:] if a not in ("--verbose", "-v")]
    root = logging.getLogger("navinib")
    root.setLevel(logging.DEBUG)
    for h in list(root.handlers):          # safe to call more than once
        root.removeHandler(h)
        h.close()
    console = logging.StreamHandler()
    console.setLevel(logging.DEBUG if verbose else logging.WARNING)
    console.setFormatter(logging.Formatter("%(levelname)-7s %(name)-24s %(message)s"))
    root.addHandler(console)
    if log_file:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        f = logging.FileHandler(log_file, encoding="utf-8")
        f.setLevel(logging.DEBUG)
        f.setFormatter(logging.Formatter(FORMAT, "%Y-%m-%d %H:%M:%S"))
        root.addHandler(f)
    root.propagate = False
    root.info("---- run: %s", " ".join([Path(sys.argv[0]).name] + sys.argv[1:]))
    return root
