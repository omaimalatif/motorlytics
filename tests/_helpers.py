"""Shared test setup: a fresh temporary database + calibration store per run, project root on sys.path.

Import this BEFORE anything from core/ or ui/ (they read MOTORLYTICS_DB / MOTORLYTICS_CALIB at import time).
"""
import os
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SAMPLES = ROOT / "samples"
FIXTURES = Path(__file__).resolve().parent / "fixtures"


def setup(name: str) -> Path:
    """Point the app at a new empty temp folder (database, calibration store, uploads) and return it."""
    tmp = Path(tempfile.mkdtemp(prefix=f"motorlytics_{name}_"))
    os.environ["MOTORLYTICS_DB"] = str(tmp / "test.db")
    os.environ["MOTORLYTICS_CALIB"] = str(tmp / "calibration.json")
    os.chdir(ROOT)
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    return tmp


class Checks:
    """Collect pass/fail results; call done() at the end (exit code 1 if anything failed)."""

    def __init__(self, name: str):
        self.name, self.failed = name, []

    def ok(self, cond, what: str, detail: str = "") -> bool:
        print(f"  {'PASS' if cond else 'FAIL'}  {what}" + (f"  ({detail})" if detail else ""))
        if not cond:
            self.failed.append(what)
        return bool(cond)

    def page(self, at, step: str) -> None:
        """No uncaught exception on an AppTest page."""
        exc = [e.value for e in at.exception]
        self.ok(not exc, f"page renders: {step}", exc[0][:300] if exc else "")

    def done(self) -> None:
        print(f"\n{self.name}: {'FAILED: ' + '; '.join(self.failed) if self.failed else 'all checks passed'}")
        sys.exit(1 if self.failed else 0)


def upload(user, car, paths, **kw):
    """Same call the Upload trip page makes."""
    from core import app_core
    return app_core.ingest_logs(user, car, [(Path(p).name, Path(p).read_bytes()) for p in paths], **kw)
