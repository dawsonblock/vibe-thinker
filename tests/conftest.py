"""Pytest configuration shared across the vibe-thinker test suite.

Currently this module's only job is to suppress a benign teardown noise:
on macOS Python 3.12, after pytest exits, the multiprocessing
``ResourceTracker.__del__`` can raise ``AttributeError`` because the
underlying socket is already closed during interpreter shutdown. This is
multiprocessing teardown noise, NOT a test failure, but it pollutes CI
output and can mask real failures. See AGENTS.md.

The fix: register an ``atexit`` handler (runs at interpreter shutdown,
*after* pytest's own teardown) that wraps the resource tracker's
``__del__`` to swallow the expected ``AttributeError``. The handler is
guarded so it only patches when the relevant attribute exists and only
suppresses ``AttributeError`` (any other exception type still propagates,
so we never hide a real bug).
"""

import atexit
import sys


def _suppress_resource_tracker_del_noise() -> None:
    """Wrap ResourceTracker.__del__ to swallow the benign AttributeError
    raised during interpreter shutdown on macOS Py3.12.

    The observed noise comes from TWO sources:
      - the third-party ``multiprocess`` package (a multiprocessing fork
        used by pathos/cloudpickle): ``multiprocess.resource_tracker``
        whose ``__del__`` -> ``_stop`` -> ``_stop_locked`` fails because
        the RLock has already lost ``_recursion_count`` at shutdown.
      - stdlib ``multiprocessing.resource_tracker`` (same shape, rarer).

    We patch whichever of these is importable. Safe no-op where the
    attribute doesn't exist. Only ``AttributeError`` is swallowed — any
    other exception type still propagates so we never mask a real bug.
    """
    for modname in ("multiprocess.resource_tracker",
                    "multiprocessing.resource_tracker"):
        try:
            mod = __import__(modname, fromlist=["ResourceTracker"])
        except ImportError:
            continue
        cls = getattr(mod, "ResourceTracker", None)
        if cls is None:
            continue
        _patch_one(cls)


def _patch_one(cls) -> None:
    orig_del = getattr(cls, "__del__", None)
    if orig_del is None or getattr(cls, "__del__vibe_suppressed__", False):
        return

    def _quiet_del(self):  # pragma: no cover - exercised at interpreter exit
        try:
            orig_del(self)
        except AttributeError:
            # Expected: the tracker's lock/socket is already gone at
            # shutdown. This is the exact noise we are suppressing.
            pass
        except Exception:
            # Any non-AttributeError failure still propagates so we never
            # mask a genuine bug.
            raise

    try:
        cls.__del__ = _quiet_del
        cls.__del__vibe_suppressed__ = True  # idempotency guard
    except (AttributeError, TypeError):
        # Some builds use a C-implemented class that can't be monkeypatched.
        # Nothing we can do — the noise is cosmetic anyway.
        pass


# Only register on macOS where the noise is observed; registering elsewhere
# is harmless but unnecessary. Registered at import time (pytest collection)
# so the handler runs at interpreter shutdown regardless of test outcome.
if sys.platform == "darwin":
    atexit.register(_suppress_resource_tracker_del_noise)
