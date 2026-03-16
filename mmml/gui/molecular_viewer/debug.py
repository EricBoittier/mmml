"""
Debug logging to vconsole (Linux /dev/console) or stderr.
Enable with --debug or MOLECULAR_VIEWER_DEBUG=1.
Use MOLECULAR_VIEWER_VCONSOLE=1 to try /dev/console (VR: visible on TTY).
"""

from __future__ import annotations

import os
import sys

_DEBUG = os.environ.get("MOLECULAR_VIEWER_DEBUG", "").lower() in ("1", "true", "yes")
_USE_VCONSOLE = os.environ.get("MOLECULAR_VIEWER_VCONSOLE", "").lower() in ("1", "true", "yes")
_VCONSOLE = None


def set_debug(enabled: bool) -> None:
    global _DEBUG
    _DEBUG = enabled


def _get_vconsole():
    """Get output stream: /dev/console (vconsole) if requested and writable, else stderr."""
    global _VCONSOLE
    if _VCONSOLE is not None:
        return _VCONSOLE
    if _USE_VCONSOLE:
        try:
            f = open("/dev/console", "w")
            _VCONSOLE = f
            return f
        except (OSError, PermissionError):
            pass
    _VCONSOLE = sys.stderr
    return sys.stderr


def debug(msg: str, *args, **kwargs) -> None:
    """Print debug message when MOLECULAR_VIEWER_DEBUG=1 or --debug."""
    if not _DEBUG:
        return
    out = _get_vconsole()
    text = msg if not args and not kwargs else (msg % args if args else msg)
    print(f"[molecular_viewer] {text}", file=out, flush=True, **kwargs)


def log(msg: str, *args, **kwargs) -> None:
    """Print message to vconsole/stderr (always)."""
    out = _get_vconsole()
    text = msg if not args and not kwargs else (msg % args if args else msg)
    print(f"[molecular_viewer] {text}", file=out, flush=True, **kwargs)
