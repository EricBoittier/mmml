"""
Compatibility shim for legacy ``mmml.physnetjax`` imports.

Implementation currently lives under ``mmml.models.physnetjax``.
"""

from pathlib import Path

_REAL_PKG = Path(__file__).resolve().parents[1] / "models" / "physnetjax"
__path__ = [str(_REAL_PKG)]

