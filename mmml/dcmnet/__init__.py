"""
Compatibility shim for legacy ``mmml.dcmnet`` imports.

The implementation lives under ``mmml.models.dcmnet``.
"""

from pathlib import Path

_REAL_PKG = Path(__file__).resolve().parents[1] / "models" / "dcmnet"
__path__ = [str(_REAL_PKG)]

