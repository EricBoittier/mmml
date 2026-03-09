"""
Compatibility shim for legacy ``mmml.pycharmmInterface`` imports.

The implementation moved to ``mmml.interfaces.pycharmmInterface``.
Expose that directory as this package's module search path so existing
imports like ``mmml.pycharmmInterface.mmml_calculator`` keep working.
"""

from pathlib import Path

_REAL_PKG = Path(__file__).resolve().parents[1] / "interfaces" / "pycharmmInterface"
__path__ = [str(_REAL_PKG)]

