"""jax_md must import on CI (fresh uv resolve from pyproject.toml).

JAX 0.11.2 deleted ``jax.experimental.hijax.HiPrimitive`` (renamed ``HiPrim``).
flax 0.12.x still subclasses ``HiPrimitive`` when importing ``flax.nnx``, and
``jax_md.energy`` does that at import. Keep ``jax``/``jaxlib`` ``<0.11.2`` in
``pyproject.toml`` until flax follows.
"""

from __future__ import annotations

from importlib.metadata import version

from jax.experimental import hijax
from jax_md import space
from packaging.version import Version


def test_jax_below_hiprimitive_deletion() -> None:
    assert Version(version("jax")) < Version("0.11.2")
    assert Version(version("jaxlib")) < Version("0.11.2")


def test_hijax_exports_hiprimitive_for_flax() -> None:
    assert hasattr(hijax, "HiPrimitive")


def test_jax_md_space_imports() -> None:
    assert space is not None
