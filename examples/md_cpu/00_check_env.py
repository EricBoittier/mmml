#!/usr/bin/env python3
"""Check CPU MD example environment (md-cpu extra, JAX CPU, DESdimers checkpoint)."""

from __future__ import annotations

import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


def _ok(msg: str) -> None:
    print(f"PASS: {msg}")


def _fail(msg: str) -> None:
    print(f"FAIL: {msg}")


def main() -> int:
    ok = True

    ckpt = Path(os.environ.get("MMML_CKPT", REPO_ROOT / "examples/ckpts_json/DESdimers_params.json"))
    if ckpt.is_file():
        _ok(f"checkpoint {ckpt}")
    else:
        _fail(f"checkpoint missing: {ckpt}")
        ok = False

    try:
        import jax

        devices = [str(d) for d in jax.devices()]
        if all("cpu" in d.lower() for d in devices):
            _ok(f"JAX {jax.__version__} on CPU ({devices[0]})")
        else:
            _fail(f"expected CPU JAX devices, got {devices}")
            ok = False
    except Exception as exc:
        _fail(f"jax: {exc}")
        ok = False

    try:
        import jax_md  # noqa: F401

        _ok("jax-md")
    except Exception as exc:
        _fail(f"jax-md: {exc}")
        ok = False

    try:
        import ase  # noqa: F401

        _ok("ase")
    except Exception as exc:
        _fail(f"ase: {exc}")
        ok = False

    try:
        from mmml.interfaces.pyxtal_placement import have_pyxtal

        if have_pyxtal():
            _ok("pyxtal (optional, from mmml[chem])")
        else:
            print("INFO: pyxtal not installed (optional; uv sync --extra chem)")
    except Exception as exc:
        print(f"INFO: pyxtal probe skipped: {exc}")

    try:
        from mmml.interfaces.pycharmmInterface.nl_reference import have_vesin

        if have_vesin():
            _ok("vesin (md-cpu / nl-validation)")
        else:
            _fail("vesin missing — run: uv sync --extra md-cpu")
            ok = False
    except Exception as exc:
        _fail(f"vesin probe: {exc}")
        ok = False

    try:
        import mmml

        _ok(f"mmml {getattr(mmml, '__version__', '?')}")
    except Exception as exc:
        _fail(f"mmml import: {exc}")
        ok = False

    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
