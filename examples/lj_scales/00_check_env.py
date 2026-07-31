#!/usr/bin/env python
"""Step 00 — is this environment able to run the LJ-scale ladder?

Checks the interpreter, JAX, the CGenFF master tables, and the input dataset.
Exits non-zero on anything that would break a later step, so the ladder fails
here rather than half-way through training.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

FAIL: list[str] = []
WARN: list[str] = []


def ok(label: str, value: object) -> None:
    print(f"  {label:22s} {value}")


print("=== 00: environment ===")

# --- interpreter -----------------------------------------------------------
# The venv kernelspec uv installs uses a bare "python" in its argv, so notebooks
# and stray shells can end up on a conda interpreter that cannot even parse
# mmml's `tuple[float, ...]` annotations. Catch it here with a clear message.
ok("interpreter", sys.executable)
ok("python", ".".join(str(v) for v in sys.version_info[:3]))
if sys.version_info < (3, 10):
    FAIL.append(
        f"Python {sys.version.split()[0]} is too old (need >= 3.10). "
        f"Wrong interpreter: {sys.executable}. Use the project .venv."
    )

# --- jax -------------------------------------------------------------------
try:
    import jax

    ok("jax", jax.__version__)
    ok("devices", jax.devices())
    ok("x64", jax.config.jax_enable_x64)
    want_gpu = (os.environ.get("MMML_MLPOT_DEVICE") or "cpu").lower() == "gpu"
    on_gpu = any(d.platform != "cpu" for d in jax.devices())
    if want_gpu and not on_gpu:
        WARN.append(
            "MMML_MLPOT_DEVICE=gpu but JAX sees no GPU — training will run on "
            "CPU. Install the CUDA build: make install-gpu"
        )
except Exception as exc:  # pragma: no cover - environment dependent
    FAIL.append(f"import jax failed: {exc}")

# --- CGenFF master tables --------------------------------------------------
# These define the per-type scale vector length; without them nothing downstream
# can map a trained scale back onto a CHARMM atom type.
try:
    from mmml.models.mm_lj_scales import cgenff_type_names_from_prm

    names = cgenff_type_names_from_prm()
    ok("CGenFF types", f"{len(names)} (e.g. {', '.join(names[:4])} ...)")
except Exception as exc:
    FAIL.append(f"could not load CGenFF master tables: {exc}")

# --- optax (needed by step 04 and by training) -----------------------------
try:
    import optax

    ok("optax", optax.__version__)
except Exception as exc:
    FAIL.append(f"import optax failed: {exc}")

# --- dataset ---------------------------------------------------------------
dataset = Path(os.environ.get("LJ_DATASET", ""))
if not dataset.name:
    WARN.append("LJ_DATASET unset — source examples/lj_scales/_env.sh first")
elif not dataset.is_file():
    WARN.append(
        f"dataset not found: {dataset}\n"
        "      Steps 00, 03 and 04 still run (they are self-contained);\n"
        "      steps 01, 02, 05 and 07 need it."
    )
else:
    ok("dataset", f"{dataset.name} ({dataset.stat().st_size / 1e6:.1f} MB)")

# --- PyCHARMM (only step 07 needs it) --------------------------------------
try:
    import pycharmm  # noqa: F401

    ok("pycharmm", "importable")
except Exception:
    WARN.append("PyCHARMM not importable — step 07 (MD deploy) will be skipped")

print()
for w in WARN:
    print(f"WARNING: {w}")
for f in FAIL:
    print(f"ERROR:   {f}", file=sys.stderr)

if FAIL:
    print("\n00: FAILED", file=sys.stderr)
    sys.exit(1)
print("00: OK" + (f" ({len(WARN)} warning(s))" if WARN else ""))
