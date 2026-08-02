#!/usr/bin/env python
"""Step 00 — can this environment run the dichloromethane crystal ladder?

Checks the interpreter, the numerical stack, the two bundled CIFs, and the
CGenFF tables the lattice energy reads. Nothing here needs CHARMM, a GPU, or a
trained checkpoint: the whole ladder is arithmetic on a 20-atom unit cell.
"""

from __future__ import annotations

import sys

FAIL: list[str] = []
WARN: list[str] = []


def ok(label: str, value: object) -> None:
    print(f"  {label:22s} {value}")


print("=== 00: environment ===")

ok("interpreter", sys.executable)
ok("python", ".".join(str(v) for v in sys.version_info[:3]))
if sys.version_info < (3, 10):
    FAIL.append(
        f"Python {sys.version.split()[0]} is too old (need >= 3.10). "
        f"Wrong interpreter: {sys.executable}. Use the project .venv."
    )

try:
    import ase

    ok("ase", ase.__version__)
except Exception as exc:  # pragma: no cover - environment dependent
    FAIL.append(f"import ase failed: {exc} (needed to expand the CIF symmetry)")

try:
    import scipy

    ok("scipy", scipy.__version__)
except Exception as exc:  # pragma: no cover - environment dependent
    FAIL.append(
        f"import scipy failed: {exc} (needed for erfc in the Ewald real space, "
        f"and for the cell relaxation in step 05)"
    )

try:
    import jax

    ok("jax", jax.__version__)
    ok("x64", jax.config.jax_enable_x64)
    if not jax.config.jax_enable_x64:
        WARN.append(
            "JAX_ENABLE_X64 is off. The reciprocal-space sum cancels against the "
            "self term to several digits, so single precision costs real accuracy "
            "in the lattice energy. Source _env.sh to set it."
        )
except Exception as exc:  # pragma: no cover - environment dependent
    FAIL.append(f"import jax failed: {exc}")

# --- the deposited structures ----------------------------------------------
try:
    from mmml.analysis.dcm_crystal import DCM_CRYSTAL_PHASES

    missing = [
        phase.key for phase in DCM_CRYSTAL_PHASES.values() if not phase.cif_path().is_file()
    ]
    if missing:
        FAIL.append(f"bundled CIFs missing for phases: {', '.join(missing)}")
    else:
        ok("phases", f"{len(DCM_CRYSTAL_PHASES)} CIFs (Podsiadlo et al. 2005, via COD)")
except Exception as exc:
    FAIL.append(f"could not load the DCM phase table: {exc}")

# --- CGenFF, read straight off the bundled parameter files ------------------
try:
    from mmml.data.cgenff_dataset import load_reference

    ref = load_reference()
    if "DCM" not in ref.residues:
        FAIL.append("CGenFF RTF has no RESI DCM — cannot type dichloromethane")
    else:
        ok("CGenFF", f"{len(ref.sigmas)} types; RESI DCM present")
except Exception as exc:
    FAIL.append(f"could not load the CGenFF tables: {exc}")

print()
for w in WARN:
    print(f"WARNING: {w}")
for f in FAIL:
    print(f"ERROR:   {f}", file=sys.stderr)

if FAIL:
    print("\n00: FAILED", file=sys.stderr)
    sys.exit(1)
print("00: OK" + (f" ({len(WARN)} warning(s))" if WARN else ""))
