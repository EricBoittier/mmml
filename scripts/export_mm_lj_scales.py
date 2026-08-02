#!/usr/bin/env python3
"""Export fitted MM LJ sigma/epsilon scales from a training checkpoint into a
sidecar JSON that ``mmml md-system --mm-lj-scales-file`` can consume.

This is needed because the ``hybrid_mm.json`` written next to a training
checkpoint carries only the *configuration* -- ``cgenff_type_names``, the
bounds, the trainable mask, the frame counts -- and **not** the fitted
``mm_lj_sigma_scale`` / ``mm_lj_epsilon_scale`` arrays. Those live in the
checkpoint parameters. ``load_mm_lj_scales_sidecar`` raises on such a file, so
pointing an MD run at the training ``hybrid_mm.json`` does not work.

Two modes, and the pair is the point:

``--mode trained``
    the fitted scales, read from the checkpoint.

``--mode unit``
    every scale exactly 1.0, i.e. stock CGenFF/literature LJ.

The control matters because several types are pinned at their bounds -- argon,
helium and neon sit on both floors, krypton and xenon on both ceilings -- so a
condensed-phase run with the trained scales alone cannot distinguish "the
training is wrong" from "the underlying literature parameters are wrong". Same
file format, same code path, only the numbers differ.

``ema_params`` is preferred over ``params`` by default: the EMA is what the
training reports and the two differ in the fourth decimal.

Example::

    python scripts/export_mm_lj_scales.py \\
        --checkpoint artifacts/.../ckpts/<run>/epoch-25 \\
        --hybrid-mm artifacts/.../ckpts/<run>/hybrid_mm.json \\
        --mode trained -o scales_trained.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def load_scales(ckpt: Path, group: str) -> tuple[np.ndarray, np.ndarray]:
    import orbax.checkpoint as ocp

    from mmml.utils.model_checkpoint import _restore_pytree_cpu_safe

    state = _restore_pytree_cpu_safe(ocp.PyTreeCheckpointer(), str(ckpt.absolute()))
    if group not in state:
        raise KeyError(f"{ckpt}: no '{group}' in checkpoint (have {list(state)[:8]})")
    g = state[group]
    for key in ("mm_lj_sigma_scale", "mm_lj_epsilon_scale"):
        if key not in g:
            raise KeyError(f"{ckpt}: '{group}' has no {key} -- was this run trained "
                           "with learn_mm_lj_scales=true?")
    return (
        np.asarray(g["mm_lj_sigma_scale"], dtype=np.float64).reshape(-1),
        np.asarray(g["mm_lj_epsilon_scale"], dtype=np.float64).reshape(-1),
    )


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--checkpoint", type=Path, required=True, help="epoch-N directory")
    ap.add_argument("--hybrid-mm", type=Path, required=True,
                    help="hybrid_mm.json from the same run (for type names + bounds)")
    ap.add_argument("--mode", choices=("trained", "unit"), required=True)
    ap.add_argument("--group", default="ema_params", choices=("ema_params", "params"))
    ap.add_argument("-o", "--output", type=Path, required=True)
    a = ap.parse_args(argv)

    cfg = json.loads(a.hybrid_mm.read_text())
    names = cfg.get("cgenff_type_names")
    if not names:
        raise SystemExit(f"{a.hybrid_mm}: no cgenff_type_names")
    sb = cfg.get("mm_lj_sigma_scale_bounds", [0.8, 1.2])
    eb = cfg.get("mm_lj_epsilon_scale_bounds", [0.25, 4.0])

    if a.mode == "unit":
        sig = np.ones(len(names))
        eps = np.ones(len(names))
        # The sidecar loader bounds-checks what it reads. Unit scales are the
        # control, so they must be representable under the run's own bounds --
        # if a prior excluded 1.0 the control would be silently unrunnable.
        if not (sb[0] <= 1.0 <= sb[1] and eb[0] <= 1.0 <= eb[1]):
            raise SystemExit(
                f"unit scales fall outside this run's bounds sigma={sb} eps={eb}; "
                "the control cannot be expressed against them"
            )
    else:
        sig, eps = load_scales(a.checkpoint, a.group)
        if sig.size != len(names) or eps.size != len(names):
            raise SystemExit(
                f"length mismatch: {sig.size}/{eps.size} scales vs {len(names)} "
                f"type names -- checkpoint and hybrid_mm.json are from different runs"
            )

    payload = {
        "learn_mm_lj_scales": True,
        "include_lj": bool(cfg.get("include_lj", True)),
        "lr_solver": cfg.get("lr_solver", "mic"),
        "cgenff_type_names": [str(n) for n in names],
        "mm_lj_sigma_scale": [float(x) for x in sig],
        "mm_lj_epsilon_scale": [float(x) for x in eps],
        "mm_lj_sigma_scale_bounds": [float(x) for x in sb],
        "mm_lj_epsilon_scale_bounds": [float(x) for x in eb],
        "_provenance": {
            "mode": a.mode,
            "checkpoint": str(a.checkpoint) if a.mode == "trained" else None,
            "group": a.group if a.mode == "trained" else None,
            "hybrid_mm": str(a.hybrid_mm),
        },
    }
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(payload, indent=2))

    # Round-trip through the real loader rather than trusting the file we just
    # wrote -- it bounds-checks and will reject a payload MD would reject.
    from mmml.models.mm_lj_scales import load_mm_lj_scales_sidecar

    back = load_mm_lj_scales_sidecar(a.output)
    if back is None:
        raise SystemExit(f"{a.output}: loader returned None -- payload not recognised")

    print(f"wrote {a.output}  ({a.mode}, {len(names)} types)")
    n_at = int((np.abs(sig - sb[0]) < 1e-3).sum() + (np.abs(sig - sb[1]) < 1e-3).sum())
    print(f"  sigma range {sig.min():.4f}-{sig.max():.4f}  ({n_at} within 1e-3 of a bound)")
    print(f"  eps   range {eps.min():.4f}-{eps.max():.4f}")
    for t in ("AR", "KR", "XE", "OT", "HT"):
        if t in names:
            i = names.index(t)
            print(f"    {t:3s} sigma={sig[i]:.4f} eps={eps[i]:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
