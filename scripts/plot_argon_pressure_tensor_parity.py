#!/usr/bin/env python3
"""Parity: CHARMM vs jax-md (MM) strain pressure tensors on shared frames.

``PIXX…PIYZ`` stay 0 on this KEY_LIBRARY build outside DYNA, so both engines
are probed the same way jax-md's barostat sees pressure: isotropic/anisotropic
strain derivatives of the potential at **fixed fractional coordinates**

    P = −(1/V) ∂U/∂ε     (virial only; no kinetic term)

* **MM / jax-md** — switched LJ via ``nonbonded_energy_and_forces`` (same
  cutoffs as the NpT run)
* **CHARMM** — ``ENER`` after ``crystal.define_*`` + ``build`` for the
  strained cell

Frames are taken from ``ar1_90k_n500_jaxmd_200ps``.

Example::

    uv run python scripts/plot_argon_pressure_tensor_parity.py
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from mmml.utils.plotting.styles import apply_plot_style, comparison_colors

REPO = Path(__file__).resolve().parents[1]
RUN = REPO / "artifacts/npt_argon_water/runs/ar1_90k_n500_jaxmd_200ps"
BOX = REPO / "artifacts/npt_argon_water/boxes/ar1_90k_n500"
IMG = REPO / "docs/images/npt_argon_water"
OUT_PNG = IMG / "ar1_90k_n500_pressure_tensor_parity.png"
OUT_NPZ = (
    REPO / "artifacts/npt_argon_water/runs/ar1_90k_n500_pressure_tensor_parity.npz"
)

EV_A3_TO_PA = 1.602176634e-19 / 1e-30
EV_A3_TO_BAR = EV_A3_TO_PA * 1.0e-5
KCAL_MOL_TO_EV = 1.0 / 23.060547830619026

COMPONENTS = ("xx", "yy", "zz")
COMP_TO_IJ = {
    "xx": (0, 0),
    "yy": (1, 1),
    "zz": (2, 2),
    # Off-diagonal strain needs a live triclinic crystal. On this KEY_LIBRARY
    # build ``define_tri(..., gamma=85)`` still reports γ=90 via get_unit_cell,
    # so shear FD was cubic-MIC + remapped coords — systematic slopes, not physics.
    "xy": (0, 1),
    "xz": (0, 2),
    "yz": (1, 2),
}
PLOT_COMPONENTS = ("xx", "yy", "zz")  # honest parity only

MM_SWITCH_ON = 7.0
MM_SWITCH_WIDTH = 3.39
N_FRAMES = 16
STRAIN_EPS = 3.0e-5


def _env() -> None:
    os.environ.setdefault(
        "MMML_CGENFF_EXTRA_RTF",
        str(REPO / "mmml/data/charmm/top_noble_gases_literature.rtf"),
    )
    os.environ.setdefault(
        "MMML_CGENFF_EXTRA_PRM",
        str(REPO / "mmml/data/charmm/par_noble_gases_literature.prm"),
    )
    os.environ.setdefault("CHARMM_HOME", str(REPO / "setup/charmm"))
    os.environ.setdefault("CHARMM_LIB_DIR", str(REPO / "setup/charmm/lib"))
    os.environ.setdefault("MMML_NO_CHARMM_MPI", "1")
    os.environ.setdefault("MMML_NO_MPI_RERUN", "1")
    os.environ.setdefault("JAX_PLATFORMS", "cpu")


def _box_to_abc_angles(box: np.ndarray) -> tuple[float, float, float, float, float, float]:
    """Lattice vectors as **columns** of ``box`` → (a,b,c,α,β,γ°)."""
    B = np.asarray(box, float).reshape(3, 3)
    a_v, b_v, c_v = B[:, 0], B[:, 1], B[:, 2]
    la, lb, lc = (float(np.linalg.norm(v)) for v in (a_v, b_v, c_v))

    def ang(u, v) -> float:
        c = float(np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v)))
        c = float(np.clip(c, -1.0, 1.0))
        return float(np.degrees(np.arccos(c)))

    return la, lb, lc, ang(b_v, c_v), ang(a_v, c_v), ang(a_v, b_v)


def _abc_angles_to_box(
    a: float, b: float, c: float, alpha: float, beta: float, gamma: float
) -> np.ndarray:
    """Standard crystallographic cell (columns = a,b,c). Same metric as input box."""
    al, be, ga = np.radians([alpha, beta, gamma])
    ax = float(a)
    bx = float(b) * np.cos(ga)
    by = float(b) * np.sin(ga)
    cx = float(c) * np.cos(be)
    cy = float(c) * (np.cos(al) - np.cos(be) * np.cos(ga)) / np.sin(ga)
    cz = float(np.sqrt(max(float(c) ** 2 - cx**2 - cy**2, 0.0)))
    return np.array([[ax, bx, cx], [0.0, by, cy], [0.0, 0.0, cz]], dtype=np.float64)


def _mm_energy_eV(pos: np.ndarray, box: np.ndarray, nbdata, settings) -> float:
    from mmml.interfaces.pycharmmInterface.mm_system_energy import (
        nonbonded_energy_and_forces,
    )

    terms, _ = nonbonded_energy_and_forces(
        np.asarray(pos, float),
        nbdata,
        np.asarray(box, float),
        settings,
        molecule_id=np.arange(pos.shape[0], dtype=np.int32),
    )
    return float(terms["total"]) * KCAL_MOL_TO_EV


def strain_tensor_bar(
    energy_eV_fn,
    pos: np.ndarray,
    box: np.ndarray,
    *,
    eps: float = STRAIN_EPS,
) -> np.ndarray:
    """P_ij = −(1/V) ∂U/∂ε_ij at fixed fractional coords (bar)."""
    box0 = np.asarray(box, dtype=np.float64).reshape(3, 3)
    R0 = np.asarray(pos, dtype=np.float64).reshape(-1, 3)
    frac = R0 @ np.linalg.inv(box0)
    V = abs(float(np.linalg.det(box0)))
    P = np.zeros((3, 3), dtype=np.float64)
    I = np.eye(3)
    for i in range(3):
        for j in range(i, 3):
            ep = np.zeros((3, 3))
            ep[i, j] = eps
            if i != j:
                ep[j, i] = eps  # symmetric strain
            bp = box0 @ (I + ep)
            bm = box0 @ (I - ep)
            dU = (energy_eV_fn(frac @ bp, bp) - energy_eV_fn(frac @ bm, bm)) / (
                2.0 * eps
            )
            # For off-diagonal symmetric strain, ∂U/∂ε_ij includes both ij & ji.
            scale = 1.0 if i == j else 0.5
            P[i, j] = P[j, i] = -scale * dU / V
    return P * EV_A3_TO_BAR


def _setup():
    _env()
    from mmml.interfaces.pycharmmInterface.nbonds_config import read_cgenff_toppar
    from mmml.interfaces.pycharmmInterface.mlpot.pbc_env import (
        prepare_charmm_pbc,
        apply_pbc_nbonds,
    )
    from mmml.interfaces.pycharmmInterface.mlpot.block_terms import apply_charmm_mm_block
    from mmml.interfaces.pycharmmInterface.import_pycharmm import (
        reset_block,
        reset_block_no_internal,
        safe_energy_show,
    )
    from mmml.interfaces.pycharmmInterface.mm_system_energy import (
        CharmmNbondSettings,
        NonbondedSystemData,
    )
    import pycharmm.read as read
    import pycharmm.coor as coor
    import pycharmm.crystal as crystal
    import pycharmm.lingo as lingo
    from ase.io import read as ase_read

    L0 = 28.868571
    read_cgenff_toppar()
    read.psf_card(str(BOX / "model.psf"))
    pos0 = np.asarray(ase_read(str(BOX / "model.pdb")).get_positions(), float)
    coor.set_positions(pd.DataFrame(pos0, columns=["x", "y", "z"]))
    reset_block()
    reset_block_no_internal()
    reset_block()
    apply_charmm_mm_block()
    prepare_charmm_pbc(L0)
    cuts = apply_pbc_nbonds(
        cubic_box_side_A=L0,
        mm_switch_on=MM_SWITCH_ON,
        mm_switch_width=MM_SWITCH_WIDTH,
        rebuild=True,
    )
    print("CHARMM nbond cuts", cuts, flush=True)
    safe_energy_show()

    n = pos0.shape[0]
    nbdata = NonbondedSystemData(
        charges=np.zeros(n, dtype=np.float64),
        at_codes=np.zeros(n, dtype=np.int32),
        epsilon=np.full(n, 0.238070, dtype=np.float64),
        rmin=np.full(n, 1.910990, dtype=np.float64),
        excluded_pairs=frozenset(),
        e14_pairs=frozenset(),
    )
    settings = CharmmNbondSettings(
        cutnb=float(cuts.cutnb),
        ctonnb=float(cuts.ctonnb),
        ctofnb=float(cuts.ctofnb),
    )
    build_cut = float(cuts.cutim)

    def charmm_energy_eV(pos: np.ndarray, box: np.ndarray) -> float:
        """Install ``box`` metric via crystal, evaluate ENER at matching fractionals.

        CHARMM only accepts (a,b,c,α,β,γ), which canonicalizes the Cartesian
        cell orientation. Placing ``pos`` from the raw strained box into that
        cell without remapping made shear FD apples-to-oranges (systematic
        off-diagonal slopes). We keep fractional coords and remap into the
        crystallographic cell (same metric → same MIC distances).
        """
        box = np.asarray(box, dtype=np.float64).reshape(3, 3)
        pos = np.asarray(pos, dtype=np.float64).reshape(-1, 3)
        frac = pos @ np.linalg.inv(box)
        a, b, c, alpha, beta, gamma = _box_to_abc_angles(box)
        crystal.free_crystal()
        if (
            abs(alpha - 90.0) < 1e-4
            and abs(beta - 90.0) < 1e-4
            and abs(gamma - 90.0) < 1e-4
        ):
            ok = crystal.define_ortho(a, b, c)
        else:
            ok = crystal.define_tri(a, b, c, alpha, beta, gamma)
        if not ok:
            raise RuntimeError(f"crystal define failed for box={box}")
        if not crystal.build(build_cut):
            raise RuntimeError("crystal.build failed")
        cell = _abc_angles_to_box(a, b, c, alpha, beta, gamma)
        pos_canon = frac @ cell
        coor.set_positions(pd.DataFrame(pos_canon, columns=["x", "y", "z"]))
        safe_energy_show()
        ener_kcal = float(lingo.get_energy_value("ENER"))
        return ener_kcal * KCAL_MOL_TO_EV

    def mm_energy_eV(pos: np.ndarray, box: np.ndarray) -> float:
        return _mm_energy_eV(pos, box, nbdata, settings)

    return {
        "charmm_energy_eV": charmm_energy_eV,
        "mm_energy_eV": mm_energy_eV,
        "cuts": cuts,
        "coor": coor,
        "crystal": crystal,
        "L0": L0,
        "build_cut": build_cut,
    }


def evaluate_frames(n_frames: int = N_FRAMES) -> dict:
    eng = _setup()
    z = np.load(RUN / "trajectory.npz")
    positions = np.asarray(z["positions"], float)
    boxes = np.asarray(z["boxes"], float)
    pvir = np.asarray(z["pressures_vir_bar"], float)
    idxs = np.unique(np.linspace(0, len(positions) - 1, n_frames, dtype=int))

    rows = []
    for k, iframe in enumerate(idxs):
        pos = positions[iframe]
        box = boxes[iframe]
        L = float(np.mean(np.diag(box)))
        P_mm = strain_tensor_bar(eng["mm_energy_eV"], pos, box)
        P_ch = strain_tensor_bar(eng["charmm_energy_eV"], pos, box)
        t_ps = iframe / 10.0
        row = {
            "frame": int(iframe),
            "t_ps": float(t_ps),
            "L_A": L,
            "mm_scalar_bar": float(np.trace(P_mm) / 3.0),
            "charmm_scalar_bar": float(np.trace(P_ch) / 3.0),
            "jax_traj_pvir_bar": float(pvir[iframe]),
        }
        for comp, (i, j) in COMP_TO_IJ.items():
            row[f"mm_{comp}"] = float(P_mm[i, j])
            row[f"charmm_{comp}"] = float(P_ch[i, j])
        rows.append(row)
        print(
            f"{k+1:02d}/{len(idxs)} frame={iframe} t={t_ps:6.1f}ps L={L:.3f} "
            f"P_mm={row['mm_scalar_bar']:+8.2f} P_ch={row['charmm_scalar_bar']:+8.2f} "
            f"traj_Pvir={row['jax_traj_pvir_bar']:+8.2f} bar",
            flush=True,
        )
    return {"rows": rows, "cuts": eng["cuts"]}


def plot_parity(rows: list[dict]) -> None:
    # t=0 is post-FIRE / pre-NpT; CHARMM crystal FD is unreliable there.
    rows = [r for r in rows if float(r["t_ps"]) > 0.5]
    apply_plot_style("icml")
    colors = comparison_colors("icml", n=3)
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.6))

    sc = None
    for ax, comp, color in zip(axes, PLOT_COMPONENTS, colors):
        x = np.array([r[f"mm_{comp}"] for r in rows], float)
        y = np.array([r[f"charmm_{comp}"] for r in rows], float)
        m = np.isfinite(x) & np.isfinite(y)
        vals = np.concatenate([x[m], y[m]]) if m.any() else np.array([-1.0, 1.0])
        lo, hi = float(np.min(vals)), float(np.max(vals))
        pad = 0.08 * max(hi - lo, 1.0)
        lim = (lo - pad, hi + pad)
        ax.plot(lim, lim, color="0.55", ls="--", lw=1.0, zorder=0)
        sc = ax.scatter(
            x,
            y,
            c=[r["t_ps"] for r in rows],
            cmap="viridis",
            s=32,
            edgecolors=color,
            linewidths=0.6,
            zorder=2,
        )
        if m.any():
            rms = float(np.sqrt(np.mean((y[m] - x[m]) ** 2)))
            bias = float(np.mean(y[m] - x[m]))
            ax.set_title(rf"$P_{{{comp}}}$  rms={rms:.2g}  bias={bias:+.2g}", fontsize=9)
        ax.set_xlim(lim)
        ax.set_ylim(lim)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel(r"MM strain (jax-md) / bar")
        if ax is axes[0]:
            ax.set_ylabel(r"CHARMM strain / bar")

    fig.colorbar(sc, ax=axes.tolist(), fraction=0.035, pad=0.02, label="t (ps)")
    fig.suptitle(
        r"AR1:500  $P_{ii}=-\frac{1}{V}\partial U/\partial\varepsilon_{ii}$"
        r"  (virial, ortho strain, $t>0$)",
        fontsize=11,
        y=1.02,
    )
    fig.tight_layout()
    IMG.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PNG, dpi=220, bbox_inches="tight")
    fig.savefig(OUT_PNG.with_suffix(".pdf"), bbox_inches="tight")
    print(f"wrote {OUT_PNG}")
    print(f"wrote {OUT_PNG.with_suffix('.pdf')}")


def main() -> int:
    result = evaluate_frames(N_FRAMES)
    rows = result["rows"]
    cuts = result["cuts"]
    payload = {
        "frame": np.array([r["frame"] for r in rows]),
        "t_ps": np.array([r["t_ps"] for r in rows]),
        "L_A": np.array([r["L_A"] for r in rows]),
        "mm_scalar_bar": np.array([r["mm_scalar_bar"] for r in rows]),
        "charmm_scalar_bar": np.array([r["charmm_scalar_bar"] for r in rows]),
        "jax_traj_pvir_bar": np.array([r["jax_traj_pvir_bar"] for r in rows]),
        "cuts_json": np.asarray(
            json.dumps(
                {
                    "cutnb": float(cuts.cutnb),
                    "ctonnb": float(cuts.ctonnb),
                    "ctofnb": float(cuts.ctofnb),
                }
            )
        ),
    }
    for comp in COMPONENTS:
        payload[f"mm_{comp}"] = np.array([r[f"mm_{comp}"] for r in rows])
        payload[f"charmm_{comp}"] = np.array([r[f"charmm_{comp}"] for r in rows])
    OUT_NPZ.parent.mkdir(parents=True, exist_ok=True)
    np.savez(OUT_NPZ, **payload)
    print(f"wrote {OUT_NPZ}")
    plot_parity(rows)
    for comp in COMPONENTS:
        x, y = payload[f"mm_{comp}"], payload[f"charmm_{comp}"]
        m = np.isfinite(x) & np.isfinite(y)
        if m.any():
            print(
                f"  {comp}: rms={np.sqrt(np.mean((y[m]-x[m])**2)):.2f} "
                f"bias={np.mean(y[m]-x[m]):+.2f} bar"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
