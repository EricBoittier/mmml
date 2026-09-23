#!/usr/bin/env python3
"""Argon LJ NpT: density, pressure, temperature across MD backends.

Prefers AR1:500 200 ps artifacts when present (falls back to AR1:108).
Embeds POV-Ray stills at t=0 and t=final (jax-md + PyCHARMM).

Example::

    uv run python scripts/plot_argon_npt_backend_comparison.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import image as mpimg
from matplotlib.gridspec import GridSpec

from mmml.utils.plotting.styles import apply_plot_style, comparison_colors

REPO = Path(__file__).resolve().parents[1]
RUNS = REPO / "artifacts" / "npt_argon_water" / "runs"
IMG_DIR = REPO / "docs" / "images" / "npt_argon_water"
OUT = IMG_DIR / "ar1_90k_backend_density_pressure.png"

NIST_RHO = 1.37860
NIST_P_BAR = 1.3176 * 1.01325
T_TARGET = 90.0
M_AR = 39.948
NA = 6.02214076e23
KB_EV = 8.617333262145e-5  # eV/K

# Prefer campaign-size N=500; fall back to N=108 smoke.
CONFIGS = (
    {
        "n_atoms": 500,
        "L0_A": 28.868571,
        "mm_switch": "7+3.39 Å",
        "jaxmd_dirs": ("ar1_90k_n500_jaxmd_200ps",),
        "pycharmm_dirs": ("ar1_90k_n500_pycharmm_pure_200ps",),
        "pov_before": "ar1_90k_n500_before.png",
        "pov_jax": "ar1_90k_n500_after_jaxmd.png",
        "pov_pch": "ar1_90k_n500_after_pycharmm.png",
        "pov_fallback": "ar1_90k_n500_before.png",
    },
    {
        "n_atoms": 108,
        "L0_A": 17.321,
        "mm_switch": "4+2 Å",
        "jaxmd_dirs": ("ar1_90k_n108_jaxmd_200ps", "ar1_90k_n108_jaxmd"),
        "pycharmm_dirs": (
            "ar1_90k_n108_pycharmm_pure_200ps_longcpt",
            "ar1_90k_n108_pycharmm_pure_200ps",
            "ar1_90k_n108_pycharmm_pure",
        ),
        "pov_before": "ar1_90k_n108_before.png",
        "pov_jax": "ar1_90k_n108_after_jaxmd.png",
        "pov_pch": "ar1_90k_n108_after_pycharmm.png",
        "pov_fallback": "ar1_90k_n108_box.png",
    },
)


def _first_existing(*names: str) -> Path:
    for name in names:
        path = RUNS / name
        if (path / "trajectory.npz").is_file():
            return path
    raise FileNotFoundError(f"none of {names} have trajectory.npz under {RUNS}")


def _pick_config() -> tuple[dict, Path, Path]:
    for cfg in CONFIGS:
        try:
            jax = _first_existing(*cfg["jaxmd_dirs"])
            pch = _first_existing(*cfg["pycharmm_dirs"])
            return cfg, jax, pch
        except FileNotFoundError:
            continue
    raise FileNotFoundError("no AR1 jax-md / PyCHARMM NpT trajectories found")


def _rho_from_volumes(volumes_A3: np.ndarray, n: int) -> np.ndarray:
    V = np.asarray(volumes_A3, dtype=float)
    return n * M_AR / (NA * (V * 1e-24))


def _T_from_kinetic_eV(kinetic_eV: np.ndarray, n: int) -> np.ndarray:
    K = np.asarray(kinetic_eV, dtype=float)
    return 2.0 * K / (3.0 * n * KB_EV)


def load_jaxmd(path: Path, n_atoms: int) -> dict:
    z = np.load(path / "trajectory.npz", allow_pickle=True)
    summary_path = path / "summary.json"
    summary = json.loads(summary_path.read_text()) if summary_path.is_file() else {}
    P = np.asarray(z["pressures_bar"], dtype=float)
    V = np.asarray(z["volumes_A3"], dtype=float)
    n = len(P)
    n_atoms = int(summary.get("n_atoms", n_atoms))
    if "positions" in z.files:
        n_atoms = int(np.asarray(z["positions"]).shape[1])
    ps = float(summary.get("ps", 200.0 if n > 500 else 5.0))
    # Prefer time from frame count when summary missing
    if "summary.json" not in {p.name for p in path.iterdir()} and n > 1:
        # md-system typically saves every 100 steps at dt=1 fs → n frames ≈ ps*10+1
        ps = max(ps, (n - 1) / 10.0)
    t = np.linspace(0.0, ps, n)
    T = _T_from_kinetic_eV(z["kinetic_energies"], n_atoms)
    return {
        "label": "jax-md",
        "t_ps": t,
        "rho": _rho_from_volumes(V, n_atoms),
        "P_bar": P,
        "T_K": T,
        "summary": summary,
        "L_final": float(np.mean(np.diag(np.asarray(z["boxes"][-1], float)))),
        "n_atoms": n_atoms,
    }


def load_pycharmm(path: Path) -> dict:
    z = np.load(path / "trajectory.npz", allow_pickle=True)
    summary_path = path / "summary.json"
    summary = json.loads(summary_path.read_text()) if summary_path.is_file() else {}
    return {
        "label": "PyCHARMM CPT",
        "t_ps": np.asarray(z["t_ps"], dtype=float),
        "rho": np.asarray(z["density_g_cm3"], dtype=float),
        "P_bar": np.asarray(z["pressures_bar"], dtype=float),
        "T_K": (
            np.asarray(z["temperatures_K"], dtype=float)
            if "temperatures_K" in z.files
            else None
        ),
        "summary": summary,
        "L_final": float(np.asarray(z["L_A"], float)[-1]),
        "n_atoms": int(summary.get("n_atoms", 0)),
    }


def _show_pov(ax, path: Path, title: str, *, fallback: Path, scale_xy=(0.51, 0.27)) -> None:
    img = path if path.is_file() else fallback
    if img.is_file():
        ax.imshow(mpimg.imread(img))
        ax.text(
            scale_xy[0],
            scale_xy[1],
            r"10 $\mathrm{\AA}$",
            transform=ax.transAxes,
            ha="center",
            va="top",
            fontsize=8,
            color="0.1",
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.8, "pad": 1.0},
        )
    ax.set_axis_off()
    ax.set_title(title, loc="left", fontsize=9, pad=2)


def _smooth(y: np.ndarray, k: int) -> np.ndarray:
    k = max(1, int(k))
    if k == 1 or len(y) < k:
        return np.asarray(y, dtype=float)
    kernel = np.ones(k) / k
    return np.convolve(np.asarray(y, dtype=float), kernel, mode="valid")


def main() -> int:
    apply_plot_style("icml")
    colors = comparison_colors("icml", n=3)

    cfg, jax_dir, pch_dir = _pick_config()
    n_atoms = int(cfg["n_atoms"])
    L0 = float(cfg["L0_A"])
    jax = load_jaxmd(jax_dir, n_atoms)
    pch = load_pycharmm(pch_dir)
    n_atoms = int(jax.get("n_atoms") or n_atoms)
    mode = str(pch.get("summary", {}).get("mode", ""))
    rho_pch70 = float(np.mean(pch["rho"][int(0.3 * len(pch["rho"])) :]))
    print(f"config AR1:{n_atoms}  L0={L0:.3f}")
    print(f"jax-md: {jax_dir.name} ({len(jax['t_ps'])} frames, L_f={jax['L_final']:.3f})")
    print(f"pycharmm: {pch_dir.name} ({len(pch['t_ps'])} frames, L_f={pch['L_final']:.3f})")

    pov_before = IMG_DIR / cfg["pov_before"]
    pov_jax = IMG_DIR / cfg["pov_jax"]
    pov_pch = IMG_DIR / cfg["pov_pch"]
    pov_fallback = IMG_DIR / cfg["pov_fallback"]

    fig = plt.figure(figsize=(11.0, 7.2))
    gs = GridSpec(
        3,
        3,
        figure=fig,
        width_ratios=[1.0, 1.0, 1.35],
        height_ratios=[1.0, 1.0, 1.0],
        wspace=0.22,
        hspace=0.32,
        left=0.04,
        right=0.98,
        top=0.94,
        bottom=0.08,
    )

    ax_b = fig.add_subplot(gs[0, 0])
    ax_j = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[1, 0])
    ax_c2 = fig.add_subplot(gs[1, 1])
    ax_r = fig.add_subplot(gs[0, 2])
    ax_p = fig.add_subplot(gs[1, 2], sharex=ax_r)
    ax_t = fig.add_subplot(gs[2, 2], sharex=ax_r)
    ax_note = fig.add_subplot(gs[2, 0:2])
    ax_note.axis("off")

    _show_pov(
        ax_b,
        pov_before,
        rf"t = 0,  $L={L0:.2f}\,\mathrm{{\AA}}$",
        fallback=pov_fallback,
    )
    _show_pov(
        ax_j,
        pov_jax,
        rf"jax-md 200 ps,  $L={jax['L_final']:.2f}\,\mathrm{{\AA}}$",
        fallback=pov_fallback,
    )
    if pov_pch.is_file():
        _show_pov(
            ax_c,
            pov_pch,
            rf"PyCHARMM 200 ps,  $L={pch['L_final']:.2f}\,\mathrm{{\AA}}$",
            fallback=pov_fallback,
        )
    else:
        ax_c.axis("off")
        ax_c.set_title(
            rf"PyCHARMM 200 ps,  $L={pch['L_final']:.2f}\,\mathrm{{\AA}}$",
            loc="left",
            fontsize=9,
            pad=2,
        )
        ax_c.text(
            0.5,
            0.5,
            f"L final = {pch['L_final']:.2f} Å\n"
            f"⟨ρ⟩₇₀% = {rho_pch70:.3f} g/cm³",
            transform=ax_c.transAxes,
            ha="center",
            va="center",
            fontsize=11,
            color="0.25",
        )

    ax_c2.axis("off")
    if n_atoms >= 500:
        note = (
            "CHARMM CPT (AR1:500)\n"
            "• 20 × 10 ps DynamicsScript segments\n"
            "• MMML_CPT_DYNAMICS_CHUNK_NSTEP=500k\n"
            "  (no 250-step micro-chunks / reseed)\n"
            "• mild expansion only (ρ≈1.28 vs 1.38)\n"
            "  — far better than AR1:108 gas-like\n"
            "  collapse; residual barostat/P noise."
        )
    elif "long_continuous" in mode:
        note = (
            "CHARMM CPT (this plot)\n"
            "• 20 × 10 ps DynamicsScript segments\n"
            "• MMML_CPT_DYNAMICS_CHUNK_NSTEP=500k\n"
            "  (no 250-step micro-chunks)\n"
            "• in-memory continuation (no iasvel=1)\n"
            "→ continuous velocities, but box still\n"
            "  expands (ρ → gas-like). Barostat /\n"
            "  LJ-cutoff / virial mismatch vs jax-md\n"
            "  remains — not fixed by longer DYNA."
        )
    else:
        note = (
            "CHARMM CPT note\n"
            "• every 2.5 ps outer chunk falls back to\n"
            "  iasvel=1 (Boltzmann reseed)\n"
            "• scratch restart does not restore\n"
            "  Hoover piston internals\n"
            "→ not a continuous NpT trajectory."
        )
    ax_c2.text(
        0.02,
        0.95,
        note,
        transform=ax_c2.transAxes,
        va="top",
        ha="left",
        fontsize=8.5,
        family="monospace",
        linespacing=1.35,
        color="0.15",
    )

    for series, color in ((jax, colors[0]), (pch, colors[1])):
        ax_r.plot(series["t_ps"], series["rho"], color=color, lw=1.5, label=series["label"])
        k = max(5, len(series["P_bar"]) // 40)
        p_s = _smooth(series["P_bar"], k)
        t_s = series["t_ps"][k - 1 :][: len(p_s)]
        ax_p.plot(series["t_ps"], series["P_bar"], color=color, lw=0.5, alpha=0.15, label="_nolegend_")
        ax_p.plot(t_s, p_s, color=color, lw=1.8, label=series["label"])

    ax_r.axhline(NIST_RHO, color="0.35", ls="--", lw=1.0, label="NIST")
    ax_p.axhline(NIST_P_BAR, color="0.35", ls="--", lw=1.0, label="NIST")
    ax_r.set_ylabel(r"density (g/cm$^3$)")
    ax_r.legend(frameon=False, loc="best", fontsize=8)
    ax_r.tick_params(labelbottom=False)
    ax_p.set_ylabel("pressure (bar)")
    ax_p.set_ylim(-200.0, 250.0)
    ax_p.legend(frameon=False, loc="best", fontsize=8)
    ax_p.tick_params(labelbottom=False)

    ax_t.plot(jax["t_ps"], jax["T_K"], color=colors[0], lw=0.6, alpha=0.25, label="_nolegend_")
    kT = max(5, len(jax["T_K"]) // 40)
    tT = jax["t_ps"][kT - 1 :][: len(_smooth(jax["T_K"], kT))]
    ax_t.plot(tT, _smooth(jax["T_K"], kT), color=colors[0], lw=1.8, label="jax-md")
    ax_t.axhline(T_TARGET, color="0.35", ls="--", lw=1.0, label="target 90 K")
    pch_T = pch.get("T_K")
    if pch_T is not None and np.isfinite(pch_T).any():
        ax_t.plot(
            pch["t_ps"],
            pch_T,
            color=colors[1],
            lw=1.6,
            marker="o",
            markersize=3,
            label="PyCHARMM CPT",
        )
    ax_t.set_xlabel("time (ps)")
    ax_t.set_ylabel("T (K)")
    ax_t.set_ylim(60.0, 120.0)
    ax_t.legend(frameon=False, loc="best", fontsize=8)

    ax_note.text(
        0.02,
        0.6,
        rf"AR1:{n_atoms}  ·  90 K  ·  $P_\mathrm{{sat}}=1.335$ bar  ·  mm-switch {cfg['mm_switch']}",
        transform=ax_note.transAxes,
        fontsize=9,
        color="0.25",
        va="center",
    )

    fig.savefig(OUT, dpi=220)
    fig.savefig(OUT.with_suffix(".pdf"))
    print(f"wrote {OUT}")
    print(f"wrote {OUT.with_suffix('.pdf')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
