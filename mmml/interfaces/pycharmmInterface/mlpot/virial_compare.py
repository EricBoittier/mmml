"""Three-way virial pressure comparison for CHARMM CPT vs hybrid ML/MM.

CHARMM CPT integrates ``PRSI`` from ``VIRAL`` on accumulated ``dx/dy/dz``
(including MLpot USER gradients). That is the **atomic** virial ``Σ F·r``.
JAX-MD NPT (after #249) uses **strain** ``P = -dE/dV`` at fixed fractional
coordinates. Those agree for a pair potential with no wrap; they disagree when
energy depends on ``L`` through MIC / image lists / a COM switch.

Units: energies eV, lengths Å, pressure atm.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Callable, Literal

import numpy as np

# eV/Å^3 → Pa → atm (same chain as scripts/validate_virial_vs_charmm.py)
EV_A3_TO_PA = 1.602176634e-19 / 1e-30
PA_TO_ATM = 1.0 / 101325.0
EV_A3_TO_ATM = EV_A3_TO_PA * PA_TO_ATM
KCAL_MOL_A_TO_EV_A = 1.0 / 23.060547830619026

Hypothesis = Literal["H1", "H2", "H3", "H4", "ambiguous"]


def virial_pressure_atm(
    forces_ev_a: np.ndarray,
    positions_a: np.ndarray,
    volume_a3: float,
    kinetic_ev: float = 0.0,
) -> float:
    """P = (2 KE + Σ F·r) / (3 V), in atm (CHARMM atomic-virial convention)."""
    f = np.asarray(forces_ev_a, dtype=np.float64).reshape(-1, 3)
    r = np.asarray(positions_a, dtype=np.float64).reshape(-1, 3)
    if f.shape != r.shape:
        raise ValueError(f"force/position shape mismatch: {f.shape} vs {r.shape}")
    virial = float(np.sum(f * r))
    p_ev_a3 = (2.0 * float(kinetic_ev) + virial) / (3.0 * float(volume_a3))
    return p_ev_a3 * EV_A3_TO_ATM


def wrap_delta(dr: np.ndarray, box_side: float) -> np.ndarray:
    """Minimum-image Cartesian displacement on a cubic cell."""
    l = float(box_side)
    delta = np.asarray(dr, dtype=np.float64)
    return delta - l * np.round(delta / l)


def mic_harmonic_energy_ev(
    positions_a: np.ndarray,
    box_side: float,
    *,
    k: float = 1.0,
) -> float:
    """Two-particle MIC harmonic spring, E = ½ k |r₁₂^MIC|² (eV)."""
    pos = np.asarray(positions_a, dtype=np.float64).reshape(-1, 3)
    if pos.shape[0] < 2:
        raise ValueError("mic_harmonic_energy_ev needs at least two atoms")
    d = wrap_delta(pos[1] - pos[0], box_side)
    return 0.5 * float(k) * float(np.dot(d, d))


def mic_harmonic_forces_stopgrad_ev(
    positions_a: np.ndarray,
    box_side: float,
    *,
    k: float = 1.0,
) -> np.ndarray:
    """Atomic forces with the lattice shift treated as constant (stop-grad wrap).

    Same force convention CHARMM ``VIRAL`` sees: ``F = -∇_r E`` at fixed ``L``,
    no ``d(shift)/dr`` term.
    """
    pos = np.asarray(positions_a, dtype=np.float64).reshape(-1, 3)
    l = float(box_side)
    n = np.round((pos[1] - pos[0]) / l)
    d = pos[1] - pos[0] - l * n
    f0 = float(k) * d
    f1 = -float(k) * d
    out = np.zeros_like(pos)
    out[0] = f0
    out[1] = f1
    return out


def strain_pressure_atm(
    energy_ev_of_real: Callable[[np.ndarray, float], float],
    frac: np.ndarray,
    box_side: float,
    *,
    rel_dv: float = 1.0e-4,
) -> float:
    """P_strain = -dE/dV (atm) at fixed fractional coordinates.

    Isotropic ``V → V(1 ± δ)``, ``L → L (V'/V)^{1/3}``. Same reference as
    ``jaxmd_runner.independent_dE_dV``.
    """
    s = np.asarray(frac, dtype=np.float64).reshape(-1, 3)
    l = float(box_side)
    volume = l**3
    energies: list[float] = []
    for sgn in (1.0, -1.0):
        vol_s = volume * (1.0 + sgn * float(rel_dv))
        side_s = l * (vol_s / volume) ** (1.0 / 3.0)
        energies.append(float(energy_ev_of_real(s * side_s, side_s)))
    d_e_d_v = (energies[0] - energies[1]) / (2.0 * float(rel_dv) * volume)
    return -d_e_d_v * EV_A3_TO_ATM


def pressures_agree(
    a: float,
    b: float,
    *,
    rel: float = 0.15,
    abs_atm: float = 50.0,
) -> bool:
    """True when two pressures match within a few percent or ``abs_atm``."""
    scale = max(abs(float(a)), abs(float(b)), 1.0)
    return abs(float(a) - float(b)) <= max(float(abs_atm), float(rel) * scale)


def classify_cpt_ml_virial_hypothesis(
    *,
    p_prsi_atm: float,
    p_atomic_atm: float,
    p_strain_atm: float,
    liquid_atm: float = 50.0,
    large_atm: float = 200.0,
) -> tuple[Hypothesis, str]:
    """Map the three-way comparison onto the CHARMM CPT diagnostic table."""
    prsi = float(p_prsi_atm)
    atomic = float(p_atomic_atm)
    strain = float(p_strain_atm)
    prsi_atomic = pressures_agree(prsi, atomic)
    atomic_strain = pressures_agree(atomic, strain)
    if abs(prsi) < liquid_atm and abs(atomic) > large_atm:
        return (
            "H1",
            "PRSI ≈ 0 while Σ F·r is large: USER gradients may not reach VIRAL",
        )
    if prsi_atomic and not atomic_strain:
        extra = ""
        if abs(strain) < liquid_atm:
            extra = "; P_strain is near 1 atm (CPT follows the wrong pressure)"
        return (
            "H2",
            "PRSI ≈ P_atomic ≠ P_strain: atomic virial is not -dE/dV" + extra,
        )
    if prsi_atomic and atomic_strain and abs(strain) > large_atm:
        return (
            "H3",
            "All three pressures agree and are ≫ 1 atm: Hamiltonian wants a larger box",
        )
    if abs(strain) < liquid_atm and abs(prsi) < liquid_atm:
        return (
            "H4",
            "P_strain and PRSI are both near 1 atm: not a virial-convention walk",
        )
    return (
        "ambiguous",
        "Pressures do not match a single row of the diagnostic table",
    )


def wrap_positions_to_frac(positions_a: np.ndarray, box_side: float) -> np.ndarray:
    """Fractional coordinates in [0, 1) for an isotropic strain FD."""
    l = float(box_side)
    pos = np.asarray(positions_a, dtype=np.float64).reshape(-1, 3)
    return np.mod(pos / l, 1.0)


def report_to_json(report: dict[str, Any]) -> str:
    return json.dumps(report, indent=2)


def collect_toy_mic_pressures(
    frac: np.ndarray,
    box_side: float,
    *,
    k: float = 1.0,
    rel_dv: float = 1.0e-4,
) -> dict[str, float]:
    """Atomic vs strain pressure for the MIC harmonic toy (no CHARMM)."""
    l = float(box_side)
    pos = np.asarray(frac, dtype=np.float64).reshape(-1, 3) * l
    forces = mic_harmonic_forces_stopgrad_ev(pos, l, k=k)
    p_atomic = virial_pressure_atm(forces, pos, l**3)
    p_strain = strain_pressure_atm(
        lambda r, side: mic_harmonic_energy_ev(r, side, k=k),
        np.asarray(frac, dtype=np.float64),
        l,
        rel_dv=rel_dv,
    )
    return {
        "p_atomic_atm": float(p_atomic),
        "p_strain_atm": float(p_strain),
        "energy_ev": float(mic_harmonic_energy_ev(pos, l, k=k)),
    }


def _charmm_energy_getter(name: str) -> float | None:
    import pycharmm.lingo as lingo

    try:
        val = float(lingo.get_energy_value(name))
    except Exception:
        return None
    return val if math.isfinite(val) else None


def collect_live_cpt_ml_virial_report(
    mlpot_ctx: Any,
    *,
    box_side: float,
    rel_dv: float = 1.0e-4,
) -> dict[str, Any]:
    """PRSI / atomic / strain / VIRE / VIRI after a registered MLpot ENER FORCE."""
    from mmml.interfaces.pycharmmInterface.mlpot.cli_common import (
        mlpot_spherical_energy_forces_ev_angstrom,
        refresh_mlpot_energy_and_grms,
    )
    from mmml.interfaces.pycharmmInterface.mlpot.pressure_tensor import (
        read_instantaneous_scalar_pressure_atm,
    )
    from mmml.interfaces.pycharmmInterface.mlpot.setup import (
        get_charmm_forces_array,
        get_charmm_positions_array,
    )

    refresh_mlpot_energy_and_grms(
        mlpot_ctx,
        context="CPT ML virial diagnose",
        silent_charmm=False,
    )
    pos = np.asarray(get_charmm_positions_array(), dtype=np.float64)
    forces_kcal = np.asarray(get_charmm_forces_array(), dtype=np.float64).reshape(-1, 3)
    forces_ev = forces_kcal * KCAL_MOL_A_TO_EV_A
    volume = float(box_side) ** 3
    p_atomic = virial_pressure_atm(forces_ev, pos, volume, kinetic_ev=0.0)
    p_prsi = float(
        read_instantaneous_scalar_pressure_atm(
            refresh_energy=False,
            mlpot_ctx=mlpot_ctx,
            quiet=True,
        )
    )
    vire = _charmm_energy_getter("VIRE")
    viri = _charmm_energy_getter("VIRI")

    p_strain: float | None = None
    strain_error: str | None = None
    frac = wrap_positions_to_frac(pos, box_side)

    def _energy_ev(real_pos: np.ndarray, side: float) -> float:
        out = mlpot_spherical_energy_forces_ev_angstrom(
            mlpot_ctx.pyCModel,
            positions=real_pos,
            use_pbc=True,
            box_A=float(side),
        )
        if out is None:
            raise RuntimeError(
                "hybrid spherical_fn unavailable; cannot evaluate P_strain"
            )
        return float(out[0])

    try:
        p_strain = float(strain_pressure_atm(_energy_ev, frac, box_side, rel_dv=rel_dv))
    except Exception as exc:
        strain_error = f"{type(exc).__name__}: {exc}"

    hypothesis: Hypothesis = "ambiguous"
    reason = "P_strain unavailable"
    if p_strain is not None:
        hypothesis, reason = classify_cpt_ml_virial_hypothesis(
            p_prsi_atm=p_prsi,
            p_atomic_atm=p_atomic,
            p_strain_atm=p_strain,
        )

    return {
        "n_atoms": int(pos.shape[0]),
        "box_side_A": float(box_side),
        "volume_A3": volume,
        "p_prsi_atm": p_prsi,
        "p_atomic_atm": p_atomic,
        "p_strain_atm": p_strain,
        "vire_kcal": vire,
        "viri_kcal": viri,
        "virial_sum_F_dot_r_eV": float(np.sum(forces_ev * pos)),
        "hypothesis": hypothesis,
        "reason": reason,
        "strain_error": strain_error,
        "rel_dv": float(rel_dv),
    }


def run_live_cpt_ml_virial(
    *,
    psf: Path,
    crd: Path,
    checkpoint: Path,
    box_side: float,
    composition: str,
    mm_switch_width: float,
    continue_from: Path | None = None,
    output_dir: Path | None = None,
    rel_dv: float = 1.0e-4,
) -> dict[str, Any]:
    """Register the campaign hybrid, ENER FORCE once, return the three pressures.

    No CPT / ``dyna``. Strain FD uses the JAX hybrid energy at a scaled box so
    CHARMM's crystal is not rebuilt after MLpot registration.
    """
    from mmml.cli.run.md_handoff import load_handoff, set_handoff_in
    from mmml.cli.run.md_system import build_parser
    from mmml.interfaces.pycharmmInterface.mlpot.pbc_env import setup_charmm_environment
    from mmml.interfaces.pycharmmInterface.mlpot.run_workflow import (
        _register_mlpot_context,
    )
    from mmml.interfaces.pycharmmInterface.mlpot.setup import (
        get_charmm_positions_array,
        load_cluster_from_artifacts,
        sync_charmm_positions,
    )
    from mmml.interfaces.pycharmmInterface.mlpot.staged_workflow import (
        _load_or_build_cluster,
    )
    from mmml.interfaces.pycharmmInterface.utils import get_Z_from_psf

    out = Path(output_dir) if output_dir is not None else Path.cwd() / "cpt_ml_virial"
    out.mkdir(parents=True, exist_ok=True)

    argv = [
        "--backend",
        "pycharmm",
        "--setup",
        "pbc_npt",
        "--from-psf",
        str(Path(psf).expanduser().resolve()),
        "--from-crd",
        str(Path(crd).expanduser().resolve()),
        "--skip-cluster-build",
        "--box-size",
        str(float(box_side)),
        "--composition",
        str(composition),
        "--checkpoint",
        str(Path(checkpoint).expanduser().resolve()),
        "--mm-switch-width",
        str(float(mm_switch_width)),
        "--no-calculator-pre-minimize",
        "--no-charmm-pre-minimize",
        "--no-monomer-physnet-mini",
        "--no-mc-density-equalize",
        "--md-stages",
        "equi",
        "--output-dir",
        str(out),
        "--job-name",
        "cpt_ml_virial",
    ]
    if continue_from is not None:
        argv.extend(["--continue-from", str(Path(continue_from).expanduser().resolve())])

    args = build_parser().parse_args(argv)
    if continue_from is not None:
        set_handoff_in(load_handoff(Path(continue_from).expanduser().resolve()))

    import mmml.interfaces.pycharmmInterface.import_pycharmm  # noqa: F401

    z, r, n_mol, _tag = _load_or_build_cluster(args)
    if z is None or len(z) == 0:
        z, r, n_mol, _tag = load_cluster_from_artifacts(args)
    setup_charmm_environment(
        use_pbc=True,
        cubic_box_side_A=float(box_side),
        workflow_args=args,
    )
    if continue_from is not None:
        from mmml.cli.run.md_handoff import get_handoff_in

        ho = get_handoff_in()
        if ho is not None:
            r = np.asarray(ho.positions, dtype=np.float64)
    sync_charmm_positions(np.asarray(r, dtype=np.float64))
    z_live = np.asarray(get_Z_from_psf(), dtype=int)
    if z_live.size == int(np.asarray(r).shape[0]):
        z = z_live
    pos = get_charmm_positions_array()
    ctx = _register_mlpot_context(
        np.asarray(z, dtype=int),
        np.asarray(pos, dtype=float),
        Path(checkpoint).expanduser().resolve(),
        int(pos.shape[0]),
        int(n_mol),
        cubic_box_side_A=float(box_side),
        mlpot_use_pbc=True,
        verbose=True,
        args=args,
        topology_psf=Path(psf).expanduser().resolve(),
    )
    try:
        report = collect_live_cpt_ml_virial_report(
            ctx,
            box_side=float(box_side),
            rel_dv=rel_dv,
        )
    finally:
        ctx.unset()
    report["composition"] = str(composition)
    report["mm_switch_width"] = float(mm_switch_width)
    report["checkpoint"] = str(Path(checkpoint).expanduser().resolve())
    report["continue_from"] = (
        str(Path(continue_from).expanduser().resolve()) if continue_from else None
    )
    return report
