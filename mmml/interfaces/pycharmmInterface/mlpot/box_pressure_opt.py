"""Cubic-box optimization toward a target pressure (CHARMM-default NpT prep).

Mixes Monte Carlo volume moves, a 1D scalar refine on ``L``, and an optional
short CPT MD refine. Intended as a **pre-hybrid / MM** stage: crystal/IMAGE
rebuilds are unsafe after MLpot registration; certify ``box.json`` then run
hybrid heat/NVE at fixed ``L``.

Pressure evaluation is injected (CHARMM virial, synthetic model, …) so the
core loop stays unit-testable without libcharmm.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from mmml.interfaces.pycharmmInterface.mlpot.mc_density import (
    density_g_cm3_for_box,
    monomer_offsets_from_atoms_per,
    scale_molecule_coms_with_cubic_box,
)
from mmml.utils.geometry_checks import find_worst_intermonomer_overlap

# --- Defaults (named; override via BoxPressureOptConfig) ---------------------

# Metropolis MC volume proposals.
DEFAULT_BOX_PRESSURE_MC_STEPS = 64
DEFAULT_BOX_PRESSURE_MC_STEP_SCALE = 0.04
DEFAULT_BOX_PRESSURE_MC_TEMPERATURE = 1.0  # in pressure-objective units (atm)
DEFAULT_BOX_PRESSURE_MC_MIN_SCALE = 0.70
DEFAULT_BOX_PRESSURE_MC_MAX_SCALE = 1.40
DEFAULT_BOX_PRESSURE_OBJECTIVE_SCALE_ATM = 1.0

# 1D golden-section refine on cubic side after MC.
DEFAULT_BOX_PRESSURE_1D_MAX_ITERS = 24
DEFAULT_BOX_PRESSURE_1D_BRACKET_FRAC = 0.08
DEFAULT_BOX_PRESSURE_1D_TOL_A = 1.0e-3

# Short CPT refine plan (CHARMM Hoover CPT smoke lengths).
DEFAULT_BOX_PRESSURE_CPT_NSTEP = 500
DEFAULT_BOX_PRESSURE_CPT_TIMESTEP_PS = 0.001
DEFAULT_BOX_PRESSURE_CPT_TEMPERATURE_K = 300.0
DEFAULT_BOX_PRESSURE_TARGET_ATM = 1.0
DEFAULT_BOX_PRESSURE_CPT_PGAMMA = 5.0

# Geometry guard during volume proposals.
DEFAULT_BOX_PRESSURE_MIN_INTERMONOMER_A = 0.8

BOX_PRESSURE_OPT_SCHEMA = "mmml.box_pressure_opt.v1"

PressureFn = Callable[[np.ndarray, float], float]
CptRefineFn = Callable[
    [np.ndarray, float],
    tuple[np.ndarray, float, Mapping[str, Any]],
]


@dataclass(frozen=True)
class BoxPressureOptConfig:
    """Resolved settings for pressure-targeted cubic box optimization."""

    target_pressure_atm: float = DEFAULT_BOX_PRESSURE_TARGET_ATM
    temperature_K: float = DEFAULT_BOX_PRESSURE_CPT_TEMPERATURE_K
    mc_steps: int = DEFAULT_BOX_PRESSURE_MC_STEPS
    mc_step_scale: float = DEFAULT_BOX_PRESSURE_MC_STEP_SCALE
    mc_temperature: float = DEFAULT_BOX_PRESSURE_MC_TEMPERATURE
    mc_min_scale: float = DEFAULT_BOX_PRESSURE_MC_MIN_SCALE
    mc_max_scale: float = DEFAULT_BOX_PRESSURE_MC_MAX_SCALE
    objective_scale_atm: float = DEFAULT_BOX_PRESSURE_OBJECTIVE_SCALE_ATM
    min_intermonomer_distance_A: float = DEFAULT_BOX_PRESSURE_MIN_INTERMONOMER_A
    run_1d_refine: bool = True
    refine_1d_max_iters: int = DEFAULT_BOX_PRESSURE_1D_MAX_ITERS
    refine_1d_bracket_frac: float = DEFAULT_BOX_PRESSURE_1D_BRACKET_FRAC
    refine_1d_tol_A: float = DEFAULT_BOX_PRESSURE_1D_TOL_A
    run_cpt_refine: bool = False
    cpt_nstep: int = DEFAULT_BOX_PRESSURE_CPT_NSTEP
    cpt_timestep_ps: float = DEFAULT_BOX_PRESSURE_CPT_TIMESTEP_PS
    cpt_pgamma: float = DEFAULT_BOX_PRESSURE_CPT_PGAMMA
    seed: int = 123
    min_box_side_A: float | None = None
    max_box_side_A: float | None = None


@dataclass(frozen=True)
class McPressureResult:
    """Outcome of pressure-objective Monte Carlo volume moves."""

    ran: bool
    reason: str
    initial_box_A: float
    final_box_A: float
    target_pressure_atm: float
    initial_pressure_atm: float | None
    final_pressure_atm: float | None
    accepted_moves: int = 0
    attempted_moves: int = 0
    best_objective: float | None = None
    min_intermonomer_distance_A: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class Refine1DResult:
    """Outcome of the post-MC 1D box-side refine."""

    ran: bool
    reason: str
    initial_box_A: float
    final_box_A: float
    initial_pressure_atm: float | None
    final_pressure_atm: float | None
    n_evals: int = 0
    best_objective: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class BoxPressureOptResult:
    """Full pressure box-opt outcome + certified ``box.json`` payload fields."""

    status: str
    schema: str = BOX_PRESSURE_OPT_SCHEMA
    composition: str | None = None
    n_molecules: int = 0
    n_atoms: int = 0
    box_side_A: float | None = None
    final_cubic_side_A: float | None = None
    density_g_cm3: float | None = None
    target_pressure_atm: float | None = None
    final_pressure_atm: float | None = None
    temperature_K: float | None = None
    mc: dict[str, Any] | None = None
    refine_1d: dict[str, Any] | None = None
    cpt: dict[str, Any] | None = None
    steps_applied: list[str] = field(default_factory=list)
    message: str = ""
    box_json_path: Path | None = None

    def to_box_json(self) -> dict[str, Any]:
        side = self.box_side_A
        return {
            "schema": self.schema,
            "status": self.status,
            "composition": self.composition,
            "n_molecules": int(self.n_molecules),
            "n_atoms": int(self.n_atoms),
            "box_side_A": side,
            "final_cubic_side_A": (
                self.final_cubic_side_A if self.final_cubic_side_A is not None else side
            ),
            "density_g_cm3": self.density_g_cm3,
            "target_pressure_atm": self.target_pressure_atm,
            "final_pressure_atm": self.final_pressure_atm,
            "temperature_K": self.temperature_K,
            "mc_pressure": self.mc,
            "refine_1d": self.refine_1d,
            "cpt": self.cpt,
            "steps_applied": list(self.steps_applied),
            "message": self.message,
            "generated_at": datetime.now(timezone.utc).isoformat(),
        }


def pressure_objective(
    pressure_atm: float,
    *,
    target_pressure_atm: float,
    scale_atm: float = DEFAULT_BOX_PRESSURE_OBJECTIVE_SCALE_ATM,
) -> float:
    """Non-negative objective: ``|P - P_target| / scale`` (scale in atm)."""
    scale = float(scale_atm)
    if scale <= 0.0:
        raise ValueError(f"objective scale_atm must be positive, got {scale}")
    return abs(float(pressure_atm) - float(target_pressure_atm)) / scale


def build_cpt_box_refine_dynamics_kw(
    config: BoxPressureOptConfig,
) -> dict[str, Any]:
    """Serializable CHARMM CPT keyword plan for a short pressure refine."""
    from mmml.interfaces.pycharmmInterface.mlpot.pressure_tensor import (
        apply_npt_pressure_reference,
    )

    kw: dict[str, Any] = {
        "nstep": int(config.cpt_nstep),
        "timestep": float(config.cpt_timestep_ps),
        "finalt": float(config.temperature_K),
        "firstt": float(config.temperature_K),
        "cpt": True,
        "pgamma": float(config.cpt_pgamma),
        "thermostat": "hoover",
    }
    apply_npt_pressure_reference(
        kw,
        pref=float(config.target_pressure_atm),
        pressure_tensor=None,
    )
    return kw


def write_box_pressure_opt_json(
    result: BoxPressureOptResult,
    path: Path | str,
) -> Path:
    """Write certified ``box.json`` (includes ``final_cubic_side_A`` alias)."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as fh:
        json.dump(result.to_box_json(), fh, indent=2, sort_keys=True)
        fh.write("\n")
    result.box_json_path = out.resolve()
    return result.box_json_path


def run_mc_pressure_box_moves(
    positions: np.ndarray,
    *,
    atoms_per_list: Sequence[int],
    box_side_A: float,
    pressure_fn: PressureFn,
    config: BoxPressureOptConfig,
) -> tuple[np.ndarray, float, McPressureResult]:
    """Metropolis volume moves minimizing :func:`pressure_objective`."""
    pos = np.asarray(positions, dtype=np.float64).reshape(-1, 3)
    initial_L = float(box_side_A)
    if initial_L <= 0.0:
        raise ValueError(f"box_side_A must be positive, got {box_side_A}")
    offsets = monomer_offsets_from_atoms_per(atoms_per_list)
    if int(offsets[-1]) != int(pos.shape[0]):
        raise ValueError(
            f"atoms_per_list sums to {int(offsets[-1])}, but positions has "
            f"{pos.shape[0]} rows"
        )

    steps = int(config.mc_steps)
    if steps <= 0:
        p0 = float(pressure_fn(pos, initial_L))
        return pos, initial_L, McPressureResult(
            ran=False,
            reason="zero_steps",
            initial_box_A=initial_L,
            final_box_A=initial_L,
            target_pressure_atm=float(config.target_pressure_atm),
            initial_pressure_atm=p0,
            final_pressure_atm=p0,
        )

    min_scale = float(config.mc_min_scale)
    max_scale = float(config.mc_max_scale)
    if min_scale <= 0.0 or max_scale <= 0.0 or min_scale > max_scale:
        raise ValueError("mc_min_scale/mc_max_scale must be positive with min <= max")
    min_L = initial_L * min_scale
    max_L = initial_L * max_scale
    if config.min_box_side_A is not None:
        min_L = max(min_L, float(config.min_box_side_A))
    if config.max_box_side_A is not None:
        max_L = min(max_L, float(config.max_box_side_A))
    if min_L > max_L:
        min_L = max_L

    rng = np.random.default_rng(int(config.seed))
    step_scale = max(0.0, float(config.mc_step_scale))
    temperature = max(1.0e-12, float(config.mc_temperature))
    min_contact = float(config.min_intermonomer_distance_A)
    target_p = float(config.target_pressure_atm)
    scale_atm = float(config.objective_scale_atm)

    current_L = initial_L
    current_pos = pos.copy()
    current_p = float(pressure_fn(current_pos, current_L))
    current_obj = pressure_objective(
        current_p, target_pressure_atm=target_p, scale_atm=scale_atm
    )
    best_L = current_L
    best_pos = current_pos.copy()
    best_p = current_p
    best_obj = current_obj
    best_contact: float | None = None
    accepted = 0

    if len(offsets) > 2:
        cell = np.diag([current_L, current_L, current_L])
        best_contact, _ = find_worst_intermonomer_overlap(current_pos, offsets, cell=cell)

    for _ in range(steps):
        proposal_log_L = np.log(current_L)
        if step_scale > 0.0:
            proposal_log_L += float(rng.normal(0.0, step_scale))
        proposal_L = float(np.clip(np.exp(proposal_log_L), min_L, max_L))
        if abs(proposal_L - current_L) < 1.0e-12:
            continue
        proposal_pos = scale_molecule_coms_with_cubic_box(
            current_pos,
            offsets,
            old_box_A=current_L,
            new_box_A=proposal_L,
        )
        contact = float("inf")
        if len(offsets) > 2 and min_contact > 0.0:
            contact, _ = find_worst_intermonomer_overlap(
                proposal_pos,
                offsets,
                cell=np.diag([proposal_L, proposal_L, proposal_L]),
            )
            if contact < min_contact:
                continue
        proposal_p = float(pressure_fn(proposal_pos, proposal_L))
        proposal_obj = pressure_objective(
            proposal_p, target_pressure_atm=target_p, scale_atm=scale_atm
        )
        delta = proposal_obj - current_obj
        if delta <= 0.0 or float(rng.random()) < float(np.exp(-delta / temperature)):
            current_L = proposal_L
            current_pos = proposal_pos
            current_p = proposal_p
            current_obj = proposal_obj
            accepted += 1
        if proposal_obj < best_obj:
            best_L = proposal_L
            best_pos = proposal_pos
            best_p = proposal_p
            best_obj = proposal_obj
            best_contact = None if not np.isfinite(contact) else float(contact)

    return best_pos, best_L, McPressureResult(
        ran=True,
        reason="pressure_mc",
        initial_box_A=initial_L,
        final_box_A=best_L,
        target_pressure_atm=target_p,
        initial_pressure_atm=float(pressure_fn(pos, initial_L)),
        final_pressure_atm=best_p,
        accepted_moves=accepted,
        attempted_moves=steps,
        best_objective=best_obj,
        min_intermonomer_distance_A=best_contact,
    )


def refine_box_side_1d(
    positions: np.ndarray,
    *,
    atoms_per_list: Sequence[int],
    box_side_A: float,
    pressure_fn: PressureFn,
    config: BoxPressureOptConfig,
) -> tuple[np.ndarray, float, Refine1DResult]:
    """Golden-section minimize of ``(P(L)-P_target)^2`` on cubic side.

    Derivative-free 1D search (BFGS-like role for scalar ``L``). Scales molecule
    COMs when evaluating trial sides.
    """
    pos = np.asarray(positions, dtype=np.float64).reshape(-1, 3)
    L0 = float(box_side_A)
    if not bool(config.run_1d_refine):
        p0 = float(pressure_fn(pos, L0))
        return pos, L0, Refine1DResult(
            ran=False,
            reason="disabled",
            initial_box_A=L0,
            final_box_A=L0,
            initial_pressure_atm=p0,
            final_pressure_atm=p0,
        )

    offsets = monomer_offsets_from_atoms_per(atoms_per_list)
    target_p = float(config.target_pressure_atm)
    scale_atm = float(config.objective_scale_atm)
    bracket = max(1.0e-6, float(config.refine_1d_bracket_frac))
    lo = L0 * (1.0 - bracket)
    hi = L0 * (1.0 + bracket)
    if config.min_box_side_A is not None:
        lo = max(lo, float(config.min_box_side_A))
    if config.max_box_side_A is not None:
        hi = min(hi, float(config.max_box_side_A))
    if hi <= lo:
        p0 = float(pressure_fn(pos, L0))
        return pos, L0, Refine1DResult(
            ran=False,
            reason="empty_bracket",
            initial_box_A=L0,
            final_box_A=L0,
            initial_pressure_atm=p0,
            final_pressure_atm=p0,
        )

    n_evals = 0

    def _eval(L: float) -> tuple[float, np.ndarray, float]:
        nonlocal n_evals
        trial_pos = scale_molecule_coms_with_cubic_box(
            pos, offsets, old_box_A=L0, new_box_A=float(L)
        )
        p = float(pressure_fn(trial_pos, float(L)))
        n_evals += 1
        obj = pressure_objective(p, target_pressure_atm=target_p, scale_atm=scale_atm)
        return obj, trial_pos, p

    # Golden-section search on [lo, hi].
    phi = 0.5 * (3.0 - np.sqrt(5.0))
    a, b = float(lo), float(hi)
    c = a + phi * (b - a)
    d = a + (1.0 - phi) * (b - a)
    fc, pos_c, p_c = _eval(c)
    fd, pos_d, p_d = _eval(d)
    best_obj, best_L, best_pos, best_p = fc, c, pos_c, p_c
    if fd < best_obj:
        best_obj, best_L, best_pos, best_p = fd, d, pos_d, p_d

    tol = float(config.refine_1d_tol_A)
    for _ in range(max(1, int(config.refine_1d_max_iters))):
        if abs(b - a) <= tol:
            break
        if fc < fd:
            b, d, fd, pos_d, p_d = d, c, fc, pos_c, p_c
            c = a + phi * (b - a)
            fc, pos_c, p_c = _eval(c)
            if fc < best_obj:
                best_obj, best_L, best_pos, best_p = fc, c, pos_c, p_c
        else:
            a, c, fc, pos_c, p_c = c, d, fd, pos_d, p_d
            d = a + (1.0 - phi) * (b - a)
            fd, pos_d, p_d = _eval(d)
            if fd < best_obj:
                best_obj, best_L, best_pos, best_p = fd, d, pos_d, p_d

    p0 = float(pressure_fn(pos, L0))
    return best_pos, float(best_L), Refine1DResult(
        ran=True,
        reason="golden_section",
        initial_box_A=L0,
        final_box_A=float(best_L),
        initial_pressure_atm=p0,
        final_pressure_atm=float(best_p),
        n_evals=n_evals,
        best_objective=float(best_obj),
    )


def run_box_pressure_opt(
    positions: np.ndarray,
    *,
    atoms_per_list: Sequence[int],
    box_side_A: float,
    pressure_fn: PressureFn,
    config: BoxPressureOptConfig | None = None,
    composition: Mapping[str, int] | str | None = None,
    cpt_refine_fn: CptRefineFn | None = None,
    output_dir: Path | str | None = None,
) -> tuple[np.ndarray, float, BoxPressureOptResult]:
    """MC → 1D refine → optional CPT; write ``box.json`` when ``output_dir`` set."""
    cfg = config or BoxPressureOptConfig()
    steps: list[str] = []
    pos = np.asarray(positions, dtype=np.float64).reshape(-1, 3)
    L = float(box_side_A)

    pos, L, mc = run_mc_pressure_box_moves(
        pos,
        atoms_per_list=atoms_per_list,
        box_side_A=L,
        pressure_fn=pressure_fn,
        config=cfg,
    )
    if mc.ran:
        steps.append("mc_pressure")

    refine: Refine1DResult | None = None
    if cfg.run_1d_refine:
        pos, L, refine = refine_box_side_1d(
            pos,
            atoms_per_list=atoms_per_list,
            box_side_A=L,
            pressure_fn=pressure_fn,
            config=cfg,
        )
        if refine.ran:
            steps.append("refine_1d")

    cpt_summary: dict[str, Any] | None = None
    if cfg.run_cpt_refine:
        if cpt_refine_fn is None:
            cpt_summary = {
                "ran": False,
                "reason": "no_cpt_refine_fn",
                "plan": build_cpt_box_refine_dynamics_kw(cfg),
            }
        else:
            pos, L, cpt_summary = cpt_refine_fn(pos, L)
            cpt_summary = dict(cpt_summary)
            cpt_summary.setdefault("ran", True)
            cpt_summary.setdefault("plan", build_cpt_box_refine_dynamics_kw(cfg))
            steps.append("cpt_refine")

    final_p = float(pressure_fn(pos, L))
    comp_str: str | None
    comp_dict: dict[str, int] | None
    if isinstance(composition, str):
        comp_str = composition
        from mmml.interfaces.pycharmmInterface.mlpot.box_sizing import (
            parse_composition_dict,
        )

        comp_dict = parse_composition_dict(composition)
    elif composition is None:
        comp_str = None
        comp_dict = None
    else:
        comp_dict = {str(k): int(v) for k, v in composition.items()}
        comp_str = ":".join(f"{k}:{v}" for k, v in sorted(comp_dict.items()))

    density = None
    if comp_dict is not None:
        try:
            density = density_g_cm3_for_box(comp_dict, L)
        except ValueError:
            density = None

    n_mol = int(sum(comp_dict.values())) if comp_dict else int(len(atoms_per_list))
    result = BoxPressureOptResult(
        status="pass",
        composition=comp_str,
        n_molecules=n_mol,
        n_atoms=int(pos.shape[0]),
        box_side_A=float(L),
        final_cubic_side_A=float(L),
        density_g_cm3=density,
        target_pressure_atm=float(cfg.target_pressure_atm),
        final_pressure_atm=final_p,
        temperature_K=float(cfg.temperature_K),
        mc=mc.to_dict(),
        refine_1d=None if refine is None else refine.to_dict(),
        cpt=cpt_summary,
        steps_applied=steps,
        message=(
            f"box pressure opt: L={L:.4f} Å, P≈{final_p:.4g} atm "
            f"(target {cfg.target_pressure_atm:g} atm)"
        ),
    )
    if output_dir is not None:
        write_box_pressure_opt_json(result, Path(output_dir) / "box.json")
    return pos, float(L), result


def synthetic_inverse_cube_pressure_fn(
    *,
    target_box_side_A: float,
    target_pressure_atm: float = DEFAULT_BOX_PRESSURE_TARGET_ATM,
) -> PressureFn:
    """``P = P_target * (L_target / L)^3`` — offline/unit-test pressure model.

    Calibrate so the known liquid-box side is the pressure optimum. Used when
    CHARMM virial pressure is unavailable (CI / preflight).
    """
    L_star = float(target_box_side_A)
    p_star = float(target_pressure_atm)
    if L_star <= 0.0 or p_star == 0.0:
        raise ValueError("target_box_side_A and target_pressure_atm must be nonzero")
    k = p_star * (L_star**3)

    def _fn(_positions: np.ndarray, box_side_A: float) -> float:
        return float(k) / (float(box_side_A) ** 3)

    return _fn


def run_box_pressure_opt_from_box_json(
    liquid_box_dir: Path | str,
    *,
    output_dir: Path | str | None = None,
    config: BoxPressureOptConfig | None = None,
    pressure_fn: PressureFn | None = None,
    cpt_refine_fn: CptRefineFn | None = None,
) -> BoxPressureOptResult:
    """Load certified ``box.json`` (+ optional CRD) and run pressure box-opt.

    When ``pressure_fn`` is omitted, uses
    :func:`synthetic_inverse_cube_pressure_fn` calibrated to the certified side
    (pipeline smoke). Pass :func:`charmm_pressure_fn` for real virial pressure.
    """
    root = Path(liquid_box_dir).expanduser().resolve()
    box_path = root / "box.json"
    if not box_path.is_file():
        raise FileNotFoundError(f"missing certified box.json under {root}")
    payload = json.loads(box_path.read_text(encoding="utf-8"))
    side = payload.get("box_side_A", payload.get("final_cubic_side_A"))
    if side is None:
        raise ValueError(f"{box_path} has no box_side_A / final_cubic_side_A")
    L0 = float(side)
    cfg = config or BoxPressureOptConfig()
    n_atoms = int(payload.get("n_atoms") or 0)
    n_mol = int(payload.get("n_molecules") or 0)
    composition = payload.get("composition")
    if n_mol <= 0 and isinstance(composition, str) and ":" in composition:
        from mmml.interfaces.pycharmmInterface.mlpot.box_sizing import (
            parse_composition_dict,
        )

        comp = parse_composition_dict(composition) or {}
        n_mol = int(sum(comp.values()))
    if n_atoms <= 0 and n_mol > 0:
        # TIP3-like default when only molecule count is known.
        n_atoms = 3 * n_mol
    if n_mol <= 0 or n_atoms <= 0 or n_atoms % n_mol != 0:
        raise ValueError(
            f"cannot infer atoms_per_list from box.json (n_mol={n_mol}, n_atoms={n_atoms})"
        )
    atoms_each = n_atoms // n_mol
    atoms_per_list = [atoms_each] * n_mol

    # Prefer real coordinates when model.crd exists; else a loose lattice for MC.
    positions = None
    crd = root / "model.crd"
    if not crd.is_file():
        artifacts = payload.get("artifacts") or {}
        crd_raw = artifacts.get("model_crd")
        if crd_raw:
            crd = Path(str(crd_raw))
    if crd.is_file():
        try:
            from mmml.interfaces.pycharmmInterface.mlpot.dynamics_validation import (
                read_crd_coordinates,
            )

            positions = read_crd_coordinates(crd)
        except Exception:
            positions = None
    if positions is None:
        positions = _placeholder_monomer_lattice(atoms_per_list, L0)

    p_fn = pressure_fn or synthetic_inverse_cube_pressure_fn(
        target_box_side_A=L0,
        target_pressure_atm=float(cfg.target_pressure_atm),
    )
    out = Path(output_dir) if output_dir is not None else root / "box_pressure_opt"
    _pos, _L, result = run_box_pressure_opt(
        positions,
        atoms_per_list=atoms_per_list,
        box_side_A=L0,
        pressure_fn=p_fn,
        config=cfg,
        composition=composition if isinstance(composition, (str, dict)) else None,
        cpt_refine_fn=cpt_refine_fn,
        output_dir=out,
    )
    return result


def _placeholder_monomer_lattice(
    atoms_per_list: Sequence[int],
    box_side_A: float,
) -> np.ndarray:
    """Deterministic non-overlapping monomer COMs for offline smoke (no CRD)."""
    counts = [int(x) for x in atoms_per_list]
    n_mol = len(counts)
    L = float(box_side_A)
    n_side = int(np.ceil(n_mol ** (1.0 / 3.0)))
    spacing = L / max(n_side, 1)
    frames: list[np.ndarray] = []
    for mi, n_at in enumerate(counts):
        ix = mi % n_side
        iy = (mi // n_side) % n_side
        iz = mi // (n_side * n_side)
        com = np.array(
            [(ix + 0.5) * spacing, (iy + 0.5) * spacing, (iz + 0.5) * spacing],
            dtype=np.float64,
        )
        local = np.zeros((n_at, 3), dtype=np.float64)
        if n_at >= 1:
            local[0] = 0.0
        if n_at >= 2:
            local[1] = [0.96, 0.0, 0.0]
        if n_at >= 3:
            local[2] = [-0.24, 0.93, 0.0]
        for j in range(3, n_at):
            local[j] = [0.2 * j, 0.1 * j, 0.0]
        frames.append(local + com)
    return np.vstack(frames)


def charmm_pressure_fn(
    *,
    temperature_K: float,
    mlpot_ctx: Any | None = None,
) -> PressureFn:
    """Build a ``pressure_fn`` that syncs coords into CHARMM and reads ``PRSI``.

    Caller must already have CHARMM PBC installed. Each evaluation pushes
    ``positions`` and cubic ``L`` then reads the virial pressure.
    """

    def _fn(positions: np.ndarray, box_side_A: float) -> float:
        from mmml.interfaces.pycharmmInterface.mlpot.pbc_env import (
            push_charmm_cubic_box_side_A,
        )
        from mmml.interfaces.pycharmmInterface.mlpot.pressure_tensor import (
            read_instantaneous_scalar_pressure_atm,
        )
        from mmml.interfaces.pycharmmInterface.mlpot.setup import sync_charmm_positions

        push_charmm_cubic_box_side_A(float(box_side_A), quiet=True)
        sync_charmm_positions(np.asarray(positions, dtype=np.float64))
        # temperature_K reserved for future kinetic contribution / reporting
        _ = temperature_K
        return read_instantaneous_scalar_pressure_atm(
            refresh_energy=True,
            mlpot_ctx=mlpot_ctx,
            quiet=True,
        )

    return _fn
