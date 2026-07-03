"""Compare DCM MP2 reference NPZ geometries to CHARMM / JAX MM energies and forces."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from mmml.analysis.npz_comparison import compute_scalar_metrics
from mmml.data.units import (
    convert_forces,
    energy_to_ev,
    infer_reference_energy_unit,
    infer_reference_force_unit,
)

DCM_PSF_MONOMER_PERM = np.array([0, 3, 4, 1, 2], dtype=int)
ATOMS_PER_DCM = 5
HYBRID_CALCULATOR_CHOICES = ("checkpoint", "hybrid-ml", "hybrid-monomer")
# Vacuum MP2 reference: keep dimer ML active (production mm_switch_on tapers ml_2b to 0 beyond R_COM).
VACUUM_REF_MM_SWITCH_ON_A = 99.0


def dcm_dimer_com_distance_A(
    positions: np.ndarray,
    *,
    atoms_per_monomer: int = ATOMS_PER_DCM,
) -> float:
    """COM–COM distance (Å) for a DCM dimer in PSF order."""
    pos = np.asarray(positions, dtype=np.float64).reshape(-1, 3)
    n = int(atoms_per_monomer)
    if pos.shape[0] < 2 * n:
        raise ValueError(f"need at least {2 * n} atoms for a dimer, got {pos.shape[0]}")
    com_a = pos[:n].mean(axis=0)
    com_b = pos[n : 2 * n].mean(axis=0)
    return float(np.linalg.norm(com_b - com_a))


def default_model_cutoff_from_checkpoint(checkpoint: Path | str) -> float:
    from mmml.interfaces.calculators.checkpoint_loading import load_checkpoint_bundle

    bundle = load_checkpoint_bundle(Path(checkpoint))
    return float(bundle.config.get("physnet_config", {}).get("cutoff", 6.0))


@dataclass(frozen=True, slots=True)
class HybridEvalResult:
    calculator: str
    energy_eV: float
    forces_ev_A: np.ndarray
    interaction_eV: float | None = None
    mp2_interaction_eV: float | None = None


def _calculator_slug(name: str) -> str:
    return str(name).replace("-", "_")


class DcmHybridEvaluator:
    """ASE hybrid / checkpoint calculators for DCM (energies and forces in eV)."""

    def __init__(
        self,
        checkpoint: Path | str,
        *,
        model_cutoff: float = 6.0,
        mm_switch_on: float = VACUUM_REF_MM_SWITCH_ON_A,
    ) -> None:
        self.checkpoint = Path(checkpoint)
        self.model_cutoff = float(model_cutoff)
        self.mm_switch_on = float(mm_switch_on)
        self._checkpoint_calc: Any | None = None
        self._hybrid_cache: dict[tuple[int, bool], Any] = {}

    def _checkpoint_calculator(self) -> Any:
        if self._checkpoint_calc is None:
            from mmml.interfaces.calculators.checkpoint_loading import (
                create_calculator_from_checkpoint,
            )

            self._checkpoint_calc = create_calculator_from_checkpoint(
                self.checkpoint,
                cutoff=self.model_cutoff,
            )
        return self._checkpoint_calc

    def _hybrid_calculator(self, n_atoms: int, *, do_ml_dimer: bool) -> Any:
        n_monomers = int(n_atoms) // ATOMS_PER_DCM
        if n_monomers * ATOMS_PER_DCM != int(n_atoms):
            raise ValueError(
                f"hybrid calculators expect DCM {ATOMS_PER_DCM}-atom monomers, got N={n_atoms}"
            )
        key = (n_monomers, bool(do_ml_dimer))
        if key not in self._hybrid_cache:
            from mmml.interfaces.pycharmmInterface.cutoffs import CutoffParameters
            from mmml.interfaces.pycharmmInterface.mmml_calculator import setup_calculator

            factory = setup_calculator(
                ATOMS_PER_MONOMER=[ATOMS_PER_DCM] * n_monomers,
                N_MONOMERS=n_monomers,
                doML=True,
                doMM=False,
                doML_dimer=do_ml_dimer,
                model_restart_path=str(self.checkpoint),
                MAX_ATOMS_PER_SYSTEM=int(n_atoms),
                ml_sparse_dimers=False,
                mm_switch_on=self.mm_switch_on,
                verbose=False,
            )
            cutoff_params = CutoffParameters(
                ml_switch_width=0.01,
                mm_switch_on=self.mm_switch_on,
                mm_switch_width=0.0,
            )
            calc, _spherical_fn, _get_update_fn = factory(
                atomic_numbers=np.ones(int(n_atoms), dtype=np.int32),
                atomic_positions=np.zeros((int(n_atoms), 3), dtype=np.float64),
                n_monomers=n_monomers,
                cutoff_params=cutoff_params,
                doML=True,
                doMM=False,
                doML_dimer=do_ml_dimer,
                backprop=False,
                debug=False,
                verbose=False,
            )
            self._hybrid_cache[key] = calc
        return self._hybrid_cache[key]

    def _ase_calculator(self, name: str, n_atoms: int) -> Any:
        if name == "checkpoint":
            return self._checkpoint_calculator()
        if name == "hybrid-ml":
            return self._hybrid_calculator(n_atoms, do_ml_dimer=True)
        if name == "hybrid-monomer":
            return self._hybrid_calculator(n_atoms, do_ml_dimer=False)
        raise ValueError(f"Unknown hybrid calculator: {name!r}")

    def evaluate(
        self,
        name: str,
        numbers: np.ndarray,
        positions: np.ndarray,
    ) -> tuple[float, np.ndarray]:
        from ase import Atoms

        pos = np.asarray(positions, dtype=np.float64)
        z = np.asarray(numbers, dtype=np.int32)
        atoms = Atoms(numbers=z, positions=pos)
        atoms.calc = self._ase_calculator(name, len(z))
        energy_eV = float(atoms.get_potential_energy())
        forces_ev_A = np.asarray(atoms.get_forces(), dtype=np.float64)
        return energy_eV, forces_ev_A

    def evaluate_frame(
        self,
        frame: Mp2Frame,
        calculators: tuple[str, ...],
        *,
        compute_interaction: bool,
        mp2_interaction_eV: float | None,
    ) -> list[HybridEvalResult]:
        results: list[HybridEvalResult] = []
        mono_cache: dict[tuple[int, ...], float] = {}
        for name in calculators:
            energy_eV, forces_ev_A = self.evaluate(name, frame.z, frame.r)
            interaction_eV: float | None = None
            if compute_interaction and frame.n_atoms == 10 and name in ("hybrid-ml", "checkpoint"):
                key_a = tuple(frame.r[:5].ravel().tolist())
                key_b = tuple(frame.r[5:].ravel().tolist())
                if key_a not in mono_cache:
                    e_a, _ = self.evaluate("hybrid-monomer", frame.z[:5], frame.r[:5])
                    mono_cache[key_a] = e_a
                if key_b not in mono_cache:
                    e_b, _ = self.evaluate("hybrid-monomer", frame.z[5:], frame.r[5:])
                    mono_cache[key_b] = e_b
                interaction_eV = float(energy_eV - mono_cache[key_a] - mono_cache[key_b])
            results.append(
                HybridEvalResult(
                    calculator=name,
                    energy_eV=energy_eV,
                    forces_ev_A=forces_ev_A,
                    interaction_eV=interaction_eV,
                    mp2_interaction_eV=mp2_interaction_eV,
                )
            )
        return results


@dataclass(frozen=True, slots=True)
class Mp2Frame:
    index: int
    source_index: int
    n_atoms: int
    z: np.ndarray
    r: np.ndarray
    e_ref_raw: float
    e_ref_eV: float
    f_ref_ev_A: np.ndarray | None


@dataclass(frozen=True, slots=True)
class MmFrameResult:
    index: int
    jax_energy_kcal: float
    charmm_energy_kcal: float
    jax_forces_kcal_A: np.ndarray
    charmm_forces_kcal_A: np.ndarray
    interaction_energy_kcal: float | None = None
    mp2_interaction_eV: float | None = None


def parse_monomer_permutation(text: str) -> np.ndarray:
    values = [int(part.strip()) for part in str(text).split(",") if part.strip()]
    if sorted(values) != list(range(len(values))):
        raise ValueError("permutation must be zero-based, e.g. 0,3,4,1,2")
    return np.asarray(values, dtype=int)


def repeat_monomer_permutation(active_n: int, monomer_perm: np.ndarray) -> np.ndarray:
    if active_n % len(monomer_perm) != 0:
        raise ValueError(f"N={active_n} is not divisible by monomer size {len(monomer_perm)}")
    return np.concatenate(
        [monomer_perm + offset for offset in range(0, active_n, len(monomer_perm))]
    )


def apply_atom_permutation(
    z: np.ndarray,
    r: np.ndarray,
    f: np.ndarray | None,
    perm: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    z_out = np.asarray(z, dtype=np.int32)[perm]
    r_out = np.asarray(r, dtype=np.float64)[perm]
    f_out = None
    if f is not None:
        f_out = np.asarray(f, dtype=np.float64)[perm]
    return z_out, r_out, f_out


def select_frame_indices(
    n_frames: int,
    *,
    max_frames: int | None,
    stride: int,
    seed: int,
) -> np.ndarray:
    indices = np.arange(0, n_frames, int(stride), dtype=int)
    if max_frames is not None and len(indices) > int(max_frames):
        rng = np.random.default_rng(seed)
        indices = np.sort(rng.choice(indices, size=int(max_frames), replace=False))
    return indices


def load_mp2_frames(
    path: Path | str,
    *,
    n_atoms: int = 10,
    monomer_permutation: np.ndarray | None = DCM_PSF_MONOMER_PERM,
    reference_energy_unit: str | None = None,
    reference_force_unit: str | None = None,
    max_frames: int | None = None,
    stride: int = 1,
    seed: int = 31,
) -> tuple[list[Mp2Frame], dict[str, Any]]:
    """Load MP2 frames with ``N == n_atoms`` (default 10 = DCM dimer)."""
    ref_path = Path(path)
    e_unit = reference_energy_unit or infer_reference_energy_unit(ref_path)
    f_unit = reference_force_unit or infer_reference_force_unit(ref_path)
    with np.load(ref_path, allow_pickle=True) as data:
        required = {"N", "Z", "R", "E"}
        missing = sorted(required.difference(data.files))
        if missing:
            raise KeyError(f"{ref_path} missing keys: {missing}")
        counts = np.asarray(data["N"], dtype=int)
        mask = counts == int(n_atoms)
        if not np.any(mask):
            raise ValueError(f"No frames with N={n_atoms} in {ref_path}")
        local_indices = np.where(mask)[0]
        selected = select_frame_indices(len(local_indices), max_frames=max_frames, stride=stride, seed=seed)
        global_indices = local_indices[selected]
        source = (
            np.asarray(data["source_indices"], dtype=int)
            if "source_indices" in data.files
            else np.arange(counts.shape[0], dtype=int)
        )
        perm = None
        if monomer_permutation is not None:
            perm = repeat_monomer_permutation(int(n_atoms), np.asarray(monomer_permutation, dtype=int))
        frames: list[Mp2Frame] = []
        for local_idx, global_idx in enumerate(global_indices):
            n = int(n_atoms)
            z = np.asarray(data["Z"][global_idx, :n], dtype=np.int32)
            r = np.asarray(data["R"][global_idx, :n], dtype=np.float64)
            f_raw = (
                np.asarray(data["F"][global_idx, :n], dtype=np.float64)
                if "F" in data.files
                else None
            )
            if perm is not None:
                z, r, f_raw = apply_atom_permutation(z, r, f_raw, perm)
            e_raw = float(np.asarray(data["E"][global_idx], dtype=np.float64))
            frames.append(
                Mp2Frame(
                    index=int(local_idx),
                    source_index=int(source[global_idx]),
                    n_atoms=n,
                    z=z,
                    r=r,
                    e_ref_raw=e_raw,
                    e_ref_eV=float(energy_to_ev(np.array(e_raw), e_unit)),
                    f_ref_ev_A=(
                        np.asarray(convert_forces(f_raw, f_unit, "ev_angstrom"), dtype=np.float64)
                        if f_raw is not None
                        else None
                    ),
                )
            )
    meta = {
        "path": str(ref_path),
        "n_atoms": int(n_atoms),
        "n_available": int(np.sum(mask)),
        "n_loaded": len(frames),
        "reference_energy_unit": e_unit,
        "reference_force_unit": f_unit,
        "monomer_permutation": None if monomer_permutation is None else list(map(int, monomer_permutation)),
    }
    return frames, meta


@dataclass(frozen=True, slots=True)
class DcmVacuumMmSession:
    psf_path: Path
    cgenff_prm: Path
    cell: np.ndarray
    cutnb: float
    ctonnb: float
    ctofnb: float
    workdir: Path


def _charmm_mm_unavailable_message() -> str:
    import os

    from mmml.interfaces.pycharmmInterface.charmm_mpi import charmm_lib_available

    lib_dir = (os.environ.get("CHARMM_LIB_DIR") or "").strip()
    if charmm_lib_available():
        return (
            "PyCHARMM failed to initialize. Run via ./scripts/mmml-charmm-mpirun.sh "
            f"(CHARMM_LIB_DIR={lib_dir!r})."
        )
    return (
        "CHARMM is not available for MM evaluation "
        f"(CHARMM_LIB_DIR={lib_dir!r}; libcharmm.so not found). "
        "Export CHARMM_LIB_DIR and use ./scripts/mmml-charmm-mpirun.sh for MM, "
        "or pass --hybrid-only to compare ML without CHARMM."
    )


def _require_charmm_for_mm() -> None:
    from mmml.interfaces.pycharmmInterface import import_pycharmm as ipy

    if not ipy.ensure_pycharmm_loaded():
        raise RuntimeError(_charmm_mm_unavailable_message())


_active_mm_session_psf: str | None = None


def activate_dcm_vacuum_mm_session(session: DcmVacuumMmSession) -> None:
    """Load ``session.psf_path`` into the active PyCHARMM session (vacuum nbonds)."""
    global _active_mm_session_psf

    psf_key = str(session.psf_path.resolve())
    if _active_mm_session_psf == psf_key:
        return

    from mmml.interfaces.pycharmmInterface import import_pycharmm as ipy
    from mmml.interfaces.pycharmmInterface.cgenff_bonded_reference import read_psf_card_file
    from mmml.interfaces.pycharmmInterface.mlpot.cgenff_prm_swap import mark_cgenff_params_full
    from mmml.interfaces.pycharmmInterface.mlpot.setup import prepare_charmm_vacuum
    from mmml.interfaces.pycharmmInterface.nbonds_config import (
        apply_nbonds_kwargs,
        vacuum_nbond_kwargs,
    )

    ipy.pycharmm.lingo.charmm_script("DELETE ATOM SELE ALL END")
    ipy.reset_block()
    prepare_charmm_vacuum()
    read_psf_card_file(session.psf_path)
    apply_nbonds_kwargs(vacuum_nbond_kwargs())
    mark_cgenff_params_full()
    _active_mm_session_psf = psf_key


def reset_active_mm_session_cache() -> None:
    """Clear PSF activation cache (for tests)."""
    global _active_mm_session_psf
    _active_mm_session_psf = None


def build_dcm_vacuum_mm_session(
    workdir: Path | str,
    *,
    n_monomers: int = 2,
    spacing_A: float = 8.0,
    vacuum_box_side_A: float = 40.0,
) -> DcmVacuumMmSession:
    """Build a vacuum DCM cluster PSF for MM evaluation (no MLpot)."""
    _require_charmm_for_mm()

    from mmml.cli.run.md_pbc_suite.ase import _build_cluster_from_composition
    from mmml.interfaces.pycharmmInterface import import_pycharmm as ipy
    from mmml.interfaces.pycharmmInterface.mlpot.cgenff_prm_swap import mark_cgenff_params_full
    from mmml.interfaces.pycharmmInterface.mlpot.setup import prepare_charmm_vacuum
    from mmml.interfaces.pycharmmInterface.nbonds_config import (
        VACUUM_CTONNB,
        VACUUM_CTOFNB,
        VACUUM_CUTNB,
        apply_nbonds_kwargs,
        vacuum_nbond_kwargs,
    )

    out = Path(workdir)
    out.mkdir(parents=True, exist_ok=True)

    ipy.pycharmm.lingo.charmm_script("DELETE ATOM SELE ALL END")
    ipy.reset_block()
    prepare_charmm_vacuum()
    _build_cluster_from_composition(
        composition=[("DCM", int(n_monomers))],
        spacing=float(spacing_A),
    )
    apply_nbonds_kwargs(vacuum_nbond_kwargs())
    mark_cgenff_params_full()

    psf_path = out / f"dcm{int(n_monomers)}_vacuum.psf"
    import os

    prev = os.getcwd()
    try:
        os.chdir(out)
        import pycharmm.write as write

        write.psf_card(psf_path.name)
    finally:
        os.chdir(prev)

    side = float(vacuum_box_side_A)
    cell = np.diag([side, side, side])
    global _active_mm_session_psf
    _active_mm_session_psf = str(psf_path.resolve())
    return DcmVacuumMmSession(
        psf_path=psf_path.resolve(),
        cgenff_prm=Path(ipy.CGENFF_PRM),
        cell=cell,
        cutnb=float(VACUUM_CUTNB),
        ctonnb=float(VACUUM_CTONNB),
        ctofnb=float(VACUUM_CTOFNB),
        workdir=out.resolve(),
    )


def load_monomer_mean_energy_eV(
    path: Path | str,
    *,
    reference_energy_unit: str = "hartree",
) -> float:
    """Mean MP2 total energy (eV) for ``N=5`` monomer frames in the dataset."""
    with np.load(path, allow_pickle=True) as data:
        counts = np.asarray(data["N"], dtype=int)
        mask = counts == ATOMS_PER_DCM
        if not np.any(mask):
            raise ValueError(f"No N={ATOMS_PER_DCM} monomer frames in {path}")
        e_raw = np.asarray(data["E"][mask], dtype=np.float64)
    return float(np.mean(energy_to_ev(e_raw, reference_energy_unit)))


def evaluate_mm_at_positions(
    session: DcmVacuumMmSession,
    positions: np.ndarray,
    *,
    mono_session: DcmVacuumMmSession | None = None,
    mp2_interaction_eV: float | None = None,
) -> MmFrameResult:
    """Evaluate JAX + CHARMM MM at PSF-ordered positions (kcal/mol, kcal/mol/Å)."""
    from mmml.interfaces.pycharmmInterface.cgenff_bonded_reference import (
        run_charmm_bonded_ener_force,
        set_charmm_positions,
    )
    from mmml.interfaces.pycharmmInterface.jax_x64_config import ensure_jax_x64
    from mmml.interfaces.pycharmmInterface.mm_system_energy import (
        CharmmNbondSettings,
        load_bonded_system_from_psf,
        load_nonbonded_system_from_charmm,
        mm_system_energy_and_forces,
    )
    from mmml.interfaces.pycharmmInterface.mlpot.block_terms import apply_charmm_mm_block
    from mmml.interfaces.pycharmmInterface.mlpot.cli_common import charmm_total_forces_kcalmol_A
    import pycharmm.energy as energy

    ensure_jax_x64(context="evaluate_mm_at_positions")
    pos = np.asarray(positions, dtype=np.float64)
    activate_dcm_vacuum_mm_session(session)
    set_charmm_positions(pos)
    apply_charmm_mm_block()
    run_charmm_bonded_ener_force(silent=True)
    charmm_e = float(energy.get_total())
    charmm_f = np.asarray(charmm_total_forces_kcalmol_A(), dtype=np.float64)

    settings = CharmmNbondSettings(
        cutnb=float(session.cutnb),
        ctonnb=float(session.ctonnb),
        ctofnb=float(session.ctofnb),
    )
    bonded = load_bonded_system_from_psf(
        session.psf_path,
        pos,
        prm_file=session.cgenff_prm,
    )
    nbond = load_nonbonded_system_from_charmm(session.psf_path, session.cgenff_prm)
    jax_result = mm_system_energy_and_forces(
        pos,
        bonded,
        nbond,
        session.cell,
        settings,
        prm_file=session.cgenff_prm,
        include_cmap=False,
        lr_solver="mic",
    )

    interaction_kcal: float | None = None
    if mono_session is not None and pos.shape[0] == 10:
        e_a = evaluate_mm_at_positions(mono_session, pos[:5]).jax_energy_kcal
        e_b = evaluate_mm_at_positions(mono_session, pos[5:]).jax_energy_kcal
        interaction_kcal = float(jax_result.total_energy - e_a - e_b)

    return MmFrameResult(
        index=-1,
        jax_energy_kcal=float(jax_result.total_energy),
        charmm_energy_kcal=charmm_e,
        jax_forces_kcal_A=np.asarray(jax_result.forces, dtype=np.float64),
        charmm_forces_kcal_A=charmm_f,
        interaction_energy_kcal=interaction_kcal,
        mp2_interaction_eV=mp2_interaction_eV,
    )


def forces_kcal_to_ev(forces_kcal_A: np.ndarray) -> np.ndarray:
    return np.asarray(
        convert_forces(forces_kcal_A, "kcal_mol_angstrom", "ev_angstrom"),
        dtype=np.float64,
    )


def _force_rmse_ev_A(pred_ev_A: np.ndarray, ref_ev_A: np.ndarray) -> float:
    d = np.asarray(pred_ev_A, dtype=np.float64) - np.asarray(ref_ev_A, dtype=np.float64)
    return float(np.sqrt(np.mean(d**2)))


def compare_mm_to_mp2_frame(
    frame: Mp2Frame,
    *,
    mm: MmFrameResult | None = None,
    hybrid: list[HybridEvalResult] | None = None,
) -> dict[str, Any]:
    out: dict[str, Any] = {
        "index": frame.index,
        "source_index": frame.source_index,
        "mp2_energy_eV": frame.e_ref_eV,
    }
    if frame.n_atoms == 10:
        out["dimer_com_distance_A"] = dcm_dimer_com_distance_A(frame.r)
    if mm is not None:
        out["jax_energy_kcal"] = mm.jax_energy_kcal
        out["charmm_energy_kcal"] = mm.charmm_energy_kcal
        out["jax_charmm_energy_delta_kcal"] = mm.jax_energy_kcal - mm.charmm_energy_kcal
        jax_f_ev = forces_kcal_to_ev(mm.jax_forces_kcal_A)
        ch_f_ev = forces_kcal_to_ev(mm.charmm_forces_kcal_A)
        out["jax_charmm_force_rmse_ev_A"] = float(
            np.sqrt(np.mean((jax_f_ev - ch_f_ev) ** 2))
        )
        if frame.f_ref_ev_A is not None:
            mp2_f = np.asarray(frame.f_ref_ev_A, dtype=np.float64)
            d_jax = jax_f_ev - mp2_f
            d_ch = ch_f_ev - mp2_f
            out["mp2_jax_force_rmse_ev_A"] = float(np.sqrt(np.mean(d_jax**2)))
            out["mp2_charmm_force_rmse_ev_A"] = float(np.sqrt(np.mean(d_ch**2)))
            out["mp2_jax_force_mae_ev_A"] = float(np.mean(np.abs(d_jax)))
            out["mp2_charmm_force_mae_ev_A"] = float(np.mean(np.abs(d_ch)))
        if mm.interaction_energy_kcal is not None and mm.mp2_interaction_eV is not None:
            from mmml.data.units import convert_energy

            mm_int_ev = float(convert_energy(mm.interaction_energy_kcal, "kcal_mol", "ev"))
            out["jax_interaction_eV"] = mm_int_ev
            out["mp2_interaction_eV"] = float(mm.mp2_interaction_eV)
            out["interaction_delta_eV"] = mm_int_ev - float(mm.mp2_interaction_eV)
    if hybrid:
        out["hybrid"] = {}
        for hres in hybrid:
            slug = _calculator_slug(hres.calculator)
            block: dict[str, Any] = {
                "energy_eV": hres.energy_eV,
                "mp2_energy_delta_eV": hres.energy_eV - frame.e_ref_eV,
            }
            if frame.f_ref_ev_A is not None:
                block["mp2_force_rmse_ev_A"] = _force_rmse_ev_A(
                    hres.forces_ev_A, frame.f_ref_ev_A
                )
                block["mp2_force_mae_ev_A"] = float(
                    np.mean(np.abs(hres.forces_ev_A - frame.f_ref_ev_A))
                )
            if hres.interaction_eV is not None and hres.mp2_interaction_eV is not None:
                block["interaction_eV"] = hres.interaction_eV
                block["mp2_interaction_eV"] = float(hres.mp2_interaction_eV)
                block["interaction_delta_eV"] = hres.interaction_eV - float(
                    hres.mp2_interaction_eV
                )
            out["hybrid"][hres.calculator] = block
            out[f"hybrid_{slug}_mp2_force_rmse_ev_A"] = block.get("mp2_force_rmse_ev_A")
            out[f"hybrid_{slug}_interaction_delta_eV"] = block.get("interaction_delta_eV")
    return out


COM_BIN_EDGES_A: tuple[tuple[float, float], ...] = (
    (0.0, 6.0),
    (6.0, 8.0),
    (8.0, 10.0),
    (10.0, float("inf")),
)


def com_binned_force_rmse(
    rows: list[dict[str, Any]],
    rmse_key: str,
    *,
    com_key: str = "dimer_com_distance_A",
    bins: tuple[tuple[float, float], ...] = COM_BIN_EDGES_A,
) -> list[dict[str, Any]]:
    """Bin MP2 force RMSE by dimer COM distance (Å)."""
    if not rows or com_key not in rows[0] or rmse_key not in rows[0]:
        return []
    com = np.asarray([r[com_key] for r in rows], dtype=np.float64)
    rmse = np.asarray([r.get(rmse_key, float("nan")) for r in rows], dtype=np.float64)
    out: list[dict[str, Any]] = []
    for lo, hi in bins:
        mask = (com >= lo) & (com < hi)
        if not np.any(mask):
            continue
        vals = rmse[mask]
        finite = vals[np.isfinite(vals)]
        hi_label = "∞" if not np.isfinite(hi) else f"{hi:g}"
        out.append(
            {
                "com_min_A": float(lo),
                "com_max_A": float(hi) if np.isfinite(hi) else None,
                "com_bin_label": f"[{lo:g},{hi_label})",
                "n": int(finite.size),
                "median_rmse_ev_A": float(np.median(finite)) if finite.size else float("nan"),
                "mean_rmse_ev_A": float(np.mean(finite)) if finite.size else float("nan"),
                "p90_rmse_ev_A": float(np.percentile(finite, 90)) if finite.size else float("nan"),
            }
        )
    return out


def hybrid_calculator_pairwise_summary(
    rows: list[dict[str, Any]],
    calc_a: str,
    calc_b: str,
) -> dict[str, Any] | None:
    """Compare per-frame MP2 force RMSE between two hybrid calculators."""
    key_a = f"hybrid_{_calculator_slug(calc_a)}_mp2_force_rmse_ev_A"
    key_b = f"hybrid_{_calculator_slug(calc_b)}_mp2_force_rmse_ev_A"
    if not rows or key_a not in rows[0] or key_b not in rows[0]:
        return None
    a = np.asarray([r.get(key_a, float("nan")) for r in rows], dtype=np.float64)
    b = np.asarray([r.get(key_b, float("nan")) for r in rows], dtype=np.float64)
    mask = np.isfinite(a) & np.isfinite(b)
    if not np.any(mask):
        return None
    a = a[mask]
    b = b[mask]
    delta = a - b
    abs_delta = np.abs(delta)
    com = (
        np.asarray([r["dimer_com_distance_A"] for r in rows], dtype=np.float64)[mask]
        if rows and "dimer_com_distance_A" in rows[0]
        else None
    )
    corr_com = float("nan")
    if com is not None and com.size >= 2:
        corr_com = float(np.corrcoef(com, a)[0, 1])
    worst_idx = int(np.argmax(abs_delta))
    row_at_worst = rows[int(np.where(mask)[0][worst_idx])]
    return {
        "calculator_a": calc_a,
        "calculator_b": calc_b,
        "n": int(a.size),
        "mean_rmse_a_ev_A": float(np.mean(a)),
        "mean_rmse_b_ev_A": float(np.mean(b)),
        "median_rmse_a_ev_A": float(np.median(a)),
        "median_rmse_b_ev_A": float(np.median(b)),
        "mean_abs_delta_ev_A": float(np.mean(abs_delta)),
        "max_abs_delta_ev_A": float(np.max(abs_delta)),
        "rmse_delta_ev_A": float(np.sqrt(np.mean(delta**2))),
        "corr_com_rmse_a": corr_com,
        "worst_abs_delta": {
            "source_index": int(row_at_worst["source_index"]),
            "rmse_a_ev_A": float(a[worst_idx]),
            "rmse_b_ev_A": float(b[worst_idx]),
            "delta_ev_A": float(delta[worst_idx]),
        },
    }


def aggregate_comparison(rows: list[dict[str, Any]]) -> dict[str, Any]:
    def _agg(key: str) -> dict[str, float | int]:
        vals = np.asarray([r[key] for r in rows if key in r and np.isfinite(r[key])], dtype=np.float64)
        if vals.size == 0:
            return {"n": 0, "mean": float("nan"), "rmse": float("nan"), "max_abs": float("nan")}
        return {
            "n": int(vals.size),
            "mean": float(np.mean(vals)),
            "rmse": float(np.sqrt(np.mean(vals**2))),
            "max_abs": float(np.max(np.abs(vals))),
        }

    summary = {
        "n_frames": len(rows),
        "jax_charmm_energy_delta_kcal": _agg("jax_charmm_energy_delta_kcal"),
        "jax_charmm_force_rmse_ev_A": _agg("jax_charmm_force_rmse_ev_A"),
        "mp2_jax_force_rmse_ev_A": _agg("mp2_jax_force_rmse_ev_A"),
        "mp2_charmm_force_rmse_ev_A": _agg("mp2_charmm_force_rmse_ev_A"),
        "interaction_delta_eV": _agg("interaction_delta_eV"),
    }
    hybrid_names = sorted(
        {
            calc_name
            for row in rows
            for calc_name in (row.get("hybrid") or {})
        }
    )
    for calc_name in hybrid_names:
        slug = _calculator_slug(calc_name)
        summary[f"hybrid_{slug}_mp2_force_rmse_ev_A"] = _agg(
            f"hybrid_{slug}_mp2_force_rmse_ev_A"
        )
        summary[f"hybrid_{slug}_interaction_delta_eV"] = _agg(
            f"hybrid_{slug}_interaction_delta_eV"
        )
    if rows and "jax_energy_kcal" in rows[0]:
        e_mp2 = np.asarray([r["mp2_energy_eV"] for r in rows], dtype=np.float64)
        e_jax = np.asarray([r["jax_energy_kcal"] for r in rows], dtype=np.float64)
        from mmml.data.units import convert_energy

        e_jax_ev = convert_energy(e_jax, "kcal_mol", "ev")
        summary["total_energy_vs_mp2"] = compute_scalar_metrics(e_jax_ev, e_mp2).to_dict()
        summary["total_energy_note"] = (
            "Absolute totals differ by QM/MM offset; use interaction_delta_eV for physics."
        )
    if rows and "dimer_com_distance_A" in rows[0]:
        com = np.asarray([r["dimer_com_distance_A"] for r in rows], dtype=np.float64)
        summary["dimer_com_distance_A"] = {
            "mean": float(np.mean(com)),
            "median": float(np.median(com)),
            "p90": float(np.percentile(com, 90)),
            "max": float(np.max(com)),
        }
        for calc_name in hybrid_names:
            slug = _calculator_slug(calc_name)
            key = f"hybrid_{slug}_mp2_force_rmse_ev_A"
            bins = com_binned_force_rmse(rows, key)
            if bins:
                summary[f"hybrid_{slug}_com_binned_rmse_ev_A"] = bins
            rmse_vals = np.asarray(
                [r[key] for r in rows if key in r and np.isfinite(r[key])],
                dtype=np.float64,
            )
            if rmse_vals.size >= 2:
                summary[f"hybrid_{slug}_corr_com_rmse"] = float(
                    np.corrcoef(com[: rmse_vals.size], rmse_vals)[0, 1]
                )
    pair = hybrid_calculator_pairwise_summary(rows, "checkpoint", "hybrid-ml")
    if pair is not None:
        summary["hybrid_checkpoint_vs_hybrid_ml"] = pair
    return summary


def run_dcm_mp2_mm_comparison(
    npz_path: Path | str,
    output_dir: Path | str,
    *,
    workdir: Path | str | None = None,
    reference_energy_unit: str = "hartree",
    reference_force_unit: str = "ev_angstrom",
    max_frames: int = 200,
    stride: int = 10,
    seed: int = 31,
    compute_interaction: bool = True,
    monomer_permutation: np.ndarray | None = DCM_PSF_MONOMER_PERM,
    checkpoint: Path | str | None = None,
    hybrid_calculators: tuple[str, ...] = ("hybrid-ml",),
    hybrid_model_cutoff: float | None = None,
    hybrid_mm_switch_on: float = VACUUM_REF_MM_SWITCH_ON_A,
    run_mm: bool = True,
) -> dict[str, Any]:
    if not run_mm and checkpoint is None:
        raise ValueError("--hybrid-only requires --checkpoint")
    if run_mm:
        _require_charmm_for_mm()

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    wdir = Path(workdir or out / "charmm_work")

    frames, load_meta = load_mp2_frames(
        npz_path,
        n_atoms=10,
        monomer_permutation=monomer_permutation,
        reference_energy_unit=reference_energy_unit,
        reference_force_unit=reference_force_unit,
        max_frames=max_frames,
        stride=stride,
        seed=seed,
    )
    session: DcmVacuumMmSession | None = None
    mono_session: DcmVacuumMmSession | None = None
    mono_mean_eV: float | None = None
    if run_mm:
        session = build_dcm_vacuum_mm_session(wdir, n_monomers=2)
        if compute_interaction:
            mono_session = build_dcm_vacuum_mm_session(wdir / "mono_pool", n_monomers=1)
    if compute_interaction:
        mono_mean_eV = load_monomer_mean_energy_eV(
            npz_path, reference_energy_unit=reference_energy_unit
        )

    hybrid_eval: DcmHybridEvaluator | None = None
    if checkpoint is not None:
        from mmml.interfaces.pycharmmInterface.jax_x64_config import ensure_jax_x64

        ensure_jax_x64(context="run_dcm_mp2_mm_comparison")
        model_cutoff = (
            float(hybrid_model_cutoff)
            if hybrid_model_cutoff is not None
            else default_model_cutoff_from_checkpoint(checkpoint)
        )
        hybrid_eval = DcmHybridEvaluator(
            checkpoint,
            model_cutoff=model_cutoff,
            mm_switch_on=hybrid_mm_switch_on,
        )

    rows: list[dict[str, Any]] = []
    for frame in frames:
        mp2_int = (
            float(frame.e_ref_eV - 2.0 * mono_mean_eV)
            if mono_mean_eV is not None
            else None
        )
        mm: MmFrameResult | None = None
        if session is not None:
            mm = evaluate_mm_at_positions(
                session,
                frame.r,
                mono_session=mono_session,
                mp2_interaction_eV=mp2_int,
            )
        hybrid_results: list[HybridEvalResult] | None = None
        if hybrid_eval is not None:
            hybrid_results = hybrid_eval.evaluate_frame(
                frame,
                hybrid_calculators,
                compute_interaction=compute_interaction,
                mp2_interaction_eV=mp2_int,
            )
        row = compare_mm_to_mp2_frame(frame, mm=mm, hybrid=hybrid_results)
        rows.append(row)

    summary = aggregate_comparison(rows)
    if hybrid_eval is not None:
        summary["hybrid_mm_switch_on_A"] = float(hybrid_eval.mm_switch_on)
        summary["hybrid_model_cutoff_A"] = float(hybrid_eval.model_cutoff)
    payload = {
        "load": load_meta,
        "mm": {"enabled": bool(run_mm)},
        "hybrid": (
            None
            if checkpoint is None
            else {
                "checkpoint": str(Path(checkpoint)),
                "model_cutoff_A": float(hybrid_eval.model_cutoff),
                "mm_switch_on_A": float(hybrid_eval.mm_switch_on),
                "calculators": list(hybrid_calculators),
                "note": (
                    "Vacuum MP2 reference uses large mm_switch_on so dimer ML is not "
                    "tapered by COM distance (production MD uses ~6–8 Å)."
                ),
            }
        ),
        "summary": summary,
        "frames": rows,
    }
    (out / "comparison.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    _write_comparison_md(out / "report.md", load_meta, summary, rows)
    return payload


def _write_comparison_md(
    path: Path,
    load_meta: dict[str, Any],
    summary: dict[str, Any],
    rows: list[dict[str, Any]],
) -> None:
    lines = [
        "# DCM MP2 vs CHARMM/JAX MM and hybrid ML",
        "",
        f"Dataset: `{load_meta['path']}` ({load_meta['n_loaded']} dimer frames, "
        f"N={load_meta['n_atoms']}).",
        f"Reference: E in **{load_meta['reference_energy_unit']}**, "
        f"F in **{load_meta['reference_force_unit']}**.",
        "",
    ]
    has_mm = any("jax_energy_kcal" in row for row in rows)
    if has_mm:
        lines.extend(
            [
                "## MM summary",
                "",
                "| Metric | n | mean | RMSE | max |",
                "|--------|---|------|------|-----|",
            ]
        )
    else:
        lines.extend(["MM evaluation skipped (hybrid-only run).", ""])
    mm_rows = (
        [
            ("jax_charmm_force_rmse_ev_A", "JAX vs CHARMM force RMSE (eV/Å)"),
            ("mp2_jax_force_rmse_ev_A", "MP2 vs JAX force RMSE (eV/Å)"),
            ("mp2_charmm_force_rmse_ev_A", "MP2 vs CHARMM force RMSE (eV/Å)"),
            ("interaction_delta_eV", "Interaction E: JAX MM−MP2 (eV)"),
        ]
        if has_mm
        else []
    )
    for key, label in mm_rows:
        block = summary.get(key, {})
        lines.append(
            f"| {label} | {block.get('n', 0)} | "
            f"{block.get('mean', float('nan')):.4g} | "
            f"{block.get('rmse', float('nan')):.4g} | "
            f"{block.get('max_abs', float('nan')):.4g} |"
        )
    hybrid_calc_names = sorted(
        {
            calc_name
            for row in rows
            for calc_name in (row.get("hybrid") or {})
        }
    )
    if hybrid_calc_names:
        lines.extend(["", "## Hybrid calculators vs MP2", ""])
        mm_on = summary.get("hybrid_mm_switch_on_A")
        if mm_on is not None:
            lines.append(
                f"Hybrid `mm_switch_on` = **{mm_on:g} Å** "
                f"(model cutoff **{summary.get('hybrid_model_cutoff_A', float('nan')):g} Å**). "
                "Values ≪ COM distance are required for dimer ML; production MD uses ~6–8 Å."
            )
            lines.append("")
        com_block = summary.get("dimer_com_distance_A", {})
        if com_block:
            lines.append(
                f"Dimer COM distance: median **{com_block.get('median', float('nan')):.2f} Å**, "
                f"p90 **{com_block.get('p90', float('nan')):.2f} Å**, "
                f"max **{com_block.get('max', float('nan')):.2f} Å**."
            )
            lines.append("")
        for calc_name in hybrid_calc_names:
            slug = _calculator_slug(calc_name)
            force_block = summary.get(f"hybrid_{slug}_mp2_force_rmse_ev_A", {})
            int_block = summary.get(f"hybrid_{slug}_interaction_delta_eV", {})
            lines.append(f"### `{calc_name}`")
            lines.append(
                f"- MP2 force RMSE: mean={force_block.get('mean', float('nan')):.4g} eV/Å, "
                f"median={force_block.get('rmse', float('nan')):.4g} eV/Å "
                f"(n={force_block.get('n', 0)})"
            )
            corr = summary.get(f"hybrid_{slug}_corr_com_rmse")
            if corr is not None and np.isfinite(corr):
                lines.append(f"- corr(COM, RMSE) = {corr:.4g}")
            if int_block.get("n", 0):
                lines.append(
                    f"- Interaction Δ (hybrid−MP2): mean={int_block.get('mean', float('nan')):.4g} eV"
                )
            lines.append("")
        # COM-binned RMSE (first hybrid calculator with bins, or each)
        for calc_name in hybrid_calc_names:
            slug = _calculator_slug(calc_name)
            bins = summary.get(f"hybrid_{slug}_com_binned_rmse_ev_A", [])
            if not bins:
                continue
            lines.extend(
                [
                    f"### MP2 force RMSE by COM (`{calc_name}`)",
                    "",
                    "| COM bin (Å) | n | median | mean | p90 |",
                    "|-------------|---|--------|------|-----|",
                ]
            )
            for block in bins:
                lines.append(
                    f"| {block['com_bin_label']} | {block['n']} | "
                    f"{block['median_rmse_ev_A']:.4f} | "
                    f"{block['mean_rmse_ev_A']:.4f} | "
                    f"{block['p90_rmse_ev_A']:.4f} |"
                )
            lines.append("")
            break
        pair = summary.get("hybrid_checkpoint_vs_hybrid_ml")
        if pair is not None:
            lines.extend(
                [
                    "## Checkpoint vs hybrid-ml",
                    "",
                    "| Metric | checkpoint | hybrid-ml |",
                    "|--------|------------|-----------|",
                    f"| Mean MP2 force RMSE (eV/Å) | {pair['mean_rmse_a_ev_A']:.4g} | "
                    f"{pair['mean_rmse_b_ev_A']:.4g} |",
                    f"| Median MP2 force RMSE (eV/Å) | {pair['median_rmse_a_ev_A']:.4g} | "
                    f"{pair['median_rmse_b_ev_A']:.4g} |",
                    f"| Mean |Δ| per frame (eV/Å) | — | {pair['mean_abs_delta_ev_A']:.4g} |",
                    f"| Max |Δ| per frame (eV/Å) | — | {pair['max_abs_delta_ev_A']:.4g} |",
                    f"| RMSE of Δ (eV/Å) | — | {pair['rmse_delta_ev_A']:.4g} |",
                    "",
                ]
            )
            worst = pair.get("worst_abs_delta", {})
            if worst:
                lines.append(
                    f"Largest |Δ| at source_index **{worst.get('source_index')}**: "
                    f"checkpoint={worst.get('rmse_a_ev_A', float('nan')):.4f}, "
                    f"hybrid-ml={worst.get('rmse_b_ev_A', float('nan')):.4f} eV/Å "
                    f"(Δ={worst.get('delta_ev_A', float('nan')):.4g})."
                )
                lines.append("")
    lines.extend(
        [
            "",
            "Note: absolute total energies are not comparable (MP2 electronic vs MM force-field).",
            "Use **interaction** and **force** metrics for MM quality vs MP2.",
            "",
        ]
    )
    if has_mm and rows:
        worst = sorted(rows, key=lambda r: r.get("mp2_jax_force_rmse_ev_A", 0.0), reverse=True)[:5]
        lines.extend(
            [
                "## Worst MP2 vs JAX MM force RMSE (top 5)",
                "",
                "| source_index | MP2−JAX RMSE | MP2−CHARMM RMSE | JAX−CHARMM RMSE |",
                "|--------------|--------------|-----------------|-----------------|",
            ]
        )
        for row in worst:
            lines.append(
                f"| {row['source_index']} | "
                f"{row.get('mp2_jax_force_rmse_ev_A', float('nan')):.4f} | "
                f"{row.get('mp2_charmm_force_rmse_ev_A', float('nan')):.4f} | "
                f"{row.get('jax_charmm_force_rmse_ev_A', float('nan')):.4f} |"
            )
        if hybrid_calc_names:
            for calc_name in hybrid_calc_names:
                key = f"hybrid_{_calculator_slug(calc_name)}_mp2_force_rmse_ev_A"
                worst_h = sorted(
                    rows,
                    key=lambda r, k=key: r.get(k) or 0.0,
                    reverse=True,
                )[:5]
                lines.extend(
                    [
                        "",
                        f"## Worst MP2 vs `{calc_name}` force RMSE (top 5)",
                        "",
                        "| source_index | MP2−hybrid RMSE |",
                        "|--------------|-----------------|",
                    ]
                )
                for row in worst_h:
                    lines.append(
                        f"| {row['source_index']} | {row.get(key, float('nan')):.4f} |"
                    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
