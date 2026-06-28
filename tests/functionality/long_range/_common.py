"""Shared helpers for long-range Coulomb validation (MIC, jax-pme, ScaFaCoS).

Reference patterns follow lab-cosmo/jax-pme ``tests/test_ewald.py`` (Madelung
crystals, ion clusters, method cross-checks).  Units throughout: positions Å,
charges e, energy kcal/mol, forces kcal/mol/Å (CHARMM convention).
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Sequence

import jax
import jax.numpy as jnp
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from mmml.interfaces.pycharmmInterface.long_range_backend import CHARMM_COULOMB_KCAL
from mmml.interfaces.pycharmmInterface.pbc_utils_jax import mic_displacement

jax.config.update("jax_enable_x64", True)

# Default MM real-space Coulomb cutoff (mm_switch_on + mm_switch_width).
DEFAULT_MM_COULOMB_CUTOFF_A = 13.0

BackendName = Literal["mic", "mic_trunc", "jax_ewald", "jax_pme", "jax_p3m", "scafacos"]


@dataclass(frozen=True)
class CoulombSystem:
    """Point-charge test system in a cubic periodic box."""

    name: str
    positions_A: np.ndarray  # (N, 3)
    charges_e: np.ndarray  # (N,)
    box_length_A: float
    madelung_ref: float | None = None
    num_formula_units: int = 1


@dataclass(frozen=True)
class CoulombResult:
    energy_kcalmol: float
    forces_kcalmol_A: np.ndarray


def print_header(title: str) -> None:
    bar = "=" * len(title)
    print(f"\n{bar}\n{title}\n{bar}")


def print_pass(msg: str) -> None:
    from mmml.utils.rich_report import emit_status

    emit_status(True, msg)


def print_fail(msg: str) -> None:
    from mmml.utils.rich_report import emit_status

    emit_status(False, msg)


def have_jax_pme_package() -> bool:
    try:
        import jaxpme  # noqa: F401

        return True
    except ImportError:
        return False


def have_scafacos_library() -> bool:
    from mmml.interfaces.scafacosInterface import have_scafacos

    return have_scafacos()


def scafacos_integration_enabled() -> bool:
    """True when ScaFaCoS at ``~/.local/scafacos`` (or SCAFACOS_LIB) passes a smoke run."""
    if os.environ.get("MMML_SCAFACOS_TESTS", "").strip().lower() in ("0", "no", "false"):
        return False
    from mmml.interfaces.scafacosInterface.scafacos_session import scafacos_runtime_ok

    return scafacos_runtime_ok(method="ewald")


def ion_dimer_system(
    *,
    separation_A: float = 5.0,
    box_length_A: float = 30.0,
    charges: tuple[float, float] = (1.0, -1.0),
) -> CoulombSystem:
    """Two opposite charges along x in a cubic box."""
    pos = np.array(
        [[0.0, 0.0, 0.0], [separation_A, 0.0, 0.0]],
        dtype=np.float64,
    )
    chg = np.array(charges, dtype=np.float64)
    return CoulombSystem(
        name=f"ion_dimer_r{separation_A:.1f}A",
        positions_A=pos,
        charges_e=chg,
        box_length_A=float(box_length_A),
    )


def cscl_crystal(*, box_length_A: float = 1.0) -> CoulombSystem:
    """CsCl primitive cell (1 formula unit). Madelung ref ≈ 2.035361.

    Default unit cell edge 1.0 matches jax-pme ``define_crystal("CsCl")``.
    """
    frac = np.array([[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]], dtype=np.float64)
    pos = frac * float(box_length_A)
    chg = np.array([-1.0, 1.0], dtype=np.float64)
    return CoulombSystem(
        name="CsCl",
        positions_A=pos,
        charges_e=chg,
        box_length_A=float(box_length_A),
        madelung_ref=2.035361,
        num_formula_units=1,
    )


def nacl_cubic(*, box_length_A: float = 2.0) -> CoulombSystem:
    """NaCl rocksalt cubic cell (4 formula units). Madelung ref ≈ 1.747565.

    Matches jax-pme ``define_crystal("NaCl_cubic")``: integer site coords in a
    cell of edge 2 (Å or arbitrary units); scale ``box_length_A`` uniformly.
    """
    scale = float(box_length_A) / 2.0
    positions = scale * np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 1.0, 0.0],
            [1.0, 0.0, 1.0],
            [0.0, 1.0, 1.0],
            [1.0, 1.0, 1.0],
        ],
        dtype=np.float64,
    )
    chg = np.array([1.0, -1.0, -1.0, -1.0, 1.0, 1.0, 1.0, -1.0], dtype=np.float64)
    return CoulombSystem(
        name="NaCl_cubic",
        positions_A=positions,
        charges_e=chg,
        box_length_A=float(box_length_A),
        madelung_ref=1.747565,
        num_formula_units=4,
    )


def random_neutral_cluster(
    *,
    n_atoms: int = 8,
    box_length_A: float = 12.0,
    seed: int = 42,
) -> CoulombSystem:
    """Random neutral ±1 charge cluster (4 cations, 4 anions by default)."""
    rng = np.random.default_rng(seed)
    pos = rng.uniform(0.5, box_length_A - 0.5, (n_atoms, 3))
    chg = np.ones(n_atoms, dtype=np.float64)
    chg[n_atoms // 2 :] *= -1.0
    rng.shuffle(chg)
    return CoulombSystem(
        name=f"random_neutral_{n_atoms}",
        positions_A=pos,
        charges_e=chg,
        box_length_A=float(box_length_A),
    )


def _mic_displacement_np(ri: np.ndarray, rj: np.ndarray, cell: np.ndarray) -> np.ndarray:
    d = mic_displacement(jnp.asarray(ri), jnp.asarray(rj), jnp.asarray(cell))
    return np.asarray(d, dtype=np.float64)


def mic_coulomb_energy_forces(
    system: CoulombSystem,
    *,
    cutoff_A: float | None = None,
    constant: float = CHARMM_COULOMB_KCAL,
) -> CoulombResult:
    """All-pairs MIC Coulomb (MMML pair convention: sum over i<j, no extra ½)."""
    pos = np.asarray(system.positions_A, dtype=np.float64)
    chg = np.asarray(system.charges_e, dtype=np.float64)
    cell = np.eye(3, dtype=np.float64) * float(system.box_length_A)
    n = pos.shape[0]
    energy = 0.0
    forces = np.zeros((n, 3), dtype=np.float64)
    eps = 1e-10

    for i in range(n):
        for j in range(i + 1, n):
            dR = _mic_displacement_np(pos[i], pos[j], cell)
            r = float(np.linalg.norm(dR))
            if cutoff_A is not None and r > cutoff_A:
                continue
            r_safe = max(r, eps)
            qq = chg[i] * chg[j]
            e_ij = constant * qq / r_safe
            energy += e_ij
            f_ij = constant * qq / (r_safe**3) * dR
            forces[i] -= f_ij
            forces[j] += f_ij

    return CoulombResult(energy_kcalmol=energy, forces_kcalmol_A=forces)


def jax_pme_coulomb_energy_forces(
    system: CoulombSystem,
    *,
    method: Literal["ewald", "pme", "p3m"] = "ewald",
    sr_cutoff_A: float = 6.0,
) -> CoulombResult:
    """Full periodic Coulomb via jax-pme (Ewald / PME / P3M)."""
    from ase import Atoms
    from jaxpme import Ewald, P3M, PME
    from jaxpme import prefactors as jpref

    pos = np.asarray(system.positions_A, dtype=np.float64)
    chg = np.asarray(system.charges_e, dtype=np.float64)
    L = float(system.box_length_A)
    atoms = Atoms(positions=pos, cell=np.eye(3) * L, pbc=True)
    atoms.set_initial_charges(chg)

    calc_map = {"ewald": Ewald, "pme": PME, "p3m": P3M}
    calc = calc_map[method](prefactor=jpref.kcalmol_A)
    smearing = sr_cutoff_A / 5.0
    mesh_spacing = smearing / 8.0
    lr_wavelength = smearing / 2.0

    if method == "ewald":
        inputs = calc.prepare(
            atoms,
            chg,
            cutoff=sr_cutoff_A,
            smearing=smearing,
            lr_wavelength=lr_wavelength,
        )
    else:
        inputs = calc.prepare(
            atoms,
            chg,
            cutoff=sr_cutoff_A,
            mesh_spacing=mesh_spacing,
            smearing=smearing,
        )

    energy, forces = calc.energy_forces(*inputs)
    return CoulombResult(
        energy_kcalmol=float(energy),
        forces_kcalmol_A=np.asarray(forces, dtype=np.float64),
    )


def scafacos_coulomb_energy_forces(
    system: CoulombSystem,
    *,
    method: str = "ewald",
    parameters: dict[str, str | float | int] | None = None,
) -> CoulombResult:
    """Full periodic Coulomb via ScaFaCoS libfcs."""
    from mmml.interfaces.scafacosInterface import compute_scafacos_coulomb

    out = compute_scafacos_coulomb(
        system.positions_A,
        system.charges_e,
        box_length_A=system.box_length_A,
        method=method,
        parameters=parameters,
    )
    return CoulombResult(
        energy_kcalmol=float(out.energy_kcalmol),
        forces_kcalmol_A=np.asarray(out.forces_kcalmol_A, dtype=np.float64),
    )


def evaluate_backend(
    system: CoulombSystem,
    backend: BackendName,
    *,
    cutoff_A: float | None = DEFAULT_MM_COULOMB_CUTOFF_A,
    scafacos_method: str = "ewald",
    sr_cutoff_A: float = 6.0,
) -> CoulombResult:
    if backend == "mic":
        return mic_coulomb_energy_forces(system, cutoff_A=None)
    if backend == "mic_trunc":
        return mic_coulomb_energy_forces(system, cutoff_A=cutoff_A)
    if backend.startswith("jax_"):
        method = backend.removeprefix("jax_")
        return jax_pme_coulomb_energy_forces(
            system, method=method, sr_cutoff_A=sr_cutoff_A  # type: ignore[arg-type]
        )
    if backend == "scafacos":
        return scafacos_coulomb_energy_forces(system, method=scafacos_method)
    raise ValueError(f"unknown backend {backend!r}")


def madelung_constant(result: CoulombResult, system: CoulombSystem) -> float:
    """Madelung constant from total energy (jax-pme convention, unit cell size).

    Uses ``M = -E / (k * N_formula)`` with systems built at jax-pme reference
    cell sizes (CsCl ``L=1``, NaCl cubic ``L=2``).
    """
    return -result.energy_kcalmol / (
        CHARMM_COULOMB_KCAL * system.num_formula_units
    )


def compare_results(
    ref: CoulombResult,
    test: CoulombResult,
    *,
    energy_rtol: float = 1e-3,
    force_rtol: float = 5e-3,
    label: str = "",
) -> None:
    np.testing.assert_allclose(
        test.energy_kcalmol,
        ref.energy_kcalmol,
        rtol=energy_rtol,
        err_msg=f"{label} energy mismatch",
    )
    np.testing.assert_allclose(
        test.forces_kcalmol_A,
        ref.forces_kcalmol_A,
        rtol=force_rtol,
        err_msg=f"{label} force mismatch",
    )


def available_scafacos_methods() -> list[str]:
    from mmml.interfaces.scafacosInterface.scafacos_session import (
        SCAFACOS_DEFAULT_METHODS,
        scafacos_runtime_ok,
    )

    if not have_scafacos_library():
        return []
    return [m for m in SCAFACOS_DEFAULT_METHODS if scafacos_runtime_ok(method=m)]


def describe_environment() -> str:
    from mmml.interfaces.pycharmmInterface.long_range_backend import describe_lr_solver

    lines = [describe_lr_solver()]
    lines.append(f"jaxpme={'yes' if have_jax_pme_package() else 'no'}")
    lines.append(f"scafacos_lib={'yes' if have_scafacos_library() else 'no'}")
    if have_scafacos_library():
        methods = available_scafacos_methods()
        lines.append(f"scafacos_methods={','.join(methods) if methods else 'none probed'}")
    return "\n".join(lines)
