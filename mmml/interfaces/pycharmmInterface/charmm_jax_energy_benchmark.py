"""CHARMM vs JAX-MM energy benchmarks (PyCHARMM reference, jax-md parsers).

Reports per-term energy deltas and force RMS for supported CGENFF systems.
User-facing CLI: ``scripts/benchmark_charmm_jax_energy.py``.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal

import jax.numpy as jnp
import numpy as np

LayerName = Literal["bonded", "nonbonded", "total_mm"]

BONDED_TERM_MAP: dict[str, str] = {
    "bond": "bond",
    "angle": "angl",
    "urey": "urey",
    "torsion": "dihe",
    "improper": "impr",
    "cmap": "cmap",
}

DEFAULT_TOLERANCES: dict[LayerName, dict[str, float]] = {
    "bonded": {
        "energy_rtol": 2e-4,
        "energy_atol": 1e-4,
        "force_rtol": 5e-3,
        "force_atol": 1e-3,
    },
    "nonbonded": {
        "energy_rtol": 5e-4,
        "energy_atol": 1e-3,
        "force_rtol": 5e-3,
        "force_atol": 5e-3,
    },
    "total_mm": {
        "energy_rtol": 5e-4,
        "energy_atol": 2e-2,
        "force_rtol": 5e-3,
        "force_atol": 5e-2,
    },
}


@dataclass(frozen=True, slots=True)
class TermDelta:
    """Single energy term: PyCHARMM vs JAX."""

    term: str
    charmm_kcal: float
    jax_kcal: float
    abs_diff: float
    rel_diff: float

    @classmethod
    def from_pair(cls, term: str, charmm_kcal: float, jax_kcal: float) -> TermDelta:
        return cls(
            term=term,
            charmm_kcal=float(charmm_kcal),
            jax_kcal=float(jax_kcal),
            abs_diff=float(jax_kcal) - float(charmm_kcal),
            rel_diff=_relative_diff(float(charmm_kcal), float(jax_kcal)),
        )


@dataclass(frozen=True, slots=True)
class ForceDelta:
    force_rms: float
    force_max: float


@dataclass(frozen=True, slots=True)
class LayerBenchmark:
    layer: LayerName
    n_atoms: int
    terms: tuple[TermDelta, ...]
    forces: ForceDelta | None
    passed: bool
    message: str = ""


@dataclass(frozen=True, slots=True)
class SystemBenchmark:
    name: str
    description: str
    n_atoms: int
    layers: tuple[LayerBenchmark, ...]
    metadata: dict[str, Any] = field(default_factory=dict)


def _relative_diff(reference: float, value: float) -> float:
    denom = max(abs(reference), 1e-12)
    return (value - reference) / denom


def _force_delta(charmm_forces: np.ndarray, jax_forces: np.ndarray) -> ForceDelta:
    delta = np.asarray(jax_forces, dtype=np.float64) - np.asarray(charmm_forces, dtype=np.float64)
    return ForceDelta(
        force_rms=float(np.sqrt(np.mean(delta * delta))),
        force_max=float(np.max(np.abs(delta))),
    )


def _term_deltas_from_maps(
    jax_terms: dict[str, float],
    charmm_terms: dict[str, float],
    *,
    mapping: dict[str, str] | None = None,
) -> tuple[TermDelta, ...]:
    mapping = mapping or {k: k for k in jax_terms}
    out: list[TermDelta] = []
    for jax_key, charmm_key in mapping.items():
        if jax_key not in jax_terms or charmm_key not in charmm_terms:
            continue
        out.append(
            TermDelta.from_pair(jax_key, charmm_terms[charmm_key], jax_terms[jax_key])
        )
    return tuple(out)


def _assert_within_tolerance(
    layer: LayerName,
    jax_terms: dict[str, Any],
    jax_forces: np.ndarray,
    *,
    tolerances: dict[LayerName, dict[str, float]] | None = None,
    ignore_charmm_bonded_terms: tuple[str, ...] = ("cmap",),
) -> tuple[bool, str]:
    tol = (tolerances or DEFAULT_TOLERANCES)[layer]
    try:
        if layer == "bonded":
            from mmml.interfaces.pycharmmInterface.cgenff_bonded_reference import (
                compare_bonded_to_charmm,
            )

            compare_bonded_to_charmm(
                jax_terms,
                jax_forces,
                energy_rtol=tol["energy_rtol"],
                energy_atol=tol["energy_atol"],
                force_rtol=tol["force_rtol"],
                force_atol=tol["force_atol"],
            )
        elif layer == "nonbonded":
            from mmml.interfaces.pycharmmInterface.cgenff_bonded_reference import (
                compare_nonbonded_to_charmm,
            )

            compare_nonbonded_to_charmm(
                jax_terms,
                jax_forces,
                energy_rtol=tol["energy_rtol"],
                energy_atol=tol["energy_atol"],
                force_rtol=tol["force_rtol"],
                force_atol=tol["force_atol"],
            )
        else:
            from mmml.interfaces.pycharmmInterface.cgenff_bonded_reference import (
                compare_mm_system_to_charmm,
            )
            from mmml.interfaces.pycharmmInterface.mm_system_energy import (
                MmSystemEnergyResult,
            )

            result = MmSystemEnergyResult(
                bonded={},
                nonbonded={},
                total_energy=float(jax_terms["total"]),
                forces=np.asarray(jax_forces),
            )
            compare_mm_system_to_charmm(
                result,
                energy_rtol=tol["energy_rtol"],
                energy_atol=tol["energy_atol"],
                force_rtol=tol["force_rtol"],
                force_atol=tol["force_atol"],
                ignore_charmm_bonded_terms=ignore_charmm_bonded_terms,
            )
        return True, ""
    except AssertionError as exc:
        return False, str(exc)


def benchmark_bonded_layer(
    positions: np.ndarray,
    psf_path: Path | str,
    *,
    prm_path: Path | str | None = None,
) -> LayerBenchmark:
    """Bonded-only CHARMM BLOCK vs JAX ``bonded_energy_and_forces``."""
    from mmml.interfaces.pycharmmInterface.cgenff_bonded import bonded_energy_and_forces
    from mmml.interfaces.pycharmmInterface.cgenff_bonded_reference import (
        charmm_bonded_energy_components_kcalmol,
        charmm_bonded_forces_kcalmol_A,
        run_charmm_bonded_ener_force,
        set_charmm_positions,
        setup_bonded_only_charmm,
    )
    from mmml.interfaces.pycharmmInterface.mm_system_energy import load_bonded_system_from_psf

    pos = np.asarray(positions, dtype=np.float64)
    set_charmm_positions(pos)
    setup_bonded_only_charmm()
    run_charmm_bonded_ener_force(silent=True)
    charmm_terms = charmm_bonded_energy_components_kcalmol()
    charmm_forces = charmm_bonded_forces_kcalmol_A()

    bonded = load_bonded_system_from_psf(psf_path, pos, prm_file=prm_path)
    jax_terms_raw, jax_forces = bonded_energy_and_forces(
        jnp.asarray(pos),
        bonded.topology,
        bonded.bonded,
        urey_k=bonded.urey_k,
        urey_r0=bonded.urey_r0,
        energy_unit="kcal/mol",
    )
    jax_terms = {k: float(v) for k, v in jax_terms_raw.items()}
    jax_forces_np = np.asarray(jax_forces, dtype=np.float64)

    terms_list: list[TermDelta] = []
    for jax_key, charmm_key in BONDED_TERM_MAP.items():
        if jax_key not in jax_terms:
            continue
        if jax_key == "urey":
            charmm_val = float(charmm_terms.get("urey", 0.0)) + float(
                charmm_terms.get("ub", 0.0)
            )
            if "urey" not in charmm_terms and "ub" not in charmm_terms:
                continue
        elif charmm_key not in charmm_terms:
            continue
        else:
            charmm_val = float(charmm_terms[charmm_key])
        terms_list.append(
            TermDelta.from_pair(jax_key, charmm_val, jax_terms[jax_key])
        )
    if "total" in jax_terms:
        charmm_mapped_total = 0.0
        for k, charmm_key in BONDED_TERM_MAP.items():
            if k not in jax_terms:
                continue
            if k == "urey":
                charmm_mapped_total += float(charmm_terms.get("urey", 0.0)) + float(
                    charmm_terms.get("ub", 0.0)
                )
            elif charmm_key in charmm_terms:
                charmm_mapped_total += float(charmm_terms[charmm_key])
        terms_list.append(
            TermDelta.from_pair("total", charmm_mapped_total, jax_terms["total"])
        )
    terms = tuple(terms_list)

    passed, message = _assert_within_tolerance("bonded", jax_terms, jax_forces_np)
    return LayerBenchmark(
        layer="bonded",
        n_atoms=int(pos.shape[0]),
        terms=terms,
        forces=_force_delta(charmm_forces, jax_forces_np),
        passed=passed,
        message=message,
    )


def benchmark_nonbonded_layer(
    positions: np.ndarray,
    psf_path: Path | str,
    prm_path: Path | str,
    cell: np.ndarray,
    nbond_settings: Any,
) -> LayerBenchmark:
    """Nonbonded-only CHARMM BLOCK vs JAX ``nonbonded_energy_and_forces``."""
    from mmml.interfaces.pycharmmInterface.cgenff_bonded_reference import (
        charmm_bonded_forces_kcalmol_A,
        charmm_nonbonded_energy_components_kcalmol,
        run_charmm_nonbonded_ener_force,
        set_charmm_positions,
        setup_nonbonded_only_charmm,
    )
    from mmml.interfaces.pycharmmInterface.mm_system_energy import (
        load_nonbonded_system_from_charmm,
        nonbonded_energy_and_forces,
    )

    pos = np.asarray(positions, dtype=np.float64)
    set_charmm_positions(pos)
    setup_nonbonded_only_charmm()
    run_charmm_nonbonded_ener_force(silent=True)
    charmm_terms = charmm_nonbonded_energy_components_kcalmol()
    charmm_forces = charmm_bonded_forces_kcalmol_A()

    nbond_data = load_nonbonded_system_from_charmm(psf_path, prm_path)
    jax_terms_raw, jax_forces = nonbonded_energy_and_forces(
        pos,
        nbond_data,
        cell,
        nbond_settings,
    )
    jax_terms = {k: float(v) for k, v in jax_terms_raw.items()}
    jax_forces_np = np.asarray(jax_forces, dtype=np.float64)

    terms = _term_deltas_from_maps(jax_terms, charmm_terms)
    passed, message = _assert_within_tolerance("nonbonded", jax_terms, jax_forces_np)
    return LayerBenchmark(
        layer="nonbonded",
        n_atoms=int(pos.shape[0]),
        terms=terms,
        forces=_force_delta(charmm_forces, jax_forces_np),
        passed=passed,
        message=message,
    )


def benchmark_total_mm_layer(
    positions: np.ndarray,
    psf_path: Path | str,
    prm_path: Path | str,
    cell: np.ndarray,
    nbond_settings: Any,
) -> LayerBenchmark:
    """Full MM (bonded + MIC switched nonbonded) vs PyCHARMM ``ENER FORCE``."""
    import pycharmm.energy as energy

    from mmml.interfaces.pycharmmInterface.cgenff_bonded_reference import (
        charmm_bonded_energy_components_kcalmol,
        charmm_bonded_forces_kcalmol_A,
        charmm_nonbonded_energy_components_kcalmol,
        run_charmm_bonded_ener_force,
        set_charmm_positions,
    )
    from mmml.interfaces.pycharmmInterface.mlpot.block_terms import apply_charmm_mm_block
    from mmml.interfaces.pycharmmInterface.mm_system_energy import (
        load_bonded_system_from_psf,
        load_nonbonded_system_from_charmm,
        mm_system_energy_and_forces,
    )

    pos = np.asarray(positions, dtype=np.float64)
    set_charmm_positions(pos)
    apply_charmm_mm_block()
    run_charmm_bonded_ener_force(silent=True)

    charmm_bonded = charmm_bonded_energy_components_kcalmol()
    charmm_nb = charmm_nonbonded_energy_components_kcalmol()
    ignored_cmap = float(charmm_bonded.get("cmap", 0.0))
    charmm_total = float(energy.get_total()) - ignored_cmap
    charmm_forces = charmm_bonded_forces_kcalmol_A()

    bonded = load_bonded_system_from_psf(psf_path, pos, prm_file=prm_path)
    nbond_data = load_nonbonded_system_from_charmm(psf_path, prm_path)
    result = mm_system_energy_and_forces(
        pos,
        bonded,
        nbond_data,
        cell,
        nbond_settings,
    )
    jax_forces_np = np.asarray(result.forces, dtype=np.float64)
    jax_terms = {
        "bonded_total": float(sum(result.bonded.values())),
        "vdw": float(result.nonbonded["vdw"]),
        "elec": float(result.nonbonded["elec"]),
        "total": float(result.total_energy),
    }
    charmm_terms = {
        "bonded_total": float(charmm_bonded.get("total", 0.0)) - ignored_cmap,
        "vdw": float(charmm_nb["vdw"]),
        "elec": float(charmm_nb["elec"]),
        "total": charmm_total,
    }
    terms = _term_deltas_from_maps(jax_terms, charmm_terms)
    passed, message = _assert_within_tolerance(
        "total_mm",
        {"total": jax_terms["total"]},
        jax_forces_np,
    )
    return LayerBenchmark(
        layer="total_mm",
        n_atoms=int(pos.shape[0]),
        terms=terms,
        forces=_force_delta(charmm_forces, jax_forces_np),
        passed=passed,
        message=message,
    )


def perturb_positions(positions: np.ndarray, *, seed: int = 19, scale: float = 0.02) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return np.asarray(positions, dtype=np.float64) + rng.normal(scale=scale, size=positions.shape)


def load_tip3_monomer_from_charmm(
    *,
    workdir: Path | None = None,
) -> tuple[Path, np.ndarray]:
    """Load CGENFF TIP3 PSF + coordinates from the committed functionality fixture.

    Uses ``tests/functionality/pycharmmETC/pdb/initial.pdb`` (real CHARMM-minimized
    TIP3) instead of ``setupRes.generate_coordinates()`` so the benchmark is fast
    and does not depend on live minimization.
    """
    import pycharmm.generate as generate
    import pycharmm.read as read
    import pycharmm.settings as settings
    import pycharmm.write as write

    from mmml.interfaces.pycharmmInterface.cgenff_bonded_reference import (
        charmm_positions_xyz_array,
        read_pdb_file,
    )
    from mmml.interfaces.pycharmmInterface.charmm_levels import charmm_relaxed_bomlev
    from mmml.interfaces.pycharmmInterface.import_pycharmm import (
        CGENFF_PRM,
        CGENFF_RTF,
        crystal_free_charmm_for_param_append,
        pycharmm,
    )

    fixture_pdb = (
        Path(__file__).resolve().parents[3]
        / "tests"
        / "functionality"
        / "pycharmmETC"
        / "pdb"
        / "initial.pdb"
    )
    if not fixture_pdb.is_file():
        raise FileNotFoundError(f"Missing TIP3 fixture PDB: {fixture_pdb}")

    out_dir = Path(workdir or Path.cwd())
    out_dir.mkdir(parents=True, exist_ok=True)
    psf_path = out_dir / "tip3-1.psf"

    crystal_free_charmm_for_param_append()
    pycharmm.lingo.charmm_script("DELETE ATOM SELE ALL END")
    with charmm_relaxed_bomlev():
        read.rtf(CGENFF_RTF)
        read.prm(CGENFF_PRM)
    settings.set_verbosity(0)
    read.sequence_string("TIP3")
    generate.new_segment(seg_name="TIP3", setup_ic=False)
    read_pdb_file(fixture_pdb, resid=True)
    write.psf_card(str(psf_path))

    return psf_path.resolve(), charmm_positions_xyz_array()


def build_tip3_water_box(
    *,
    n_waters: int = 10,
    box_side_A: float = 28.0,
    seed: int = 11,
    workdir: Path | None = None,
    skip_reset_block: bool = True,
) -> tuple[Path, np.ndarray, np.ndarray, Any]:
    """Periodic TIP3-only box (grid placement, same recipe as tri-alanine waters)."""
    import os

    import pandas as pd
    import pycharmm.coor as coor
    import pycharmm.generate as generate
    import pycharmm.read as read
    import pycharmm.settings as settings
    import pycharmm.write as write

    from mmml.interfaces.pycharmmInterface.import_pycharmm import (
        crystal_free_charmm_for_param_append,
        pycharmm,
        reset_block,
    )
    from mmml.interfaces.pycharmmInterface.mlpot.pbc_env import (
        apply_pbc_nbonds,
        prepare_charmm_pbc,
    )
    from mmml.interfaces.pycharmmInterface.nbonds_config import PbcNbondCutoffs
    from mmml.interfaces.pycharmmInterface.trialanine_water_box import (
        _grid_oxygen_sites,
        _load_cgenff_with_trialanine,
        _tip3_template,
    )

    crystal_free_charmm_for_param_append()
    pycharmm.lingo.charmm_script("DELETE ATOM SELE ALL END")
    if not skip_reset_block:
        reset_block()

    _load_cgenff_with_trialanine()
    settings.set_verbosity(0)

    rng = np.random.default_rng(seed)
    tip3 = _tip3_template()
    tip3_com = tip3.mean(axis=0)
    existing = np.empty((0, 3), dtype=np.float64)
    oxygen_sites = _grid_oxygen_sites(
        n_waters=n_waters,
        box_side_A=box_side_A,
        spacing_A=2.85,
        margin_A=3.0,
        existing=existing,
        min_dist_A=2.4,
        rng=rng,
        water_template=tip3,
    )
    water_coords = np.vstack([site + (tip3 - tip3_com) for site in oxygen_sites])

    read.sequence_string(" ".join(["TIP3"] * n_waters))
    generate.new_segment(seg_name="SOLV", setup_ic=False)
    coor.set_positions(pd.DataFrame(water_coords, columns=["x", "y", "z"]))

    prepare_charmm_pbc(box_side_A)
    nbond_cutoffs = apply_pbc_nbonds(nbxmod=5, cubic_box_side_A=box_side_A)

    out_dir = Path(workdir or Path.cwd())
    out_dir.mkdir(parents=True, exist_ok=True)
    psf_path = out_dir / "tip3-water-box.psf"
    prev_cwd = os.getcwd()
    try:
        os.chdir(out_dir)
        write.psf_card(psf_path.name)
    finally:
        os.chdir(prev_cwd)

    positions = coor.get_positions()[["x", "y", "z"]].to_numpy(dtype=float)
    cell = np.diag([float(box_side_A)] * 3)
    return psf_path.resolve(), positions, cell, nbond_cutoffs


def _nbond_settings_from_cutoffs(cuts: Any) -> Any:
    from mmml.interfaces.pycharmmInterface.mm_system_energy import CharmmNbondSettings

    return CharmmNbondSettings(
        cutnb=float(cuts.cutnb),
        ctonnb=float(cuts.ctonnb),
        ctofnb=float(cuts.ctofnb),
    )


def run_tip3_monomer_benchmark(
    *,
    seed: int = 11,
    prm_path: Path | str | None = None,
    workdir: Path | None = None,
) -> SystemBenchmark:
    from mmml.interfaces.pycharmmInterface.import_pycharmm import CGENFF_PRM

    psf_path, positions = load_tip3_monomer_from_charmm(workdir=workdir)
    pos = perturb_positions(positions, seed=seed, scale=0.02)
    prm = prm_path or CGENFF_PRM
    bonded = benchmark_bonded_layer(pos, psf_path, prm_path=prm)
    return SystemBenchmark(
        name="tip3_monomer",
        description="CGENFF TIP3 monomer (make-res) — bonded terms only",
        n_atoms=int(pos.shape[0]),
        layers=(bonded,),
        metadata={"psf": str(psf_path), "seed": seed},
    )


def run_tip3_water_box_benchmark(
    *,
    n_waters: int = 10,
    box_side_A: float = 28.0,
    seed: int = 11,
    workdir: Path | None = None,
    prm_path: Path | str | None = None,
) -> SystemBenchmark:
    from mmml.interfaces.pycharmmInterface.import_pycharmm import CGENFF_PRM

    psf_path, positions, cell, cuts = build_tip3_water_box(
        n_waters=n_waters,
        box_side_A=box_side_A,
        seed=seed,
        workdir=workdir,
    )
    pos = perturb_positions(positions, seed=seed + 2, scale=0.02)
    prm = prm_path or CGENFF_PRM
    nb_settings = _nbond_settings_from_cutoffs(cuts)
    layers = (
        benchmark_bonded_layer(pos, psf_path, prm_path=prm),
        benchmark_nonbonded_layer(pos, psf_path, prm, cell, nb_settings),
        benchmark_total_mm_layer(pos, psf_path, prm, cell, nb_settings),
    )
    return SystemBenchmark(
        name="tip3_water_box",
        description=f"CGENFF {n_waters}× TIP3 in {box_side_A:.0f} Å cube (PBC)",
        n_atoms=int(pos.shape[0]),
        layers=layers,
        metadata={
            "psf": str(psf_path),
            "n_waters": n_waters,
            "box_side_A": box_side_A,
            "seed": seed,
        },
    )


def run_trialanine_water_benchmark(
    box: Any,
    *,
    seed: int = 23,
) -> SystemBenchmark:
    pos = perturb_positions(box.positions, seed=seed, scale=0.02)
    nb_settings = _nbond_settings_from_cutoffs(box.nbond_cutoffs)
    layers = (
        benchmark_bonded_layer(pos, box.psf_path, prm_path=box.cgenff_prm),
        benchmark_nonbonded_layer(pos, box.psf_path, box.cgenff_prm, box.cell, nb_settings),
        benchmark_total_mm_layer(pos, box.psf_path, box.cgenff_prm, box.cell, nb_settings),
    )
    return SystemBenchmark(
        name="trialanine_water",
        description="CGENFF TRIA peptide + TIP3 waters (28 Å cube)",
        n_atoms=int(pos.shape[0]),
        layers=layers,
        metadata={
            "psf": str(box.psf_path),
            "n_waters": int(box.n_waters),
            "box_side_A": float(box.box_side_A),
            "seed": seed,
        },
    )


SUPPORTED_CASES: tuple[str, ...] = ("tip3_monomer", "tip3_water_box", "trialanine_water")


def render_markdown_report(cases: tuple[SystemBenchmark, ...]) -> str:
    lines = [
        "# CHARMM vs JAX-MM energy benchmark",
        "",
        "PyCHARMM reference vs MMML JAX loaders (`cgenff_bonded`, `mm_system_energy`).",
        "Energies in kcal/mol; force RMS in kcal/mol/Å.",
        "",
    ]
    for case in cases:
        lines.append(f"## {case.name}")
        lines.append("")
        lines.append(case.description)
        lines.append(f"Atoms: {case.n_atoms}")
        lines.append("")
        for layer in case.layers:
            status = "PASS" if layer.passed else "FAIL"
            lines.append(f"### {layer.layer} — {status}")
            lines.append("")
            lines.append("| Term | CHARMM | JAX | Δ | rel Δ |")
            lines.append("|------|--------|-----|---|-------|")
            for term in layer.terms:
                lines.append(
                    f"| {term.term} | {term.charmm_kcal:.6f} | {term.jax_kcal:.6f} | "
                    f"{term.abs_diff:+.2e} | {term.rel_diff:+.2e} |"
                )
            if layer.forces is not None:
                lines.append("")
                lines.append(
                    f"Force RMS Δ: {layer.forces.force_rms:.4e}  "
                    f"max |ΔF|: {layer.forces.force_max:.4e}"
                )
            if layer.message:
                lines.append("")
                lines.append(f"Note: {layer.message[:200]}")
            lines.append("")
    n_pass = sum(1 for c in cases for ly in c.layers if ly.passed)
    n_total = sum(len(c.layers) for c in cases)
    lines.append(f"**Summary:** {n_pass}/{n_total} layers within tolerance")
    return "\n".join(lines) + "\n"


def render_json_report(cases: tuple[SystemBenchmark, ...]) -> str:
    payload = {
        "cases": [
            {
                **asdict(case),
                "layers": [asdict(layer) for layer in case.layers],
            }
            for case in cases
        ],
        "summary": {
            "layers_passed": sum(1 for c in cases for ly in c.layers if ly.passed),
            "layers_total": sum(len(c.layers) for c in cases),
        },
    }
    return json.dumps(payload, indent=2)


def all_layers_passed(cases: tuple[SystemBenchmark, ...]) -> bool:
    return all(layer.passed for case in cases for layer in case.layers)
