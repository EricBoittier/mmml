"""CHARMM36 protein toppar helpers and small peptide builders (PyCHARMM)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True, slots=True)
class ProteinToppar:
    """Paths to CHARMM all36 protein RTF/PRM under ``CHARMM_HOME/toppar``."""

    rtf: Path
    prm: Path


@dataclass(frozen=True, slots=True)
class AladBuildResult:
    """Alanine dipeptide (ACE–ALA–CT3) built in the active PyCHARMM session."""

    positions: np.ndarray
    n_atoms: int
    segment: str = "ALAD"


def protein_toppar_paths() -> ProteinToppar:
    """Return protein ``top_all36_prot.rtf`` + ``par_all36m_prot.prm`` (or ``par_all36_prot.prm``)."""
    from mmml.interfaces.pycharmmInterface.import_pycharmm import CHARMM_HOME

    base = Path(CHARMM_HOME) / "toppar"
    rtf = base / "top_all36_prot.rtf"
    prm = base / "par_all36m_prot.prm"
    if not prm.is_file():
        prm = base / "par_all36_prot.prm"
    if not rtf.is_file() or not prm.is_file():
        raise FileNotFoundError(
            f"Protein toppar not found under {base}. "
            "Set CHARMM_HOME to a full CHARMM installation with protein parameters."
        )
    return ProteinToppar(rtf=rtf, prm=prm)


def build_alad_dipeptide(
    *,
    minimize: bool = True,
    mini_steps: int = 500,
    cutnb: float = 16.0,
) -> AladBuildResult:
    """Build ACE–ALA–CT3 (segment ``ALAD``) in the active PyCHARMM session."""
    from pycharmm import generate, ic, minimize, read, settings
    from pycharmm.scripts import NonBondedScript

    toppar = protein_toppar_paths()
    settings.set_verbosity(5)
    settings.set_warn_level(-5)
    read.rtf(str(toppar.rtf))
    read.prm(str(toppar.prm))
    read.sequence_string("ALA")
    generate.new_segment(
        seg_name="ALAD",
        first_patch="ACE",
        last_patch="CT3",
        setup_ic=True,
    )
    ic.prm_fill(replace_all=True)
    ic.seed(1, "CAY", 1, "CY", 1, "N")
    ic.build()
    NonBondedScript(
        cutnb=cutnb,
        ctofnb=min(cutnb - 2.0, 14.0),
        ctonnb=min(cutnb - 4.0, 12.0),
        atom=True,
        vatom=True,
        eps=1,
        switch=True,
        vswitch=True,
        cdie=True,
    ).run()
    if minimize:
        minimize.run_abnr(nstep=int(mini_steps), tolenr=1e-3, tolgrd=1e-3)

    import pycharmm.coor as coor

    positions = coor.get_positions()[["x", "y", "z"]].to_numpy(dtype=np.float64)
    return AladBuildResult(positions=positions, n_atoms=int(positions.shape[0]))


def write_alad_artifacts(
    output_dir: Path | str,
    *,
    minimize: bool = True,
) -> tuple[Path, Path, AladBuildResult]:
    """Build ALAD, write ``alad.pdb`` + ``alad.psf`` under ``output_dir``."""
    import pycharmm.write as write

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    result = build_alad_dipeptide(minimize=minimize)
    pdb_path = out / "alad.pdb"
    psf_path = out / "alad.psf"
    write.pdb(str(pdb_path))
    write.psf_card(str(psf_path))
    return pdb_path, psf_path, result


def charmm_total_energy_kcalmol() -> float:
    """Return CHARMM ``ENER`` total (kcal/mol) for the active structure."""
    from pycharmm import energy

    return float(energy.get_total())
