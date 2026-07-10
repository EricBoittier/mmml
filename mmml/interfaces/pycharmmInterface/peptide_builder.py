"""General peptide building, solvation, and QC validation helper functions in PyCHARMM."""

from __future__ import annotations

import os
import sys
import shutil
import warnings
import subprocess
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from ase import Atoms

from mmml.interfaces.pycharmmInterface.import_pycharmm import (
    ensure_pycharmm_loaded,
    CGENFF_PRM,
    CGENFF_RTF,
)
from mmml.interfaces.pycharmmInterface.protein_charmm_build import (
    ProteinToppar,
    protein_toppar_paths,
)
from mmml.interfaces.pycharmmInterface.nbonds_config import PbcNbondCutoffs
from mmml.interfaces.pycharmmInterface.utils import get_Z_from_psf
from mmml.interfaces.pycharmmInterface.charmm_levels import charmm_relaxed_bomlev
from mmml.interfaces.pycharmmInterface.import_pycharmm import CHARMM_HOME

ONE_TO_THREE = {
    "A": "ALA", "R": "ARG", "N": "ASN", "D": "ASP", "C": "CYS",
    "E": "GLU", "Q": "GLN", "G": "GLY", "H": "HSD", "I": "ILE",
    "L": "LEU", "K": "LYS", "M": "MET", "F": "PHE", "P": "PRO",
    "S": "SER", "T": "THR", "W": "TRP", "Y": "TYR", "V": "VAL"
}

THREE_TO_ONE = {value: key for key, value in ONE_TO_THREE.items()}
THREE_TO_ONE.update({"HSE": "H", "HSP": "H"})
PDB_RESNAME_TO_CHARMM = {
    "HIS": "HSD",
}





@dataclass(frozen=True, slots=True)
class PeptideBuildResult:
    """Peptide coordinates and metadata built in PyCHARMM."""
    positions: np.ndarray
    n_atoms: int
    segment: str
    sequence: list[str]
    psf_path: Path | None = None
    pdb_path: Path | None = None


@dataclass(frozen=True, slots=True)
class SolvatedPeptideBox:
    """Peptide + TIP3 water box built and PBC-configured in PyCHARMM."""
    positions: np.ndarray
    psf_path: Path
    pdb_path: Path
    box_side_A: float
    n_peptide_atoms: int
    n_waters: int
    nbond_cutoffs: PbcNbondCutoffs
    sequence: list[str]

    @property
    def cell(self) -> np.ndarray:
        side = float(self.box_side_A)
        return np.diag([side, side, side])


@dataclass
class QcReport:
    """Detailed Quality Control report of the built system."""
    is_valid: bool
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "is_valid": self.is_valid,
            "errors": self.errors,
            "warnings": self.warnings,
            "details": self.details,
        }


def parse_sequence(sequence: str | list[str]) -> list[str]:
    """Parse sequence string (1-letter/3-letter/hyphen/space separated) into list of 3-letter uppercase residues."""
    if isinstance(sequence, list):
        return [res.upper() for res in sequence]

    sequence = sequence.strip()
    if not sequence:
        return []

    # Try split by hyphen or space
    if "-" in sequence:
        parts = [p.strip().upper() for p in sequence.split("-") if p.strip()]
    elif " " in sequence:
        parts = [p.strip().upper() for p in sequence.split() if p.strip()]
    else:
        parts = []

    if not parts:
        seq_upper = sequence.upper()
        # Single 3-letter residue
        if len(seq_upper) == 3 and (seq_upper in ONE_TO_THREE.values() or seq_upper in ("HSD", "HSE", "HSP")):
            return [seq_upper]
        
        # 1-letter code sequence
        is_one_letter = True
        for char in seq_upper:
            if char not in ONE_TO_THREE:
                is_one_letter = False
                break
        if is_one_letter:
            return [ONE_TO_THREE[char] for char in seq_upper]

        # 3-letter chunk sequence (e.g. ALAPHEGLY)
        if len(sequence) % 3 == 0:
            chunks = [sequence[i : i + 3].upper() for i in range(0, len(sequence), 3)]
            # Validate that all chunks are valid 3-letter residue names
            all_valid = True
            for chunk in chunks:
                if chunk not in ONE_TO_THREE.values() and chunk not in ("HSD", "HSE", "HSP"):
                    all_valid = False
                    break
            if all_valid:
                return chunks

        raise ValueError(f"Could not parse residue sequence: {sequence}")

    res_list = []
    for p in parts:
        p_upper = p.upper()
        if len(p_upper) == 1 and p_upper in ONE_TO_THREE:
            res_list.append(ONE_TO_THREE[p_upper])
        elif len(p_upper) == 3 and (p_upper in ONE_TO_THREE.values() or p_upper in ("HSD", "HSE", "HSP")):
            res_list.append(p_upper)
        else:
            raise ValueError(f"Unknown residue code or representation: {p}")
    return res_list


def parse_peptide_patch_spec(spec: str | dict[str, Any]) -> tuple[str, str, dict[str, Any]]:
    """Normalize a CHARMM patch spec into patch name, sites, and options."""
    if isinstance(spec, str):
        parts = spec.strip().split(maxsplit=1)
        if len(parts) != 2:
            raise ValueError(
                "Peptide patch strings must look like 'PATCH SEGID RESID [SEGID RESID ...]'."
            )
        return parts[0].upper(), parts[1], {}

    if not isinstance(spec, dict):
        raise TypeError(f"Peptide patch specs must be strings or dictionaries, got {type(spec)!r}")

    name = str(spec.get("name") or spec.get("patch") or "").strip().upper()
    sites = str(spec.get("sites") or spec.get("site") or "").strip()
    if not name or not sites:
        raise ValueError("Peptide patch dictionaries require 'name'/'patch' and 'sites'.")

    options = spec.get("options") or {}
    if not isinstance(options, dict):
        raise TypeError("Peptide patch 'options' must be a dictionary when provided.")
    return name, sites, dict(options)


def _pdb_atom_chain_id(line: str) -> str:
    return line[21].strip() if len(line) > 21 else ""


def _pdb_atom_residue_key(line: str) -> tuple[str, str, str, str]:
    chain_id = _pdb_atom_chain_id(line)
    resname = line[17:20].strip().upper()
    resseq = line[22:26].strip()
    icode = line[26].strip() if len(line) > 26 else ""
    return chain_id, resseq, icode, resname


def _pdb_atom_altloc(line: str) -> str:
    return line[16].strip() if len(line) > 16 else ""


def first_protein_chain_id(pdb_path: Path | str) -> str:
    """Return the first chain ID containing a supported protein residue."""
    first_model_done = False
    with Path(pdb_path).open(encoding="utf-8", errors="replace") as handle:
        for line in handle:
            record = line[:6]
            if record.startswith("ENDMDL"):
                first_model_done = True
                continue
            if first_model_done:
                continue
            if not record.startswith("ATOM"):
                continue
            _chain, _resseq, _icode, resname = _pdb_atom_residue_key(line)
            if resname in THREE_TO_ONE:
                return _chain
    raise ValueError(f"No supported protein chain found in {pdb_path}.")


def download_rcsb_pdb(pdb_id: str, out_dir: Path | str) -> Path:
    """Download a legacy PDB file from RCSB into ``out_dir``."""
    pdb_code = pdb_id.strip().upper()
    if len(pdb_code) != 4 or not pdb_code.isalnum():
        raise ValueError(f"RCSB PDB IDs must be four alphanumeric characters, got {pdb_id!r}")

    out_path = Path(out_dir) / f"{pdb_code}.pdb"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    url = f"https://files.rcsb.org/download/{pdb_code}.pdb"
    urllib.request.urlretrieve(url, out_path)  # noqa: S310
    return out_path


def parse_pdb_chain_sequence(pdb_path: Path | str, chain_id: str | None = None) -> list[str]:
    """Return the protein residue sequence for one chain from a legacy PDB file."""
    selected_chain = chain_id.strip() if chain_id is not None else None
    residues: list[str] = []
    seen: set[tuple[str, str, str]] = set()
    first_model_done = False

    with Path(pdb_path).open(encoding="utf-8", errors="replace") as handle:
        for line in handle:
            record = line[:6]
            if record.startswith("ENDMDL"):
                first_model_done = True
                continue
            if first_model_done:
                continue
            if not record.startswith("ATOM"):
                continue
            altloc = _pdb_atom_altloc(line)
            if altloc not in ("", "A"):
                continue
            chain, resseq, icode, resname = _pdb_atom_residue_key(line)
            if selected_chain is not None and chain != selected_chain:
                continue
            charmm_resname = PDB_RESNAME_TO_CHARMM.get(resname, resname)
            if charmm_resname not in THREE_TO_ONE:
                raise ValueError(
                    f"Unsupported protein residue {resname!r} at chain {chain!r} "
                    f"residue {resseq}{icode} in {pdb_path}."
                )
            residue_key = (chain, resseq, icode)
            if residue_key not in seen:
                residues.append(charmm_resname)
                seen.add(residue_key)

    if not residues:
        chain_label = f" chain {selected_chain!r}" if selected_chain is not None else ""
        raise ValueError(f"No protein ATOM residues found in{chain_label} {pdb_path}.")
    return residues


def parse_pdb_ssbond_patches(
    pdb_path: Path | str,
    *,
    chain_id: str | None = None,
    seg_name: str = "PEPT",
) -> list[str]:
    """Infer CHARMM DISU patches from legacy PDB SSBOND records."""
    selected_chain = chain_id.strip() if chain_id is not None else None
    patches: list[str] = []

    with Path(pdb_path).open(encoding="utf-8", errors="replace") as handle:
        for line in handle:
            if not line.startswith("SSBOND"):
                continue
            parts = line.split()
            if len(parts) < 8:
                continue
            chain1, resid1 = parts[3], parts[4]
            chain2, resid2 = parts[6], parts[7]
            if selected_chain is not None and (chain1 != selected_chain or chain2 != selected_chain):
                continue
            patches.append(f"DISU {seg_name} {resid1} {seg_name} {resid2}")
    return patches


def write_pdb_chain_subset(
    pdb_path: Path | str,
    out_path: Path | str,
    *,
    chain_id: str | None = None,
) -> Path:
    """Write a first-model, protein-only, single-chain PDB coordinate file."""
    selected_chain = chain_id.strip() if chain_id is not None else None
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    wrote_atom = False
    first_model_done = False

    with Path(pdb_path).open(encoding="utf-8", errors="replace") as source, out.open(
        "w", encoding="utf-8"
    ) as dest:
        for line in source:
            record = line[:6]
            if record.startswith("ENDMDL"):
                first_model_done = True
                continue
            if first_model_done:
                continue
            if not record.startswith("ATOM"):
                continue
            altloc = _pdb_atom_altloc(line)
            if altloc not in ("", "A"):
                continue
            chain, _resseq, _icode, resname = _pdb_atom_residue_key(line)
            if selected_chain is not None and chain != selected_chain:
                continue
            if resname not in THREE_TO_ONE:
                continue
            if altloc == "A":
                line = line[:16] + " " + line[17:]
            dest.write(line)
            wrote_atom = True
        dest.write("END\n")

    if not wrote_atom:
        chain_label = f" chain {selected_chain!r}" if selected_chain is not None else ""
        raise ValueError(f"No protein ATOM records written for{chain_label} {pdb_path}.")
    return out


def prepare_rcsb_peptide_input(
    pdb_id: str,
    *,
    chain_id: str | None = None,
    workdir: Path | str | None = None,
    seg_name: str = "PEPT",
) -> dict[str, Any]:
    """Fetch an RCSB PDB entry and prepare sequence, patches, and coordinates for CHARMM."""
    out_dir = Path(workdir or Path.cwd())
    pdb_path = download_rcsb_pdb(pdb_id, out_dir)
    selected_chain = chain_id.strip() if chain_id is not None else first_protein_chain_id(pdb_path)
    sequence = parse_pdb_chain_sequence(pdb_path, chain_id=selected_chain)
    patches = parse_pdb_ssbond_patches(pdb_path, chain_id=selected_chain, seg_name=seg_name)
    chain_pdb_path = write_pdb_chain_subset(
        pdb_path,
        out_dir / f"{pdb_id.strip().upper()}_{selected_chain or 'blank'}_protein.pdb",
        chain_id=selected_chain,
    )
    return {
        "pdb_path": pdb_path,
        "chain_pdb_path": chain_pdb_path,
        "chain_id": selected_chain,
        "sequence": sequence,
        "peptide_patches": patches,
    }


def build_peptide_in_charmm(
    sequence: str | list[str],
    *,
    first_patch: str | None = "ACE",
    last_patch: str | None = "CT3",
    peptide_patches: list[str | dict[str, Any]] | None = None,
    initial_pdb_path: Path | str | None = None,
    seg_name: str = "PEPT",
    minimize: bool = True,
    mini_steps: int = 500,
    toppar: ProteinToppar | None = None,
    extra_rtfs: list[Path | str] | None = None,
    extra_prms: list[Path | str] | None = None,
    seed: int = 42,
    workdir: Path | None = None,
) -> PeptideBuildResult:
    """Build a peptide segment in PyCHARMM from a residue sequence."""
    if not ensure_pycharmm_loaded():
        raise RuntimeError("PyCHARMM not available. Check CHARMM_LIB_DIR and CHARMM_HOME.")

    import pycharmm.generate as generate
    import pycharmm.ic as ic
    import pycharmm.read as read
    import pycharmm.settings as settings
    import pycharmm.coor as coor
    import pycharmm.minimize as mini
    import pycharmm.lingo

    # Clear existing atoms in the CHARMM session to avoid conflicts
    pycharmm.lingo.charmm_script("DELETE ATOM SELE ALL END")

    res_list = parse_sequence(sequence)
    if not res_list:
        raise ValueError("Sequence cannot be empty.")

    # Prepare output directory
    out_dir = Path(workdir or Path.cwd())
    out_dir.mkdir(parents=True, exist_ok=True)

    # Initialize CHARMM settings
    settings.set_verbosity(0)
    settings.set_warn_level(-2)

    # Load parameters & generate segment under relaxed bomlev to prevent termination on toppar warnings
    from mmml.interfaces.pycharmmInterface.charmm_levels import charmm_relaxed_bomlev
    with charmm_relaxed_bomlev(-5):
        # 1. Load CGENFF topology and parameters first
        from mmml.interfaces.pycharmmInterface.import_pycharmm import CGENFF_RTF
        from mmml.interfaces.pycharmmInterface.nbonds_config import (
            read_cgenff_prm,
            _rtf_path_without_drude_autogen,
        )
        read.rtf(_rtf_path_without_drude_autogen(CGENFF_RTF))
        read_cgenff_prm(bomlev=False)

        # 2. Append standard protein topology and parameters second
        tp = toppar or protein_toppar_paths()
        read.rtf(str(tp.rtf), append=True)
        read.prm(str(tp.prm), append=True)

        # 3. Load extra RTF/PRM files if provided
        if extra_rtfs:
            for rtf_file in extra_rtfs:
                read.rtf(str(rtf_file), append=True)
        if extra_prms:
            for prm_file in extra_prms:
                read.prm(str(prm_file), append=True)

        # Read sequence and build
        read.sequence_string(" ".join(res_list))
        generate.new_segment(
            seg_name=seg_name,
            first_patch=first_patch,
            last_patch=last_patch,
            setup_ic=True,
        )
        for patch_spec in peptide_patches or []:
            patch_name, patch_sites, patch_options = parse_peptide_patch_spec(patch_spec)
            generate.patch(patch_name, patch_sites, **patch_options)

        if initial_pdb_path is not None:
            read.pdb(str(initial_pdb_path), resid=True)

    # Seed the IC table and build
    # If using acetylated/neutral terminus, we seed on ACE (res 1), otherwise on standard N-terminus
    if first_patch == "ACE":
        ic.seed(1, "CAY", 1, "CY", 1, "N")
    else:
        ic.seed(1, "N", 1, "CA", 1, "C")

    # Fill ICs from parameters and build coordinates
    from mmml.interfaces.pycharmmInterface.nbonds_config import ic_prm_fill
    ic_prm_fill(replace_all=True)
    ic.build()

    # Check for unbuilt/collapsed coordinates and seed randomly if needed
    pos = coor.get_positions()[["x", "y", "z"]].to_numpy(dtype=float).copy()
    if np.any(np.abs(pos) > 9000.0) or float(np.std(pos)) < 0.05:
        # Uniform random distribution to separate atoms before minimization
        rng = np.random.default_rng(seed)
        mask = np.any(np.abs(pos) > 9000.0, axis=1) | (float(np.std(pos)) < 0.05)
        pos[mask] = rng.uniform(0.5, 2.5, size=(np.sum(mask), 3))
        coor.set_positions(pd.DataFrame(pos, columns=["x", "y", "z"]))

    # Optional vacuum minimization to relax IC layout
    if minimize:
        from mmml.interfaces.pycharmmInterface.nbonds_config import (
            apply_nbonds_kwargs,
            vacuum_nbond_kwargs,
        )
        apply_nbonds_kwargs(vacuum_nbond_kwargs(nbxmod=1))
        mini.run_abnr(nstep=int(mini_steps), tolenr=1e-3, tolgrd=1e-3)
        
        # Jitter slightly and do full nonbonded minimization to avoid saddle points
        pos = coor.get_positions()[["x", "y", "z"]].to_numpy(dtype=float).copy()
        rng = np.random.default_rng(seed + 1)
        pos += rng.uniform(-0.02, 0.02, size=pos.shape)
        coor.set_positions(pd.DataFrame(pos, columns=["x", "y", "z"]))
        
        apply_nbonds_kwargs(vacuum_nbond_kwargs(nbxmod=5))
        mini.run_abnr(nstep=int(mini_steps), tolenr=1e-3, tolgrd=1e-3)

    import pycharmm.write as write
    pdb_path = out_dir / f"{seg_name.lower()}.pdb"
    psf_path = out_dir / f"{seg_name.lower()}.psf"

    write.coor_pdb(str(pdb_path))
    write.psf_card(str(psf_path))

    pos = coor.get_positions()[["x", "y", "z"]].to_numpy(dtype=float)
    return PeptideBuildResult(
        positions=pos,
        n_atoms=int(pos.shape[0]),
        segment=seg_name,
        sequence=res_list,
        psf_path=psf_path,
        pdb_path=pdb_path,
    )


def solvate_peptide_in_charmm(
    peptide_result: PeptideBuildResult,
    *,
    box_side_A: float = 28.0,
    n_waters: int | None = None,
    margin_A: float = 1.5,
    water_spacing_A: float = 1.85,
    min_peptide_water_dist_A: float = 1.4,
    seed: int = 42,
    workdir: Path | None = None,
    use_packmol: bool = True,
) -> SolvatedPeptideBox:
    """Solvate the built peptide in a periodic water box."""
    import pycharmm.coor as coor
    import pycharmm.generate as generate
    import pycharmm.read as read
    import pycharmm.write as write
    import pycharmm.settings as settings

    from mmml.interfaces.pycharmmInterface.mlpot.pbc_env import (
        apply_pbc_nbonds,
        prepare_charmm_pbc,
    )
    from mmml.interfaces.pycharmmInterface.trialanine_water_box import (
        _grid_oxygen_sites,
        _tip3_template,
    )

    out_dir = Path(workdir or Path.cwd())
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)

    # 1. Translate peptide to origin (0, 0, 0)
    peptide_pos = coor.get_positions()[["x", "y", "z"]].to_numpy(dtype=float).copy()
    peptide_center = peptide_pos.mean(axis=0)
    peptide_pos -= peptide_center
    coor.set_positions(pd.DataFrame(peptide_pos, columns=["x", "y", "z"]))

    # 2. Check box suitability & warn if peptide spans too much
    spans = np.ptp(peptide_pos, axis=0)
    max_span = np.max(spans)
    required_box = max_span + 12.0 # 6 Å buffer on each side
    if box_side_A < required_box:
        warnings.warn(
            f"Box side length ({box_side_A:.2f} Å) may be too small for the peptide. "
            f"Peptide maximum span is {max_span:.2f} Å. "
            f"Recommended box side is at least {required_box:.2f} Å to prevent periodic image interactions.",
            UserWarning,
        )

    # 3. Determine number of waters if not provided
    # Standard TIP3 density is ~0.0334 molecules per Å^3
    if n_waters is None:
        peptide_vol = len(peptide_pos) * 15.0 # approximate volume per peptide atom
        box_vol = box_side_A ** 3
        water_vol = max(0.0, box_vol - peptide_vol)
        n_waters = int(0.0334 * water_vol)
        n_waters = max(10, n_waters)

    water_coords = None
    use_grid_fallback = not use_packmol

    if use_packmol:
        from mmml.interfaces.pycharmmInterface.packmol_placement import packmol_executable
        from ase.io import write as ase_write
        from ase.io import read as ase_read
        from mmml.paths import bundled_file

        try:
            packmol_bin = packmol_executable()
            pep_z = get_Z_from_psf()
            pep_atoms = Atoms(pep_z, peptide_pos)
            pep_pdb_path = out_dir / "solute_peptide.pdb"
            ase_write(str(pep_pdb_path), pep_atoms)

            tip3_pdb_src = bundled_file("data", "charmm", "tip3.pdb")
            shutil.copy(tip3_pdb_src, out_dir / tip3_pdb_src.name)

            packmol_inp_path = out_dir / "packmol_solvate.inp"
            packmol_out_path = out_dir / "packmol_solvate_out.pdb"
            h_side = box_side_A / 2 - margin_A

            packmol_input = f"""
tolerance 2.0
filetype pdb
output {packmol_out_path.name}
seed {rng.integers(1000000)}

structure {pep_pdb_path.name}
  number 1
  fixed 0.0 0.0 0.0 0.0 0.0 0.0
end structure

structure {tip3_pdb_src.name}
  number {n_waters}
  inside box {-h_side:.2f} {-h_side:.2f} {-h_side:.2f} {h_side:.2f} {h_side:.2f} {h_side:.2f}
end structure
"""
            packmol_inp_path.write_text(packmol_input, encoding="utf-8")
            subprocess.run([packmol_bin, "-i", packmol_inp_path.name], cwd=out_dir, check=True, capture_output=True)

            packed_atoms = ase_read(str(packmol_out_path))
            water_coords = packed_atoms.get_positions()[len(peptide_pos):]
        except Exception as exc:
            warnings.warn(
                f"Packmol solvation failed, falling back to grid-based placement. Error: {exc}",
                UserWarning,
            )
            use_grid_fallback = True

    if use_grid_fallback:
        tip3 = _tip3_template()
        tip3_com = tip3.mean(axis=0)
        # Shift existing peptide_pos to be positive for the grid placement check
        shift = np.array([box_side_A / 2, box_side_A / 2, box_side_A / 2])
        oxygen_sites = _grid_oxygen_sites(
            n_waters=n_waters,
            box_side_A=box_side_A,
            spacing_A=water_spacing_A,
            margin_A=margin_A,
            existing=peptide_pos + shift,
            min_dist_A=min_peptide_water_dist_A,
            rng=rng,
            water_template=tip3,
        )
        water_coords = np.vstack([site - shift + (tip3 - tip3_com) for site in oxygen_sites])

    # 4. Load waters in CHARMM and apply coordinates
    read.sequence_string(" ".join(["TIP3"] * n_waters))
    generate.new_segment(seg_name="SOLV", setup_ic=False)
    all_pos = np.vstack([peptide_pos, water_coords])
    coor.set_positions(pd.DataFrame(all_pos, columns=["x", "y", "z"]))

    # 5. Set up PBC in CHARMM
    prepare_charmm_pbc(box_side_A)
    nbond_cutoffs = apply_pbc_nbonds(nbxmod=5, cubic_box_side_A=box_side_A)

    # 6. Save PSF and PDB cards
    psf_path = out_dir / f"{peptide_result.segment.lower()}-water.psf"
    pdb_path = out_dir / f"{peptide_result.segment.lower()}-water.pdb"

    # Avoid issues with relative directory writes in MPI setups by changing dir safely
    import os
    prev_cwd = os.getcwd()
    try:
        os.chdir(out_dir)
        write.psf_card(psf_path.name)
        write.coor_pdb(pdb_path.name)
    finally:
        os.chdir(prev_cwd)

    final_positions = coor.get_positions()[["x", "y", "z"]].to_numpy(dtype=float)

    return SolvatedPeptideBox(
        positions=final_positions,
        psf_path=psf_path,
        pdb_path=pdb_path,
        box_side_A=float(box_side_A),
        n_peptide_atoms=peptide_result.n_atoms,
        n_waters=n_waters,
        nbond_cutoffs=nbond_cutoffs,
        sequence=peptide_result.sequence,
    )


def parse_psf_details(psf_path: Path) -> dict[str, Any]:
    """Parse atom elements, masses, segment IDs, charges, and bonds from a CHARMM PSF file."""
    atoms = []
    bonds = []
    
    with open(psf_path, "r", encoding="utf-8", errors="replace") as f:
        lines = f.readlines()
        
    idx = 0
    while idx < len(lines):
        line = lines[idx].strip()
        if not line:
            idx += 1
            continue
            
        if "!NATOM" in line:
            n_atoms = int(line.split()[0])
            idx += 1
            for _ in range(n_atoms):
                atom_line = lines[idx].strip()
                parts = atom_line.split()
                # Determine element from mass
                mass = float(parts[7])
                import ase.data
                mdif = (ase.data.atomic_masses_common - mass) ** 2
                atomic_num = int(np.argmin(mdif))
                
                atoms.append({
                    "id": int(parts[0]) - 1, # 0-indexed
                    "segment": parts[1],
                    "resid": parts[2],
                    "resname": parts[3],
                    "name": parts[4],
                    "type": parts[5],
                    "charge": float(parts[6]),
                    "mass": mass,
                    "element": atomic_num,
                })
                idx += 1
            continue
            
        if "!NBOND" in line:
            n_bonds = int(line.split()[0])
            idx += 1
            bond_flat = []
            while len(bond_flat) < 2 * n_bonds:
                bond_line = lines[idx].strip()
                if not bond_line:
                    idx += 1
                    continue
                bond_flat.extend([int(x) - 1 for x in bond_line.split()])
                idx += 1
            
            for i in range(0, len(bond_flat), 2):
                bonds.append((bond_flat[i], bond_flat[i+1]))
            continue
            
        idx += 1
        
    return {
        "atoms": atoms,
        "bonds": bonds,
    }


def infer_charge_and_spin_from_psf(psf_path: Path | str) -> tuple[int, float]:
    """Infer the total system charge and spin multiplicity from a PSF file.

    Returns
    -------
    total_charge : int
        The total charge rounded to the nearest integer.
    multiplicity : float
        The spin multiplicity (1.0 for even electron count, 2.0 for odd).
    """
    psf_data = parse_psf_details(Path(psf_path))
    atoms = psf_data["atoms"]

    total_q = sum(a["charge"] for a in atoms)
    total_charge = int(round(total_q))

    total_z = sum(a["element"] for a in atoms)
    n_electrons = total_z - total_charge

    multiplicity = 1.0 if n_electrons % 2 == 0 else 2.0
    print("psf: ", psf_path)
    print("total_charge: ", total_charge)
    print("multiplicity: ", multiplicity)
    return total_charge, multiplicity



def qc_built_system(
    positions: np.ndarray,
    psf_path: Path | str,
    *,
    box_side_A: float | None = None,
    min_nonbonded_dist_A: float = 0.85,
    min_hx_bond_A: float = 0.75,
    max_hx_bond_A: float = 1.30,
    min_heavy_bond_A: float = 1.00,
    max_heavy_bond_A: float = 2.10,
    check_energy: bool = True,
) -> QcReport:
    """Perform Quality Control checking on PSF, atom coordinates, bonds, and clashes."""
    psf_p = Path(psf_path)
    if not psf_p.is_file():
        return QcReport(is_valid=False, errors=[f"PSF file not found: {psf_path}"])

    errors: list[str] = []
    warnings: list[str] = []
    details: dict[str, Any] = {}

    # 1. Parse PSF
    psf_data = parse_psf_details(psf_p)
    atoms = psf_data["atoms"]
    bonds = psf_data["bonds"]

    # 2. Check atom count match
    if positions.shape[0] != len(atoms):
        errors.append(
            f"Positions atom count ({positions.shape[0]}) does not match PSF NATOM ({len(atoms)})"
        )
        return QcReport(is_valid=False, errors=errors)

    # 3. Check undefined/placeholder coordinates (9999.0, NaN, Inf)
    placeholder_indices = np.where(np.any(np.abs(positions) > 9000.0, axis=1) | np.isnan(positions).any(axis=1) | np.isinf(positions).any(axis=1))[0]
    if len(placeholder_indices) > 0:
        errors.append(
            f"Found {len(placeholder_indices)} atoms with placeholder, NaN, or infinite coordinates. "
            f"Examples: indices {placeholder_indices[:10].tolist()}"
        )

    # 4. Check bond lengths
    bad_bonds = []
    for u, v in bonds:
        # Distance calculation
        pos_u = positions[u]
        pos_v = positions[v]
        d = float(np.linalg.norm(pos_u - pos_v))

        # Classify bond based on elements
        elem_u = atoms[u]["element"]
        elem_v = atoms[v]["element"]
        is_h_bond = (elem_u == 1) or (elem_v == 1)
        is_h_h_bond = (elem_u == 1) and (elem_v == 1)

        if is_h_h_bond:
            # Water H-H rigid constraint (should be around 1.514 Å)
            if d < 1.40 or d > 1.65:
                bad_bonds.append({
                    "atoms": (u, v),
                    "names": (atoms[u]["name"], atoms[v]["name"]),
                    "resids": (atoms[u]["resid"], atoms[v]["resid"]),
                    "elements": (elem_u, elem_v),
                    "length": d,
                    "type": "H-H",
                })
        elif is_h_bond:
            if d < min_hx_bond_A or d > max_hx_bond_A:
                bad_bonds.append({
                    "atoms": (u, v),
                    "names": (atoms[u]["name"], atoms[v]["name"]),
                    "resids": (atoms[u]["resid"], atoms[v]["resid"]),
                    "elements": (elem_u, elem_v),
                    "length": d,
                    "type": "H-X",
                })
        else:
            if d < min_heavy_bond_A or d > max_heavy_bond_A:
                bad_bonds.append({
                    "atoms": (u, v),
                    "names": (atoms[u]["name"], atoms[v]["name"]),
                    "resids": (atoms[u]["resid"], atoms[v]["resid"]),
                    "elements": (elem_u, elem_v),
                    "length": d,
                    "type": "Heavy-Heavy",
                })

    details["bond_violations"] = bad_bonds
    if bad_bonds:
        # Sort violations to report the most egregious ones first
        bad_bonds.sort(key=lambda x: abs(x["length"] - (1.0 if x["type"] == "H-X" else 1.5)), reverse=True)
        errors.append(
            f"Found {len(bad_bonds)} bond length violations (most severe: "
            f"{bad_bonds[0]['names'][0]} - {bad_bonds[0]['names'][1]} = {bad_bonds[0]['length']:.3f} Å)."
        )

    # 5. Check steric clashes (non-bonded contacts)
    clashes = []
    # Vectorized check for smaller structures or chunked check to avoid memory blow-up
    n_atoms = len(atoms)
    if n_atoms <= 4000:
        # Pairwise distance matrix
        dists = np.linalg.norm(positions[:, None, :] - positions[None, :, :], axis=-1)
        np.fill_diagonal(dists, np.inf)
        
        # Mask bonded 1-2 pairs
        for u, v in bonds:
            dists[u, v] = np.inf
            dists[v, u] = np.inf
            
        clash_indices = np.argwhere(dists < min_nonbonded_dist_A)
        # Select unique pairs (u < v)
        clash_pairs = clash_indices[clash_indices[:, 0] < clash_indices[:, 1]]
        
        for u, v in clash_pairs:
            clashes.append({
                "atoms": (int(u), int(v)),
                "names": (atoms[u]["name"], atoms[v]["name"]),
                "segments": (atoms[u]["segment"], atoms[v]["segment"]),
                "distance": float(dists[u, v]),
            })
    else:
        # Chunked or spatial checking for larger systems
        for i in range(n_atoms):
            diffs = positions[i+1:] - positions[i]
            dists = np.linalg.norm(diffs, axis=-1)
            # Find close indices
            for local_j in np.where(dists < min_nonbonded_dist_A)[0]:
                j = i + 1 + int(local_j)
                # Check if bonded
                if (i, j) in bonds or (j, i) in bonds:
                    continue
                clashes.append({
                    "atoms": (i, j),
                    "names": (atoms[i]["name"], atoms[j]["name"]),
                    "segments": (atoms[i]["segment"], atoms[j]["segment"]),
                    "distance": float(dists[local_j]),
                })

    details["steric_clashes"] = clashes
    if clashes:
        errors.append(
            f"Found {len(clashes)} steric clashes closer than {min_nonbonded_dist_A} Å "
            f"(closest: {clashes[0]['names'][0]} - {clashes[0]['names'][1]} = {clashes[0]['distance']:.3f} Å)."
        )

    # 6. Check net charges
    seg_charges: dict[str, float] = {}
    for atom in atoms:
        seg = atom["segment"]
        seg_charges[seg] = seg_charges.get(seg, 0.0) + atom["charge"]
    
    details["segment_charges"] = seg_charges
    for seg, chg in seg_charges.items():
        if not np.isclose(chg, round(chg), atol=1e-3):
            warnings.append(
                f"Segment '{seg}' has non-integer net charge: {chg:.4f} (expected integer)."
            )

    # 7. Check current energy in active PyCHARMM session if desired and active
    if check_energy and ensure_pycharmm_loaded():
        try:
            from mmml.interfaces.pycharmmInterface.protein_charmm_build import charmm_total_energy_kcalmol
            e_tot = charmm_total_energy_kcalmol()
            details["charmm_energy"] = e_tot
            if np.isnan(e_tot) or np.isinf(e_tot) or e_tot > 1e6:
                errors.append(f"PyCHARMM active system has suspicious energy: {e_tot:.3f} kcal/mol")
        except Exception as exc:
            warnings.warn(f"Could not check CHARMM total energy: {exc}", UserWarning)

    is_valid = len(errors) == 0
    return QcReport(is_valid=is_valid, errors=errors, warnings=warnings, details=details)
