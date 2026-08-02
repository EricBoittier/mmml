# Standard library imports
import os
import sys
import shutil
from pathlib import Path

# Third-party scientific computing
import numpy as np

# ASE imports
import ase
import ase.io
from ase import Atoms

from mmml.interfaces.pycharmmInterface.import_pycharmm import (
    CGENFF_RTF, CGENFF_PRM, CHARMM_HOME, CHARMM_LIB_DIR
)
from mmml.interfaces.pycharmmInterface.pycharmmCommands import pbcset
os.environ["CHARMM_HOME"] = CHARMM_HOME
os.environ["CHARMM_LIB_DIR"] = CHARMM_LIB_DIR

print(CHARMM_HOME)
print(CHARMM_LIB_DIR)
sys.path.append(str(Path(CHARMM_HOME) / "tool" / "pycharmm"))

# CHARMM imports
import pycharmm
import pycharmm.minimize as minimize
import pycharmm.write as write
import pycharmm.lingo

# import simple scripts
from mmml.interfaces.pycharmmInterface.pycharmmCommands import CLEAR_CHARMM


# unit registry
try:
    from pint import UnitRegistry
    ureg = UnitRegistry()
    Q_ = ureg.Quantity
    _PINT_AVAILABLE = True
except ImportError:
    _PINT_AVAILABLE = False
    ureg = None
    Q_ = None
    import warnings
    warnings.warn(
        "pint is not installed. Some functions requiring unit conversions "
        "(e.g., determine_n_molecules_from_density) will not work. "
        "Install pint with: pip install pint or conda install -c conda-forge pint",
        ImportWarning
    )


cwd = Path(__file__).parent

water_pdb_path = cwd / ".." / ".." / "data" / "charmm" / "tip3.pdb"
octanol_pdb_path = cwd / ".." / ".." / "data" / "charmm" / "ocoh.pdb"
ase_water = ase.io.read(str(water_pdb_path))
ase_octanol = ase.io.read(str(octanol_pdb_path))

def correct_names(atoms: Atoms) -> Atoms:
    """
    Corrects the names of the atoms in the atoms object
    """
    problem_symbols = ["CL", "HO"]
    e = atoms.get_chemical_symbols()
    e = [_[:1] if _.upper() in problem_symbols else _ for _ in e]
    e = [_ if _[0] != "H" else "H" for _ in e]
    an = [ase.data.chemical_symbols.index(_) for _ in e]
    atoms.set_atomic_numbers(an)
    return atoms

water = correct_names(ase_water)    
octanol = correct_names(ase_octanol)

# Legacy keys kept for backward compatibility; prefer CGenFF RESI names (TIP3, OCOH, …).
solvents_ase = {
    "water": water,
    "octanol": octanol,
    "TIP3": water,
    "OCOH": octanol,
}
solvents_density = {
    "water": 1000,
    "octanol": 824,
    "TIP3": 1000,
    "OCOH": 824,
}


def _normalize_solvent_key(solvent: str) -> str:
    from mmml.interfaces.pycharmmInterface.cgenff_residues import (
        require_cgenff_residue_name,
    )

    return require_cgenff_residue_name(solvent)


def _resolve_solvent_atoms(solvent: str) -> Atoms:
    """Return solvent monomer atoms for any CGenFF residue name."""
    from mmml.analysis.residue_geometry import load_residue_monomer_atoms

    name = _normalize_solvent_key(solvent)
    if name in solvents_ase:
        return solvents_ase[name].copy()
    # Legacy lowercase aliases still present in solvents_ase.
    legacy = {"TIP3": "water", "OCOH": "octanol"}.get(name)
    if legacy is not None and legacy in solvents_ase:
        return solvents_ase[legacy].copy()
    return load_residue_monomer_atoms(name, generate=True)


def _resolve_solvent_density_kg_m3(solvent: str, density: float | None) -> float:
    from mmml.analysis.residue_geometry import resolve_solvent_density_kg_m3

    name = _normalize_solvent_key(solvent)
    if density is not None:
        return resolve_solvent_density_kg_m3(name, density)
    if name in solvents_density:
        return float(solvents_density[name])
    legacy = {"TIP3": "water", "OCOH": "octanol"}.get(name)
    if legacy is not None and legacy in solvents_density:
        return float(solvents_density[legacy])
    return resolve_solvent_density_kg_m3(name, None)

def read_initial_pdb(cwd: Path) -> Atoms:
    """Reads the initial PDB file and returns an ASE Atoms object.

    Falls back to a whitespace-tolerant ATOM parser when ASE's fixed-column
    reader rejects CGenFF residue names (e.g. 5-character ``CH3CL``).
    """
    pdb_path = cwd / "pdb" / "initial.pdb"
    try:
        mol = ase.io.read(pdb_path)
    except (ValueError, IndexError, StopIteration) as exc:
        print(f"ase.io.read failed on {pdb_path} ({exc}); using split-based fallback")
        mol = _read_pdb_atoms_split(pdb_path)
    e = mol.get_chemical_symbols()
    print(mol)
    print(e)
    mol.set_chemical_symbols(
        [
            (
                _[:1]
                if _.upper()
                not in [
                    "CL",
                ]
                else _
            )
            for _ in e
        ]
    )
    return mol


def _element_from_pdb_atom_name(name: str, trailing: str | None = None) -> str:
    """Map a PDB atom name / trailing element token to an ASE chemical symbol."""
    if trailing:
        t = trailing.strip()
        if t.upper() == "CL":
            return "Cl"
        if len(t) <= 2 and t.isalpha():
            return t[0].upper() + t[1:].lower()
    n = (name or "").strip()
    if n.upper().startswith("CL"):
        return "Cl"
    if not n:
        return "X"
    return n[0].upper()


def _read_pdb_atoms_split(pdb_path: Path) -> Atoms:
    """Whitespace ATOM/HETATM reader for CGenFF names ASE rejects.

    Uses ``_parse_pdb_atoms_whitespace`` field order
    (``ATOM serial name resname [chain] resid x y z …``) so serial/resid are
    never mistaken for coordinates when occupancy/tempFactor are omitted.
    """
    from ase import Atoms as _Atoms
    from mmml.interfaces.pycharmmInterface.mlpot.setup import (
        _parse_pdb_atoms_whitespace,
    )

    names, _resnames, _resids, positions = _parse_pdb_atoms_whitespace(pdb_path)
    trailing_by_index: list[str | None] = [None] * len(names)
    atom_i = 0
    for line in Path(pdb_path).read_text(encoding="utf-8", errors="replace").splitlines():
        parts = line.split()
        if not parts or parts[0] not in ("ATOM", "HETATM"):
            continue
        if len(parts) >= 9 and parts[-1].isalpha() and len(parts[-1]) <= 2:
            if atom_i < len(trailing_by_index):
                trailing_by_index[atom_i] = parts[-1]
        atom_i += 1
    syms = [
        _element_from_pdb_atom_name(
            name, trailing_by_index[i] if i < len(trailing_by_index) else None
        )
        for i, name in enumerate(names)
    ]
    return _Atoms(symbols=syms, positions=np.asarray(positions, dtype=float))


def determine_box_size_from_mol(mol: Atoms) -> float:
    """Determines the box size based on the maximum distance between any two atoms"""
    dists = np.linalg.norm(
        mol.positions[:, None, :] - mol.positions[None, :, :], axis=-1
    )
    return np.max(dists)


def solute_radius_from_mol(mol: Atoms, buffer: float = 1.0) -> float:
    """
    Compute the radius of a sphere that encompasses the solute.

    Uses center of mass and max distance to any atom, plus a buffer for packing.
    """
    com = mol.get_center_of_mass()
    radii = np.linalg.norm(mol.positions - com, axis=1)
    return float(np.max(radii)) + buffer


def volume_per_solvent_molecule_ang3(
    solvent: str,
    density_kg_m3: float | None = None,
    atoms: Atoms | None = None,
) -> float:
    """
    Approximate volume per solvent molecule in Å³ from density and molecular weight.
    """
    name = _normalize_solvent_key(solvent)
    if atoms is None:
        atoms = _resolve_solvent_atoms(name)
    density = _resolve_solvent_density_kg_m3(name, density_kg_m3)
    density_g_cm3 = density / 1000.0
    mw = atoms.get_masses().sum()
    # cm³/mol -> Å³/molecule: (MW/density) / N_A * 1e24
    molar_vol_cm3 = mw / density_g_cm3
    vol_ang3 = molar_vol_cm3 / 6.022e23 * 1e24
    return vol_ang3


def outer_radius_from_n_solvent(
    n_molecules: int,
    inner_radius: float,
    solvent: str,
    buffer: float = 1.0,
    density_kg_m3: float | None = None,
    atoms: Atoms | None = None,
) -> float:
    """
    Outer radius of solvent shell to fit n_molecules around a solute.

    Solvent occupies a spherical shell between inner_radius and outer_radius.
    Volume of shell = (4/3)*pi*(R_outer³ - R_inner³) = n * V_solvent
    """
    vol_per_mol = volume_per_solvent_molecule_ang3(
        solvent, density_kg_m3=density_kg_m3, atoms=atoms
    )
    shell_volume = n_molecules * vol_per_mol
    # (4/3)*pi*R_outer³ = shell_volume + (4/3)*pi*R_inner³
    inner_vol = (4.0 / 3.0) * np.pi * (inner_radius**3)
    outer_vol = shell_volume + inner_vol
    outer_radius = (3.0 * outer_vol / (4.0 * np.pi)) ** (1.0 / 3.0)
    return outer_radius + buffer


# --- Packmol solvation hyper-parameters -------------------------------------
#
# ``determine_n_molecules_from_density`` sizes N from the *cubic* cell volume
# L³, so the solvent must be packed into that same cube.  Packing into the
# inscribed sphere instead (the old default) only offers pi/6 = 52% of the
# volume, i.e. every density-sized request was ~1.9x over capacity and Packmol
# terminated with "ENDED WITHOUT PERFECT PACKING" (exit 173) regardless of
# solvent.  Keep ``PACKMOL_REGION = "box"`` unless a genuine spherical droplet
# is wanted.
PACKMOL_REGION = "box"
PACKMOL_TOLERANCE = 2.0
PACKMOL_NLOOP = 200
# Slack on the ideal (bulk-density) occupancy.  Packmol's hard-sphere tolerance
# cannot reach 100% of the continuum density; ~2% headroom converges in seconds
# and the deficit is absorbed by the CHARMM minimisation / NPT equilibration.
PACKMOL_FILL_FRACTION = 0.98


def solvent_capacity(
    side_length: float,
    inner_radius: float,
    solvent: str,
    region: str = PACKMOL_REGION,
    outer_radius: float | None = None,
    density_kg_m3: float | None = None,
    atoms: Atoms | None = None,
    fill_fraction: float = PACKMOL_FILL_FRACTION,
) -> int:
    """
    Largest solvent count Packmol can place in the requested region.

    ``region="box"`` is the cubic cell minus the solute exclusion sphere;
    ``region="sphere"`` is the shell between *inner_radius* and *outer_radius*.
    *fill_fraction* derates the ideal bulk-density occupancy so Packmol has
    room to satisfy its tolerance.
    """
    vol_per_mol = volume_per_solvent_molecule_ang3(
        solvent, density_kg_m3=density_kg_m3, atoms=atoms
    )
    solute_vol = (4.0 / 3.0) * np.pi * (float(inner_radius) ** 3)
    if region == "box":
        available = float(side_length) ** 3 - solute_vol
    elif region == "sphere":
        if outer_radius is None:
            raise ValueError("region='sphere' requires outer_radius")
        available = (4.0 / 3.0) * np.pi * (float(outer_radius) ** 3) - solute_vol
    else:
        raise ValueError(f"region must be 'box' or 'sphere', got {region!r}")
    return max(0, int(available * float(fill_fraction) / vol_per_mol))


def setup_box(mol: Atoms) -> None:
    """Sets up the box"""
    box_size = determine_box_size_from_mol(mol)
    print(f"Box size: {box_size}")


def determine_n_molecules_from_density(
    density: float,
    mol: Atoms,
    side_length: float = 35,
    solvent: str = None,
) -> float:
    """
    Determine number of molecules from density.

    *density* is in kg/m³.  For a known solvent (TIP3/water, OCOH/octanol), a
    built-in density is used when *density* is omitted upstream; otherwise the
    provided value is used for any CGenFF solvent residue.

    Requires pint to be installed for unit conversions.
    Install with: pip install pint or conda install -c conda-forge pint
    """
    if not _PINT_AVAILABLE:
        raise ImportError(
            "pint is required for determine_n_molecules_from_density. "
            "Please install pint with: pip install pint or conda install -c conda-forge pint"
        )

    if solvent is not None:
        name = _normalize_solvent_key(solvent)
        atoms = _resolve_solvent_atoms(name)
        density_value = _resolve_solvent_density_kg_m3(name, density)
    else:
        atoms = mol
        density_value = float(density)
    masses = atoms.get_masses()

    molecular_weight = masses.sum()
    molecular_formula = atoms.get_chemical_formula(mode="reduce")

    # note use of two lines to keep length of line reasonable
    s = f"The molecular weight of {molecular_formula} is {molecular_weight:1.2f} gm/mol."
    print(s)

    box_size = side_length * ureg.angstrom
    volume = box_size**3  # Volume of the box in cm^3

    print("Volume of the box: ", volume)

    density_q = density_value * (ureg.kilogram / ureg.meter**3)
    molecular_weight = molecular_weight * (ureg.gram / ureg.mole)  # g/mol

    # Calculate mass of the substance in the box
    mass = density_q * volume  # mass = density * volume
    print(mass.to("g"))
    # Calculate moles in the box
    moles = mass.to("g") / molecular_weight.to("g/mol")
    print(moles)
    # Define Avogadro's number (molecules per mole)
    avogadro_number = 6.022e23 * ureg.molecule / ureg.mole

    # Calculate number of molecules
    num_molecules = moles * avogadro_number
    n_molecules = int(num_molecules.magnitude)
    # Display the result
    print(f"Number of molecules in the box: {n_molecules}")
    return n_molecules


def run_packmol_solvation(
    n_molecules: int,
    side_length: float,
    solvent: str,
    solute_mol: Atoms | None = None,
    inner_radius: float | None = None,
    outer_radius: float | None = None,
    solute_buffer: float = 1.0,
    solvent_buffer: float = 1.0,
    density_kg_m3: float | None = None,
    region: str = PACKMOL_REGION,
    tolerance: float = PACKMOL_TOLERANCE,
    nloop: int = PACKMOL_NLOOP,
    fill_fraction: float = PACKMOL_FILL_FRACTION,
    periodic: bool = True,
) -> int:
    """
    Pack 1 solute molecule surrounded by n_molecules of solvent, and return the
    count actually placed (clamped to what the region can hold).

    *solvent* may be any CGenFF RESI name (plus aliases ``water``/``octanol``).
    With ``region="box"`` (default) the solvent fills the whole cubic cell
    outside a solute exclusion sphere, matching the L³ volume that
    ``determine_n_molecules_from_density`` uses to size N; ``periodic`` then
    enables Packmol's ``pbc`` so tolerances hold across the cell faces.
    ``region="sphere"`` restores the spherical-shell (droplet) placement.
    """
    from mmml.analysis.residue_geometry import ensure_residue_pdb

    if region not in ("box", "sphere"):
        raise ValueError(f"region must be 'box' or 'sphere', got {region!r}")

    name = _normalize_solvent_key(solvent)
    solvent_atoms = _resolve_solvent_atoms(name)
    solvent_pdb_path = ensure_residue_pdb(name, generate=True)
    # Keep a stable path name for the packmol input (resi lower-case).
    # Copy the CHARMM/make-res PDB — never ase.io.write (defaults resname to MOL).
    solvent_tag = name.lower()
    staged = Path(f"pdb/{solvent_tag}.pdb")
    src = Path(solvent_pdb_path).resolve()
    if src != staged.resolve():
        Path("pdb").mkdir(exist_ok=True)
        shutil.copy2(src, staged)

    center = side_length / 2
    cx, cy, cz = center, center, center

    if inner_radius is None:
        if solute_mol is None:
            solute_mol = read_initial_pdb(Path.cwd())
        inner_radius = solute_radius_from_mol(solute_mol, buffer=solute_buffer)

    max_radius = center - 0.5
    if region == "sphere" and outer_radius is None:
        try:
            dens = _resolve_solvent_density_kg_m3(name, density_kg_m3)
            outer_radius = outer_radius_from_n_solvent(
                n_molecules,
                inner_radius,
                name,
                buffer=solvent_buffer,
                density_kg_m3=dens,
                atoms=solvent_atoms,
            )
        except ValueError:
            print(
                f"No density for solvent {name}; using box-limited outer radius "
                f"{max_radius:.2f} Å. Pass --density (kg/m³) for a denser shell estimate."
            )
            outer_radius = max_radius
    if outer_radius is not None and outer_radius > max_radius:
        print(
            f"Warning: outer_radius {outer_radius:.2f} Å exceeds box; capping to "
            f"{max_radius:.2f} Å. Consider increasing side_length."
        )
        outer_radius = max_radius

    # Never hand Packmol more molecules than the region can hold: an impossible
    # request burns the full nloop budget and then exits 173.
    n_requested = int(n_molecules)
    try:
        capacity = solvent_capacity(
            side_length,
            inner_radius,
            name,
            region=region,
            outer_radius=outer_radius,
            density_kg_m3=density_kg_m3,
            atoms=solvent_atoms,
            fill_fraction=fill_fraction,
        )
    except ValueError:
        capacity = None  # no density for this solvent; trust the caller's N
    if capacity is not None and n_requested > capacity:
        print(
            f"Reducing solvent count {n_requested} -> {capacity}: the {region} region "
            f"({side_length:.1f} Å cell, solute R={inner_radius:.2f} Å) holds at most "
            f"{capacity} {name} at {fill_fraction:.0%} of bulk density. "
            "Increase --box-size for more solvent."
        )
        n_molecules = capacity
    else:
        n_molecules = n_requested

    if region == "box":
        placement = (
            f"inside box 0.0 0.0 0.0 {side_length} {side_length} {side_length}\n"
            f"    outside sphere {cx} {cy} {cz} {inner_radius}"
        )
        print(
            f"Solvation region: cubic cell {side_length:.2f} Å, solute exclusion "
            f"R={inner_radius:.2f} Å, N={n_molecules} {name}"
        )
    else:
        placement = (
            f"outside sphere {cx} {cy} {cz} {inner_radius}\n"
            f"    inside sphere {cx} {cy} {cz} {outer_radius}"
        )
        print(
            f"Solvation radii: inner={inner_radius:.2f} Å, "
            f"outer={outer_radius:.2f} Å, N={n_molecules} {name}"
        )

    # ``pbc`` makes Packmol enforce the tolerance across the cell faces, so a
    # box packed at bulk density has no clashes with its own periodic images.
    pbc_line = (
        f"pbc 0.0 0.0 0.0 {side_length} {side_length} {side_length}\n"
        if (periodic and region == "box")
        else ""
    )
    # Do not set Packmol ``chain``: its PDB writer embeds the chain ID in column
    # 22 and truncates 5-char CGenFF names (CH3CL → CH3CA). We restore names after.
    randint = np.random.randint(1000000)
    packmol_input = f"""seed {randint}
    output pdb/init-{solvent_tag}box.pdb
    filetype pdb
    tolerance {tolerance}
    nloop {int(nloop)}
    {pbc_line}structure pdb/initial.pdb
    number 1
    resnumbers 2
    center
    fixed {cx} {cy} {cz} 0.0 0.0 0.0
    end structure
    structure pdb/{solvent_tag}.pdb
    number {n_molecules}
    resnumbers 2
    {placement}
    end structure
"""
    import os
    os.makedirs("packmol", exist_ok=True)
    with open(f"packmol/packmol-{solvent_tag}.inp", "w") as f:
        f.write(packmol_input)

    from mmml.interfaces.pycharmmInterface.packmol_placement import (
        packmol_executable,
        rewrite_packmol_pdb_resnames,
    )

    packmol_bin = packmol_executable()
    inp = f"packmol/packmol-{solvent_tag}.inp"
    print(f"{packmol_bin} < {inp}")
    status = os.system(" ".join([packmol_bin, " < ", inp]))
    # os.system returns an encoded wait status: 173 shows up as 44288 (173 << 8).
    try:
        exit_code = os.waitstatus_to_exitcode(status)
    except ValueError:  # killed by a signal, or the shell could not exec
        exit_code = status
    if exit_code != 0:
        from mmml.interfaces.pycharmmInterface.packmol_placement import (
            PACKMOL_EXIT_LABELS,
        )

        label = PACKMOL_EXIT_LABELS.get(exit_code, "unknown error")
        hint = ""
        if exit_code == 173:
            hint = (
                f"\nPacked {n_molecules} {name} into the {region} region "
                f"(L={side_length:.1f} Å, tolerance {tolerance} Å, nloop {int(nloop)}). "
                "Relax by lowering fill_fraction, lowering tolerance, raising nloop, "
                "or increasing --box-size."
            )
        raise RuntimeError(
            f"packmol solvation failed: exit {exit_code} ({label}); see {inp}{hint}"
        )
    out_pdb = Path(f"pdb/init-{solvent_tag}box.pdb")
    rewrite_packmol_pdb_resnames(
        out_pdb,
        [
            (Path("pdb/initial.pdb"), 1),
            (Path(f"pdb/{solvent_tag}.pdb"), int(n_molecules)),
        ],
    )
    # Fail fast if the solvent template was ASE ``MOL`` (CHARMM GENERATE aborts).
    from mmml.analysis.residue_geometry import _pdb_resnames

    restored = _pdb_resnames(out_pdb)
    if name not in restored:
        raise RuntimeError(
            f"After Packmol rewrite, {out_pdb} is missing solvent residue {name!r} "
            f"(found {sorted(restored)}). Solvent template pdb/{solvent_tag}.pdb "
            "likely used ASE placeholder MOL — regenerate with make-res / "
            "ensure_residue_pdb."
        )
    print(f"Generated {out_pdb} (CGenFF residue names restored)")
    return int(n_molecules)


def run_packmol(
    n_molecules: int,
    side_length: float,
    tolerance: float = PACKMOL_TOLERANCE,
    nloop: int = PACKMOL_NLOOP,
    periodic: bool = True,
) -> None:
    """
    Pack *n_molecules* copies of ``pdb/initial.pdb`` into a cubic cell.

    ``periodic`` enables Packmol's ``pbc`` so the tolerance holds across the
    cell faces: without it a box packed at bulk density is clash-free only
    within the cell, and molecules straddling opposite faces land on top of
    their own periodic images (contacts of ~0.2 Å, which blows up the CHARMM
    energy at the first minimisation step).
    """
    pbc_line = (
        f"pbc 0.0 0.0 0.0 {side_length} {side_length} {side_length}\n"
        if periodic
        else ""
    )
    randint = np.random.randint(1000000)
    packmol_input = f"""seed {randint}
    output pdb/init-packmol.pdb
    filetype pdb
    tolerance {tolerance}
    nloop {int(nloop)}
    {pbc_line}structure pdb/initial.pdb
    number {n_molecules}
    resnumbers 2
    inside box 0.0 0.0 0.0 {side_length} {side_length} {side_length}
    end structure
"""
    os.makedirs("packmol", exist_ok=True)
    with open("packmol/packmol.inp", "w") as f:
        f.write(packmol_input)

    from mmml.interfaces.pycharmmInterface.packmol_placement import (
        packmol_executable,
        rewrite_packmol_pdb_resnames,
    )

    packmol_bin = packmol_executable()
    print(f"{packmol_bin} < packmol/packmol.inp")
    status = os.system(" ".join([packmol_bin, " < ", "packmol/packmol.inp"]))
    try:
        exit_code = os.waitstatus_to_exitcode(status)
    except ValueError:  # killed by a signal, or the shell could not exec
        exit_code = status
    if exit_code != 0:
        from mmml.interfaces.pycharmmInterface.packmol_placement import (
            PACKMOL_EXIT_LABELS,
        )

        label = PACKMOL_EXIT_LABELS.get(exit_code, "unknown error")
        raise RuntimeError(
            f"packmol failed: exit {exit_code} ({label}); see packmol/packmol.inp. "
            f"Packed {n_molecules} copies into a {side_length:.1f} Å cell at "
            f"tolerance {tolerance} Å, nloop {int(nloop)}."
        )
    rewrite_packmol_pdb_resnames(
        "pdb/init-packmol.pdb",
        [(Path("pdb/initial.pdb"), int(n_molecules))],
    )
    print("Generated pdb/init-packmol.pdb (CGenFF residue names restored)")


def _ensure_crystal_image_str() -> None:
    """Copy crystal_image.str to cwd if missing (required by CHARMM for periodic images)."""
    dst = Path("crystal_image.str")
    if dst.exists():
        return
    from mmml.paths import crystal_image_str_source

    src = crystal_image_str_source()
    if src.exists():
        shutil.copy2(src, dst)
    else:
        raise FileNotFoundError(
            f"crystal_image.str not found in cwd and source {src} does not exist. "
            "CHARMM requires this file for periodic box setup."
        )


def setup_box_generic(pdb_path, rtf=CGENFF_RTF, prm=CGENFF_PRM, side_length: float = 30, tag="", skip_energy_show: bool = False):
    """
    Sets up the box

    Args:
        skip_energy_show: If True, skip energy.show() to avoid slow CHARMM energy evaluation
            (Drude setup). Use for faster startup when validation is not needed.
        rtf, prm: Retained for API compatibility; topology is loaded via
            ``read_cgenff_toppar()`` so ``MMML_CGENFF_EXTRA_RTF`` append residues
            (e.g. CH3CL) are available.
    """
    from mmml.interfaces.pycharmmInterface.charmm_levels import charmm_relaxed_bomlev
    from mmml.interfaces.pycharmmInterface.import_pycharmm import pycharmm_quiet, safe_energy_show
    from mmml.interfaces.pycharmmInterface.mlpot.pbc_env import prepare_charmm_pbc
    from mmml.interfaces.pycharmmInterface.mlpot.setup import (
        _parse_pdb_atoms_whitespace,
        _residue_sequence_from_pdb,
        sync_charmm_positions,
    )
    from mmml.interfaces.pycharmmInterface.nbonds_config import read_cgenff_toppar
    import pycharmm.generate as generate
    import pycharmm.read as read

    _ = (rtf, prm)  # API compat; EXTRA_RTF comes from the env via read_cgenff_toppar
    _ensure_crystal_image_str()
    CLEAR_CHARMM()

    # Prefer sequence_string + coord overlay over lingo READ SEQU PDB:
    # fixed columns truncate 5-char CGenFF names (CH3CL) and some KEY_LIBRARY
    # builds abort on sequence_pdb.
    res_seq = _residue_sequence_from_pdb(pdb_path)
    _names, _resnames, _resids, pdb_xyz = _parse_pdb_atoms_whitespace(pdb_path)
    print(
        f"setup_box_generic: sequence_string {' '.join(res_seq)} "
        f"({len(pdb_xyz)} atoms) from {pdb_path}",
        flush=True,
    )
    read_cgenff_toppar()
    with charmm_relaxed_bomlev():
        read.sequence_string(" ".join(res_seq))
        status = generate.new_segment(
            seg_name="SYS",
            first_patch="NONE",
            last_patch="NONE",
            setup_ic=True,
        )
        if status is not None and int(status) not in (0, 1):
            raise RuntimeError(
                f"GENERATE SYS failed for {pdb_path} (status={status}; "
                f"sequence={res_seq}). Check CGenFF residue names and "
                "MMML_CGENFF_EXTRA_RTF (e.g. CH3CL)."
            )
    sync_charmm_positions(np.asarray(pdb_xyz, dtype=float))
    # KEY_LIBRARY builds do not parse lingo ``nbonds`` / ``open`` / ``crystal``;
    # use the C API path (define_cubic + build + NonBondedScript).
    prepare_charmm_pbc(float(side_length))
    if not skip_energy_show:
        safe_energy_show()
    write.psf_card(f"psf/system-{tag}.psf")
    write.coor_pdb(f"pdb/init-{tag}.pdb")
    print(f"wrote pdb/init-{tag}.pdb")

    pycharmm_quiet()
    atoms = ase.io.read(f"pdb/init-{tag}.pdb")
    atoms.set_cell(np.eye(3) * side_length)
    atoms.set_pbc(True)
    return atoms



def initialize_psf(resid: str, n_molecules: int, side_length: float, solvent: str = None, pdb_path: str = None):
    """
    Initializes the PSF file
    """
    from mmml.interfaces.pycharmmInterface.import_pycharmm import pycharmm_quiet
    CLEAR_CHARMM()
    if pdb_path is None:
        pdbfilename = "pdb/init-packmol.pdb"
    else:
        pdbfilename = pdb_path

    from mmml.interfaces.pycharmmInterface.nbonds_config import read_cgenff_toppar

    read_cgenff_toppar()

    pycharmm_quiet()
    if solvent is not None:
        " ".join([solvent.upper()]*(n_molecules-1))    
        f"{resid.upper()} {solvent.upper()}"
        pdb_path = pdbfilename
    else:
        " ".join([resid.upper()]*n_molecules)
        pdb_path = pdbfilename

    from mmml.interfaces.pycharmmInterface.charmm_levels import charmm_silent_command

    header = f"""OPEN UNIT 1 READ FORM NAME {pdb_path}
    READ SEQU PDB UNIT 1
    CLOSE UNIT 1
    GENERATE SYS FIRST NONE LAST NONE SETUP

    OPEN UNIT 1 READ FORM NAME {pdb_path}
    READ COOR PDB UNIT 1
    CLOSE UNIT 1

    """
    with charmm_silent_command():
        pycharmm.lingo.charmm_script(header)
    print("read header")
    pycharmm.lingo.charmm_script(pbcset.format(SIDELENGTH=side_length))
    print("read pbcset")
    # pycharmm.lingo.charmm_script(pbcs)
    # print("read pbcs")
    # energy.show()
    # print("read energy")
    # pycharmm.lingo.charmm_script(write_system_psf)
    
    write.psf_card("psf/init.box.psf")
    write.psf_card("psf/init.box.psf")
    write.coor_pdb("pdb/init.box.pdb")
    print("wrote pdb/init.box.pdb")


def minimize_box(skip_energy_show: bool = False, nbxmod: int = 3):
    # Nonbonds come from prepare_charmm_pbc (KEY_LIBRARY has no lingo ``nbonds``).
    # nbxmod retained for API compatibility; CGENFF NONBONDED sets it at param read.
    _ = nbxmod
    minimize.run_abnr(nstep=1000, tolenr=1e-3, tolgrd=1e-3)
    if skip_energy_show:
        print("Skipping energy.show() (--skip-energy-show).")
    else:
        from mmml.interfaces.pycharmmInterface.import_pycharmm import safe_energy_show

        safe_energy_show()


def main(density: float, side_length: float, residue: str, solvent: str):
    cwd = Path(os.getcwd())
    mol = read_initial_pdb(cwd)
    print(mol)
    print(mol.get_chemical_symbols())
    print(solvent)
    if solvent is None:
        n_molecules = determine_n_molecules_from_density(density, mol, side_length, solvent=None)
        run_packmol(n_molecules, side_length)
    else:
        n_molecules = determine_n_molecules_from_density(density, mol, side_length, solvent)
        # Packmol may clamp N to the region capacity; the PSF must match.
        n_molecules = run_packmol_solvation(
            n_molecules, side_length, solvent, solute_mol=mol, density_kg_m3=density
        )
    initialize_psf(residue, n_molecules, side_length, solvent)
    # minimize_box()


def cli():
    """Command line interface"""
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--density", type=float, required=True, 
        help="Density of the box in kg/m^3"   )
    parser.add_argument("-l", "--side_length", type=float, required=True, 
        help="Side length of the box in angstrom")
    parser.add_argument("-r", "--residue", type=str, required=True, 
        help="Residue name")
    parser.add_argument("-s", "--solvent", type=str, required=False, default=None,
        help="Solvent name")
    args = parser.parse_args()
    if args.solvent == "None":
        args.solvent = None
    for arg in vars(args):
        print(f"{arg}: {getattr(args, arg)} {type(getattr(args, arg))}")
    main(args.density, args.side_length, args.residue, args.solvent)


if __name__ == "__main__":
    cli()