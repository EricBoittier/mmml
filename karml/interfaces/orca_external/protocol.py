"""Read/write helpers for ORCA's external-tool (ExtTool) file protocol."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ExtInpData:
    """Parsed contents of an ORCA ``*.extinp.tmp`` file."""

    xyz_path: Path
    charge: int
    multiplicity: int
    ncores: int
    do_gradient: bool
    pointcharges_path: Path | None = None


def _first_field(line: str) -> str:
    """Return the first whitespace-separated token, ignoring ``#`` comments."""
    stripped = line.split("#", 1)[0].strip()
    if not stripped:
        return ""
    return stripped.split()[0]


def read_extinp(inputfile: str | Path) -> ExtInpData:
    """Parse an ORCA external-tool input file."""
    input_path = Path(inputfile)
    if not input_path.is_file():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    fields: list[str] = []
    with input_path.open() as handle:
        for line in handle:
            token = _first_field(line)
            if token:
                fields.append(token)

    if len(fields) < 5:
        raise ValueError(f"ORCA extinp file has too few fields: {input_path}")

    xyz_name = fields[0]
    charge = int(fields[1])
    multiplicity = int(fields[2])
    ncores = int(fields[3])
    do_gradient_flag = int(fields[4])

    if do_gradient_flag not in (0, 1):
        raise ValueError("do_gradient from ORCA input must be 0 or 1.")
    if multiplicity < 1:
        raise ValueError("Multiplicity must be a positive integer.")
    if ncores < 1:
        raise ValueError("NCores must be a positive integer.")

    xyz_path = Path(xyz_name)
    if not xyz_path.is_absolute():
        xyz_path = (input_path.parent / xyz_path).resolve()

    pointcharges_path = None
    if len(fields) >= 6:
        pc_name = fields[5]
        pointcharges_path = Path(pc_name)
        if not pointcharges_path.is_absolute():
            pointcharges_path = (input_path.parent / pointcharges_path).resolve()

    return ExtInpData(
        xyz_path=xyz_path,
        charge=charge,
        multiplicity=multiplicity,
        ncores=ncores,
        do_gradient=bool(do_gradient_flag),
        pointcharges_path=pointcharges_path,
    )


def natoms_from_xyz(xyz_file: str | Path) -> int:
    """Return the atom count from a standard XYZ file."""
    with Path(xyz_file).open() as handle:
        return int(handle.readline().strip())


def read_xyz(xyz_file: str | Path) -> tuple[list[str], list[tuple[float, float, float]]]:
    """Read element symbols and Cartesian coordinates (Angstrom) from XYZ."""
    symbols: list[str] = []
    coordinates: list[tuple[float, float, float]] = []
    xyz_path = Path(xyz_file)
    with xyz_path.open() as handle:
        natoms = int(handle.readline().strip())
        handle.readline()  # comment line
        for _ in range(natoms):
            line = handle.readline()
            if not line:
                break
            parts = line.split()
            symbols.append(parts[0])
            coordinates.append((float(parts[1]), float(parts[2]), float(parts[3])))
    return symbols, coordinates


def write_engrad(
    filename: str | Path,
    natoms: int,
    energy_hartree: float,
    gradient_hartree_bohr: list[float] | None = None,
) -> None:
    """Write ORCA ``*.engrad`` output (energy in Eh, gradient in Eh/bohr)."""
    output_path = Path(filename)
    lines = [
        "#",
        "# Number of atoms",
        "#",
        f"{natoms}",
        "#",
        "# Total energy [Eh]",
        "#",
        f"{energy_hartree:.12e}",
    ]
    if gradient_hartree_bohr:
        lines.extend(
            [
                "#",
                "# Gradient [Eh/Bohr] A1X, A1Y, A1Z, A2X, ...",
                "#",
            ]
        )
        lines.extend(f"{value: .12e}" for value in gradient_hartree_bohr)

    try:
        output_path.write_text("\n".join(lines) + "\n")
    except OSError as exc:
        raise RuntimeError(f"Failed to write ORCA output file {output_path}: {exc}") from exc
