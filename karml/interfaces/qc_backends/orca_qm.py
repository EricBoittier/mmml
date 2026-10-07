"""ORCA QM subprocess backend for cross-check evaluation."""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
from ase import Atoms
from ase.io import write as ase_write

from mmml.interfaces.qc_backends.npz_output import stack_frame_results

_DEFAULT_SIMPLE_LINE = "! {method} {basis} EnGrad"
_DEFAULT_PAL = 1

_ENERGY_PATTERNS = (
    re.compile(r"FINAL SINGLE POINT ENERGY\s+(-?\d+\.\d+)"),
    re.compile(r"Total Energy\s+:\s+(-?\d+\.\d+)"),
)


def read_orca_engrad(path: Path) -> tuple[float, np.ndarray | None]:
    """Parse ORCA native ``*.engrad`` (energy in Eh, gradient in Eh/bohr)."""
    lines = [ln.strip() for ln in path.read_text().splitlines() if ln.strip() and not ln.strip().startswith("#")]
    if len(lines) < 2:
        raise ValueError(f"ORCA engrad file too short: {path}")

    natoms = int(float(lines[0]))
    energy = float(lines[1])
    gradient = None
    if len(lines) > 2:
        grad_vals = [float(x) for x in lines[2:]]
        if len(grad_vals) >= 3 * natoms:
            gradient = np.asarray(grad_vals[: 3 * natoms], dtype=np.float64).reshape(natoms, 3)
    return energy, gradient


def parse_orca_out_energy(text: str) -> float | None:
    """Extract final single-point energy from ORCA stdout/output file."""
    for pattern in _ENERGY_PATTERNS:
        matches = pattern.findall(text)
        if matches:
            return float(matches[-1])
    return None


def render_orca_input(
    *,
    atoms: Atoms,
    charge: int,
    multiplicity: int,
    template: str | None,
    method: str,
    basis: str,
    pal: int,
) -> str:
    """Build ORCA input deck from template or default EnGrad layout."""
    xyz_block = _atoms_to_orca_xyz_block(atoms, charge, multiplicity)
    if template:
        text = template
        return (
            text.replace("{charge}", str(charge))
            .replace("{mult}", str(multiplicity))
            .replace("{multiplicity}", str(multiplicity))
            .replace("{method}", method)
            .replace("{basis}", basis)
            .replace("{pal}", str(pal))
            .replace("{xyz}", xyz_block)
        )

    return "\n".join(
        [
            _DEFAULT_SIMPLE_LINE.format(method=method, basis=basis),
            f"%pal nprocs {pal} end",
            "* xyz {charge} {multiplicity}",
            xyz_block,
            "",
        ]
    )


def _atoms_to_orca_xyz_block(atoms: Atoms, charge: int, multiplicity: int) -> str:
    lines = [f"* xyz {charge} {multiplicity}"]
    symbols = atoms.get_chemical_symbols()
    positions = atoms.get_positions()
    for sym, (x, y, z) in zip(symbols, positions):
        lines.append(f"  {sym} {x:.8f} {y:.8f} {z:.8f}")
    return "\n".join(lines)


class OrcaQMBackend:
    """Run ORCA QM single-point + gradient jobs as a reference backend."""

    name = "orca"

    def __init__(
        self,
        *,
        method: str = "PBE",
        basis: str = "def2-SVP",
        charge: int = 0,
        multiplicity: int = 1,
        pal: int = _DEFAULT_PAL,
        orca_exe: str | None = None,
        template: str | None = None,
        template_path: Path | None = None,
        run_orca: Any | None = None,
    ) -> None:
        self.method = method
        self.basis = basis
        self.charge = charge
        self.multiplicity = multiplicity
        self.pal = pal
        self.orca_exe = orca_exe or os.environ.get("ORCA", "orca")
        self.template = template
        if template_path is not None:
            self.template = Path(template_path).read_text()
        self._run_orca = run_orca

    @property
    def method_label(self) -> str:
        return f"ORCA/{self.method}/{self.basis}"

    @property
    def energy_unit(self) -> str:
        return "hartree"

    @property
    def force_unit(self) -> str:
        return "hartree_bohr"

    def _run_single(self, atoms: Atoms, workdir: Path) -> tuple[float, np.ndarray | None]:
        if self._run_orca is not None:
            return self._run_orca(atoms, workdir, self)

        inp_text = render_orca_input(
            atoms=atoms,
            charge=self.charge,
            multiplicity=self.multiplicity,
            template=self.template,
            method=self.method,
            basis=self.basis,
            pal=self.pal,
        )
        job_base = workdir / "job"
        inp_path = job_base.with_suffix(".inp")
        inp_path.write_text(inp_text)
        ase_write(str(job_base.with_suffix(".xyz")), atoms)

        cmd = [self.orca_exe, str(inp_path.name)]
        proc = subprocess.run(
            cmd,
            cwd=str(workdir),
            capture_output=True,
            text=True,
            check=False,
        )
        out_text = proc.stdout + proc.stderr
        (workdir / "job.out").write_text(out_text)
        if proc.returncode != 0:
            raise RuntimeError(
                f"ORCA failed (exit {proc.returncode}) in {workdir}:\n{out_text[-2000:]}"
            )

        engrad_path = job_base.with_suffix(".engrad")
        if engrad_path.is_file():
            energy, gradient = read_orca_engrad(engrad_path)
            return energy, gradient

        energy = parse_orca_out_energy(out_text)
        if energy is None:
            raise RuntimeError(f"Could not parse ORCA energy from {workdir / 'job.out'}")
        return energy, None

    def evaluate_batch(
        self,
        frames: list[Atoms],
        *,
        properties: frozenset[str],
    ) -> dict[str, np.ndarray]:
        want_forces = "forces" in properties or "F" in properties
        energies: list[float] = []
        forces: list[np.ndarray] | None = [] if want_forces else None
        frames_z: list[np.ndarray] = []
        frames_r: list[np.ndarray] = []

        for atoms in frames:
            frames_z.append(np.asarray(atoms.get_atomic_numbers(), dtype=np.int32))
            frames_r.append(np.asarray(atoms.get_positions(), dtype=np.float64))
            with tempfile.TemporaryDirectory(prefix="mmml_orca_qm_") as tmp:
                workdir = Path(tmp)
                energy, gradient = self._run_single(atoms, workdir)
                energies.append(float(energy))
                if forces is not None and gradient is not None:
                    forces.append(-np.asarray(gradient, dtype=np.float64))

        return stack_frame_results(
            energies=energies,
            forces=forces,
            dipoles=None,
            frames_z=frames_z,
            frames_r=frames_r,
        )


def build_orca_backend(options: dict[str, Any]) -> OrcaQMBackend:
    template_path = options.get("template") or options.get("orca_template")
    return OrcaQMBackend(
        method=str(options.get("method") or options.get("functional") or "PBE"),
        basis=str(options.get("basis") or "def2-SVP"),
        charge=int(options.get("charge", 0)),
        multiplicity=int(options.get("multiplicity", 1)),
        pal=int(options.get("pal", _DEFAULT_PAL)),
        orca_exe=options.get("orca_exe"),
        template_path=Path(template_path) if template_path else None,
        run_orca=options.get("run_orca"),
    )


def orca_available(exe: str | None = None) -> bool:
    """Return True if the ORCA executable is on PATH or ORCA env."""
    path = exe or os.environ.get("ORCA", "orca")
    return shutil.which(path) is not None
