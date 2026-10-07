"""Molpro subprocess backend for cross-check evaluation."""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
from ase import Atoms

from mmml.data.xml_to_npz import MolproConverter
from mmml.interfaces.qc_backends.npz_output import stack_frame_results

_DEFAULT_BASIS = "cc-pVDZ"
_DEFAULT_METHOD_BLOCK = "rhf"


def render_molpro_input(
    *,
    atoms: Atoms,
    charge: int,
    multiplicity: int,
    basis: str,
    method_block: str,
    template: str | None,
) -> str:
    """Build a minimal Molpro single-point + gradient input deck."""
    geom_lines = []
    symbols = atoms.get_chemical_symbols()
    positions = atoms.get_positions()
    for sym, (x, y, z) in zip(symbols, positions):
        geom_lines.append(f"{sym},{x},{y},{z}")

    geom_block = ";\n     ".join(geom_lines)
    if template:
        return (
            template.replace("{charge}", str(charge))
            .replace("{mult}", str(multiplicity))
            .replace("{multiplicity}", str(multiplicity))
            .replace("{basis}", basis)
            .replace("{method}", method_block)
            .replace("{geometry}", geom_block)
        )

    return f"""***, MMML cross-check
gdirect;
geometry={{
     {geom_block}
}}
basis={basis}
set,charge={charge}
set,spin={max(0, multiplicity - 1)}
{method_block}
force
"""


class MolproBackend:
    """Run Molpro QM jobs and parse XML output."""

    name = "molpro"

    def __init__(
        self,
        *,
        basis: str = _DEFAULT_BASIS,
        method_block: str = _DEFAULT_METHOD_BLOCK,
        charge: int = 0,
        multiplicity: int = 1,
        molpro_exe: str | None = None,
        template: str | None = None,
        template_path: Path | None = None,
        run_molpro: Any | None = None,
        converter: MolproConverter | None = None,
    ) -> None:
        self.basis = basis
        self.method_block = method_block
        self.charge = charge
        self.multiplicity = multiplicity
        self.molpro_exe = molpro_exe or os.environ.get("MOLPRO", "molpro")
        self.template = template
        if template_path is not None:
            self.template = Path(template_path).read_text()
        self._run_molpro = run_molpro
        self._converter = converter or MolproConverter(verbose=False, padding_atoms=200)

    @property
    def method_label(self) -> str:
        return f"Molpro/{self.method_block}/{self.basis}"

    @property
    def energy_unit(self) -> str:
        return "hartree"

    @property
    def force_unit(self) -> str:
        return "hartree_bohr"

    def _run_single(self, atoms: Atoms, workdir: Path) -> dict[str, np.ndarray]:
        if self._run_molpro is not None:
            return self._run_molpro(atoms, workdir, self)

        inp_text = render_molpro_input(
            atoms=atoms,
            charge=self.charge,
            multiplicity=self.multiplicity,
            basis=self.basis,
            method_block=self.method_block,
            template=self.template,
        )
        inp_path = workdir / "job.inp"
        out_path = workdir / "job.out"
        xml_path = workdir / "job.xml"
        inp_path.write_text(inp_text)

        cmd = [self.molpro_exe, "-o", str(out_path), str(inp_path)]
        proc = subprocess.run(
            cmd,
            cwd=str(workdir),
            capture_output=True,
            text=True,
            check=False,
        )
        if proc.returncode != 0:
            raise RuntimeError(
                f"Molpro failed (exit {proc.returncode}) in {workdir}:\n"
                f"{(proc.stdout + proc.stderr)[-2000:]}"
            )

        xml_candidates = sorted(workdir.glob("*.xml"))
        if not xml_candidates:
            raise RuntimeError(f"No Molpro XML output found in {workdir}")
        xml_file = xml_candidates[0]
        if xml_path != xml_file:
            xml_path = xml_file

        npz_data = self._converter.convert_single(xml_path)
        if npz_data is None:
            raise RuntimeError(f"Failed to parse Molpro XML: {xml_path}")
        return npz_data

    def evaluate_batch(
        self,
        frames: list[Atoms],
        *,
        properties: frozenset[str],
    ) -> dict[str, np.ndarray]:
        want_forces = "forces" in properties or "F" in properties
        energies: list[float] = []
        forces: list[np.ndarray] | None = [] if want_forces else None
        dipoles: list[np.ndarray] | None = []
        frames_z: list[np.ndarray] = []
        frames_r: list[np.ndarray] = []

        for atoms in frames:
            n = len(atoms)
            frames_z.append(np.asarray(atoms.get_atomic_numbers(), dtype=np.int32))
            frames_r.append(np.asarray(atoms.get_positions(), dtype=np.float64))
            with tempfile.TemporaryDirectory(prefix="mmml_molpro_") as tmp:
                workdir = Path(tmp)
                npz_data = self._run_single(atoms, workdir)
                energies.append(float(np.asarray(npz_data["E"]).reshape(())))
                if forces is not None and "F" in npz_data:
                    f = np.asarray(npz_data["F"][0, :n], dtype=np.float64)
                    forces.append(f)
                if "Dxyz" in npz_data:
                    dipoles.append(np.asarray(npz_data["Dxyz"]).reshape(-1, 3)[0])
                elif "D" in npz_data:
                    dipoles.append(np.asarray(npz_data["D"]).reshape(-1, 3)[0])

        return stack_frame_results(
            energies=energies,
            forces=forces,
            dipoles=dipoles if dipoles else None,
            frames_z=frames_z,
            frames_r=frames_r,
        )


def build_molpro_backend(options: dict[str, Any]) -> MolproBackend:
    template_path = options.get("template") or options.get("molpro_template")
    return MolproBackend(
        basis=str(options.get("basis") or _DEFAULT_BASIS),
        method_block=str(options.get("method") or options.get("method_block") or _DEFAULT_METHOD_BLOCK),
        charge=int(options.get("charge", 0)),
        multiplicity=int(options.get("multiplicity", 1)),
        molpro_exe=options.get("molpro_exe"),
        template_path=Path(template_path) if template_path else None,
        run_molpro=options.get("run_molpro"),
        converter=options.get("converter"),
    )


def molpro_available(exe: str | None = None) -> bool:
    path = exe or os.environ.get("MOLPRO", "molpro")
    return shutil.which(path) is not None
