"""Batched TorchScript evaluation of a metatomic teacher (energy eV, forces eV/Å).

The ASE ``MetatomicCalculator`` evaluates one ``Atoms`` per call. Teacher
labelling for distillation is thousands of 10-40 atom gas-phase structures, so
per-call overhead dominates. Here many structures go through one
``AtomisticModel.forward`` as a ``list[System]``; forces are ``-dE/dR`` from a
single backward pass over the summed batch energy (systems are independent, so
each system's gradient is its own force).

Batches are packed by total atom count (``max_atoms_per_batch``) so a larger
PET (PET-MAD s/m, PET-OMAD, ...) can be run with a smaller budget on the same
GPU. Imports torch / metatomic lazily.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np

DEFAULT_MAX_ATOMS_PER_BATCH = 4096
DEFAULT_MAX_SYSTEMS_PER_BATCH = 512


def pack_batches(
    n_atoms: Sequence[int],
    *,
    max_atoms: int = DEFAULT_MAX_ATOMS_PER_BATCH,
    max_systems: int = DEFAULT_MAX_SYSTEMS_PER_BATCH,
) -> list[list[int]]:
    """Group indices (in order) so each batch stays under both budgets.

    A single structure larger than ``max_atoms`` gets a batch of its own.
    """
    if int(max_atoms) < 1 or int(max_systems) < 1:
        raise ValueError("max_atoms and max_systems must be >= 1")
    batches: list[list[int]] = []
    cur: list[int] = []
    cur_atoms = 0
    for idx, n in enumerate(n_atoms):
        n = int(n)
        if cur and (cur_atoms + n > int(max_atoms) or len(cur) >= int(max_systems)):
            batches.append(cur)
            cur, cur_atoms = [], 0
        cur.append(idx)
        cur_atoms += n
    if cur:
        batches.append(cur)
    return batches


class BatchedMetatomicTeacher:
    """Evaluate a metatomic ``.pt`` on many non-periodic structures per forward."""

    def __init__(
        self,
        checkpoint: Path | str,
        *,
        device: str | None = None,
        max_atoms_per_batch: int = DEFAULT_MAX_ATOMS_PER_BATCH,
        max_systems_per_batch: int = DEFAULT_MAX_SYSTEMS_PER_BATCH,
        extensions_directory: Path | str | None = None,
    ) -> None:
        import torch
        from metatomic.torch import load_atomistic_model

        from mmml.interfaces.calculators.metatomic import (
            metatomic_device_name,
            resolve_metatomic_model_path,
        )

        # Batches have variable atom counts; static fusion re-specializes per
        # shape (and upet notes CUDA 13 "Global alloc" failures). Same as upet.
        torch.jit.set_fusion_strategy([("DYNAMIC", 10)])
        self.model_path = resolve_metatomic_model_path(checkpoint)
        self.device = torch.device(metatomic_device_name(device=device))
        self.model = load_atomistic_model(
            str(self.model_path),
            extensions_directory=(
                None if extensions_directory is None else str(extensions_directory)
            ),
        ).to(self.device)
        caps = self.model.capabilities()
        self.dtype = getattr(torch, str(caps.dtype))
        self.interaction_range = float(caps.interaction_range)
        self.max_atoms_per_batch = int(max_atoms_per_batch)
        self.max_systems_per_batch = int(max_systems_per_batch)
        if "energy" not in caps.outputs:
            raise ValueError(f"{self.model_path} has no 'energy' output")
        from vesin.metatomic import neighbor_lists_for_model

        # skin=0: every structure is new, so no Verlet-list reuse across calls.
        self._nl_calculators = neighbor_lists_for_model(
            "angstrom", self.model, skin=0.0
        )

    def _systems(self, structures: Sequence[tuple[np.ndarray, np.ndarray]]):
        import torch
        from metatomic.torch import System

        systems = []
        for numbers, positions in structures:
            pos = torch.tensor(
                np.asarray(positions, dtype=np.float64),
                dtype=self.dtype,
                device=self.device,
                requires_grad=True,
            )
            systems.append(
                System(
                    types=torch.tensor(
                        np.asarray(numbers, dtype=np.int32), device=self.device
                    ),
                    positions=pos,
                    # zero cell + pbc False: gas-phase structure
                    cell=torch.zeros((3, 3), dtype=self.dtype, device=self.device),
                    pbc=torch.zeros(3, dtype=torch.bool, device=self.device),
                )
            )
        return systems

    def _attach_neighbors(self, systems) -> None:
        for calculator in self._nl_calculators:
            calculator.add_neighbor_list(systems)

    def _forward(self, structures: Sequence[tuple[np.ndarray, np.ndarray]]):
        import torch
        from metatomic.torch import ModelEvaluationOptions, ModelOutput

        systems = self._systems(structures)
        self._attach_neighbors(systems)
        options = ModelEvaluationOptions(
            length_unit="angstrom",
            outputs={"energy": ModelOutput(quantity="energy", unit="eV", per_atom=False)},
        )
        out = self.model(systems, options, check_consistency=False)
        energy = out["energy"].block().values.reshape(-1)
        grads = torch.autograd.grad(energy.sum(), [s.positions for s in systems])
        e_np = energy.detach().to(torch.float64).cpu().numpy()
        f_np = [(-g).detach().to(torch.float64).cpu().numpy() for g in grads]
        return e_np, f_np

    def evaluate(
        self, structures: Sequence[tuple[np.ndarray, np.ndarray]]
    ) -> list[tuple[float, np.ndarray]]:
        """Return ``(E eV, F eV/Å)`` per ``(numbers, positions)`` in input order."""
        order = sorted(range(len(structures)), key=lambda i: len(structures[i][0]))
        n_atoms = [len(structures[i][0]) for i in order]
        results: list[tuple[float, np.ndarray] | None] = [None] * len(structures)
        for batch in pack_batches(
            n_atoms,
            max_atoms=self.max_atoms_per_batch,
            max_systems=self.max_systems_per_batch,
        ):
            idx = [order[b] for b in batch]
            energies, forces = self._forward([structures[i] for i in idx])
            for k, i in enumerate(idx):
                results[i] = (float(energies[k]), forces[k])
        return [r for r in results if r is not None]
