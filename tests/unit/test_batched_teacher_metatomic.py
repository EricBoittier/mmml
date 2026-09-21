"""BatchedMetatomicTeacher against the ASE MetatomicCalculator on a real TorchScript model.

The model is a tiny smooth pair potential exported as a metatomic
``AtomisticModel`` inside the test (no download): it requests a half neighbor
list and sums ``w_ij * eps * exp(-r^2 / s^2) * fc(r)`` over pairs, with a
cosine cutoff ``fc`` and a type-dependent weight ``w_ij``. Energies eV,
lengths Å. Covers gas-phase clusters of several sizes batched into one
forward and a periodic cubic cell.
"""

# No ``from __future__ import annotations``: AtomisticModel checks the real
# forward() annotations (List[System], ...), not strings.
import math
from typing import Dict, List, Optional

import numpy as np
import pytest

torch = pytest.importorskip("torch")
mta = pytest.importorskip("metatomic.torch")
pytest.importorskip("vesin.metatomic")
pytest.importorskip("metatomic_ase")
pytest.importorskip("metatensor.torch")

from ase import Atoms  # noqa: E402
from metatensor.torch import Labels, TensorBlock, TensorMap  # noqa: E402
from metatomic.torch import ModelOutput, NeighborListOptions, System  # noqa: E402

CUTOFF_A = 4.0


class SmoothPairEnergy(torch.nn.Module):
    """Sum of type-weighted Gaussians over neighbor pairs, cosine-cut at ``cutoff``."""

    def __init__(self, cutoff: float, eps: float = 0.3, sigma: float = 1.7) -> None:
        super().__init__()
        self.cutoff = cutoff
        self.eps = eps
        self.sigma = sigma
        self._nl = NeighborListOptions(cutoff=cutoff, full_list=False, strict=True)

    def requested_neighbor_lists(self) -> List[NeighborListOptions]:
        return [self._nl]

    def forward(
        self,
        systems: List[System],
        outputs: Dict[str, ModelOutput],
        selected_atoms: Optional[Labels] = None,
    ) -> Dict[str, TensorMap]:
        results: Dict[str, TensorMap] = {}
        if "energy" not in outputs:
            return results
        energies: List[torch.Tensor] = []
        for system in systems:
            nl = system.get_neighbor_list(self._nl)
            i = nl.samples.column("first_atom")
            j = nl.samples.column("second_atom")
            r = torch.linalg.vector_norm(nl.values.reshape(-1, 3), dim=1)
            z = system.types.to(r.dtype)
            w = 0.5 + z[i] * z[j] / 64.0
            fc = 0.5 * (torch.cos(math.pi * r / self.cutoff) + 1.0)
            fc = torch.where(r < self.cutoff, fc, torch.zeros_like(fc))
            e = self.eps * w * torch.exp(-(r * r) / (self.sigma * self.sigma)) * fc
            energies.append(e.sum().reshape(1))
        values = torch.cat(energies).reshape(-1, 1)
        block = TensorBlock(
            values=values,
            samples=Labels(
                ["system"],
                torch.arange(len(systems), dtype=torch.int32, device=values.device).reshape(-1, 1),
            ),
            components=torch.jit.annotate(List[Labels], []),
            properties=Labels(["energy"], torch.tensor([[0]], device=values.device)),
        )
        results["energy"] = TensorMap(
            keys=Labels(["_"], torch.tensor([[0]], device=values.device)), blocks=[block]
        )
        return results


@pytest.fixture(scope="module")
def pair_model_path(tmp_path_factory) -> str:
    capabilities = mta.ModelCapabilities(
        outputs={"energy": ModelOutput(unit="eV", sample_kind="system")},
        atomic_types=[1, 6, 8],
        interaction_range=CUTOFF_A,
        length_unit="angstrom",
        supported_devices=["cpu"],
        dtype="float64",
    )
    metadata = mta.ModelMetadata(name="mmml-test-smooth-pair", authors=["mmml tests"])
    model = mta.AtomisticModel(SmoothPairEnergy(CUTOFF_A).eval(), metadata, capabilities)
    path = tmp_path_factory.mktemp("metatomic") / "smooth_pair.pt"
    model.save(str(path))
    return str(path)


def _cluster(rng: np.random.Generator, n: int) -> tuple[np.ndarray, np.ndarray]:
    numbers = rng.choice([1, 6, 8], size=n)
    # Random cluster, dense enough that most atoms have neighbors within the cutoff.
    positions = rng.uniform(0.0, 1.6 * n ** (1.0 / 3.0) + 1.0, size=(n, 3))
    return numbers.astype(int), positions


def _ase_reference(path: str, numbers, positions, cell=None) -> tuple[float, np.ndarray]:
    from mmml.interfaces.calculators.metatomic import load_metatomic_calculator

    calc = load_metatomic_calculator(path, device="cpu")
    atoms = Atoms(numbers=numbers, positions=positions)
    if cell is not None:
        atoms.set_cell(cell)
        atoms.set_pbc(True)
    atoms.calc = calc
    return float(atoms.get_potential_energy()), np.asarray(atoms.get_forces(), dtype=np.float64)


def test_batched_teacher_matches_ase_calculator(pair_model_path: str) -> None:
    from mmml.distill.batched_teacher import BatchedMetatomicTeacher

    rng = np.random.default_rng(7)
    gas = [_cluster(rng, n) for n in (3, 7, 10, 20, 33)]
    n_box = 24
    cell = np.eye(3) * 7.0
    box = (
        rng.choice([1, 6, 8], size=n_box).astype(int),
        rng.uniform(0.0, 7.0, size=(n_box, 3)),
        cell,
    )
    structures = [*gas, box]

    # Small atom budget: several forwards, mixed sizes, order restored.
    teacher = BatchedMetatomicTeacher(
        pair_model_path, device="cpu", max_atoms_per_batch=40, max_systems_per_batch=3
    )
    got = teacher.evaluate(structures)
    assert len(got) == len(structures)

    for item, (e_b, f_b) in zip(structures, got):
        e_ref, f_ref = _ase_reference(pair_model_path, *item)
        assert f_b.shape == (len(item[0]), 3)
        assert abs(e_b - e_ref) < 1e-5 * max(1.0, abs(e_ref))
        np.testing.assert_allclose(f_b, f_ref, atol=1e-5, rtol=0)
        assert abs(e_ref) > 1e-3  # the model sees neighbors (non-trivial check)

    # One forward over everything gives the same answer as the chunked run.
    single = BatchedMetatomicTeacher(pair_model_path, device="cpu").evaluate(structures)
    for (e_a, f_a), (e_b, f_b) in zip(got, single):
        assert e_a == pytest.approx(e_b, abs=1e-10)
        np.testing.assert_allclose(f_a, f_b, atol=1e-10)


def test_periodic_images_change_the_energy(pair_model_path: str) -> None:
    """The cell is used: a small periodic box differs from the same atoms in vacuum."""
    from mmml.distill.batched_teacher import BatchedMetatomicTeacher

    rng = np.random.default_rng(3)
    numbers = np.array([8, 1, 1, 6, 8], dtype=int)
    positions = rng.uniform(0.0, 4.5, size=(5, 3))
    cell = np.eye(3) * 4.5
    teacher = BatchedMetatomicTeacher(pair_model_path, device="cpu")
    (e_gas, _), (e_pbc, f_pbc) = teacher.evaluate(
        [(numbers, positions), (numbers, positions, cell)]
    )
    assert abs(e_pbc - e_gas) > 1e-3
    e_ref, f_ref = _ase_reference(pair_model_path, numbers, positions, cell)
    assert e_pbc == pytest.approx(e_ref, abs=1e-5)
    np.testing.assert_allclose(f_pbc, f_ref, atol=1e-5)
    assert np.allclose(f_pbc.sum(axis=0), 0.0, atol=1e-8)  # translation invariance
