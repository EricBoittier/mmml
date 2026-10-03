"""Torch ``nn.Module`` building blocks shared by the test_torch_*.py files.

Each of the harmonic torch tests defines a near-identical ``torch.nn.Module``
subclass inline. Lift those out into one place so that adding a new variant
(or fixing a bug in the shared graph) doesn't fan out across five files.

Four variants live here:

    SimpleHarmonic        forward(positions) -> energy
    SimpleHarmonicwForce  forward(positions) -> (energy, force)
    ScaledHarmonic        forward(positions, k) -> energy
    ScaledHarmonicwForce  forward(positions, k) -> (energy, force)

All four take ``coordinate_mask`` (bool/int index array of restrained atoms)
and ``reference_coords`` in Angstroms; the constructor converts to nanometers
for OpenMM compatibility (positions are nm in the OpenMM convention).

Importing this module triggers ``import torch``; the test files all gate
that with ``pytest.importorskip("torch.nn")`` and import this module
*after* the importorskip, so the helper isn't loaded on a torch-less build.
The submodule rather than ``torch`` itself is what the guard has to ask for:
a leftover ``site-packages/torch/`` with no ``__init__.py`` is a namespace
package, so ``import torch`` succeeds and the ``pt.nn.Module`` subclasses
below would fail at import instead.
"""

import torch as pt


class SimpleHarmonic(pt.nn.Module):
    """Central harmonic potential, energy-only forward (forces autograd'd)."""

    def __init__(self, coordinate_mask, reference_coords, device="cpu"):
        super().__init__()
        self.device = device
        self.coordinate_mask = pt.tensor(coordinate_mask).to(self.device)
        self.reference_coords = pt.tensor(reference_coords / 10).to(self.device)

    def forward(self, positions):
        delta_pos = positions[self.coordinate_mask] - self.reference_coords
        energy = pt.sum(delta_pos**2)
        return energy


class SimpleHarmonicwForce(pt.nn.Module):
    """Same potential as SimpleHarmonic, returning (energy, analytic force)."""

    def __init__(self, coordinate_mask, reference_coords, device="cpu"):
        super().__init__()
        self.coordinate_mask = pt.tensor(coordinate_mask).to(device)
        self.reference_coords = pt.tensor(reference_coords / 10).to(device)

    def forward(self, positions):
        delta_pos = positions[self.coordinate_mask] - self.reference_coords
        deriv_all = pt.zeros_like(positions)
        deriv_all[self.coordinate_mask] -= 2 * delta_pos
        energy = pt.sum(delta_pos**2)
        force = deriv_all
        return (energy, force)


class ScaledHarmonic(pt.nn.Module):
    """Harmonic potential scaled by a global parameter ``k``."""

    def __init__(self, coordinate_mask, reference_coords, device="cpu"):
        super().__init__()
        self.device = device
        self.coordinate_mask = pt.tensor(coordinate_mask).to(self.device)
        self.reference_coords = pt.tensor(reference_coords / 10).to(self.device)

    def forward(self, positions, k):
        delta_pos = positions[self.coordinate_mask] - self.reference_coords
        energy = k * pt.sum(delta_pos**2)
        return energy


class ScaledHarmonicwForce(pt.nn.Module):
    """Scaled harmonic potential returning (energy, analytic force)."""

    def __init__(self, coordinate_mask, reference_coords, device="cpu"):
        super().__init__()
        self.coordinate_mask = pt.tensor(coordinate_mask).to(device)
        self.reference_coords = pt.tensor(reference_coords / 10).to(device)

    def forward(self, positions, k):
        delta_pos = positions[self.coordinate_mask] - self.reference_coords
        deriv_all = pt.zeros_like(positions)
        deriv_all[self.coordinate_mask] -= 2 * delta_pos
        energy = k * pt.sum(delta_pos**2)
        force = k * deriv_all
        return (energy, force)
