"""A TorchScript stand-in potential for test_openmm_python_force_torch.py.

The model under test there is not a particular published potential -- it is
whatever a user loads with ``torch.jit.load``. This stands in for one, with an
energy simple enough that its value and its gradient can be worked out by hand,
so the test can check the number rather than check that nothing crashed:

    E(r) = 0.5 * K * sum_i |r_i|^2      r in ANGSTROM, E in "eV"

The units are deliberately not OpenMM's. A real ML potential wants angstroms
and returns eV, Hartree or kcal/mol, and the conversion on the way in and out
is where these scripts go wrong, so the test should have to do it.

Why this lives in a module of its own rather than inline in the test: the test
body runs in a subprocess launched with ``python -c``, and
``torch.jit.script`` needs real source to read -- given a class defined in a
command-line string it fails with "could not get source code". Keeping the
model here also means the test goes through ``torch.jit.save`` and
``torch.jit.load``, which is the path a user's own checkpoint takes.

Importing this module triggers ``import torch``; gate it with
``pytest.importorskip("torch.nn")`` first, for the reason spelled out at the
top of _torch_helpers.py.
"""

import torch

#: Force constant, eV/A^2. Small, so adding this potential to a molecule that
#: CHARMM has already built does not distort the geometry it was built with.
K_EV_PER_A2 = 1.0e-4

#: The conversions the caller has to get right. Kept here so the test asserts
#: against the same numbers the stand-in is defined in.
EV_TO_KJ_PER_MOL = 96.485
ANGSTROM_PER_NM = 10.0


class HarmonicToOrigin(torch.nn.Module):
    """Every atom harmonically attracted to the origin.

    Takes ``(coords, numbers)`` positionally, like a plain TorchScript
    potential. ``numbers`` is accepted and ignored: a real potential needs the
    elements, and taking the argument keeps the call signature honest.
    """

    def __init__(self, k: float = K_EV_PER_A2):
        super().__init__()
        self.k = k

    def forward(self, coords: torch.Tensor,
                numbers: torch.Tensor) -> torch.Tensor:
        return 0.5 * self.k * (coords * coords).sum()


def write_scripted_model(path) -> str:
    """Script the stand-in and save it, returning the path as a string."""
    torch.jit.script(HarmonicToOrigin()).save(str(path))
    return str(path)
