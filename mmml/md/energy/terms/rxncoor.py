"""Harmonic umbrella bias on a reaction coordinate built from distances.

The existing ``smd`` term biases a single interatomic distance, which cannot
express an SN2 reaction coordinate: methyl transfer is described by the
antisymmetric stretch ``xi = r(C-X) - r(C-N)``, where forming and breaking bonds
move in opposite directions. This term biases any
:class:`~mmml.md.restraints.LinearDistanceCV`, so the solvated umbrella windows
use exactly the same coordinate definition as the gas-phase packed sampler in
:mod:`mmml.umbrella` -- two profiles computed against differently-defined
coordinates would not be comparable, which is the whole point of the campaign.

The CV is minimum-image aware so a solute that straddles a periodic boundary
still reports the intramolecular distance rather than a box-length artefact.
``target`` may be overridden per step via a ``lambda_t`` kwarg, matching the
``smd`` convention, so the same term drives steered pulling as well as fixed
windows.
"""

from __future__ import annotations

from typing import Any

from mmml.md.energy.registry import EnergyContext, TermFns, register_term
from mmml.md.energy.terms._common import ase_contribution_from_jax
from mmml.md.restraints import (
    AngleWall,
    BondRetentionWall,
    FlatBottomWall,
    LinearDistanceCV,
)
from mmml.md.system import MolecularSystem

__all__ = ["ReactionCoordinateBiasTerm"]


@register_term("rxncoor")
class ReactionCoordinateBiasTerm:
    """``0.5 k (xi(R) - xi_0)^2`` on a linear combination of distances.

    ``k_ev_per_A2`` is in eV/A^2 to match the rest of the jax energy stack. The
    Menshutkin literature quotes force constants in kcal/mol/A^2 (Turan et al.
    use 150), so divide by 23.06 when porting a value across.
    """

    name = "rxncoor"

    def __init__(
        self,
        cv: Any = None,
        target: float | None = None,
        k_ev_per_A2: float = 6.505,
        pairs: Any = None,
        coefficients: Any = None,
        walls: Any = None,
    ):
        if cv is None:
            if pairs is None or coefficients is None:
                raise ValueError(
                    "rxncoor needs a cv (LinearDistanceCV / spec) or pairs + coefficients"
                )
            cv = {"pairs": pairs, "coefficients": coefficients}
        self.cv = LinearDistanceCV.from_spec(cv)
        self.target = None if target is None else float(target)
        self.k_ev_per_A2 = float(k_ev_per_A2)
        if self.k_ev_per_A2 < 0:
            raise ValueError(
                f"force constant must be non-negative (got {self.k_ev_per_A2})"
            )
        # Flat-bottom walls on *other* coordinates, applied alongside the bias.
        #
        # A harmonic bias on xi constrains only xi. For an antisymmetric stretch
        # that leaves the sum r(C-X) + r(C-N) completely free, and the bias
        # exerts no force along it whatsoever -- so the methyl can drift away
        # from both partners at once while xi sits exactly on its target. This
        # was measured: a window centred at xi = +0.35 held xi to within 0.15 A
        # while the sum walked from 6.25 to 5.58 A, against a training set whose
        # sum never exceeds 4.44 A anywhere near that xi. The ML energy then went
        # below its training floor and the run diverged.
        #
        # The walls belong here rather than in a separate term because they are
        # part of defining the restrained ensemble: they are zero inside the
        # allowed region, so they do not bias the sampling that MBAR unbiases,
        # but without them the sampled ensemble is not the intended one.
        # A wall is either a flat-bottom band on a linear CV or a
        # bond-retention bound on the shortest of several distances; both
        # expose .energy(R, cell=) and .validate_against(n).
        self.walls = tuple(
            w
            if isinstance(w, (FlatBottomWall, BondRetentionWall, AngleWall))
            else (
                AngleWall.from_spec(w)
                if isinstance(w, dict) and "atoms" in w
                else BondRetentionWall.from_spec(w)
                if isinstance(w, dict) and "r_max" in w
                else FlatBottomWall.from_spec(w)
            )
            for w in (walls or ())
        )

    def neighbor_request(self, system: MolecularSystem):
        return None  # a handful of named atoms; no pair list

    def make(self, system: MolecularSystem, ctx: EnergyContext) -> TermFns:
        import jax.numpy as jnp

        self.cv.validate_against(system.n_atoms)
        for w in self.walls:
            w.validate_against(system.n_atoms)
        cv = self.cv
        walls = self.walls
        k = self.k_ev_per_A2
        cell = None if system.box is None else jnp.asarray(system.box)

        # Default the window center to the CV of the built geometry, so an
        # unspecified target is a restraint at the starting value rather than a
        # silent pull toward zero.
        fixed_target = (
            self.target
            if self.target is not None
            else float(cv.value_numpy(system.R, cell=None if system.box is None else system.box))
        )

        def energy_fn(R, *, lambda_t: Any = None, box=None, **kwargs) -> Any:
            del kwargs
            cell_used = cell if box is None else jnp.asarray(box)
            target = fixed_target if lambda_t is None else lambda_t
            value = cv.value(R, cell=cell_used)
            total = 0.5 * k * jnp.square(value - target)
            for w in walls:
                total = total + w.energy(R, cell=cell_used)
            return total

        energy_fn.cv = cv  # type: ignore[attr-defined]
        energy_fn.target = fixed_target  # type: ignore[attr-defined]
        energy_fn.k_ev_per_A2 = k  # type: ignore[attr-defined]
        energy_fn.walls = walls  # type: ignore[attr-defined]

        return TermFns(
            jax_energy_fn=energy_fn,
            ase_contribution=ase_contribution_from_jax(energy_fn),
            neighbor_request=None,
        )
