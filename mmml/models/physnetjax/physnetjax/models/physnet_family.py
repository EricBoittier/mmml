"""PhysNet family facade: shared mixin, config flags, and class resolution.

Phase 2 of MPNN harmonization. Full Flax-module merge of PhysNet into
SpookyPhysNet would rename parameter trees and break checkpoints, so the
family stays as:

- :class:`PhysNet` — no Q/S conditioning (plain E/F)
- :class:`SpookyPhysNet` — production Q/S-conditioned hybrid path
- :class:`PhysNetChargeSpin` — **deprecated** third fork; do not extend

Shared non-parametric methods live on :class:`PhysNetFamilyMixin` and call
:mod:`mpnn_kernels`. New code should use :func:`resolve_physnet_class` rather
than importing ChargeSpin.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Type

import jax.numpy as jnp
from flax import linen as nn

from mmml.models.physnetjax.physnetjax.models.mpnn_kernels import (
    calc_electrostatics_switches,
    encode_geometry_and_basis,
    molecular_dipole_from_charges,
    pair_electrostatics_energy,
)

__all__ = [
    "PhysNetFamilyConfig",
    "PhysNetFamilyMixin",
    "resolve_physnet_class",
]


@dataclass(frozen=True)
class PhysNetFamilyConfig:
    """Flags describing which PhysNet-family module to construct.

    These do **not** change Flax parameter layouts by themselves; they select
    among existing checkpoint-compatible classes.
    """

    condition_on_charge_spin: bool = False
    predict_charges: bool = False
    include_electrostatics: bool = True

    def __post_init__(self) -> None:
        if self.include_electrostatics and not self.predict_charges:
            # Electrostatics in this family is assembled from predicted charges.
            object.__setattr__(self, "predict_charges", True)


def resolve_physnet_class(config: PhysNetFamilyConfig) -> Type[nn.Module]:
    """Return the canonical Flax module class for ``config``.

    Q/S conditioning always resolves to :class:`SpookyPhysNet` (not the
    deprecated ChargeSpin fork).
    """
    if config.condition_on_charge_spin:
        from mmml.models.physnetjax.physnetjax.models.spooky_model import (
            SpookyPhysNet,
        )

        return SpookyPhysNet
    from mmml.models.physnetjax.physnetjax.models.model import PhysNet

    return PhysNet


class PhysNetFamilyMixin:
    """Shared method bodies for PhysNet / SpookyPhysNet (and ChargeSpin).

    Expects the host ``nn.Module`` to define the usual hyperparameter fields
    (``cutoff``, ``switch_*``, ``num_basis_functions``, …).
    """

    def _calculate_geometric_features(
        self,
        positions: jnp.ndarray,
        dst_idx: jnp.ndarray,
        src_idx: jnp.ndarray,
        batch_mask: Optional[jnp.ndarray] = None,
        cell: Optional[jnp.ndarray] = None,
        edge_mask: Optional[jnp.ndarray] = None,
    ):
        return encode_geometry_and_basis(
            positions,
            dst_idx,
            src_idx,
            num_basis_functions=self.num_basis_functions,
            max_degree=self.max_degree,
            cutoff=self.cutoff,
            batch_mask=batch_mask,
            edge_mask=edge_mask,
            cell=cell,
            use_pbc=bool(getattr(self, "use_pbc", False)),
        )

    def _calc_switches(self, displacements: jnp.ndarray, batch_mask: jnp.ndarray):
        return calc_electrostatics_switches(
            displacements,
            batch_mask,
            switch_start=self.switch_start,
            switch_end=self.switch_end,
            electrostatics_off_start=self.electrostatics_off_start,
            electrostatics_off_end=self.electrostatics_off_end,
            electrostatics_damping_sigma=float(
                getattr(self, "electrostatics_damping_sigma", 0.0)
            ),
        )

    def _calculate_electrostatics(
        self,
        atomic_charges: jnp.ndarray,
        r: jnp.ndarray,
        off_dist: jnp.ndarray,
        eshift: jnp.ndarray,
        dst_idx: jnp.ndarray,
        src_idx: jnp.ndarray,
        atom_mask: jnp.ndarray,
        batch_mask: jnp.ndarray,
        batch_segments: jnp.ndarray,
        batch_size: int,
    ):
        del atom_mask  # call-site compatibility
        return pair_electrostatics_energy(
            atomic_charges,
            r,
            off_dist,
            eshift,
            dst_idx,
            src_idx,
            batch_mask,
            batch_segments,
            batch_size,
        )

    def _calculate_dipole(
        self,
        positions: jnp.ndarray,
        atomic_numbers: jnp.ndarray,
        charges: jnp.ndarray,
        batch_segments: jnp.ndarray,
        batch_size: int,
    ) -> Any:
        return molecular_dipole_from_charges(
            positions,
            atomic_numbers,
            charges,
            batch_segments,
            batch_size,
        )
