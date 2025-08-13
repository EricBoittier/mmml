import functools

import jax
import jax.numpy as jnp
import orbax

from physnetjax.training.loss import (
    mean_absolute_error,
    mean_squared_loss,
    mean_squared_loss_QD,
)

DTYPE = jnp.float32


@functools.partial(
    jax.jit, static_argnames=("model_apply",
                              "batch_size",
                              "charges"))
def eval_step(
    model_apply,
    batch,
    batch_size,
    charges,
    energy_weight,
    forces_weight,
    dipole_weight,
    charges_weight,
    params,
):
    if charges:
        output = model_apply(
            params,
            atomic_numbers=batch["Z"],
            positions=batch["R"],
            dst_idx=batch["dst_idx"],
            src_idx=batch["src_idx"],
            batch_segments=batch["batch_segments"],
            batch_size=batch_size,
            batch_mask=batch["batch_mask"],
            atom_mask=batch["atom_mask"],
        )

        loss = mean_squared_loss_QD(
            energy_prediction=output["energy"],
            energy_target=batch["E"],
            energy_weight=energy_weight,
            forces_prediction=output["forces"],
            forces_target=batch["F"],
            forces_weight=forces_weight,
            dipole_prediction=output["dipoles"],
            dipole_target=batch["D"],
            dipole_weight=dipole_weight,
            total_charges_prediction=output["sum_charges"],
            total_charge_target=jnp.zeros_like(output["sum_charges"]),
            total_charge_weight=charges_weight,
            atomic_mask=batch["atom_mask"],
        )

        energy_mae = mean_absolute_error(output["energy"], batch["E"], batch_size)
        forces_mae = mean_absolute_error(
            output["forces"] * batch["atom_mask"][..., None],
            batch["F"] * batch["atom_mask"][..., None],
            batch["atom_mask"].sum() * 3,
        )
        dipole_mae = mean_absolute_error(output["dipoles"], batch["D"], batch_size)
        return loss, energy_mae, forces_mae, dipole_mae
    else:
        output = model_apply(
            params,
            atomic_numbers=batch["Z"],
            positions=batch["R"],
            dst_idx=batch["dst_idx"],
            src_idx=batch["src_idx"],
            batch_segments=batch["batch_segments"],
            batch_size=batch_size,
            batch_mask=batch["batch_mask"],
            atom_mask=batch["atom_mask"],
        )
        loss = mean_squared_loss(
            energy_prediction=output["energy"],
            energy_target=batch["E"],
            forces_prediction=output["forces"],
            forces_target=batch["F"],
            forces_weight=forces_weight,
            energy_weight=energy_weight,
            atomic_mask=batch["atom_mask"],
        )
        energy_mae = mean_absolute_error(output["energy"], batch["E"], batch_size)
        forces_mae = mean_absolute_error(
            output["forces"] * batch["atom_mask"][..., None],
            batch["F"] * batch["atom_mask"][..., None],
            batch["atom_mask"].sum() * 3,
        )
        return loss, energy_mae, forces_mae, 0
