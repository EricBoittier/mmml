import functools

import jax
import jax.numpy as jnp
import optax
from optax import tree_utils as otu


from physnetjax.training.loss import (
    mean_absolute_error,
    mean_squared_loss,
    mean_squared_loss_QD,
)

DTYPE = jnp.float32


@functools.partial(
    jax.jit,
    static_argnames=(
        "model_apply",
        "optimizer_update",
        "batch_size",
        "doCharges",
        "debug",
    ),
)
def train_step(
    model_apply,
    optimizer_update,
    transform_state,
    batch,
    batch_size,
    doCharges,
    energy_weight,
    forces_weight,
    dipole_weight,
    charges_weight,
    opt_state,
    params,
    ema_params,
    debug=False,
    ema_decay=0.999,
):
    if doCharges:

        def loss_fn(params):
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
            return loss, (
                output["energy"],
                output["forces"],
                output["charges"],
                output["dipoles"],
            )

    else:

        def loss_fn(params):
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
            return loss, (output["energy"], output["forces"])

    if doCharges:
        (loss, (energy, forces, charges, dipole)), grad = jax.value_and_grad(
            loss_fn, has_aux=True
        )(params)
    else:
        (loss, (energy, forces)), grad = jax.value_and_grad(loss_fn, has_aux=True)(
            params
        )

    updates, opt_state = optimizer_update(grad, opt_state, params)
    # check for nans in updates
    # print("Checking for nans in updates")
    # import lovely_jax as lj
    # lj.monkey_patch()
    # jax.debug.print("updates {x}", x=updates)

    # update "reduce on plateau" state
    updates = otu.tree_scalar_mul(transform_state.scale, updates)
    params = optax.apply_updates(params, updates)

    energy_mae = mean_absolute_error(
        energy,
        batch["E"],
        batch_size,
    )
    forces_mae = mean_absolute_error(
        forces * batch["atom_mask"][..., None],
        batch["F"] * batch["atom_mask"][..., None],
        batch["atom_mask"].sum() * 3,
    )
    if doCharges:
        dipole_mae = mean_absolute_error(dipole, batch["D"], batch_size)
    else:
        dipole_mae = 0

    # Update EMA weights

    ema_params = jax.tree_map(
        lambda ema, new: ema_decay * ema + (1 - ema_decay) * new,
        ema_params,
        params,
    )

    return (
        params,
        ema_params,
        opt_state,
        transform_state,
        loss,
        energy_mae,
        forces_mae,
        dipole_mae,
    )
