import functools

from typing import Any

try:
    import jax  # type: ignore
    import jax.numpy as jnp  # type: ignore
except ModuleNotFoundError:  # pragma: no cover - exercised during doc builds
    jax = None  # type: ignore[assignment]
    jnp = None  # type: ignore[assignment]

try:
    import optax  # type: ignore
    from optax import tree_utils as otu  # type: ignore
except ModuleNotFoundError:  # pragma: no cover
    optax = None  # type: ignore[assignment]
    otu = None  # type: ignore[assignment]

from mmml.models.physnetjax.physnetjax.training.loss import (
    mean_absolute_error,
    mean_squared_loss,
    mean_squared_loss_distill,
    mean_squared_loss_QD,
    mean_squared_loss_QD_distill,
)


TRAIN_STEP_DOC = """Single training step for PhysNetJax model.

Performs forward pass, computes loss, calculates gradients, updates
parameters, and maintains exponential moving average (EMA) of parameters.
Supports both standard energy/force prediction and charge/dipole prediction.
"""


def _forward(model_apply, params, batch, batch_size):
    return model_apply(
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


if jax is None or jnp is None or optax is None or otu is None:  # pragma: no cover

    def train_step(*_args: Any, **_kwargs: Any) -> Any:
        """Single training step for PhysNetJax model (requires jax/optax)."""

        raise ModuleNotFoundError(
            "jax and optax must be installed to use the training utilities"
        )

else:

    DTYPE = jnp.float32

    @functools.partial(
        jax.jit,
        static_argnames=(
            "model_apply",
            "optimizer_update",
            "batch_size",
            "doCharges",
            "doDistill",
            "distill_energy",
            "distill_forces",
            "distill_dipole",
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
        teacher_params=None,
        distill_alpha=1.0,
        doDistill=False,
        distill_energy=True,
        distill_forces=True,
        distill_dipole=True,
        debug: bool = False,
        ema_decay: float = 0.999,
    ):
        """Implementation of :data:`TRAIN_STEP_DOC`."""

        teacher_output = None
        if doDistill and teacher_params is not None:
            teacher_output = jax.lax.stop_gradient(
                _forward(model_apply, teacher_params, batch, batch_size)
            )

        if doCharges:

            def loss_fn(params):
                output = _forward(model_apply, params, batch, batch_size)
                if doDistill and teacher_output is not None:
                    loss = mean_squared_loss_QD_distill(
                        energy_prediction=output["energy"],
                        forces_prediction=output["forces"],
                        dipole_prediction=output["dipoles"],
                        total_charges_prediction=output["sum_charges"],
                        energy_target_gt=batch["E"],
                        forces_target_gt=batch["F"],
                        dipole_target_gt=batch["D"],
                        total_charge_target=jnp.zeros_like(output["sum_charges"]),
                        energy_target_teacher=teacher_output["energy"],
                        forces_target_teacher=teacher_output["forces"],
                        dipole_target_teacher=teacher_output["dipoles"],
                        energy_weight=energy_weight,
                        forces_weight=forces_weight,
                        dipole_weight=dipole_weight,
                        total_charge_weight=charges_weight,
                        atomic_mask=batch["atom_mask"],
                        distill_alpha=distill_alpha,
                        distill_energy=distill_energy,
                        distill_forces=distill_forces,
                        distill_dipole=distill_dipole,
                    )
                else:
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
                output = _forward(model_apply, params, batch, batch_size)
                if doDistill and teacher_output is not None:
                    loss = mean_squared_loss_distill(
                        energy_prediction=output["energy"],
                        forces_prediction=output["forces"],
                        energy_target_gt=batch["E"],
                        forces_target_gt=batch["F"],
                        energy_target_teacher=teacher_output["energy"],
                        forces_target_teacher=teacher_output["forces"],
                        energy_weight=energy_weight,
                        forces_weight=forces_weight,
                        atomic_mask=batch["atom_mask"],
                        distill_alpha=distill_alpha,
                        distill_energy=distill_energy,
                        distill_forces=distill_forces,
                    )
                else:
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
            (loss, (energy, forces)), grad = jax.value_and_grad(
                loss_fn, has_aux=True
            )(params)

        updates, opt_state = optimizer_update(grad, opt_state, params)

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

        ema_params = jax.tree_util.tree_map(
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

    train_step.__doc__ = TRAIN_STEP_DOC
