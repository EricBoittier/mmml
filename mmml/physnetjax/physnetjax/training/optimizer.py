import jax.numpy as jnp
import optax
import optax.contrib

base_learning_rate = 0.001

base_schedule_fn = optax.schedules.warmup_exponential_decay_schedule(
    init_value=base_learning_rate,
    peak_value=base_learning_rate * 1.05,
    warmup_steps=10,
    transition_steps=10,
    decay_rate=0.999,
)
base_optimizer = optax.chain(
    #    optax.adaptive_grad_clip(1.0),
    optax.clip_by_global_norm(10.0),
    optax.amsgrad(learning_rate=base_schedule_fn, b1=0.9, b2=0.99, eps=1e-3),
)

base_transform = optax.contrib.reduce_on_plateau(
    patience=5,
    cooldown=5,
    factor=0.99,
    rtol=1e-4,
    accumulation_size=5,
    min_scale=0.01,
)


def get_optimizer(
    learning_rate: float = 0.001,
    schedule_fn: optax.Schedule | str | None = None,
    optimizer: optax.GradientTransformation | str | None = None,
    transform: optax.GradientTransformation | str | None = None,
    clip_global: bool | float = True,
    patience: int = 5,
    cooldown: int = 5,
    factor: float = 0.9,
    rtol: float = 1e-4,
    accumulation_size: int = 5,
    min_scale: float = 0.01,
    **kwargs,
):

    if isinstance(clip_global, bool):
        clip_global = 10.0 if clip_global else None
    elif isinstance(clip_global, float) and clip_global > 0:
        pass
    else:
        raise ValueError("clip_global must be a bool or positive float.")

    if schedule_fn is None:
        _schedule_fn = optax.schedules.constant_schedule(learning_rate)
    elif isinstance(schedule_fn, str):
        if schedule_fn == "warmup":
            _schedule_fn = optax.schedules.warmup_exponential_decay_schedule(
                init_value=learning_rate,
                peak_value=learning_rate * 3,
                warmup_steps=100,
                transition_steps=10,
                decay_rate=0.9999,
            )
        elif schedule_fn == "cosine_annealing":
            cosine_dicts = [
                {
                    "transition_steps": 5000,
                    "peak_value": learning_rate * 1.5,
                    "decay_steps": 5000,
                    "alpha": 0.3,
                }
                for _ in range(5)
            ]
            _schedule_fn = optax.schedules.sgdr_schedule(cosine_dicts=cosine_dicts)
        elif schedule_fn == "exponential":
            _schedule_fn = optax.schedules.exponential_decay_schedule(
                init_value=learning_rate, decay_rate=0.995
            )
        elif schedule_fn == "polynomial":
            _schedule_fn = optax.schedules.polynomial_decay_schedule(
                init_value=learning_rate, end_value=0.0, power=1.0, transition_steps=100
            )
        elif schedule_fn == "cosine":
            _schedule_fn = optax.schedules.cosine_decay_schedule(
                init_value=learning_rate, decay_steps=5000, alpha=0.3
            )
        elif schedule_fn == "warmup_cosine":
            _schedule_fn = optax.schedules.warmup_cosine_decay_schedule(
                init_value=learning_rate,
                peak_value=learning_rate * 1.5,
                end_value=learning_rate * 0.1,
                warmup_steps=100,
                decay_steps=5000,
            )
        elif schedule_fn == "constant":
            _schedule_fn = optax.schedules.constant_schedule(learning_rate)
        else:
            raise ValueError(
                f"Invalid schedule_fn: {schedule_fn}. Must be None, a valid optax.Schedule object, "
                f"or one of the supported string options ('warmup', 'cosine_annealing', 'exponential', etc.)."
            )
    else:
        _schedule_fn = schedule_fn
        _schedule_fn = optax.schedules.constant_schedule(learning_rate)

    if optimizer is None:
        _optimizer = optax.chain(
            optax.clip_by_global_norm(clip_global),
            optax.amsgrad(learning_rate=_schedule_fn, b1=0.9, b2=0.99, eps=1e-7),
        )
    elif isinstance(optimizer, str):
        _chain = []
        if clip_global:
            if not isinstance(clip_global, float):
                clip_global = 10.0
            _chain.append(optax.clip_by_global_norm(clip_global))
            # _chain.append(optax.adaptive_grad_clip(clip_global))
        if optimizer == "adam":
            _chain.append(optax.adam(learning_rate=_schedule_fn))
        elif optimizer == "adamw":
            _chain.append(optax.adamw(learning_rate=_schedule_fn))
        elif optimizer == "amsgrad":
            _chain.append(
                optax.amsgrad(learning_rate=_schedule_fn, b1=0.9, b2=0.99, eps=1e-3)
            )
        _optimizer = optax.chain(*_chain)
    else:
        _optimizer = optimizer
    # else:
    #     raise ValueError(
    #         f"Invalid optimizer: {optimizer}. Must be None, a valid optax.GradientTransformation object, "
    #         f"or one of the supported string options ('adam', 'adamw', 'amsgrad')."
    #     )

    if transform is None:
        _transform = optax.contrib.reduce_on_plateau(
            patience=patience,
            cooldown=cooldown,
            factor=factor,
            rtol=rtol,
            accumulation_size=accumulation_size,
            min_scale=min_scale,
        )

    elif isinstance(transform, str):
        if transform == "reduce_on_plateau":
            _transform = optax.contrib.reduce_on_plateau(
                patience=5,
                cooldown=5,
                factor=0.9,
                rtol=1e-4,
                accumulation_size=5,
                min_scale=0.01,
            )
        else:
            raise ValueError(
                f"Invalid transform: {transform}. Must be None, a valid optax.GradientTransformation object, "
                f"or one of the supported string options ('reduce_on_plateau')."
            )

    else:
        _transform = optax.contrib.reduce_on_plateau(
            patience=5,
            cooldown=5,
            factor=0.9,
            rtol=1e-4,
            accumulation_size=5,
            min_scale=0.01,
        )

    optimizer_kwargs = {
        "optimizer": optimizer,
        "optimized_chain": _optimizer,
        "schedule_fn": schedule_fn,
        "scheduling_function": _schedule_fn,
        "transform": transform,
        "reduce_transform": _transform,
        "clip_global": clip_global,
        "b1": 0.9,
        "b2": 0.99,
        "eps": 1e-3,
    }

    return _optimizer, _transform, _schedule_fn, optimizer_kwargs


def cycled_cosine_annealing_schedule(init_lr, period=200):
    """
    Creates a cosine annealing learning rate schedule with repeated cycles.

    Args:
        init_lr (float): Initial learning rate at the start of each cycle.
        period (int): The number of steps in each cycle.
    Returns:
        optax.Schedule: A cycled cosine annealing learning rate schedule.
    """

    # Adjust step to account for the starting step
    num_cycles = 200
    print(period, num_cycles)
    lr_schedule = optax.join_schedules(
        schedules=[
            optax.cosine_onecycle_schedule(
                transition_steps=period // 2,
                peak_value=init_lr * (0.99**i),
                div_factor=1.3,
                final_div_factor=1.6,
            )
            for i in range(num_cycles)
        ],
        boundaries=jnp.cumsum(jnp.array([period] * num_cycles)),
    )

    return lr_schedule
