#!/usr/bin/env python3
"""Profile PhysNetJax epoch components: batch prep, train, valid, checkpoint.

Example:
  JAX_PLATFORMS=cpu python scripts/profile_physnet_training.py --epochs 3
  MMML_PHYSNET_PROFILE_EPOCH_TIMING=1 mmml physnet-train --config train.yaml
"""

from __future__ import annotations

import argparse
import time

import jax
import jax.numpy as jnp

from mmml.models.physnetjax.physnetjax.data.batches import (
    _pair_indices,
    prepare_batches_fast,
    prepare_batches_jit,
)
from mmml.models.physnetjax.physnetjax.models.model import EF


def _mock_data(n: int, natoms: int):
    key = jax.random.PRNGKey(0)
    return {
        "R": jax.random.normal(key, (n, natoms, 3)),
        "F": jax.random.normal(key, (n, natoms, 3)),
        "E": jax.random.normal(key, (n, 1)),
        "Z": jax.random.randint(key, (n, natoms), 1, 10),
        "N": jnp.full((n,), natoms, dtype=jnp.int32),
    }


def profile_batch_prep(n: int, natoms: int, batch_size: int, repeats: int) -> None:
    data = _mock_data(n, natoms)
    keys = ("R", "Z", "F", "E", "N")
    key = jax.random.PRNGKey(1)
    pair_cache = _pair_indices(natoms, batch_size)

    for label, fn, kwargs in (
        ("jit", prepare_batches_jit, {}),
        ("fast+cache", prepare_batches_fast, {"pair_cache": pair_cache}),
    ):
        fn(key, data, batch_size, data_keys=keys, num_atoms=natoms, **kwargs)
        t0 = time.perf_counter()
        for i in range(repeats):
            fn(
                jax.random.fold_in(key, i),
                data,
                batch_size,
                data_keys=keys,
                num_atoms=natoms,
                **kwargs,
            )
        ms = (time.perf_counter() - t0) / repeats * 1000.0
        print(f"batch_prep/{label}: {ms:.2f} ms/epoch ({n // batch_size} batches)")


def profile_model(features: int, natoms: int, batch_size: int, steps: int) -> None:
    from mmml.models.physnetjax.physnetjax.training.trainstep import train_step
    import optax

    data = _mock_data(max(steps * batch_size, batch_size), natoms)
    pair_cache = _pair_indices(natoms, batch_size)
    batches = prepare_batches_fast(
        jax.random.PRNGKey(2),
        data,
        batch_size,
        data_keys=("R", "Z", "F", "E", "N"),
        num_atoms=natoms,
        pair_cache=pair_cache,
    )[:steps]

    model = EF(
        features=features,
        max_degree=0,
        num_iterations=2,
        num_basis_functions=16,
        cutoff=6.0,
        max_atomic_number=35,
        charges=False,
        natoms=natoms,
        n_res=3,
        zbl=True,
        debug=False,
        efa=False,
        use_energy_bias=True,
        use_pbc=False,
        include_electrostatics=True,
    )
    dst_idx, src_idx = batches[0]["dst_idx"], batches[0]["src_idx"]
    key = jax.random.PRNGKey(3)
    params = model.init(
        key,
        atomic_numbers=data["Z"][0],
        positions=data["R"][0],
        dst_idx=dst_idx,
        src_idx=src_idx,
    )
    from mmml.models.physnetjax.physnetjax.training.optimizer import (
        base_transform,
        get_optimizer,
    )

    optimizer, transform, schedule_fn, _ = get_optimizer(learning_rate=1e-3)
    opt_state = optimizer.init(params)
    ema_params = params
    transform_state = transform.init(params)

    batch = batches[0]
    _ = train_step(
        model_apply=model.apply,
        optimizer_update=optimizer.update,
        transform_state=transform_state,
        batch=batch,
        batch_size=batch_size,
        energy_weight=1.0,
        forces_weight=52.91,
        dipole_weight=27.21,
        charges_weight=14.39,
        opt_state=opt_state,
        doCharges=False,
        params=params,
        ema_params=ema_params,
        debug=False,
    )
    jax.block_until_ready(params)

    t0 = time.perf_counter()
    for batch in batches:
        params, ema_params, opt_state, transform_state, *_rest = train_step(
            model_apply=model.apply,
            optimizer_update=optimizer.update,
            transform_state=transform_state,
            batch=batch,
            batch_size=batch_size,
            energy_weight=1.0,
            forces_weight=52.91,
            dipole_weight=27.21,
            charges_weight=14.39,
            opt_state=opt_state,
            doCharges=False,
            params=params,
            ema_params=ema_params,
            debug=False,
        )
    jax.block_until_ready(params)
    ms = (time.perf_counter() - t0) * 1000.0
    print(f"train/features={features}: {ms:.1f} ms for {steps} step(s)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-train", type=int, default=2000)
    parser.add_argument("--natoms", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--train-steps", type=int, default=8)
    args = parser.parse_args()

    print("=== PhysNetJax training profiler ===")
    profile_batch_prep(args.n_train, args.natoms, args.batch_size, args.repeats)
    for features in (16, 64, 128):
        profile_model(features, args.natoms, args.batch_size, args.train_steps)


if __name__ == "__main__":
    main()
