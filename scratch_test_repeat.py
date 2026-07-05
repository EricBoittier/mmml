import jax
import jax.numpy as jnp

def test_func():
    mm_scale = jnp.array([1.0, 2.0])
    n_pairs_per_dimer_arr = jnp.array([0, 0], dtype=jnp.int32)
    mm_scale_expanded = jnp.repeat(
        mm_scale,
        n_pairs_per_dimer_arr,
        total_repeat_length=0
    )
    return mm_scale_expanded

print(jax.jit(test_func)())
