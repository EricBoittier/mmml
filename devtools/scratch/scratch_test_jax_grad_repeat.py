import jax
import jax.numpy as jnp
import numpy as np

def f(x, repeats, tot):
    rep = jnp.repeat(x, repeats, total_repeat_length=tot)
    return jnp.sum(rep ** 2)

x = jnp.array([1.0, 2.0, 3.0])
repeats = jnp.array([2, 1, 3])
tot = 6

try:
    grad = jax.grad(f)(x, repeats, tot)
    print("Grad worked:", grad)
except Exception as e:
    print("Grad failed:", e)
