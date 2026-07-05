import jax
import jax.numpy as jnp

def safe_norm(x, axis=None, keepdims=False):
    is_zero = jnp.all(x == 0, axis=axis, keepdims=True)
    x_safe = jnp.where(is_zero, jnp.ones_like(x), x)
    n = jnp.linalg.norm(x_safe, axis=axis, keepdims=keepdims)
    return jnp.where(jnp.squeeze(is_zero, axis=axis) if not keepdims and axis is not None else is_zero, 0.0, n)

grad_f = jax.grad(lambda x: safe_norm(x))
print(grad_f(jnp.array([0.0, 0.0])))
print(grad_f(jnp.array([1.0, 2.0])))
