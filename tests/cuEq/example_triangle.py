import jax
import jax.numpy as jnp
from cuequivariance_jax import triangle_multiplicative_update
# Create input tensor with arbitrary batch dimensions
key = jax.random.key(0)
key, subkey = jax.random.split(key)
batch_dim1, batch_dim2, seq_len, D_in = 2, 3, 128, 128
x = jax.random.normal(subkey, (batch_dim1, batch_dim2, seq_len, seq_len, D_in), dtype=jnp.float32)
# Create mask (1 for valid positions, 0 for masked)
mask = jnp.ones((batch_dim1, batch_dim2, seq_len, seq_len))
# Create weight parameters (in practice, these would be learned)
norm_in_weight = jnp.ones(D_in)
norm_in_bias = jnp.zeros(D_in)
# Optional bias parameters for projection and gating layers
p_in_bias = jnp.zeros(2 * D_in)  # Optional input projection bias
g_in_bias = jnp.zeros(2 * D_in)  # Optional input gating bias
p_out_bias = jnp.zeros(D_in)     # Optional output projection bias (would be D_out if dimension changes)
g_out_bias = jnp.zeros(D_in)     # Optional output gating bias (would be D_out if dimension changes)
# Initialize other weights using the key
key, subkey = jax.random.split(key)
# Perform triangular multiplication
output = triangle_multiplicative_update(
    x=x,
    direction="outgoing",  # or "incoming"
    key=subkey,  # Only needed if some weights are None
    mask=mask,
    norm_in_weight=norm_in_weight,
    norm_in_bias=norm_in_bias,
    p_in_bias=p_in_bias,  # Can be None to skip bias
    g_in_bias=g_in_bias,  # Can be None to skip bias
    p_out_bias=p_out_bias,  # Can be None to skip bias
    g_out_bias=g_out_bias,  # Can be None to skip bias
    # ... pass other weights or let them initialize ...
)
print(output.shape)
# (2, 3, 128, 128, 128)
# Example with dimension change: input 128 -> output 256
g_out_weight_256 = jax.random.normal(jax.random.key(1), (256, 128))
p_out_weight_256 = jax.random.normal(jax.random.key(2), (256, 128))
key, subkey2 = jax.random.split(key)
output_256 = triangle_multiplicative_update(
    x=x,
    direction="outgoing",
    key=subkey2,  # Key needed for other weight initialization
    g_out_weight=g_out_weight_256,
    p_out_weight=p_out_weight_256,
)
print(output_256.shape)
# (2, 3, 128, 128, 256)


cuequivariance_jax.triangle_attention(q, k, v, bias, mask, scale, precision=None)
# triangle attention

# Parameters
# :
# q (Array) – Query tensor of shape [B, N, H, S_qo, D].

# k (Array) – Key tensor of shape [B, N, H, S_kv, D].

# v (Array) – Value tensor of shape [B, N, H, S_kv, D].

# bias (Array) – Bias tensor of shape [B, 1, H, S_qo, S_kv].

# mask (Array) – Mask tensor of shape [B, N, 1, 1, S_kv] (boolean, True means valid).

# scale (float) – Scaling factor for the dot product.

# precision (Precision | None) – Precision for the computation (default is None).

# Returns
# :
# A tuple containing the attention output, log-sum-exp, and maximum value.

 
# where 
# , 
# , and 
#  are the query, key, and value tensors, 
#  is the mask bias, and 
#  is the triangle bias.


with cue.assume(cue.SO3, cue.ir_mul):
    x = cuex.RepArray("3x0", jnp.array([1.0, 2.0, 3.0]))
    y = cuex.RepArray("1x1", jnp.array([0.0, 0.0, 0.0]))
cuex.concatenate([x, y])
# {0: 3x0+1} [1. 2. 3. 0. 0. 0.]


# LayerNorm
# class cuequivariance_jax.flax_linen.LayerNorm
# LayerNorm(epsilon: float = 0.01, parent: Union[flax.linen.module.Module, flax.core.scope.Scope, flax.linen.module._Sentinel, NoneType] = <flax.linen.module._Sentinel object at 0x7ff1e8fdbc20>, name: Optional[str] = None)

# __init__(
# epsilon=0.01,
# parent=<flax.linen.module._Sentinel object>,
# name=None,
# )
# Parameters
# :
# epsilon (float)

# parent (Module | Scope | _Sentinel | None)

# name (str | None)

# Return type
# :
# None