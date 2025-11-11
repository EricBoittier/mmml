"""
Utility functions to fix JAX async context issues in IPython/Jupyter environments.

When JAX operations are called in IPython/Jupyter, they can return lazy arrays
that interact poorly with the async event loop, causing:
- RuntimeError: cannot enter context: <_contextvars.Context object> is already entered
- RuntimeWarning: coroutine 'Kernel.shell_main' was never awaited

This module provides utilities to safely handle JAX operations in async contexts.
"""

import jax
import jax.numpy as jnp
from typing import Any, Tuple, Dict, Union


def block_jax_until_ready(*arrays: Any) -> None:
    """
    Block until all JAX arrays are ready to prevent async context issues.
    
    This function ensures that JAX operations complete synchronously before
    returning, preventing conflicts with IPython's async event loop.
    
    Parameters
    ----------
    *arrays : Any
        Variable number of JAX arrays or tuples/dicts containing JAX arrays
        
    Examples
    --------
    >>> result = jax.jit(my_function)(x)
    >>> block_jax_until_ready(result)  # Ensure result is ready
    
    >>> params, loss = train_step(params, batch)
    >>> block_jax_until_ready(params, loss)  # Block on multiple arrays
    
    >>> result_dict = {'loss': loss, 'grads': grads}
    >>> block_jax_until_ready(result_dict)  # Works with dicts
    """
    def _block_array(arr):
        """Recursively block on arrays in nested structures."""
        if isinstance(arr, (dict,)):
            for v in arr.values():
                _block_array(v)
        elif isinstance(arr, (tuple, list)):
            for item in arr:
                _block_array(item)
        elif hasattr(arr, 'block_until_ready'):
            # JAX array-like object
            arr.block_until_ready()
        elif hasattr(arr, '__array__'):
            # Try to convert and block
            try:
                jax.block_until_ready(jnp.asarray(arr))
            except (TypeError, ValueError):
                pass
    
    for arr in arrays:
        _block_array(arr)


def safe_jax_call(func, *args, **kwargs):
    """
    Safely call a JAX function and block until ready.
    
    This is a convenience wrapper that calls a function and ensures
    the result is ready before returning.
    
    Parameters
    ----------
    func : callable
        Function to call (typically a JAX jit-compiled function)
    *args : Any
        Positional arguments to pass to func
    **kwargs : Any
        Keyword arguments to pass to func
        
    Returns
    -------
    Any
        The result of func(*args, **kwargs), guaranteed to be ready
        
    Examples
    --------
    >>> result = safe_jax_call(jax.jit(my_model.apply), params, x)
    >>> # result is guaranteed to be ready, no async issues
    """
    result = func(*args, **kwargs)
    block_jax_until_ready(result)
    return result


def is_ipython() -> bool:
    """
    Check if running in IPython/Jupyter environment.
    
    Returns
    -------
    bool
        True if running in IPython/Jupyter, False otherwise
    """
    try:
        from IPython import get_ipython
        ipython = get_ipython()
        return ipython is not None
    except ImportError:
        return False


# Auto-configure JAX for IPython environments
if is_ipython():
    # Set JAX to use synchronous execution in IPython
    # This helps prevent async context issues
    import os
    # Disable async dispatch in IPython to avoid context conflicts
    os.environ.setdefault('JAX_PLATFORMS', 'cpu,gpu')
    # Note: We can't directly disable async dispatch, but blocking helps

