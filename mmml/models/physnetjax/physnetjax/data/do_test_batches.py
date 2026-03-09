import time

import e3x.ops
import jax
import jax.numpy as jnp
import numpy as np


def generate_mock_data(data_size, num_atoms=60):
    """
    Generate mock molecular dataset for testing batch preparation.

    Args:
        data_size (int): Total number of molecular configurations
        num_atoms (int): Number of atoms per configuration

    Returns:
        dict: Simulated molecular dataset
    """
    key = jax.random.PRNGKey(42)

    return {
        "R": jax.random.normal(
            key, shape=(data_size, num_atoms, 3)
        ),  # Atomic positions
        "F": jax.random.normal(key, shape=(data_size, num_atoms, 3)),  # Forces
        "E": jax.random.normal(key, shape=(data_size, 1)),  # Energies
        "Z": jax.random.randint(
            key, shape=(data_size, num_atoms), minval=1, maxval=10
        ),  # Atomic numbers
        "N": jnp.full(
            (data_size,), num_atoms, dtype=jnp.int32
        ),  # Number of atoms per configuration
        "mono": jax.random.normal(
            key, shape=(data_size, num_atoms)
        ),  # Additional feature
    }


def performance_comparison(original_func, optimized_func, data, batch_size=32):
    """
    Compare performance and correctness of batch preparation functions.

    Args:
        original_func (callable): Original batch preparation function
        optimized_func (callable): Optimized batch preparation function
        data (dict): Input dataset
        batch_size (int): Batch size for processing
    """
    key = jax.random.PRNGKey(42)
    data_keys = list(data.keys())

    # Time original function
    start_time = time.time()
    original_batches = original_func(
        key, data, batch_size, data_keys=tuple(data_keys), num_atoms=data["R"].shape[1]
    )
    original_time = time.time() - start_time

    # Time optimized function
    start_time = time.time()
    optimized_batches = optimized_func(
        key, data, batch_size, data_keys=tuple(data_keys), num_atoms=data["R"].shape[1]
    )
    optimized_time = time.time() - start_time

    # Verify batch count and structure
    print(f"Original batches: {len(original_batches)}")
    print(f"Optimized batches: {len(optimized_batches)}")
    assert len(original_batches) == len(optimized_batches), "Batch count mismatch"

    # Compare batch contents
    for orig_batch, opt_batch in zip(original_batches, optimized_batches):
        for key in orig_batch:
            try:
                np.testing.assert_allclose(
                    orig_batch[key],
                    opt_batch[key],
                    rtol=1e-5,
                    atol=1e-5,
                    err_msg=f"Mismatch in key: {key}",
                )
            except Exception as e:
                print(f"Verification failed for key: {key}")
                raise e

    # Performance comparison
    print(f"\nPerformance Comparison:")
    print(f"Original Function Time: {original_time:.4f} seconds")
    print(f"Optimized Function Time: {optimized_time:.4f} seconds")
    print(f"Speedup: {original_time / optimized_time:.2f}x")


def main():
    # Import the original and optimized functions
    from physnetjax.data import prepare_batches as optimized_prepare_batches
    from physnetjax.data import prepare_batches_old as original_prepare_batches

    # Generate test dataset
    data_sizes = [1000, 10000, 50000]
    batch_sizes = [1, 16, 32, 64, 128]

    for data_size in data_sizes:
        print(f"\n--- Dataset Size: {data_size} ---")
        mock_data = generate_mock_data(data_size)

        for batch_size in batch_sizes:
            print(f"\nBatch Size: {batch_size}")
            performance_comparison(
                original_prepare_batches,
                optimized_prepare_batches,
                mock_data,
                batch_size,
            )


if __name__ == "__main__":
    main()
