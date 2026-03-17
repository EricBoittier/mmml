#!/usr/bin/env python
"""Check CuPy/CUDA compatibility. Run on a GPU node before using pyscf-dft."""
import sys

print("=== GPU (nvidia-smi) ===")
import subprocess
try:
    subprocess.run(["nvidia-smi", "--query-gpu=name,driver_version,compute_cap", "--format=csv"], check=False)
except FileNotFoundError:
    print("nvidia-smi not found")

print("\n=== CuPy/CUDA ===")
try:
    import cupy as cp
    p = cp.cuda.runtime.getDeviceProperties(0)
    print(f"CuPy: {cp.__version__}")
    print(f"GPU compute capability: {p.major}.{p.minor}")
    _ = cp.array([1.0])
    print("CuPy test: OK")
except Exception as e:
    print(f"Error: {e}")
    sys.exit(1)
