import os
import argparse
import pickle
import numpy as np
from pathlib import Path
from tqdm import tqdm

parser = argparse.ArgumentParser(description="Compile calculation results into a single npz file with padding.")
parser.add_argument("--input_dir", type=str, required=True, help="Directory containing result files (pickled dicts)")
parser.add_argument("--output", type=str, required=True, help="Output npz file path")
args = parser.parse_args()

input_dir = Path(args.input_dir)
output_path = Path(args.output)

compiled = []
max_n_atoms = 0
max_n_grid = 0

# First pass: determine max_n_atoms and max_n_grid
files = sorted(input_dir.glob("*.pkl"))
print(f"Scanning {len(files)} files for max shape...")
for file in tqdm(files, desc="Scanning"):
    with open(file, "rb") as f:
        d = pickle.load(f)
    n_atoms = d["R"].shape[0]
    n_grid = d["esp"].shape[0]
    if n_atoms > max_n_atoms:
        max_n_atoms = n_atoms
    if n_grid > max_n_grid:
        max_n_grid = n_grid
    compiled.append(d)

# Second pass: pad and process
processed = []
print("Padding and stacking data...")
for d in tqdm(compiled, desc="Processing"):
    # Remove unserializable keys
    d = dict(d)  # shallow copy
    d.pop("mol", None)
    d.pop("opt_callback", None)
    d.pop("calcs", None)
    # Calculate n_grid
    d["n_grid"] = len(d["esp"])
    # Rename esp_grid to vdw_surface
    if "esp_grid" in d:
        d["vdw_surface"] = d.pop("esp_grid")
    # Pad arrays
    n_atoms = d["R"].shape[0]
    n_grid = d["esp"].shape[0]
    # Pad R
    R_padded = np.zeros((max_n_atoms, 3))
    R_padded[:n_atoms] = d["R"]
    d["R"] = R_padded
    # Pad Z
    Z_padded = np.zeros((max_n_atoms,), dtype=d["Z"].dtype)
    Z_padded[:n_atoms] = d["Z"]
    d["Z"] = Z_padded
    if "mono" in d:
        # Pad mono
        mono_padded = np.zeros((max_n_atoms,), dtype=d["mono"].dtype)
        mono_padded[:n_atoms] = d["mono"]
        d["mono"] = mono_padded
    else:
        # if mono is not present, use atomic numbers as mono
        d["mono"] = d["Z"]

    # Pad gradient
    if "gradient" in d:
        grad_padded = np.zeros((max_n_atoms, 3))
        grad_padded[:n_atoms] = d["gradient"]
        d["gradient"] = grad_padded

    # Pad vdw_surface
    vdw_padded = np.full((max_n_grid, 3), 1000.0)
    vdw_padded[:n_grid] = d["vdw_surface"]
    d["vdw_surface"] = vdw_padded
    # Pad esp
    esp_padded = np.zeros((max_n_grid,), dtype=d["esp"].dtype)
    esp_padded[:n_grid] = d["esp"]
    d["esp"] = esp_padded
    # Pad Q if present (assume shape (3,3), no padding needed)
    # Pad D if present (assume shape (3,), no padding needed)
    processed.append(d)

# Stack arrays for each key
keys = processed[0].keys()
arrays = {k: np.stack([d[k] for d in processed]) for k in keys}
np.savez_compressed(output_path, **arrays)

print(f"Saved compiled results to {output_path}") 