"""
Packed Memory-Mapped Data Loader for PhysNet

This module provides efficient loading of variable-size molecular datasets
stored in packed memory-mapped format, with conversion to PhysNet-compatible batches.
"""

import os
from typing import Dict, Iterator, Optional

import e3x
import jax.numpy as jnp
import numpy as np


class PackedMemmapLoader:
    """
    Memory-mapped data loader for packed molecular datasets.
    
    This loader efficiently handles variable-size molecules stored in packed format,
    with bucketed batching to minimize padding overhead. Data is stored in a packed
    format where all molecules are concatenated, with offsets tracking boundaries.
    
    Expected directory structure:
        data_path/
            offsets.npy       - (N+1,) array of atom offsets for each molecule
            n_atoms.npy       - (N,) array of atom counts per molecule
            Z_pack.int32      - (sum_atoms,) packed atomic numbers
            R_pack.f32        - (sum_atoms, 3) packed positions
            F_pack.f32        - (sum_atoms, 3) packed forces
            E.f64             - (N,) energies
            Qtot.f64          - (N,) total charges (optional)
    
    Parameters
    ----------
    path : str
        Path to directory containing packed memmap files
    batch_size : int
        Number of molecules per batch
    shuffle : bool, optional
        Whether to shuffle data, by default True
    bucket_size : int, optional
        Size of buckets for sorting by molecule size, by default 8192.
        Larger buckets reduce padding but increase memory usage.
    seed : int, optional
        Random seed for shuffling, by default 0
        
    Attributes
    ----------
    N : int
        Total number of molecules in dataset
    n_atoms : np.ndarray
        Array of atom counts per molecule
    """
    
    def __init__(
        self,
        path: str,
        batch_size: int,
        shuffle: bool = True,
        bucket_size: int = 8192,
        seed: int = 0,
    ):
        self.path = path
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.bucket_size = bucket_size
        self.rng = np.random.default_rng(seed)

        # Load metadata FIRST to determine array shapes
        self.offsets = np.load(os.path.join(path, "offsets.npy"))
        self.n_atoms = np.load(os.path.join(path, "n_atoms.npy"))
        self.N = int(self.n_atoms.shape[0])
        sumA = int(self.offsets[-1])

        # Open read-only memmaps with explicit shapes
        self.Z_pack = np.memmap(
            os.path.join(path, "Z_pack.int32"),
            dtype=np.int32,
            mode="r",
            shape=(sumA,),
        )
        self.R_pack = np.memmap(
            os.path.join(path, "R_pack.f32"),
            dtype=np.float32,
            mode="r",
            shape=(sumA, 3),
        )
        self.F_pack = np.memmap(
            os.path.join(path, "F_pack.f32"),
            dtype=np.float32,
            mode="r",
            shape=(sumA, 3),
        )
        self.E = np.memmap(
            os.path.join(path, "E.f64"),
            dtype=np.float64,
            mode="r",
            shape=(self.N,),
        )
        
        # Qtot is optional
        qtot_path = os.path.join(path, "Qtot.f64")
        if os.path.exists(qtot_path):
            self.Qtot = np.memmap(
                qtot_path,
                dtype=np.float64,
                mode="r",
                shape=(self.N,),
            )
        else:
            self.Qtot = None

        # Validate file sizes for data integrity
        self._validate_file_sizes(sumA)

        self.indices = np.arange(self.N, dtype=np.int64)

    def _validate_file_sizes(self, sumA: int):
        """Validate that file sizes match expected dimensions."""
        def _expect_bytes(dtype, shape):
            return np.dtype(dtype).itemsize * int(np.prod(shape))

        def _filesize(filename):
            return os.path.getsize(os.path.join(self.path, filename))

        checks = [
            ("Z_pack.int32", np.int32, (sumA,)),
            ("R_pack.f32", np.float32, (sumA, 3)),
            ("F_pack.f32", np.float32, (sumA, 3)),
            ("E.f64", np.float64, (self.N,)),
        ]
        
        if self.Qtot is not None:
            checks.append(("Qtot.f64", np.float64, (self.N,)))

        for filename, dtype, shape in checks:
            expected = _expect_bytes(dtype, shape)
            actual = _filesize(filename)
            if actual != expected:
                raise ValueError(
                    f"{filename} size mismatch: expected {expected} bytes, got {actual} bytes"
                )

    def _yield_indices_bucketed(self) -> Iterator[np.ndarray]:
        """
        Yield molecule indices in buckets sorted by size.
        
        This reduces padding overhead by grouping similarly-sized molecules
        together in batches.
        
        Yields
        ------
        np.ndarray
            Indices for one batch of molecules
        """
        order = self.indices.copy()
        if self.shuffle:
            self.rng.shuffle(order)

        for start in range(0, self.N, self.bucket_size):
            chunk = order[start : start + self.bucket_size]
            # Sort by size within bucket to minimize padding
            chunk = chunk[np.argsort(self.n_atoms[chunk], kind="mergesort")]

            for bstart in range(0, len(chunk), self.batch_size):
                yield chunk[bstart : bstart + self.batch_size]

    def _slice_mol(self, k: int):
        """
        Extract a single molecule from packed arrays.
        
        Parameters
        ----------
        k : int
            Molecule index
            
        Returns
        -------
        tuple
            (Z, R, F, E, Qtot) arrays for the molecule
        """
        a0, a1 = int(self.offsets[k]), int(self.offsets[k + 1])
        qtot = self.Qtot[k] if self.Qtot is not None else 0.0
        return (
            self.Z_pack[a0:a1],
            self.R_pack[a0:a1],
            self.F_pack[a0:a1],
            self.E[k],
            qtot,
        )

    def batches(
        self, num_atoms: Optional[int] = None, physnet_format: bool = True
    ) -> Iterator[Dict[str, jnp.ndarray]]:
        """
        Generate batches of molecular data.
        
        Parameters
        ----------
        num_atoms : int, optional
            Maximum number of atoms to pad to. If None, uses max from each batch.
        physnet_format : bool, optional
            If True, includes PhysNet-specific fields (dst_idx, src_idx, batch_segments).
            If False, returns simple padded format with mask, by default True.
            
        Yields
        ------
        dict
            Batch dictionary with molecular data. Keys depend on physnet_format:
            
            If physnet_format=True:
                - Z: (B, Amax) atomic numbers
                - R: (B, Amax, 3) positions
                - F: (B, Amax, 3) forces
                - E: (B,) energies
                - N: (B,) atom counts
                - Qtot: (B,) total charges
                - dst_idx: (Amax*(Amax-1),) destination indices for pairs
                - src_idx: (Amax*(Amax-1),) source indices for pairs
                - batch_segments: (B*Amax,) batch index for each atom
                
            If physnet_format=False:
                - Z: (B, Amax) atomic numbers
                - R: (B, Amax, 3) positions
                - F: (B, Amax, 3) forces
                - E: (B,) energies
                - mask: (B, Amax) boolean mask for real atoms
                - Qtot: (B,) total charges
        """
        for batch_idx in self._yield_indices_bucketed():
            if len(batch_idx) == 0:
                continue

            Amax = int(self.n_atoms[batch_idx].max())
            if num_atoms is not None:
                Amax = num_atoms
            B = len(batch_idx)

            # Preallocate arrays
            Z = np.zeros((B, Amax), dtype=np.int32)
            R = np.zeros((B, Amax, 3), dtype=np.float32)
            F = np.zeros((B, Amax, 3), dtype=np.float32)
            N = np.zeros((B,), dtype=np.int32)
            E = np.zeros((B,), dtype=np.float64)
            Qtot = np.zeros((B,), dtype=np.float64)

            # Fill batch
            for j, k in enumerate(batch_idx):
                z, r, f, e, q = self._slice_mol(int(k))
                a = z.shape[0]
                Z[j, :a] = z
                R[j, :a] = r
                F[j, :a] = f
                N[j] = a
                E[j] = e
                Qtot[j] = q

            batch_dict = {
                "Z": jnp.array(Z),
                "R": jnp.array(R),
                "F": jnp.array(F),
                "N": jnp.array(N),
                "E": jnp.array(E),
                "Qtot": jnp.array(Qtot),
            }

            if physnet_format:
                # Generate graph indices for PhysNet
                dst_idx, src_idx = e3x.ops.sparse_pairwise_indices(Amax)
                
                # Create batch segments (which molecule each atom belongs to)
                batch_segments = np.repeat(np.arange(B), Amax).astype(np.int32)

                batch_dict.update({
                    "dst_idx": dst_idx,
                    "src_idx": src_idx,
                    "batch_segments": batch_segments,
                })
            else:
                # Simple mask format
                mask = np.zeros((B, Amax), dtype=bool)
                for j, n in enumerate(N):
                    mask[j, :n] = True
                batch_dict["mask"] = jnp.array(mask)

            yield batch_dict

    def __len__(self) -> int:
        """Return number of molecules in dataset."""
        return self.N

    def __repr__(self) -> str:
        return (
            f"PackedMemmapLoader(path={self.path}, N={self.N}, "
            f"batch_size={self.batch_size}, shuffle={self.shuffle})"
        )


def split_loader(
    loader: PackedMemmapLoader,
    train_fraction: float = 0.9,
    seed: Optional[int] = None,
) -> tuple[PackedMemmapLoader, PackedMemmapLoader]:
    """
    Split a loader into train and validation loaders.
    
    Parameters
    ----------
    loader : PackedMemmapLoader
        Loader to split
    train_fraction : float, optional
        Fraction of data for training, by default 0.9
    seed : int, optional
        Random seed for splitting, by default None (no shuffle)
        
    Returns
    -------
    tuple[PackedMemmapLoader, PackedMemmapLoader]
        (train_loader, valid_loader)
    """
    if seed is not None:
        rng = np.random.default_rng(seed)
        indices = rng.permutation(loader.N)
    else:
        indices = np.arange(loader.N)
    
    n_train = int(loader.N * train_fraction)
    
    # Create train loader
    train_loader = PackedMemmapLoader(
        path=loader.path,
        batch_size=loader.batch_size,
        shuffle=loader.shuffle,
        bucket_size=loader.bucket_size,
        seed=loader.rng.integers(0, 2**31),  # New seed
    )
    train_loader.indices = indices[:n_train]
    train_loader.N = n_train
    
    # Create validation loader
    valid_loader = PackedMemmapLoader(
        path=loader.path,
        batch_size=loader.batch_size,
        shuffle=False,  # Don't shuffle validation
        bucket_size=loader.bucket_size,
        seed=loader.rng.integers(0, 2**31),
    )
    valid_loader.indices = indices[n_train:]
    valid_loader.N = loader.N - n_train
    
    return train_loader, valid_loader

