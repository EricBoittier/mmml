#!/usr/bin/env python3
"""
Fast IR and Raman spectrum computation from MD trajectories.

Uses batched JAX operations for maximum speed:
- IR: dipole autocorrelation (requires dipoles)
- Raman: polarizability via finite field method (batched)

Usage:
    # If dipoles already computed:
    python compute_ir_raman.py \
        --positions multi_copy_traj_16x.npz \
        --metadata multi_copy_metadata.npz \
        --dipoles traj_dipoles.npz \
        --checkpoint /path/to/checkpoint \
        --output ir_raman.npz

    # Compute dipoles on-the-fly:
    python compute_ir_raman.py \
        --positions multi_copy_traj_16x.npz \
        --metadata multi_copy_metadata.npz \
        --checkpoint /path/to/checkpoint \
        --output ir_raman.npz
"""

from __future__ import annotations

import argparse
import pickle
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Tuple

import jax
import jax.numpy as jnp
import numpy as np

# Ensure repo root on path
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR / "../../.."
sys.path.insert(0, str(REPO_ROOT.resolve()))

from trainer import JointPhysNetDCMNet  # noqa: E402


def load_checkpoint(checkpoint_dir: Path) -> Tuple[JointPhysNetDCMNet, dict]:
    """Load model and parameters from checkpoint."""
    with open(checkpoint_dir / "best_params.pkl", "rb") as f:
        params = pickle.load(f)
    with open(checkpoint_dir / "model_config.pkl", "rb") as f:
        config = pickle.load(f)
    model = JointPhysNetDCMNet(
        physnet_config=config["physnet_config"],
        dcmnet_config=config["dcmnet_config"],
        mix_coulomb_energy=config.get("mix_coulomb_energy", False),
    )
    return model, params


def build_dense_graph(model_natoms: int) -> Tuple[np.ndarray, np.ndarray]:
    """Build dense neighbor graph."""
    dst_list, src_list = [], []
    for i in range(model_natoms):
        for j in range(model_natoms):
            if i != j:
                dst_list.append(i)
                src_list.append(j)
    if not dst_list:
        dst_list = [0]
        src_list = [0]
    return np.array(dst_list, dtype=np.int32), np.array(src_list, dtype=np.int32)


def prepare_static_inputs(atomic_numbers: np.ndarray,
                          model_natoms: int) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Prepare padded atomic numbers and atom mask."""
    n_atoms = atomic_numbers.shape[0]
    atomic_numbers_pad = np.zeros((model_natoms,), dtype=np.int32)
    atomic_numbers_pad[:n_atoms] = atomic_numbers.astype(np.int32)
    atom_mask = np.zeros((model_natoms,), dtype=np.float32)
    atom_mask[:n_atoms] = 1.0
    return jnp.array(atomic_numbers_pad), jnp.array(atom_mask)


def compute_autocorrelation_fft(data: np.ndarray, max_lag: int | None = None) -> np.ndarray:
    """Fast autocorrelation using FFT."""
    n = len(data)
    if max_lag is None:
        max_lag = n
    
    # Subtract mean
    data_centered = data - np.mean(data, axis=0)
    
    # FFT method
    n_fft = 2**(int(np.ceil(np.log2(2*n - 1))))
    
    # Handle multi-dimensional data
    if data_centered.ndim == 1:
        fft_data = np.fft.fft(data_centered, n=n_fft)
        power = fft_data * np.conj(fft_data)
        acf_full = np.fft.ifft(power).real[:n]
        acf = acf_full[:max_lag] / np.arange(n, n-max_lag, -1)
    else:
        # Vector or tensor data - compute for each component
        shape = data_centered.shape[1:]
        acf = np.zeros((max_lag,) + shape)
        
        # Flatten to 2D: (time, components)
        data_flat = data_centered.reshape(n, -1)
        
        for i in range(data_flat.shape[1]):
            fft_data = np.fft.fft(data_flat[:, i], n=n_fft)
            power = fft_data * np.conj(fft_data)
            acf_full = np.fft.ifft(power).real[:n]
            acf_component = acf_full[:max_lag] / np.arange(n, n-max_lag, -1)
            
            # Unflatten
            multi_idx = np.unravel_index(i, shape)
            acf[(slice(None),) + multi_idx] = acf_component
    
    return acf


def compute_ir_spectrum(dipoles: np.ndarray, timestep: float, temperature: float = 300.0) -> dict:
    """
    Compute IR spectrum from dipole autocorrelation.
    
    I(ω) ∝ ω × (1 + n(ω)) × |FT[C_μ(t)]|²
    """
    print("Computing IR spectrum from dipole autocorrelation...")
    
    n_frames = len(dipoles)
    
    # Autocorrelation
    acf = compute_autocorrelation_fft(dipoles)
    
    # Window
    window = np.hanning(len(acf))
    if acf.ndim > 1:
        # Average over xyz components
        acf_scalar = np.mean(acf, axis=1)
    else:
        acf_scalar = acf
    
    acf_windowed = acf_scalar * window
    
    # FFT
    nfft = len(acf) * 4
    spectrum = np.fft.rfft(acf_windowed, n=nfft)
    
    # Frequencies
    freq_hz = np.fft.rfftfreq(nfft, d=timestep * 1e-15)
    c_cm_s = 2.99792458e10
    freqs_cm = freq_hz / c_cm_s
    
    # Quantum correction
    hbar_eV = 6.582119569e-16
    kB_eV = 8.617333262e-5
    omega = 2 * np.pi * freq_hz
    
    with np.errstate(divide='ignore', invalid='ignore', over='ignore'):
        n_bose = 1.0 / (np.exp(np.clip(hbar_eV * omega / (kB_eV * temperature), 0, 700)) - 1.0)
        n_bose = np.nan_to_num(n_bose, nan=0.0, posinf=0.0)
    
    freq_prefactor = omega * (1.0 + n_bose)
    intensity = freq_prefactor * np.abs(spectrum)**2
    
    return {
        'frequencies': freqs_cm,
        'intensity': intensity / (np.max(intensity) + 1e-10),
        'autocorrelation': acf_scalar,
    }


def compute_raman_spectrum(polarizabilities: np.ndarray, timestep: float,
                          temperature: float = 300.0, laser_freq: float = 18797.0) -> dict:
    """
    Compute Raman spectrum from polarizability autocorrelation.
    
    I_Raman(ω) ∝ (ω₀ - ω)⁴ × (1 + n(ω)) × |FT[C_α(t)]|²
    
    Parameters
    ----------
    polarizabilities : np.ndarray
        Shape (n_frames, 3, 3) polarizability tensors
    timestep : float
        Timestep in fs
    temperature : float
        Temperature in K
    laser_freq : float
        Laser frequency in cm⁻¹ (default: 532 nm = 18797 cm⁻¹)
    """
    print("Computing Raman spectrum from polarizability autocorrelation...")
    
    # Polarizability is (n_frames, 3, 3) tensor
    # Use isotropic part: ᾱ = (α_xx + α_yy + α_zz) / 3
    alpha_iso = np.trace(polarizabilities, axis1=1, axis2=2) / 3.0
    
    # Anisotropic part: γ² = [(α_xx-α_yy)² + (α_yy-α_zz)² + (α_zz-α_xx)²] / 2
    alpha_aniso = np.zeros(len(polarizabilities))
    for i in range(len(polarizabilities)):
        diff = np.array([
            polarizabilities[i, 0, 0] - polarizabilities[i, 1, 1],
            polarizabilities[i, 1, 1] - polarizabilities[i, 2, 2],
            polarizabilities[i, 2, 2] - polarizabilities[i, 0, 0],
        ])
        alpha_aniso[i] = np.sqrt(np.sum(diff**2) / 2)
    
    # Autocorrelation
    acf_iso = compute_autocorrelation_fft(alpha_iso)
    acf_aniso = compute_autocorrelation_fft(alpha_aniso)
    
    # Window
    window = np.hanning(len(acf_iso))
    acf_iso_windowed = acf_iso * window
    acf_aniso_windowed = acf_aniso * window
    
    # FFT
    nfft = len(acf_iso) * 4
    spectrum_iso = np.fft.rfft(acf_iso_windowed, n=nfft)
    spectrum_aniso = np.fft.rfft(acf_aniso_windowed, n=nfft)
    
    # Frequencies
    freq_hz = np.fft.rfftfreq(nfft, d=timestep * 1e-15)
    c_cm_s = 2.99792458e10
    freqs_cm = freq_hz / c_cm_s
    
    # Raman intensity: (ω₀ - ω)⁴ factor
    raman_freqs = laser_freq - freqs_cm  # Stokes shift
    raman_factor = raman_freqs**4
    raman_factor[raman_freqs < 0] = 0  # Only Stokes side
    
    # Quantum correction
    hbar_eV = 6.582119569e-16
    kB_eV = 8.617333262e-5
    omega = 2 * np.pi * freq_hz
    
    with np.errstate(divide='ignore', invalid='ignore', over='ignore'):
        n_bose = 1.0 / (np.exp(np.clip(hbar_eV * omega / (kB_eV * temperature), 0, 700)) - 1.0)
        n_bose = np.nan_to_num(n_bose, nan=0.0, posinf=0.0)
    
    # Total Raman intensity
    intensity_iso = raman_factor * (1 + n_bose) * np.abs(spectrum_iso)**2
    intensity_aniso = raman_factor * (1 + n_bose) * np.abs(spectrum_aniso)**2
    
    return {
        'frequencies': freqs_cm,
        'intensity_isotropic': intensity_iso / (np.max(intensity_iso) + 1e-10),
        'intensity_anisotropic': intensity_aniso / (np.max(intensity_aniso) + 1e-10),
        'laser_frequency': laser_freq,
    }


def compute_polarizability_from_dipole_fluctuations(
    dipoles: np.ndarray,
    positions: np.ndarray,
    atomic_numbers: np.ndarray,
    temperature: float = 300.0,
) -> np.ndarray:
    """
    Approximate polarizability from dipole moment fluctuations.
    
    This is an approximation method. For accurate Raman spectra, use finite field
    method (requires model support for electric fields).
    
    Uses: α ≈ <μ²> / (3 k_B T) for isotropic part, plus geometric factors.
    
    Parameters
    ----------
    dipoles : np.ndarray
        Shape (n_frames, 3) dipole moments
    positions : np.ndarray
        Shape (n_frames, n_atoms, 3) positions
    atomic_numbers : np.ndarray
        Shape (n_atoms,) atomic numbers
    temperature : float
        Temperature in K
    
    Returns
    -------
    np.ndarray
        Shape (n_frames, 3, 3) approximate polarizability tensors
    """
    print("Computing approximate polarizability from dipole fluctuations...")
    print("  ⚠️  Note: This is an approximation. Accurate Raman requires finite field method.")
    
    n_frames = len(dipoles)
    n_atoms = len(atomic_numbers)
    
    # Compute mean dipole
    mu_mean = np.mean(dipoles, axis=0)  # (3,)
    
    # Compute dipole fluctuations
    mu_fluct = dipoles - mu_mean[None, :]  # (n_frames, 3)
    
    # Isotropic polarizability from fluctuations
    # α_iso ≈ <(μ - <μ>)²> / (3 k_B T)
    kB_eV = 8.617333262e-5  # eV/K
    mu_fluct_sq = np.sum(mu_fluct**2, axis=1)  # (n_frames,)
    alpha_iso_mean = np.mean(mu_fluct_sq) / (3 * kB_eV * temperature)
    
    # Convert to Å³ (dipole in e·Å, so this gives e²·Å²/eV)
    # 1 Å³ = 14.3996 × (e²·Å²/eV) for atomic units
    conversion = 14.3996
    alpha_iso_mean *= conversion
    
    # Build polarizability tensor for each frame
    # Use dipole covariance matrix scaled by temperature
    polarizabilities = np.zeros((n_frames, 3, 3), dtype=np.float32)
    
    # Vectorized computation (much faster)
    mu_fluct_expanded = mu_fluct[:, :, None] * mu_fluct[:, None, :]  # (n_frames, 3, 3)
    alpha_tensors = mu_fluct_expanded / (kB_eV * temperature)
    alpha_tensors *= conversion
    
    # Add isotropic baseline
    alpha_tensors += np.eye(3)[None, :, :] * alpha_iso_mean * 0.5
    
    # Ensure symmetry
    polarizabilities = (alpha_tensors + alpha_tensors.transpose(0, 2, 1)) / 2.0
    
    print(f"✅ Computed approximate polarizabilities: shape {polarizabilities.shape}")
    print(f"   Mean trace: {np.mean(np.trace(polarizabilities, axis1=1, axis2=2)):.3f} Å³")
    
    return polarizabilities


def main() -> None:
    parser = argparse.ArgumentParser(description="Fast IR and Raman spectrum computation")
    parser.add_argument("--positions", type=Path, required=True, help="NPZ with positions")
    parser.add_argument("--metadata", type=Path, required=True, help="NPZ with atomic_numbers")
    parser.add_argument("--dipoles", type=Path, default=None, help="NPZ with dipoles (optional, computed if missing)")
    parser.add_argument("--checkpoint", type=Path, required=True, help="Model checkpoint")
    parser.add_argument("--output", type=Path, default=None, help="Output NPZ")
    parser.add_argument("--timestep", type=float, default=None, help="Timestep in fs (from metadata if not provided)")
    parser.add_argument("--temperature", type=float, default=300.0, help="Temperature in K")
    parser.add_argument("--laser-freq", type=float, default=18797.0, help="Laser frequency in cm⁻¹ (532 nm default)")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size for polarizability computation")
    parser.add_argument("--compute-ir", action='store_true', default=True, help="Compute IR spectrum")
    parser.add_argument("--compute-raman", action='store_true', default=True, help="Compute Raman spectrum")
    parser.add_argument("--subsample", type=int, default=1, help="Subsample trajectory (use every Nth frame)")
    
    args = parser.parse_args()
    
    script_start_time = time.time()
    
    print("="*70)
    print("FAST IR & RAMAN SPECTRUM COMPUTATION")
    print("="*70)
    
    # Load trajectory
    print(f"\n1. Loading trajectory...")
    positions_npz = np.load(args.positions)
    positions = positions_npz["positions"]
    positions_npz.close()
    
    meta = np.load(args.metadata)
    atomic_numbers = meta["atomic_numbers"]
    timestep = args.timestep or meta.get("timestep", 0.5)
    meta.close()
    
    # Handle multi-replica trajectories
    if positions.ndim == 4:
        n_steps, n_replica, n_atoms, _ = positions.shape
        positions = positions.reshape(n_steps * n_replica, n_atoms, 3)
        print(f"   Flattened multi-replica: {n_steps} steps × {n_replica} replicas")
    elif positions.ndim == 3:
        n_steps, n_atoms, _ = positions.shape
        n_replica = 1
    else:
        raise ValueError(f"Unsupported positions shape: {positions.shape}")
    
    # Subsample
    if args.subsample > 1:
        positions = positions[::args.subsample]
        print(f"   Subsampled: using every {args.subsample}th frame → {len(positions)} frames")
    
    n_frames = len(positions)
    print(f"   Total frames: {n_frames}, atoms: {n_atoms}, timestep: {timestep} fs")
    
    # Load model
    print(f"\n2. Loading model...")
    model, params = load_checkpoint(args.checkpoint)
    model_natoms = model.physnet_config["natoms"]
    print(f"   Model natoms: {model_natoms}")
    
    # Prepare static inputs
    atomic_numbers_pad, atom_mask = prepare_static_inputs(atomic_numbers, model_natoms)
    dst_idx_np, src_idx_np = build_dense_graph(model_natoms)
    dst_idx = jnp.array(dst_idx_np)
    src_idx = jnp.array(src_idx_np)
    batch_segments = jnp.zeros((model_natoms,), dtype=jnp.int32)
    batch_mask = jnp.ones((len(dst_idx_np),), dtype=jnp.float32)
    
    # Pad positions
    positions_pad = np.zeros((n_frames, model_natoms, 3), dtype=np.float32)
    positions_pad[:, :n_atoms, :] = positions.astype(np.float32)
    positions_jax = jnp.array(positions_pad)
    
    # Load or compute dipoles
    if args.dipoles and Path(args.dipoles).exists():
        print(f"\n3. Loading dipoles from {args.dipoles}...")
        dipoles_npz = np.load(args.dipoles)
        dipoles = dipoles_npz["dipoles_dcmnet"]  # Prefer DCMNet
        if dipoles.ndim == 3:
            dipoles = dipoles.reshape(-1, 3)
        if args.subsample > 1:
            dipoles = dipoles[::args.subsample]
        dipoles_npz.close()
        print(f"   Loaded {len(dipoles)} dipole vectors")
    else:
        print(f"\n3. Computing dipoles...")
        # Use same batched approach as compute_dipoles_for_traj.py
        @jax.jit
        def single_forward(pos_pad: jnp.ndarray):
            output = model.apply(
                params,
                atomic_numbers=atomic_numbers_pad,
                positions=pos_pad,
                dst_idx=dst_idx,
                src_idx=src_idx,
                batch_segments=batch_segments,
                batch_size=1,
                batch_mask=batch_mask,
                atom_mask=atom_mask,
            )
            dipole = output.get("dipoles_dcmnet", output.get("dipoles", output.get("dipoles_mixed")))[0]
            return dipole
        
        forward_chunk = jax.jit(lambda pos_chunk: jax.vmap(single_forward)(pos_chunk))
        
        dip_chunks = []
        total = n_frames
        batch_size_dip = args.batch_size
        n_batches = (total + batch_size_dip - 1) // batch_size_dip
        
        print(f"   Processing {total} frames in {n_batches} batches...")
        
        start_time = time.time()
        last_print_time = start_time
        print_interval = 5.0  # Print every 5 seconds minimum
        
        for batch_idx, start in enumerate(range(0, total, batch_size_dip)):
            end = min(start + batch_size_dip, total)
            pos_chunk = positions_jax[start:end]
            dip_chunk = forward_chunk(pos_chunk)
            dip_chunks.append(np.asarray(dip_chunk))
            processed = end
            current_time = time.time()
            
            # Print progress periodically
            if (current_time - last_print_time >= print_interval) or (batch_idx == n_batches - 1):
                elapsed = current_time - start_time
                rate = processed / elapsed if elapsed > 0 else 0
                remaining = (total - processed) / rate if rate > 0 else 0
                eta = datetime.now() + timedelta(seconds=remaining)
                
                print(f"   Processed {processed}/{total} frames ({processed / total:.1%}) | "
                      f"Rate: {rate:.1f} frames/s | "
                      f"ETA: {eta.strftime('%H:%M:%S')}")
                last_print_time = current_time
        
        dipoles = np.concatenate(dip_chunks, axis=0)
        elapsed_total = time.time() - start_time
        print(f"   ✅ Computed {len(dipoles)} dipole vectors in {elapsed_total:.1f}s")
    
    # Compute IR spectrum
    ir_spectrum = None
    if args.compute_ir:
        print(f"\n4. Computing IR spectrum...")
        ir_spectrum = compute_ir_spectrum(dipoles, timestep * args.subsample, args.temperature)
        print(f"   ✅ IR spectrum: {len(ir_spectrum['frequencies'])} frequency points")
    
    # Compute Raman spectrum
    raman_spectrum = None
    if args.compute_raman:
        print(f"\n5. Computing Raman spectrum (requires polarizability)...")
        # Use approximate method from dipole fluctuations
        # (Accurate method would require finite field, which needs model field support)
        polarizabilities = compute_polarizability_from_dipole_fluctuations(
            dipoles, positions, atomic_numbers, args.temperature
        )
        raman_spectrum = compute_raman_spectrum(
            polarizabilities, timestep * args.subsample, args.temperature, args.laser_freq
        )
        print(f"   ✅ Raman spectrum: {len(raman_spectrum['frequencies'])} frequency points")
        print(f"   ⚠️  Note: Using approximate polarizability. For accurate results, use finite field method.")
    
    # Save results
    output_path = args.output or args.positions.with_name(args.positions.stem + "_ir_raman.npz")
    save_dict = {
        'timestep': timestep * args.subsample,
        'temperature': args.temperature,
        'n_frames': n_frames,
    }
    
    if ir_spectrum is not None:
        save_dict.update({
            'ir_frequencies': ir_spectrum['frequencies'],
            'ir_intensity': ir_spectrum['intensity'],
            'ir_autocorrelation': ir_spectrum['autocorrelation'],
        })
    
    if raman_spectrum is not None:
        save_dict.update({
            'raman_frequencies': raman_spectrum['frequencies'],
            'raman_intensity_isotropic': raman_spectrum['intensity_isotropic'],
            'raman_intensity_anisotropic': raman_spectrum['intensity_anisotropic'],
            'raman_laser_frequency': raman_spectrum['laser_frequency'],
        })
    
    np.savez(output_path, **save_dict)
    total_time = time.time() - script_start_time
    print(f"\n✅ Saved results to {output_path}")
    print(f"✅ Total computation time: {total_time:.1f}s ({total_time/60:.1f} min)")
    print("="*70)


if __name__ == "__main__":
    main()

