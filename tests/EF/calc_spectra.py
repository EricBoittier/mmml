#!/usr/bin/env python3
"""
Calculate IR, Raman, and VCD spectra from the electric field model.

Pipeline:
  1. Hessian  d²E/dR²           → normal modes (frequencies + eigenvectors)
  2. APT      dμ/dR             → IR intensities
  3. dα/dQ    (finite diff)     → Raman activities
  4. AAT      (Lorentz / Born)  → VCD rotational strengths

Usage:
    python calc_spectra.py --params params.json --data data-full.npz
    python calc_spectra.py --params params.json --data data-full.npz --raman --vcd
    python calc_spectra.py --params params.json --data data-full.npz --all
"""

import os
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", ".99")

import argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

import ase
from ase.data import atomic_masses as ASE_ATOMIC_MASSES
from ase.optimize import BFGS

import sys
sys.path.insert(0, str(Path(__file__).parent))

from ase_calc_EF import AseCalculatorEF

# =====================================================================
# Physical constants & unit conversions
# =====================================================================
EV_TO_J   = 1.602176634e-19       # eV → J
ANG_TO_M  = 1e-10                 # Å  → m
AMU_TO_KG = 1.66053906660e-27     # amu → kg
C_CMS     = 2.99792458e10         # speed of light in cm/s

# sqrt(eV / (Å² · amu))  →  cm⁻¹
FREQ_FACTOR = np.sqrt(EV_TO_J / (ANG_TO_M**2 * AMU_TO_KG)) / (2 * np.pi * C_CMS)
# ≈ 521.47


# =====================================================================
# Hessian via finite differences of forces  (memory-safe)
# =====================================================================
def hessian_fd(calc, atoms, delta=0.001):
    """Build the Hessian column-by-column from central finite differences
    of the forces.

        H[i, j] = -( F_j(R + δ e_i) - F_j(R - δ e_i) ) / (2δ)

    Each evaluation uses exactly the same GPU memory as a single
    energy+forces call, so this never OOMs.

    Parameters
    ----------
    calc  : ASE calculator (must return forces)
    atoms : ASE Atoms (positions will be restored after)
    delta : finite-difference step in Å

    Returns
    -------
    hessian : (N, 3, N, 3)  in eV/Å²
    """
    import copy
    N = len(atoms)
    ndof = 3 * N
    pos0 = atoms.get_positions().copy()
    hess = np.zeros((ndof, ndof))

    # We need a scratch atoms object so we don't trigger recalculations
    # on the original one
    work = atoms.copy()
    work.calc = calc
    work.info.update(atoms.info)

    for i in range(ndof):
        s, c = divmod(i, 3)          # atom index, Cartesian component
        if (i + 1) % 10 == 0 or i == 0:
            print(f"    Hessian col {i+1}/{ndof}  (atom {s}, {'xyz'[c]})")

        # +δ
        pos_p = pos0.copy()
        pos_p[s, c] += delta
        work.set_positions(pos_p)
        f_p = work.get_forces().flatten()   # (3N,)

        # –δ
        pos_m = pos0.copy()
        pos_m[s, c] -= delta
        work.set_positions(pos_m)
        f_m = work.get_forces().flatten()   # (3N,)

        hess[i, :] = -(f_p - f_m) / (2.0 * delta)

    # Restore original positions
    atoms.set_positions(pos0)
    # Symmetrise
    hess = 0.5 * (hess + hess.T)
    return hess.reshape(N, 3, N, 3)


# =====================================================================
# Normal-mode analysis
# =====================================================================
def compute_normal_modes(hessian, masses):
    """Diagonalise the mass-weighted Hessian.

    Parameters
    ----------
    hessian : (N, 3, N, 3)  in eV/Å²
    masses  : (N,)           in amu

    Returns
    -------
    frequencies  : (3N,)  cm⁻¹  (negative ⇒ imaginary)
    eigvecs_mw   : (3N, 3N)  mass-weighted eigenvectors  (columns)
    eigvecs_cart : (3N, 3N)  Cartesian eigenvectors       (columns)
        eigvecs_cart[:, k] = eigvecs_mw[:, k] / sqrt(m)
        NOT normalised — correct for  dμ/dQ_k = APT_flat @ eigvecs_cart[:, k]
    """
    N = len(masses)
    ndof = 3 * N

    hess_flat = hessian.reshape(ndof, ndof)
    hess_flat = 0.5 * (hess_flat + hess_flat.T)  # symmetrise

    masses_3n   = np.repeat(masses, 3)
    inv_sqrt_m  = 1.0 / np.sqrt(masses_3n)

    hess_mw = hess_flat * np.outer(inv_sqrt_m, inv_sqrt_m)

    eigenvalues, eigvecs_mw = np.linalg.eigh(hess_mw)

    frequencies = np.sign(eigenvalues) * np.sqrt(np.abs(eigenvalues)) * FREQ_FACTOR

    # Cartesian displacement per unit normal coordinate
    eigvecs_cart = eigvecs_mw * inv_sqrt_m[:, None]

    return frequencies, eigvecs_mw, eigvecs_cart


# =====================================================================
# IR intensities from APT
# =====================================================================
def compute_ir(apt, eigvecs_cart):
    """IR intensities  I_k = |dμ/dQ_k|².

    Parameters
    ----------
    apt          : (3, N, 3)   P_{dipole_α, atom_s, cart_β}
    eigvecs_cart : (3N, 3N)

    Returns
    -------
    ir_intensities : (3N,)
    dmu_dQ         : (3, 3N)   dipole derivative per mode
    """
    ndof = eigvecs_cart.shape[0]
    apt_flat = apt.reshape(3, ndof)          # (3_dip, 3N)
    dmu_dQ   = apt_flat @ eigvecs_cart       # (3, 3N)
    ir_intensities = np.sum(dmu_dQ**2, axis=0)
    return ir_intensities, dmu_dQ


# =====================================================================
# Raman activities  (finite-difference polarizability derivative)
# =====================================================================
def compute_raman(calc, atoms, eigvecs_cart, frequencies,
                  freq_threshold=50.0, delta=0.005):
    """Raman activity  S_k = 45·(ᾱ')² + 7·(γ')².

    Parameters
    ----------
    calc, atoms : calculator & geometry
    eigvecs_cart : (3N, 3N)
    frequencies  : (3N,)
    freq_threshold : skip modes with |ν| below this (cm⁻¹)
    delta : Å  finite-difference step

    Returns
    -------
    raman_activities : (3N,)
    dalpha_dQ        : (3N, 3, 3)
    """
    N = len(atoms)
    ndof = 3 * N
    raman_activities = np.zeros(ndof)
    dalpha_dQ = np.zeros((ndof, 3, 3))

    positions = atoms.get_positions().copy()
    ef = atoms.info.get('electric_field', [0, 0, 0])

    active = np.where(np.abs(frequencies) > freq_threshold)[0]
    print(f"    Computing Raman for {len(active)}/{ndof} modes "
          f"(|ν| > {freq_threshold} cm⁻¹) ...")

    for count, k in enumerate(active):
        if (count + 1) % 10 == 0 or count == 0:
            print(f"      mode {count+1}/{len(active)}  "
                  f"(ν = {frequencies[k]:.1f} cm⁻¹)")

        disp = eigvecs_cart[:, k].reshape(N, 3) * delta

        atoms_p = atoms.copy();  atoms_p.set_positions(positions + disp)
        atoms_p.info['electric_field'] = ef
        alpha_p = calc.get_polarizability(atoms_p)

        atoms_m = atoms.copy();  atoms_m.set_positions(positions - disp)
        atoms_m.info['electric_field'] = ef
        alpha_m = calc.get_polarizability(atoms_m)

        da = (alpha_p - alpha_m) / (2.0 * delta)
        dalpha_dQ[k] = da

        alpha_bar = np.trace(da) / 3.0
        gamma_sq  = 0.5 * (3.0 * np.sum(da * da) - np.trace(da)**2)
        raman_activities[k] = 45.0 * alpha_bar**2 + 7.0 * gamma_sq

    return raman_activities, dalpha_dQ


# =====================================================================
# VCD rotational strengths from APT + AAT
# =====================================================================
def compute_vcd(apt, aat, eigvecs_cart):
    """VCD rotational strengths  R_k = S_k · M_k.

    S_k (electric dipole TM)  from APT,
    M_k (magnetic dipole TM)  from AAT.

    Parameters
    ----------
    apt          : (3, N, 3)
    aat          : (N, 3, 3)   aat[s, α, β]
    eigvecs_cart : (3N, 3N)

    Returns
    -------
    rot_strengths : (3N,)
    S_k, M_k     : (3, 3N)
    """
    ndof = eigvecs_cart.shape[0]

    apt_flat = apt.reshape(3, ndof)
    S_k = apt_flat @ eigvecs_cart                          # (3, 3N)

    # aat[s, α, β] → (α, s*3+β) = (3, 3N)
    aat_flat = aat.transpose(1, 0, 2).reshape(3, ndof)
    M_k = aat_flat @ eigvecs_cart                          # (3, 3N)

    rot_strengths = np.sum(S_k * M_k, axis=0)             # (3N,)
    return rot_strengths, S_k, M_k


# =====================================================================
# Spectrum broadening
# =====================================================================
def lorentzian(x, x0, gamma):
    return gamma / (np.pi * ((x - x0)**2 + gamma**2))


def broaden(freq_axis, stick_freqs, stick_intensities, gamma=10.0,
            freq_min=10.0):
    """Broaden a stick spectrum with Lorentzian lineshape.

    Only includes modes with frequency > freq_min (real, positive).
    """
    spec = np.zeros_like(freq_axis)
    for f, I in zip(stick_freqs, stick_intensities):
        if f > freq_min:
            spec += I * lorentzian(freq_axis, f, gamma)
    return spec


# =====================================================================
# CLI
# =====================================================================
def get_args():
    p = argparse.ArgumentParser(
        description="Calculate IR, Raman, and VCD spectra")
    p.add_argument("--params",      default="params.json")
    p.add_argument("--config",      default=None)
    p.add_argument("--data",        default="data-full.npz")
    p.add_argument("--index",       type=int, default=0)
    p.add_argument("--field-scale", type=float, default=0.001)
    p.add_argument("--electric-field", type=float, nargs=3, default=None,
                   metavar=("EX", "EY", "EZ"),
                   help="Override electric field (in model input units, i.e. ×0.001 au). "
                        "Default: use the field from the dataset.")

    p.add_argument("--optimize", action="store_true",
                   help="Minimise geometry before computing spectra")
    p.add_argument("--fmax", type=float, default=0.001,
                   help="Force convergence for optimisation (eV/Å)")
    p.add_argument("--opt-steps", type=int, default=500,
                   help="Max optimisation steps")

    p.add_argument("--raman", action="store_true",
                   help="Compute Raman (slower — finite diff)")
    p.add_argument("--vcd",   action="store_true",
                   help="Compute VCD")
    p.add_argument("--all",   action="store_true",
                   help="Compute IR + Raman + VCD")

    p.add_argument("--hessian-method", choices=["fd", "ad"], default="fd",
                   help="Hessian method: fd=finite diff (safe), ad=jax.hessian (fast but OOM-prone)")
    p.add_argument("--hessian-delta", type=float, default=0.001,
                   help="Finite-diff step for Hessian (Å)")
    p.add_argument("--raman-delta", type=float, default=0.005,
                   help="Finite-diff step for Raman (Å)")
    p.add_argument("--freq-min",  type=float, default=0.0)
    p.add_argument("--freq-max",  type=float, default=4000.0)
    p.add_argument("--broadening", type=float, default=10.0,
                   help="Lorentzian HWHM (cm⁻¹)")
    p.add_argument("--output-dir", default="spectra")
    return p.parse_args()


# =====================================================================
# Main
# =====================================================================
def main():
    args = get_args()
    if args.all:
        args.raman = args.vcd = True
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("  Vibrational Spectroscopy from Electric-Field Model")
    print("=" * 70)

    # ---- load structure ------------------------------------------------
    dataset = np.load(args.data, allow_pickle=True)
    idx = args.index
    Z  = np.asarray(dataset["Z"][idx], dtype=int)
    R  = np.asarray(dataset["R"][idx], dtype=float)
    if R.ndim == 3 and R.shape[0] == 1:
        R = R.squeeze(0)
    Ef = np.asarray(dataset["Ef"][idx], dtype=float)

    # Override electric field if requested
    if args.electric_field is not None:
        Ef = np.asarray(args.electric_field, dtype=float)
        print(f"  Electric field overridden → {Ef}")

    atoms = ase.Atoms(numbers=Z, positions=R)
    atoms.info['electric_field'] = Ef
    N = len(atoms)
    masses = ASE_ATOMIC_MASSES[Z]

    print(f"  Structure : {N} atoms, dataset index {idx}")
    print(f"  Ef (input): {Ef}  (physical = {Ef * args.field_scale} au)")
    print(f"  Masses    : {masses}")

    # ---- calculator ----------------------------------------------------
    calc = AseCalculatorEF(
        params_path=args.params, config_path=args.config,
        field_scale=args.field_scale,
    )
    atoms.calc = calc

    # ================================================================
    # 0.  (optional) Geometry optimisation
    # ================================================================
    if args.optimize:
        traj_path = str(out / "opt.traj")
        print(f"\n[0] Geometry optimisation  (BFGS, fmax={args.fmax} eV/Å, "
              f"max {args.opt_steps} steps)")
        print(f"    Trajectory → {traj_path}")
        # logfile='-' prints to stdout (Step  Time  Energy  fmax)
        opt = BFGS(atoms, logfile='-', trajectory=traj_path)
        converged = opt.run(fmax=args.fmax, steps=args.opt_steps)
        n_steps = opt.get_number_of_steps()
        forces = atoms.get_forces()
        fmax_final = float(np.max(np.linalg.norm(forces, axis=1)))
        energy_final = float(atoms.get_potential_energy())
        if converged:
            print(f"    ✓ Converged in {n_steps} steps")
        else:
            print(f"    ✗ NOT converged after {n_steps} steps "
                  f"(fmax target {args.fmax}, got {fmax_final:.6f} eV/Å)")
            print(f"      Try increasing --opt-steps or relaxing --fmax")
        print(f"    Final energy  = {energy_final:.6f} eV")
        print(f"    Final max |F| = {fmax_final:.6f} eV/Å")
    else:
        # Check whether structure is likely a minimum
        f = atoms.get_forces()
        fmax = np.max(np.linalg.norm(f, axis=1))
        if fmax > 0.05:
            print(f"\n  ⚠  max |F| = {fmax:.4f} eV/Å  — structure is NOT "
                  f"at a minimum!")
            print(f"     Consider adding --optimize for meaningful spectra.")

    # ================================================================
    # 1.  Hessian  →  normal modes
    # ================================================================
    if args.hessian_method == "fd":
        print(f"\n[1] Hessian via finite differences (δ = {args.hessian_delta} Å, "
              f"2×{3*N} = {6*N} force evaluations) ...")
        hessian = hessian_fd(calc, atoms, delta=args.hessian_delta)
    else:
        print("\n[1] Hessian via jax.hessian (AD) ...")
        hessian = calc.get_hessian(atoms)
    print(f"    shape {hessian.shape},  max |H| = {np.max(np.abs(hessian)):.4f} eV/Å²")

    print("\n[2] Normal-mode analysis ...")
    frequencies, eigvecs_mw, eigvecs_cart = compute_normal_modes(hessian, masses)

    n_real = int(np.sum(frequencies > 1.0))
    n_imag = int(np.sum(frequencies < -1.0))
    n_zero = 3 * N - n_real - n_imag
    print(f"    {n_real} real,  {n_zero} near-zero (trans/rot),  {n_imag} imaginary")
    if n_real > 0:
        print(f"    range: {frequencies[frequencies > 1.0].min():.1f} – "
              f"{frequencies.max():.1f} cm⁻¹")
    if n_imag > 6:
        print(f"    WARNING: {n_imag} imaginary modes — structure is not at "
              f"a minimum.  Use --optimize.")

    # ================================================================
    # 2.  APT  →  IR
    # ================================================================
    print("\n[3] APT (dμ/dR)  →  IR intensities ...")
    apt = calc.get_atomic_polar_tensor(atoms)
    ir_int, dmu_dQ = compute_ir(apt, eigvecs_cart)
    print(f"    APT shape {apt.shape}")

    # ================================================================
    # 3.  Raman  (optional)
    # ================================================================
    raman_act = np.zeros(3 * N)
    dalpha_dQ = np.zeros((3 * N, 3, 3))
    if args.raman:
        print(f"\n[4] Raman (finite diff, δ = {args.raman_delta} Å) ...")
        raman_act, dalpha_dQ = compute_raman(
            calc, atoms, eigvecs_cart, frequencies,
            delta=args.raman_delta,
        )
    else:
        print("\n[4] Raman — skipped  (use --raman or --all)")

    # ================================================================
    # 4.  AAT  →  VCD  (optional)
    # ================================================================
    rot_str = np.zeros(3 * N)
    if args.vcd:
        print(f"\n[5] AAT  →  VCD ...")
        # Try ML charges first, fall back to Born charges
        try:
            aat, ml_q = calc.get_aat_ml_charges(atoms)
            print(f"    Using ML-charge AAT  (charges: min={ml_q.min():.3f}  "
                  f"max={ml_q.max():.3f}  sum={ml_q.sum():.3f})")
        except Exception:
            aat, q_eff = calc.get_aat_born(atoms)
            print(f"    Using Born-charge AAT  (q_eff: {q_eff})")
        rot_str, S_k, M_k = compute_vcd(apt, aat, eigvecs_cart)
    else:
        print("\n[5] VCD — skipped  (use --vcd or --all)")

    # ================================================================
    # Summary table
    # ================================================================
    print("\n" + "=" * 70)
    hdr = f"{'#':>4s} {'ν (cm⁻¹)':>11s} {'IR':>12s}"
    if args.raman:  hdr += f" {'Raman':>12s}"
    if args.vcd:    hdr += f" {'VCD R':>12s}"
    print(hdr)
    print("-" * len(hdr))

    for k in range(3 * N):
        if abs(frequencies[k]) < 10.0:
            continue
        row = f"{k+1:4d} {frequencies[k]:11.2f} {ir_int[k]:12.6f}"
        if args.raman:  row += f" {raman_act[k]:12.6f}"
        if args.vcd:    row += f" {rot_str[k]:12.8f}"
        print(row)

    # ================================================================
    # Save data
    # ================================================================
    npz_path = out / "spectral_data.npz"
    np.savez(
        npz_path,
        frequencies=frequencies,
        ir_intensities=ir_int,
        raman_activities=raman_act,
        vcd_rotational_strengths=rot_str,
        apt=apt, hessian=hessian,
        eigvecs_cart=eigvecs_cart, eigvecs_mw=eigvecs_mw,
        masses=masses,
    )
    print(f"\nData  →  {npz_path}")

    txt_path = out / "modes.txt"
    with open(txt_path, 'w') as f:
        f.write(f"# {'Mode':>4s} {'Freq(cm-1)':>12s} {'IR':>14s}")
        if args.raman:  f.write(f" {'Raman':>14s}")
        if args.vcd:    f.write(f" {'VCD_R':>14s}")
        f.write("\n")
        for k in range(3 * N):
            line = f"  {k+1:4d} {frequencies[k]:12.4f} {ir_int[k]:14.8f}"
            if args.raman:  line += f" {raman_act[k]:14.8f}"
            if args.vcd:    line += f" {rot_str[k]:14.8f}"
            f.write(line + "\n")
    print(f"Table →  {txt_path}")

    # ================================================================
    # Plot
    # ================================================================
    freq_ax = np.linspace(args.freq_min, args.freq_max, 4000)
    gamma   = args.broadening

    n_panels = 1 + int(args.raman) + int(args.vcd)
    fig, axes = plt.subplots(n_panels, 1, figsize=(12, 3.5 * n_panels),
                             sharex=True)
    if n_panels == 1:
        axes = [axes]

    pi = 0  # panel index

    # Annotation for quality
    n_imag = int(np.sum(frequencies < -1.0))
    n_real = int(np.sum(frequencies > 1.0))
    quality_note = (f"{n_real} real modes, {n_imag} imaginary"
                    if n_imag > 6 else f"{n_real} real modes")

    # --- IR ---
    ir_spec = broaden(freq_ax, frequencies, ir_int, gamma)
    ax = axes[pi]
    ax.plot(freq_ax, ir_spec, 'b-', lw=1.5)
    ax.fill_between(freq_ax, ir_spec, alpha=0.25)
    # Stick markers for real modes
    real_mask = frequencies > 10.0
    ax.stem(frequencies[real_mask], ir_int[real_mask],
            linefmt='r-', markerfmt='', basefmt='', label='stick')
    ax.set_ylabel('IR intensity (arb. u.)')
    ax.set_title(f'Infrared Spectrum  ({quality_note})')
    ax.invert_xaxis()
    pi += 1

    # --- Raman ---
    if args.raman:
        ram_spec = broaden(freq_ax, frequencies, raman_act, gamma)
        ax = axes[pi]
        ax.plot(freq_ax, ram_spec, 'g-', lw=1.5)
        ax.fill_between(freq_ax, ram_spec, alpha=0.25, color='green')
        ax.stem(frequencies[real_mask], raman_act[real_mask],
                linefmt='r-', markerfmt='', basefmt='')
        ax.set_ylabel('Raman activity (arb. u.)')
        ax.set_title('Raman Spectrum')
        ax.invert_xaxis()
        pi += 1

    # --- VCD ---
    if args.vcd:
        vcd_spec = broaden(freq_ax, frequencies, rot_str, gamma)
        ax = axes[pi]
        ax.plot(freq_ax, vcd_spec, 'k-', lw=1.5)
        ax.fill_between(freq_ax, 0, vcd_spec,
                        where=(vcd_spec >= 0), color='red',   alpha=0.3)
        ax.fill_between(freq_ax, 0, vcd_spec,
                        where=(vcd_spec < 0),  color='blue',  alpha=0.3)
        ax.axhline(0, color='grey', lw=0.5)
        ax.stem(frequencies[real_mask], rot_str[real_mask],
                linefmt='grey', markerfmt='', basefmt='')
        ax.set_ylabel('VCD rot. strength (arb. u.)')
        ax.set_title('VCD Spectrum')
        ax.invert_xaxis()
        pi += 1

    axes[-1].set_xlabel('Frequency (cm⁻¹)')
    plt.tight_layout()
    fig_path = out / "spectra.png"
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    print(f"Plot  →  {fig_path}")

    print(f"\n{'=' * 70}")
    print("  Done!")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
