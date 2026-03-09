#!/usr/bin/env python3
"""
Calculate simulation requirements for IR spectroscopy.

Frequency resolution in FFT-based IR spectra:
  Δν (cm⁻¹) = 1 / (T_total × c)
  
where:
  T_total = n_steps × timestep (in seconds)
  c = 2.99792458e10 cm/s (speed of light)

For good IR spectra:
  - Resolution: 1-5 cm⁻¹ is excellent, 5-10 cm⁻¹ is good
  - Maximum frequency: ν_max = 1 / (2 × timestep × c)
"""

import numpy as np

# Constants
c_cm_per_s = 2.99792458e10  # cm/s
target_max_freq = 4500  # cm⁻¹

print("="*70)
print("IR SPECTROSCOPY SIMULATION REQUIREMENTS")
print("="*70)

# Common timesteps
timesteps_fs = [0.1, 0.5, 1.0]

print(f"\nTarget frequency range: 0 - {target_max_freq} cm⁻¹")
print(f"\n{'Timestep (fs)':<15} {'Max freq (cm⁻¹)':<20} {'Resolution':<15} {'Time needed':<20} {'Steps needed':<15}")
print("-"*85)

for dt_fs in timesteps_fs:
    dt_s = dt_fs * 1e-15
    
    # Maximum frequency (Nyquist)
    nyquist_freq_hz = 1.0 / (2 * dt_s)
    max_freq_cm = nyquist_freq_hz / c_cm_per_s
    
    # Check if timestep is sufficient
    if max_freq_cm < target_max_freq:
        status = "⚠️  TOO LARGE"
        resolution_str = "N/A"
        time_needed_str = "N/A"
        steps_needed_str = "N/A"
    else:
        # Calculate time needed for different resolutions
        resolutions = [1.0, 2.0, 5.0, 10.0]  # cm⁻¹
        best_res = resolutions[0]
        
        # Time for 1 cm⁻¹ resolution
        T_total_s = 1.0 / (best_res * c_cm_per_s)
        T_total_fs = T_total_s / 1e-15
        n_steps = int(T_total_fs / dt_fs)
        
        resolution_str = f"{best_res:.1f} cm⁻¹"
        time_needed_str = f"{T_total_fs/1000:.1f} ps"
        steps_needed_str = f"{n_steps:,}"
    
    print(f"{dt_fs:<15.1f} {max_freq_cm:<20.0f} {resolution_str:<15} {time_needed_str:<20} {steps_needed_str:<15}")

print("\n" + "="*70)
print("RECOMMENDATIONS")
print("="*70)

print("\nFor IR spectrum 0-4500 cm⁻¹:")
print("\n1. TIMESTEP:")
print("   ✅ Use 0.5 fs (recommended)")
print("   ✅ Use 0.1 fs (more accurate, slower)")
print("   ❌ Avoid 1.0 fs (may alias high frequencies)")

print("\n2. SIMULATION LENGTH:")
print("   • Excellent resolution (1 cm⁻¹):")
print("     - 0.5 fs timestep: ~33 ps = ~66,000 steps")
print("     - 0.1 fs timestep: ~33 ps = ~330,000 steps")
print("")
print("   • Good resolution (5 cm⁻¹):")
print("     - 0.5 fs timestep: ~6.7 ps = ~13,400 steps")
print("     - 0.1 fs timestep: ~6.7 ps = ~67,000 steps")
print("")
print("   • Moderate resolution (10 cm⁻¹):")
print("     - 0.5 fs timestep: ~3.3 ps = ~6,700 steps")
print("     - 0.1 fs timestep: ~3.3 ps = ~33,000 steps")

print("\n3. PRACTICAL RECOMMENDATION:")
print("   For production IR spectra:")
print("   • Timestep: 0.5 fs")
print("   • Duration: 20-50 ps")
print("   • Steps: 40,000 - 100,000")
print("   • Resolution: ~1-2 cm⁻¹ (excellent!)")
print("")
print("   For quick tests:")
print("   • Timestep: 0.5 fs")
print("   • Duration: 5-10 ps")
print("   • Steps: 10,000 - 20,000")
print("   • Resolution: ~3-7 cm⁻¹ (good)")

print("\n4. SAVING FRAMES:")
print("   Save every frame (save_interval=1) for best frequency resolution.")
print("   If storage is limited, save_interval=2-5 is acceptable.")

print("\n" + "="*70)
print("EXAMPLE COMMAND")
print("="*70)
print("""
# For excellent IR resolution (1 cm⁻¹):
python jaxmd_dynamics.py \\
    --checkpoint /path/to/checkpoint \\
    --molecule CO2 \\
    --multi-replicas 16 \\
    --temperature 300 \\
    --timestep 0.5 \\
    --nsteps 66000 \\
    --output-dir ./ir_sim

# Then compute dipoles and IR:
python compute_dipoles_for_traj.py \\
    --positions ./ir_sim/multi_copy_traj_16x.npz \\
    --metadata ./ir_sim/multi_copy_metadata.npz \\
    --checkpoint /path/to/checkpoint \\
    --output ./ir_sim/dipoles.npz

python compute_ir_raman.py \\
    --positions ./ir_sim/multi_copy_traj_16x.npz \\
    --metadata ./ir_sim/multi_copy_metadata.npz \\
    --dipoles ./ir_sim/dipoles.npz \\
    --checkpoint /path/to/checkpoint \\
    --output ./ir_sim/ir_spectrum.npz
""")

print("="*70)

