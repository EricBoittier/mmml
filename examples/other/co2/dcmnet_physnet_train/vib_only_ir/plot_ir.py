#!/usr/bin/env python3
"""Quick IR plotting script"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

# Load data
data = np.load('md_ir_spectrum.npz')
print(f"Data keys: {list(data.keys())}")

freqs = data['frequencies']
int_phys = data['intensity_physnet']
int_dcm = data['intensity_dcmnet']

# Create figure
fig, axes = plt.subplots(2, 1, figsize=(14, 10))

# Plot 1: Full range
ax = axes[0]
mask = (freqs > 0) & (freqs < 4000)
ax.plot(freqs[mask], int_phys[mask], 'b-', lw=1.5, label='PhysNet', alpha=0.8)
ax.plot(freqs[mask], int_dcm[mask], 'c--', lw=1.5, label='DCMNet', alpha=0.8)

# Experimental CO2
exp = [('ν2 bend', 667.4), ('ν1 sym', 1388.2), ('ν3 asym', 2349.2)]
for name, f in exp:
    ax.axvline(f, color='red', ls=':', lw=2, alpha=0.6)
    ax.text(f, ax.get_ylim()[1]*0.9, name, rotation=90, va='bottom', fontsize=9, color='red')

ax.set_xlabel('Frequency (cm⁻¹)', fontsize=12, weight='bold')
ax.set_ylabel('Intensity', fontsize=12, weight='bold')
ax.set_title('MD IR Spectrum (rotations removed)', fontsize=14, weight='bold')
ax.legend()
ax.grid(True, alpha=0.3)
ax.set_xlim(0, 4000)

# Plot 2: Zoomed
ax = axes[1]
mask2 = (freqs > 500) & (freqs < 2800)
ax.plot(freqs[mask2], int_phys[mask2], 'b-', lw=2, label='PhysNet')
ax.plot(freqs[mask2], int_dcm[mask2], 'c--', lw=2, label='DCMNet')

for name, f in exp:
    ax.axvline(f, color='red', ls=':', lw=2, alpha=0.6)

ax.set_xlabel('Frequency (cm⁻¹)', fontsize=12, weight='bold')
ax.set_ylabel('Intensity', fontsize=12, weight='bold')
ax.set_title('CO2 Range (500-2800 cm⁻¹)', fontsize=14, weight='bold')
ax.legend()
ax.grid(True, alpha=0.3)
ax.set_xlim(500, 2800)

plt.tight_layout()
plt.savefig('ir_comparison.png', dpi=300, bbox_inches='tight')
print(f"\n✅ Saved: ir_comparison.png")

# Find and report peaks with very sensitive threshold
mask_p = (freqs > 400) & (freqs < 3000)
peaks_p, props_p = find_peaks(int_phys[mask_p], height=1e-10, prominence=1e-11, distance=10)
peaks_d, props_d = find_peaks(int_dcm[mask_p], height=1e-10, prominence=1e-11, distance=10)

print(f"\nPhysNet peaks (showing top 20 by intensity):")
# Sort peaks by intensity
if len(peaks_p) > 0:
    peak_intensities = props_p['peak_heights']
    sort_idx = np.argsort(peak_intensities)[::-1][:20]  # Top 20
    for i in sort_idx:
        idx = peaks_p[i]
        print(f"  {freqs[mask_p][idx]:7.1f} cm⁻¹ (intensity: {peak_intensities[i]:.6e})")
else:
    top = np.argsort(int_phys[mask_p])[-20:][::-1]
    print(f"  No peaks found. Top 20 maxima:")
    for idx in top:
        print(f"    {freqs[mask_p][idx]:7.1f} cm⁻¹ (intensity: {int_phys[mask_p][idx]:.6e})")

print(f"\nDCMNet peaks (showing top 20 by intensity):")
if len(peaks_d) > 0:
    peak_intensities = props_d['peak_heights']
    sort_idx = np.argsort(peak_intensities)[::-1][:20]
    for i in sort_idx:
        idx = peaks_d[i]
        print(f"  {freqs[mask_p][idx]:7.1f} cm⁻¹ (intensity: {peak_intensities[i]:.6e})")
else:
    top = np.argsort(int_dcm[mask_p])[-20:][::-1]
    print(f"  No peaks found. Top 20 maxima:")
    for idx in top:
        print(f"    {freqs[mask_p][idx]:7.1f} cm⁻¹ (intensity: {int_dcm[mask_p][idx]:.6e})")

# Summary statistics
print(f"\nIntensity statistics (CO2 range 400-3000 cm⁻¹):")
print(f"  PhysNet: max={int_phys[mask_p].max():.6e}, mean={int_phys[mask_p].mean():.6e}")
print(f"  DCMNet:  max={int_dcm[mask_p].max():.6e}, mean={int_dcm[mask_p].mean():.6e}")
print(f"  Overall max: {max(int_phys.max(), int_dcm.max()):.6e}")

print(f"\nExperimental CO2 for reference:")
for name, f in exp:
    print(f"  {name:15s}: {f:7.1f} cm⁻¹")

plt.show()

