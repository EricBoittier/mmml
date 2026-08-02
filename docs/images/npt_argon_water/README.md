# Argon MM-only NpT pressure

Pure Lennard-Jones (`AR1:500`, literature noble-gas parameters, **unit** LJ
scales, **no ML**). Certified box at 90 K / saturation pressure
(1.3176 atm ≈ 1.335 bar).

![pressure](ar1_90k_mmonly_pressure.png)

| | value |
|---|---:|
| run | `artifacts/npt_argon_water/runs/ar1_90k_mmonly_unit/` |
| length | 20 ps, dt = 1 fs, jaxmd-unified |
| ⟨P⟩ (all) | −0.48 bar (σ = 118 bar) |
| ⟨P⟩ (last 70%) | **1.00 bar** vs target **1.335 bar** |
| ⟨P_kin⟩ / ⟨P_vir⟩ (prod) | ~317 / ~−316 bar |
| ρ (prod) | 1.685 ± 0.002 g/cm³ vs NIST 1.379 (**not equilibrated**, +22%) |

**Pressure takeaway:** virial is live. Kinetic-only pressure would sit near
~300 bar; here P_kin and P_vir cancel and production ⟨P⟩ ≈ 1.0 bar tracks the
1.3 bar target within noise.

Density is still drifting on this 20 ps smoke — do not quote it. Longer
EQ/PROD (campaign script) is needed for a density claim.
