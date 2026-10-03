"""pycharmm.correl -- trajectory analysis examples

Demonstrates all four phases:
  Phase 1: Geometry extraction (distance, angle, dihedral, phi/psi)
  Phase 2: MANTIM transforms (smoothing, derivatives, histograms)
  Phase 3: Correlation and spectral analysis (ACF, CCF, MSD, spectrum)
  Phase 4: Energy decomposition and live dynamics collection

Prerequisites: a CHARMM system must be set up (PSF loaded) before
creating a Trajectory from a DCD file. The from_coordinates() path
works without CHARMM for testing.
"""

# ===========================================================================
# Examples requiring a loaded CHARMM system + DCD trajectory
# ===========================================================================

# Example 1: Ramachandran plot
# ----------------------------
# from pycharmm.correl import Trajectory
# traj = Trajectory("dynamics.dcd")             # direct Fortran reader
# phi, psi = traj.ramachandran(resid=5)
# import matplotlib.pyplot as plt
# plt.scatter(phi.values, psi.values, s=1, alpha=0.3)
# plt.xlabel("phi (degrees)"); plt.ylabel("psi (degrees)")
# plt.savefig("ramachandran.png")

# Example 2: Dihedral transition analysis pipeline
# -------------------------------------------------
# phi = traj.phi(5)
# phi_cont = phi.cont()                         # unwrap periodic jumps
# phi_smooth = phi_cont.movi(20)                # 20-frame running average
# omega = phi_smooth.deri(dt=0.002)             # angular velocity (deg/ps)
# print(f"Max angular velocity: {omega.abs().max():.1f} deg/ps")
# transition = omega.abs().stat(50, 1e6)        # 1 when |omega| > 50 deg/ps
# print(f"Fraction in transition: {transition.mean():.3f}")

# Example 3: Distance monitoring with statistics
# -----------------------------------------------
# d = traj.distance(42, 187)
# d_smooth = d.movi(50)
# print(f"Distance: {d.mean():.2f} +/- {d.std():.2f} Ang")
# print(f"Min: {d.min():.2f}, Max: {d.max():.2f}")
# probs = d.hist(bins=50, range=(2, 10))

# Example 4: Order parameter from bond vector reorientation
# ----------------------------------------------------------
# # P2 = <3*cos^2(theta) - 1>/2 for each residue
# for resid in range(2, 20):
#     phi = traj.phi(resid)
#     p2_inst = phi.cos2()  # 3*cos^2 - 1 at each frame
#     print(f"Res {resid}: <P2> = {p2_inst.mean()/2:.3f}")

# Example 5: Energy decomposition from trajectory (Phase 4)
# ----------------------------------------------------------
# traj = Trajectory("dynamics.dcd")
# energies = traj.energy_series(terms=["ENER", "BOND", "VDW", "ELEC"])
# print(f"Mean total energy: {energies['ENER'].mean():.1f} kcal/mol")
# # Correlate VDW and ELEC fluctuations
# ccf_ve = energies["VDW"].ccf(energies["ELEC"], maxlag=200)

# Example 6: Live dynamics collection (Phase 4)
# -----------------------------------------------
# from pycharmm.correl import Collector
# monitors = [
#     {"type": "phi", "resid": 5},
#     {"type": "psi", "resid": 5},
#     {"type": "energy", "term": "ENER"},
#     {"type": "distance", "i": 42, "j": 187},
# ]
# collector = Collector(nsteps=100000, nsavc=100, monitors=monitors)
# results = collector.run()
# phi = results["phi(5)"]
# ener = results["ENER"]
# # Full analysis pipeline on live-collected data
# phi_acf = phi.cont().dave().acf(maxlag=200)
# freqs, spectrum = ener.dave().spectrum()


# ===========================================================================
# Runnable example: synthetic trajectory (no CHARMM needed)
# ===========================================================================

if __name__ == "__main__":
    import numpy as np
    from pycharmm.correl import Series, Trajectory

    print("=" * 60)
    print("pycharmm.correl Phase 2 Demo: MANTIM Transforms")
    print("=" * 60)

    # --- Synthetic dihedral trajectory: transition between two states ---
    np.random.seed(42)
    nframes = 2000
    dt = 0.002  # ps

    # Generate dihedral that jumps between -60 and 60 degrees
    state = np.zeros(nframes)
    state[0] = -60.0
    for i in range(1, nframes):
        # Rare transitions (1% chance per step)
        if np.random.random() < 0.01:
            state[i] = 60.0 if state[i-1] < 0 else -60.0
        else:
            state[i] = state[i-1]
    # Add Gaussian noise
    dihedral_raw = state + np.random.randn(nframes) * 8.0
    phi = Series(dihedral_raw, name="phi(5)", units="degrees", dt=dt)

    print(f"\n1. Raw dihedral: {phi}")
    print(f"   Range: [{phi.min():.1f}, {phi.max():.1f}] degrees")

    # --- Transform pipeline ---
    print("\n2. Transform pipeline:")

    # Detrend
    phi_centered = phi.dave()
    print(f"   dave (detrend):    mean = {phi_centered.mean():.6f}")

    # Smooth with moving average
    phi_smooth = phi.movi(50)
    print(f"   movi(50) smooth:   std reduced {phi.std():.1f} -> "
          f"{phi_smooth.std():.1f} deg")

    # Numerical derivative = angular velocity
    omega = phi_smooth.deri()
    print(f"   deri (velocity):   max |omega| = "
          f"{omega.abs().max():.1f} deg/ps")

    # Detect transitions via thresholding angular velocity
    transitions = omega.abs().stat(5.0, 1e6)
    n_trans = int(transitions.values.sum())
    print(f"   stat (threshold):  {n_trans} frames with |omega| > 5 deg/ps")

    # Block average to reduce data
    phi_blocked = phi.aver(100)
    print(f"   aver(100):         {len(phi)} -> {len(phi_blocked)} points, "
          f"dt: {phi.dt} -> {phi_blocked.dt} ps")

    # --- Trig transforms ---
    print("\n3. Trigonometric transforms:")
    cos_phi = phi.cos()
    print(f"   cos(phi):  <cos(phi)> = {cos_phi.mean():.4f}")
    p2 = phi.cos2()
    print(f"   cos2(phi): <P2>       = {p2.mean()/2:.4f}")

    # --- Histogram analysis ---
    print("\n4. Histogram analysis:")
    probs = phi.hist(bins=36, range=(-180, 180))
    peak_bin = np.argmax(probs.values)
    bin_center = -180 + (peak_bin + 0.5) * 10
    print(f"   Most populated 10-deg bin centered at {bin_center:.0f} deg "
          f"(p = {probs.values[peak_bin]:.3f})")

    # --- Method chaining demo ---
    print("\n5. Method chaining:")
    result = phi.dave().movi(20).square().sqrt()
    print(f"   phi.dave().movi(20).square().sqrt() = "
          f"abs(smoothed_fluctuation)")
    print(f"   Mean magnitude: {result.mean():.2f} deg")

    # --- Calculus round-trip ---
    print("\n6. Calculus:")
    t = np.linspace(0, 4 * np.pi, 500)
    signal = Series(np.sin(t), name="sin(t)", dt=t[1]-t[0])
    deriv = signal.deri()
    print(f"   d/dt sin(t) at t=0: {deriv[0]:.4f} "
          f"(expected: {np.cos(t[0]):.4f})")
    integral = signal.inte()
    print(f"   integral sin(t) at t=4pi: {integral[-1]:.4f} "
          f"(expected: ~0)")

    # --- Geometry with synthetic trajectory ---
    print("\n7. Synthetic trajectory geometry:")
    natom = 4
    x = np.zeros((nframes, natom))
    y = np.zeros((nframes, natom))
    z = np.zeros((nframes, natom))
    for f in range(nframes):
        angle = np.radians(dihedral_raw[f])
        x[f, 0] = 0; y[f, 0] = 1; z[f, 0] = 0
        x[f, 1] = 0; y[f, 1] = 0; z[f, 1] = 0
        x[f, 2] = 1; y[f, 2] = 0; z[f, 2] = 0
        x[f, 3] = 1; y[f, 3] = np.cos(angle); z[f, 3] = np.sin(angle)

    traj = Trajectory.from_coordinates(x, y, z)
    d_series = traj.dihedral(1, 2, 3, 4)
    print(f"   Dihedral series: {d_series}")
    print(f"   Matches input: max error = "
          f"{np.max(np.abs(d_series.values - dihedral_raw)):.2e} deg")

    # ==================================================================
    # Phase 3: Correlation and spectral analysis
    # ==================================================================

    print("\n" + "=" * 60)
    print("Phase 3 Demo: Correlation & Spectral Analysis")
    print("=" * 60)

    # --- Damped harmonic oscillator (Langevin dynamics) ---
    # Simulates velocity of a particle in a harmonic well with friction.
    # v(t+dt) = v(t) - omega^2 * x(t) * dt - gamma * v(t) * dt + noise
    print("\n8. Langevin harmonic oscillator:")
    np.random.seed(7)
    n_steps = 20000
    dt_sim = 0.005  # ps
    omega = 20.0    # angular frequency (1/ps) -> ~3.2 ps period
    gamma = 2.0     # friction coefficient (1/ps)
    kT = 1.0        # thermal energy (arbitrary units)

    x_osc = np.zeros(n_steps)
    v_osc = np.zeros(n_steps)
    v_osc[0] = np.random.randn() * np.sqrt(kT)

    for i in range(1, n_steps):
        noise = np.sqrt(2 * gamma * kT * dt_sim) * np.random.randn()
        v_osc[i] = v_osc[i-1] - omega**2 * x_osc[i-1] * dt_sim \
                    - gamma * v_osc[i-1] * dt_sim + noise
        x_osc[i] = x_osc[i-1] + v_osc[i] * dt_sim

    vel = Series(v_osc, name="velocity", units="A/ps", dt=dt_sim)
    pos = Series(x_osc, name="position", units="A", dt=dt_sim)
    print(f"   Simulated {n_steps} steps, dt={dt_sim} ps")
    print(f"   Position: mean={pos.mean():.3f}, std={pos.std():.3f} A")
    print(f"   Velocity: mean={vel.mean():.3f}, std={vel.std():.3f} A/ps")

    # --- Velocity autocorrelation function (VACF) ---
    print("\n9. Velocity autocorrelation:")
    vacf = vel.acf(maxlag=1000)
    print(f"   VACF(0) = {vacf[0]:.4f} (should be 1.0)")
    # Find first zero crossing -> half-period
    first_zero = 0
    for i in range(1, len(vacf)):
        if vacf[i] <= 0:
            first_zero = i
            break
    if first_zero > 0:
        half_period = first_zero * dt_sim
        freq_from_acf = 1.0 / (2 * half_period)
        print(f"   First zero crossing at lag {first_zero} "
              f"({half_period:.3f} ps)")
        print(f"   Estimated frequency: {freq_from_acf:.1f} /ps "
              f"(expected: ~{omega/(2*np.pi):.1f} /ps)")

    # --- Power spectrum ---
    print("\n10. Power spectrum:")
    freqs, power = vel.spectrum(window="cosine", pad_factor=8)
    peak_idx = np.argmax(power.values[1:]) + 1  # skip DC
    peak_freq = freqs[peak_idx]
    expected_freq = omega / (2 * np.pi)
    print(f"   Peak frequency: {peak_freq:.2f} /ps "
          f"(expected: {expected_freq:.2f} /ps)")
    print(f"   Frequency resolution: {freqs[1]:.4f} /ps")

    # --- Cross-correlation ---
    print("\n11. Cross-correlation (position vs velocity):")
    ccf_xv = pos.ccf(vel, maxlag=500)
    # For harmonic oscillator: x and v are 90 degrees out of phase
    peak_lag = np.argmax(np.abs(ccf_xv.values))
    print(f"   Peak |CCF| at lag {peak_lag} "
          f"({peak_lag * dt_sim:.3f} ps)")

    # --- Mean squared displacement ---
    print("\n12. Mean squared displacement:")
    msd_pos = pos.msd(maxlag=500)
    print(f"   MSD(0) = {msd_pos[0]:.2e} (should be ~0)")
    print(f"   MSD(100) = {msd_pos[100]:.4f} A^2")
    # For confined motion (harmonic well), MSD plateaus at 2*<x^2>
    plateau = 2 * np.mean(x_osc**2)
    print(f"   Expected plateau: {plateau:.4f} A^2")

    # --- Full pipeline: dihedral ACF ---
    print("\n13. Dihedral autocorrelation pipeline:")
    phi_acf = phi.dave().acf(maxlag=200)
    # Estimate decorrelation time (lag where ACF drops below 1/e)
    decorr = 0
    for i in range(len(phi_acf)):
        if phi_acf[i] < 1.0 / np.e:
            decorr = i
            break
    print(f"   phi ACF decorrelation time: ~{decorr} frames "
          f"({decorr * dt:.1f} ps)")

    print("\n" + "=" * 60)
    print("All examples completed successfully.")
    print("=" * 60)
