"""Analyze the long NPT LJ trajectory: volume, density, energy, virial pressure
(equilibration), and the radial distribution function g(r)."""
import warnings, numpy as np
warnings.simplefilter("ignore")
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from mmml.utils.plotting.styles import apply_plot_style, default_cmap

SP = Path(__file__).parent
apply_plot_style("icml")
d = np.load(SP / "npt_lj_series.npz")
t, vol, Lt, E, rho = d["t_ps"], d["vol"], d["Lt"], d["energies"], d["g_per_cm3"]
pos, boxes, N = d["positions"], d["boxes"], int(d["N"])
EPS, RMIN_HALF, T = float(d["eps"]), float(d["rmin_half"]), float(d["temperature_K"])
CUT = float(d["ctofnb"])
RMIN = 2 * RMIN_HALF                 # pair Rmin (Å); eps_pair = EPS
KB = 1.380649e-23                    # J/K
KCAL = 4184.0 / 6.02214076e23       # J per kcal/mol-particle
BAR = 1e5                           # Pa

seq = default_cmap("sequential")


def _mic_pairs(R, L):
    """All i<j MIC displacement magnitudes within CUT for a cubic box side L."""
    diff = R[:, None, :] - R[None, :, :]
    diff -= L * np.round(diff / L)
    r = np.sqrt((diff ** 2).sum(-1))
    iu = np.triu_indices(len(R), k=1)
    rr = r[iu]
    return rr[rr < CUT]


def virial_pressure(R, L):
    """Configurational LJ pressure (bar) at set-point T: P = (N kB T + W/3)/V."""
    r = _mic_pairs(R, L)
    x6 = (RMIN / r) ** 6
    x12 = x6 * x6
    W_kcal = np.sum(EPS * (12 * x12 - 12 * x6))          # sum r*f (kcal/mol)
    V_m3 = (L * 1e-10) ** 3
    W_J = W_kcal * KCAL
    P = (N * KB * T + W_J / 3.0) / V_m3
    return P / BAR


press = np.array([virial_pressure(pos[i], Lt[i]) for i in range(len(pos))])

# g(r) from the equilibrated second half
half = len(pos) // 2
nbins = 120
rmax = min(CUT, Lt[half:].min() / 2)
edges = np.linspace(0.0, rmax, nbins + 1)
centers = 0.5 * (edges[:-1] + edges[1:])
hist = np.zeros(nbins)
for i in range(half, len(pos)):
    rr = _mic_pairs(pos[i], Lt[i])
    hist += np.histogram(rr[rr < rmax], bins=edges)[0]
n_used = len(pos) - half
shell_vol = 4 / 3 * np.pi * (edges[1:] ** 3 - edges[:-1] ** 3)
dens = N / (Lt[half:].mean() ** 3)
ideal = shell_vol * dens * N / 2 * n_used     # ideal pair count per shell
gr = hist / np.maximum(ideal, 1e-12)

half_mean = lambda x: float(np.mean(x[half:]))
half_std = lambda x: float(np.std(x[half:]))
print(f"equilibrated (last half): V={half_mean(vol):.0f}±{half_std(vol):.0f} Å³  "
      f"rho={half_mean(rho):.3f} g/cm³  P={half_mean(press):.0f}±{half_std(press):.0f} bar  "
      f"E={half_mean(E):.3f} eV")
np.savez(SP / "npt_analysis.npz", t=t, vol=vol, rho=rho, E=E, press=press,
         gr_r=centers, gr=gr, half=half)

# --- figure -----------------------------------------------------------------
fig = plt.figure(figsize=(13, 8.5))
gs = fig.add_gridspec(2, 3)


def _panel(ax, y, ylabel, title, c, target=None):
    ax.plot(t, y, color=c, lw=1.2)
    ax.axvspan(t[half], t[-1], color="#8883", lw=0, label="equilibrated window")
    m = half_mean_arr = np.mean(y[half:])
    ax.axhline(m, ls="--", c="#333", lw=1, label=f"mean {m:.3g}")
    if target is not None:
        ax.axhline(target, ls=":", c="#c0392b", lw=1.2, label=f"target {target:g}")
    ax.set(xlabel="time (ps)", ylabel=ylabel, title=title)
    ax.legend(frameon=False, fontsize=8, loc="best")


_panel(fig.add_subplot(gs[0, 0]), vol, "volume V (Å³)", "Box volume", seq(0.5))
_panel(fig.add_subplot(gs[0, 1]), rho, "density (g/cm³)", "Mass density", seq(0.7))
_panel(fig.add_subplot(gs[0, 2]), press, "pressure (bar)", "Virial pressure", seq(0.35), target=1.0)
_panel(fig.add_subplot(gs[1, 0]), E, "potential energy (eV)", "MM (LJ) potential energy", seq(0.55))

# g(r)
axg = fig.add_subplot(gs[1, 1:])
axg.plot(centers, gr, color=seq(0.4), lw=1.8)
axg.axhline(1.0, ls="--", c="#888", lw=1)
axg.axvline(RMIN, ls=":", c="#c0392b", lw=1.2, label=f"pair R$_{{min}}$={RMIN:.2f} Å")
axg.set(xlabel="r (Å)", ylabel="g(r)",
        title="Radial distribution function (equilibrated half)")
axg.legend(frameon=False, fontsize=9)

fig.suptitle(f"NPT Lennard-Jones fluid (jax-md): {N} argon atoms, T={T:.0f} K, "
             f"P=1 bar, {t[-1]:.0f} ps", y=1.0, fontsize=14)
fig.tight_layout()
fig.savefig(SP / "npt_analysis.png", dpi=200, bbox_inches="tight")
print("wrote npt_analysis.png")
