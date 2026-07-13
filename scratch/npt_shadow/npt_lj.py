"""Short NPT trajectory of a monatomic Lennard-Jones (argon-like) fluid on the
MM forcefield (mm_nonbonded), illustrating box-volume relaxation.

Each atom is its own molecule, so the intermolecular-only mm_nonbonded term
governs the whole system (no missing intramolecular bonds). Started slightly
compressed at ~liquid density so the barostat visibly relaxes the volume.
"""
import os, warnings, numpy as np
os.environ["JAX_ENABLE_X64"] = "1"
warnings.simplefilter("ignore")
from pathlib import Path

from mmml.md import EnsembleSpec, RunConfig, SystemSpec, assemble_and_run
from mmml.md.energy import EnergyContext
from mmml.md.system import FFParams, MolecularSystem

SP = Path(__file__).parent

# --- argon-like LJ fluid on a jittered simple-cubic lattice -----------------
n_side = 5                      # 125 atoms
L0 = 17.5                       # Å; spacing 3.5 Å < pair Rmin (3.82) -> mildly
box_len = L0                    # compressed, so the box expands to equilibrium
EPS = 0.238                     # kcal/mol  (argon)
RMIN_HALF = 1.9075              # Å         (sigma = 3.40 Å)
MASS = 39.95                    # amu       (argon)
rng = np.random.default_rng(0)

spacing = box_len / n_side
grid = (np.arange(n_side) + 0.5) * spacing
R = np.array([[x, y, z] for x in grid for y in grid for z in grid])
R += rng.normal(scale=0.15, size=R.shape)       # small jitter to break symmetry
N = len(R)
ff = FFParams(
    charges=np.zeros(N), epsilon=np.full(N, EPS), rmin_half=np.full(N, RMIN_HALF),
    at_codes=np.zeros(N, dtype=np.int32),
    exclusions=np.empty((0, 2), dtype=np.int32), e14_pairs=np.empty((0, 2), dtype=np.int32),
)
system = MolecularSystem(
    R=R, Z=np.full(N, 18), box=np.diag([box_len] * 3),
    mol_id=np.arange(N, dtype=np.int32),                 # each atom its own molecule
    monomer_indices=[np.array([i]) for i in range(N)],
    ff_params=ff,
)

# cutoffs < L/2 (=8.75) so the periodic pair build is valid as the box breathes
ctx = EnergyContext(options={"cutnb": 8.0, "ctonnb": 7.0, "ctofnb": 8.0})

N_STEPS = 30000        # 60 ps at 2 fs — long enough for volume/pressure to plateau
REC = 60               # record every 60 steps -> 500 frames
cfg = RunConfig(
    system=SystemSpec(builder="psf"), terms=("mm_nonbonded",),
    ensemble=EnsembleSpec(
        ensemble="npt", temperature_K=120.0, pressure_bar=1.0, dt_fs=2.0, n_steps=N_STEPS,
        params={"seed": 0, "float64": True, "masses": np.full(N, MASS)},
    ),
    backend="jaxmd", output_dir=SP,
)

from mmml.md.drivers import JaxmdDriver
from mmml.md.assemble import build_hybrid_energy, _auto_neighbor_fn

energy = build_hybrid_energy(system, cfg.terms, ctx)
nfn = _auto_neighbor_fn(system, energy, cfg)
driver = JaxmdDriver(neighbor_fn=nfn, record_every=REC, output_path=SP / "npt_lj.npz")
print(f"running NPT: {N} atoms, L0={box_len} Å, {N_STEPS} steps @ 2 fs "
      f"= {N_STEPS*2/1000:.0f} ps ...", flush=True)
traj = driver.run(system, energy, cfg.ensemble)

boxes = np.asarray(traj.metadata["boxes"])          # (n_frames, 3, 3)
energies = np.asarray(traj.metadata["energies"])    # eV
positions = np.asarray(traj.metadata["positions"])  # (n_frames, N, 3)
vol = np.array([abs(np.linalg.det(b)) for b in boxes])
Lt = vol ** (1 / 3)
t_ps = np.arange(len(vol)) * REC * 2.0 / 1000.0     # record_every*dt_fs -> ps
# number density and mass density (argon)
num_dens = N / vol                                  # atoms / Å^3
g_per_cm3 = (N * MASS / 6.02214076e23) / (vol * 1e-24)
np.savez(SP / "npt_lj_series.npz", t_ps=t_ps, vol=vol, Lt=Lt, energies=energies,
         num_dens=num_dens, g_per_cm3=g_per_cm3, positions=positions, boxes=boxes, N=N,
         eps=EPS, rmin_half=RMIN_HALF, mass=MASS, temperature_K=120.0,
         ctofnb=8.0, dt_fs=2.0, rec=REC)
print(f"frames={len(vol)}  V: {vol[0]:.0f} -> {vol[-1]:.0f} Å^3   "
      f"rho: {g_per_cm3[0]:.3f} -> {g_per_cm3[-1]:.3f} g/cm^3   "
      f"E: {energies[0]:.3f} -> {energies[-1]:.3f} eV")
print("finite:", np.all(np.isfinite(vol)) and np.all(np.isfinite(energies)))
