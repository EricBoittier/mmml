"""Static pair list vs rebuilt neighbour list: correctness and speed.

Decides the default of ``UmbrellaConfig.static_pairs``. Both feed the same
``mm_nonbonded`` kernel; they differ only in which pairs reach it:

* static  -- every intermolecular pair, uploaded once, never rebuilt
* nbrlist -- pairs within ``cutoff + skin``, rebuilt on the host

The claim under test is that the switching function makes distant pairs
contribute exactly zero, so the complete list and the cutoff list give the same
energy. Correctness therefore has three axes, not one: agreement at a fixed
configuration, sensitivity to the build cutoff, and staleness once atoms move.

CHARMM-free: TIP3P water on a jittered lattice at experimental density.
"""

from __future__ import annotations

import argparse
import json
import time

import numpy as np

# TIP3P, CHARMM convention (charges in e, epsilon in kcal/mol, Rmin/2 in A).
Q_O, Q_H = -0.834, 0.417
EPS_O, EPS_H = 0.1521, 0.046
RMIN_O, RMIN_H = 1.7682, 0.2245
R_OH, ANG_HOH = 0.9572, np.deg2rad(104.52)
M_WATER_G_MOL = 18.01528
N_AVOGADRO = 6.02214076e23


def water_box(n_mol: int, density_kg_m3: float = 997.0, seed: int = 0):
    """``n_mol`` rigid TIP3P waters on a jittered lattice at the given density."""
    rng = np.random.default_rng(seed)
    volume_A3 = n_mol * M_WATER_G_MOL / N_AVOGADRO / (density_kg_m3 * 1e-3) * 1e24
    side = float(volume_A3 ** (1.0 / 3.0))

    per_side = int(np.ceil(n_mol ** (1.0 / 3.0)))
    spacing = side / per_side
    sites = np.array(
        [(x, y, z) for x in range(per_side) for y in range(per_side) for z in range(per_side)],
        dtype=np.float64,
    )[:n_mol] * spacing
    # Jitter well inside the lattice spacing so molecules never overlap.
    sites += rng.uniform(-0.15, 0.15, size=sites.shape) * spacing

    # One rigid geometry, randomly oriented per molecule.
    local = np.array([
        [0.0, 0.0, 0.0],
        [R_OH, 0.0, 0.0],
        [R_OH * np.cos(ANG_HOH), R_OH * np.sin(ANG_HOH), 0.0],
    ])
    pos = np.empty((3 * n_mol, 3), dtype=np.float64)
    for m in range(n_mol):
        # Uniform random rotation via QR of a Gaussian matrix.
        q, r = np.linalg.qr(rng.normal(size=(3, 3)))
        q = q * np.sign(np.diag(r))
        pos[3 * m : 3 * m + 3] = local @ q.T + sites[m]

    n = 3 * n_mol
    z = np.tile([8, 1, 1], n_mol).astype(np.int32)
    mol_id = np.repeat(np.arange(n_mol), 3).astype(np.int32)
    charges = np.tile([Q_O, Q_H, Q_H], n_mol)
    epsilon = np.tile([EPS_O, EPS_H, EPS_H], n_mol)
    rmin_half = np.tile([RMIN_O, RMIN_H, RMIN_H], n_mol)
    at_codes = np.tile([0, 1, 1], n_mol).astype(np.int32)
    # Intramolecular: O-H, O-H, H-H per molecule.
    exc = np.concatenate([
        np.array([[3 * m, 3 * m + 1], [3 * m, 3 * m + 2], [3 * m + 1, 3 * m + 2]])
        for m in range(n_mol)
    ]).astype(np.int32)

    from mmml.md.system import FFParams, MolecularSystem

    ff = FFParams(
        charges=charges,
        epsilon=epsilon,
        rmin_half=rmin_half,
        at_codes=at_codes,
        exclusions=exc,
        e14_pairs=np.empty((0, 2), dtype=np.int32),
    )
    system = MolecularSystem(
        R=pos, Z=z, box=np.diag([side, side, side]), mol_id=mol_id, ff_params=ff
    )
    return system, side, n


def energy_and_grad(system, ctofnb: float = 12.0, ctonnb: float = 10.0):
    """Jitted (E, dE/dR) taking explicit pair arrays."""
    import jax
    import jax.numpy as jnp

    from mmml.interfaces.pycharmmInterface.mm_system_energy import CharmmNbondSettings
    from mmml.md.energy import EnergyContext
    from mmml.md.energy.terms import MMNonbondedTerm

    settings = CharmmNbondSettings(cutnb=ctofnb, ctonnb=ctonnb, ctofnb=ctofnb)
    fn = MMNonbondedTerm(settings).make(system, EnergyContext()).jax_energy_fn

    def e(R, pair_i, pair_j, pair_mask):
        return fn(R, pair_i=pair_i, pair_j=pair_j, pair_mask=pair_mask)

    e_jit = jax.jit(e)
    g_jit = jax.jit(jax.grad(e))

    def run(R, pairs):
        R = jnp.asarray(R)
        pi = jnp.asarray(pairs["pair_i"])
        pj = jnp.asarray(pairs["pair_j"])
        pm = jnp.asarray(pairs["pair_mask"])
        return e_jit(R, pi, pj, pm), g_jit(R, pi, pj, pm)

    return run


def static_pairs(system):
    from mmml.md.static_pairs import make_static_pair_fn

    fn = make_static_pair_fn(system, verbose=False)
    return {k: np.asarray(v) for k, v in fn(None, None).items()}


def nbr_pairs(system, positions, cutoff_A: float, skin_A: float = 0.0):
    from mmml.md.neighbors import make_intermolecular_neighbor_fn

    fn = make_intermolecular_neighbor_fn(system, cutoff_A=cutoff_A, skin_A=skin_A)
    return fn(np.asarray(positions), np.asarray(system.box))


def n_live(pairs) -> int:
    return int(np.asarray(pairs["pair_mask"]).sum())


def timeit(fn, *args, repeat: int = 5):
    """Median wall time (ms), discarding the compile/first call."""
    fn(*args)
    ts = []
    for _ in range(repeat):
        t0 = time.perf_counter()
        out = fn(*args)
        # Block on the device result so we time compute, not dispatch.
        for x in (out if isinstance(out, tuple) else (out,)):
            np.asarray(x)
        ts.append((time.perf_counter() - t0) * 1e3)
    return float(np.median(ts))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sizes", type=int, nargs="+", default=[100, 200, 400, 800, 1600])
    ap.add_argument("--cutoffs", type=float, nargs="+", default=[9.0, 10.0, 11.0, 12.0, 14.0])
    ap.add_argument("--drifts", type=float, nargs="+", default=[0.0, 0.1, 0.25, 0.5, 1.0, 2.0])
    ap.add_argument("--parity-size", type=int, default=400)
    ap.add_argument("--out", default="pair_bench.json")
    args = ap.parse_args()

    import jax

    jax.config.update("jax_enable_x64", True)
    print(f"jax devices: {jax.devices()}\n")

    results: dict = {"platform": str(jax.devices()[0].platform), "sizes": [], "cutoff": [], "drift": []}

    # ---------------------------------------------------------------- speed
    print("=== speed and pair counts vs system size ===")
    print(f"{'atoms':>7} {'box A':>7} {'static prs':>11} {'nbr prs':>9} {'nbr cap':>8} "
          f"{'build ms':>9} {'E+F stat':>9} {'E+F nbr':>9} {'dE (eV)':>12}")
    for n_mol in args.sizes:
        system, side, n = water_box(n_mol)
        run = energy_and_grad(system)
        sp = static_pairs(system)
        np_pairs = nbr_pairs(system, system.R, cutoff_A=12.0)

        build_ms = timeit(lambda: nbr_pairs(system, system.R, cutoff_A=12.0), repeat=3)
        e_s, f_s = run(system.R, sp)
        e_n, f_n = run(system.R, np_pairs)
        t_s = timeit(run, system.R, sp)
        t_n = timeit(run, system.R, np_pairs)

        de = abs(float(e_s) - float(e_n))
        df = float(np.max(np.abs(np.asarray(f_s) - np.asarray(f_n))))
        row = {
            "n_atoms": n, "box_A": side,
            "static_pairs": int(sp["pair_i"].shape[0]),
            "nbr_pairs_live": n_live(np_pairs),
            "nbr_capacity": int(np.asarray(np_pairs["pair_i"]).shape[0]),
            "host_build_ms": build_ms,
            "eval_static_ms": t_s, "eval_nbr_ms": t_n,
            "E_static_eV": float(e_s), "E_nbr_eV": float(e_n),
            "abs_dE_eV": de, "max_dF_eV_A": df,
        }
        results["sizes"].append(row)
        print(f"{n:>7} {side:>7.1f} {row['static_pairs']:>11} {row['nbr_pairs_live']:>9} "
              f"{row['nbr_capacity']:>8} {build_ms:>9.1f} {t_s:>9.2f} {t_n:>9.2f} {de:>12.2e}")

    # ------------------------------------------------------------- cutoff
    # The static list is complete, so it is the exact reference. This is the
    # axis on which the two differ at all: a neighbour list built below the
    # switch-off distance silently drops interactions that are not yet zero.
    print("\n=== truncation: neighbour list vs the complete list (exact) ===")
    system, side, n = water_box(args.parity_size)
    run = energy_and_grad(system)
    sp = static_pairs(system)
    e_ref, f_ref = run(system.R, sp)
    print(f"reference: complete list, {sp['pair_i'].shape[0]} pairs, E = {float(e_ref):.9f} eV")
    print(f"{'cutoff A':>9} {'live prs':>9} {'dE (eV)':>13} {'dE/atom (meV)':>14} {'max dF':>11}")
    for c in args.cutoffs:
        pj = nbr_pairs(system, system.R, cutoff_A=c)
        e_c, f_c = run(system.R, pj)
        de = float(e_c) - float(e_ref)
        df = float(np.max(np.abs(np.asarray(f_c) - np.asarray(f_ref))))
        results["cutoff"].append(
            {"cutoff_A": c, "live_pairs": n_live(pj), "dE_eV": de, "max_dF_eV_A": df}
        )
        print(f"{c:>9.1f} {n_live(pj):>9} {de:>13.2e} {de / n * 1e3:>14.3f} {df:>11.2e}")

    # -------------------------------------------------------------- staleness
    # A rebuilt list is only correct at the configuration it was built at. The
    # static list cannot go stale, so this error has no counterpart there.
    print("\n=== staleness: list built at t=0, energy evaluated after drift ===")
    print(f"{'RMS drift A':>12} {'dE stale (eV)':>15} {'dE/atom (meV)':>14} {'max dF':>11} {'missed prs':>11}")
    rng = np.random.default_rng(7)
    pairs_t0 = nbr_pairs(system, system.R, cutoff_A=12.0)
    for d in args.drifts:
        disp = rng.normal(scale=d / np.sqrt(3.0), size=system.R.shape) if d > 0 else 0.0
        R_t = system.R + disp
        e_exact, f_exact = run(R_t, sp)                        # complete list: exact
        e_stale, f_stale = run(R_t, pairs_t0)                  # list from t=0
        pairs_fresh = nbr_pairs(system, R_t, cutoff_A=12.0)
        de = float(e_stale) - float(e_exact)
        df = float(np.max(np.abs(np.asarray(f_stale) - np.asarray(f_exact))))
        missed = n_live(pairs_fresh) - n_live(pairs_t0)
        results["drift"].append(
            {"rms_drift_A": d, "dE_stale_eV": de, "max_dF_eV_A": df, "pair_delta": missed}
        )
        print(f"{d:>12.2f} {de:>15.2e} {de / n * 1e3:>14.3f} {df:>11.2e} {missed:>+11}")

    with open(args.out, "w") as fh:
        json.dump(results, fh, indent=2)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
