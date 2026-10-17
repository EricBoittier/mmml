#!/usr/bin/env python
"""Replica-batched periodic MD of liquid water with a packed EFieldPhysNet checkpoint (NVT Langevin → NVE).

All replicas (independent boxes, orthorhombic cell) run in one packed batch. Neighbour lists are rebuilt on
the GPU every --rebuild steps inside ``jax.lax.scan`` (all intra-box pairs, minimum image, radius
cutoff + --skin, fixed edge budget), so there is no host round trip. Message passing uses the model's smooth
cutoff, so pairs in the skin contribute exactly zero. Requires cutoff + skin < L/2 (minimum image unique).

Coulomb: the model's own Coulomb term (hard-truncated at the neighbour-list radius in training) is switched off
and re-added outside the model with the same charges and erf(r/4 Å) damping, times a C2 switch that is 1 below
--coulomb-switch-on and 0 at the cutoff (as nve_ir.py). No Ewald: real-space only, 5 Å.

Positions are never wrapped, so molecules stay whole and the model's cell dipole μ = Σ q_i (r_i − c) + Σ μ_i is
continuous; IR uses its time derivative. Optionally a second checkpoint (--alpha-ckpt) gives α = ∂μ/∂E and its own μ
on the same frames (Raman).

Modes:
  --checks  (use with --x64): finite-difference forces, wrap invariance, skin invariance, MD potential
            (switched Coulomb) vs training potential (hard-truncated) against DFT forces; writes OUT/checks.json
  default   Langevin --equil-ps then NVE --nve-ps; writes OUT/<tag>_traj.npz and OUT/<tag>_meta.json

Usage: python scripts/spice_alpha/pbc_water_md.py CKPT_DIR PARAMS OUT_DIR --data RAGGED_DIR [--frames 0 6 12 ...]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from nve_ir import ACC, KB_EV  # noqa: E402

BOHR3_PER_A3 = 6.748334


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("ckpt_dir", type=Path)
    ap.add_argument("params", type=Path)
    ap.add_argument("out_dir", type=Path)
    ap.add_argument("--data", type=Path, required=True, help="periodic ragged dir (with 'cell')")
    ap.add_argument("--frames", type=int, nargs="+", default=[0], help="start frames = replicas")
    ap.add_argument("--alpha-ckpt", type=Path, default=None)
    ap.add_argument("--alpha-params", type=Path, default=None)
    ap.add_argument("--temperature", type=float, default=300.0)
    ap.add_argument("--dt", type=float, default=0.5)
    ap.add_argument("--equil-ps", type=float, default=5.0)
    ap.add_argument("--friction", type=float, default=0.01, help="1/fs")
    ap.add_argument("--nve-ps", type=float, default=20.0)
    ap.add_argument("--save-every", type=int, default=4, help="steps between saved E, μ, α")
    ap.add_argument("--rebuild", type=int, default=20, help="steps between neighbour-list rebuilds (multiple of save-every)")
    ap.add_argument("--pos-every", type=int, default=40, help="steps between saved positions (multiple of rebuild)")
    ap.add_argument("--skin", type=float, default=0.8)
    ap.add_argument("--chunk-ps", type=float, default=1.0, help="ps per jitted call")
    ap.add_argument("--coulomb-switch-on", type=float, default=None, help="default cutoff − 0.5")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tag", default="water")
    ap.add_argument("--x64", action="store_true")
    ap.add_argument("--checks", action="store_true")
    ap.add_argument("--oh-wall", type=float, nargs=2, default=None, metavar=("R0", "K"),
                    help="topology safeguard: harmonic wall K (eV/Å²) on intermolecular O···H pairs closer than R0 Å (molecules from the bond graph of frame 0)")
    return ap.parse_args()


class System:
    """Packed periodic replicas: atoms [box0 | box1 | ... | pad], molecule slots [0..nr-1 | pad]."""

    def __init__(self, model, params, cfg, Z, L, nr, fdt, skin, r_on, n_edges_hint):
        import jax
        import jax.numpy as jnp
        from mmml.data.units import ANGSTROM_TO_BOHR, HARTREE_TO_EV
        from mmml.models.efield.packed import _forward

        self.n, self.nr = len(Z), nr
        n = self.n
        A, M = nr * n + 1, nr + 1
        self.A, self.M = A, M
        self.rc = rc = float(cfg["cutoff"])
        self.r_list = rc + skin
        assert self.r_list < 0.5 * float(np.min(L)), "cutoff + skin must be < L/2"
        self.E_max = int(np.ceil(n_edges_hint * 1.25 / 1024) * 1024)
        Lj = jnp.asarray(L, fdt)
        seg = jnp.asarray(np.r_[np.repeat(np.arange(nr), n), M - 1].astype(np.int32))
        self.static = {
            "atomic_numbers": jnp.asarray(np.r_[np.tile(Z, nr), 0].astype(np.int32)),
            "batch_segments": seg,
            "electric_field": jnp.zeros((M, 3), fdt),
        }
        dummy_c = {"coulomb_dst_idx_flat": jnp.asarray([A - 1], jnp.int32),
                   "coulomb_src_idx_flat": jnp.asarray([A - 1], jnp.int32)}
        mol = jnp.asarray(np.r_[np.ones(nr), 0.0], fdt)
        eye = jnp.eye(n, dtype=bool)
        E_max = self.E_max
        r_list = self.r_list
        sign = model.coulomb_sign

        def neighbors(pos_rep, radius=r_list, size=E_max):
            d = pos_rep[:, None, :, :] - pos_rep[:, :, None, :]           # [r, i=dst, j=src] = R_j − R_i
            img = jnp.round(d / Lj)
            r2 = jnp.sum((d - img * Lj) ** 2, -1)
            mask = (r2 < radius * radius) & ~eye[None]
            cnt = jnp.sum(mask)
            rr, ii, jj = jnp.nonzero(mask, size=size, fill_value=0)
            valid = jnp.arange(size) < cnt
            dst = jnp.where(valid, rr * n + ii, A - 1).astype(jnp.int32)
            src = jnp.where(valid, rr * n + jj, A - 1).astype(jnp.int32)
            shift = jnp.where(valid[:, None], -img[rr, ii, jj] * Lj, 0.0)
            return dst, src, shift.astype(fdt), cnt

        def flat(pos_rep):
            return jnp.concatenate([pos_rep.reshape(-1, 3), jnp.zeros((1, 3), fdt)])

        def switch(r):
            x = jnp.clip((r - r_on) / max(rc - r_on, 1e-6), 0.0, 1.0)
            return jnp.where(r < rc, 1.0 - x * x * x * (10.0 - 15.0 * x + 6.0 * x * x), 0.0)

        def energy(pos_rep, nl, prm=params):
            """MD potential: model NN energy + switched external Coulomb. Returns (sum, (e, μ, q))."""
            dst, src, shift, _ = nl
            pos = flat(pos_rep)
            b = {**self.static, **dummy_c, "positions": pos, "dst_idx_flat": dst, "src_idx_flat": src, "shifts_flat": shift}
            e, mu, state = _forward(model.apply, prm, b, M)
            q = state["intermediates"]["atomic_charges"][-1]
            valid = dst != A - 1
            disp = jnp.where(valid[:, None], pos[src] - pos[dst] + shift, 1.0)
            r = jnp.linalg.norm(disp, axis=-1)
            pair = jnp.where(valid, q[src] * q[dst] * jax.scipy.special.erf(r / 4.0) / (r * ANGSTROM_TO_BOHR) * switch(r), 0.0)
            coul = jax.ops.segment_sum(pair, seg[dst], num_segments=M) / 2.0 * HARTREE_TO_EV
            e = (e + sign * coul) * mol
            return jnp.sum(e), (e[:nr], mu[:nr], q[:nr * n])

        def energy_train(pos_rep, nl, prm=params):
            """Training potential: model's own Coulomb over the (hard, radius-rc) edge list."""
            dst, src, shift, _ = nl
            b = {**self.static, "positions": flat(pos_rep), "dst_idx_flat": dst, "src_idx_flat": src, "shifts_flat": shift}
            e, mu, _ = _forward(model.apply, prm, b, M)
            return jnp.sum(e * mol), (e[:nr], mu[:nr])

        def dipole_polar(pos_rep, nl, apply_fn, prm):
            """(μ (nr,3) e·Å, α (nr,3,3) Å³) with α_ab = ∂μ_a/∂E_b (forward mode, 3 tangents)."""
            dst, src, shift, _ = nl
            b = {**self.static, **dummy_c, "positions": flat(pos_rep), "dst_idx_flat": dst, "src_idx_flat": src,
                 "shifts_flat": shift}

            def dip(ef):
                return _forward(apply_fn, prm, {**b, "electric_field": jnp.tile(ef[None], (M, 1))}, M)[1][:nr]

            z = jnp.zeros((3,), fdt)
            mu, tang = jax.lax.map(lambda t: jax.jvp(dip, (z,), (t,)), jnp.eye(3, dtype=fdt))  # sequential: 1/3 memory
            alpha = jnp.transpose(tang, (1, 2, 0)) / 0.001 * ANGSTROM_TO_BOHR / BOHR3_PER_A3
            return mu[0], alpha

        self.neighbors, self.energy, self.energy_train, self.dipole_polar = neighbors, energy, energy_train, dipole_polar


def load_start(data_dir: Path, frames):
    from mmml.data.spice_alpha_ragged import load_ragged
    d = load_ragged(sorted(data_dir.glob("*.npz")))
    off = d["offsets"]
    Z = d["Z"][off[frames[0]]:off[frames[0] + 1]].astype(np.int32)
    R, F, E = [], [], []
    for f in frames:
        assert np.array_equal(d["Z"][off[f]:off[f + 1]], Z)
        R.append(d["R"][off[f]:off[f + 1]].astype(np.float64)); F.append(d["F"][off[f]:off[f + 1]]); E.append(d["E"][f])
    cell = d["cell"][frames[0]].astype(np.float64)
    assert np.allclose(cell, np.diag(np.diag(cell))), "orthorhombic cells only"
    assert all(np.allclose(d["cell"][f], cell) for f in frames)
    return Z, np.stack(R), np.stack(F), np.asarray(E), np.diag(cell).copy()


def run_checks(args, model, params, cfg, Z, R, F_ref, L, fdt, r_on):
    import jax
    import jax.numpy as jnp
    out = {}
    nr = len(R)
    sysm = System(model, params, cfg, Z, L, nr, fdt, args.skin, r_on, n_edges_hint=len(Z) * 120 * nr)
    pos = jnp.asarray(R, fdt)
    nl_skin = jax.jit(sysm.neighbors)(pos)
    nl_rc = jax.jit(lambda p: sysm.neighbors(p, radius=sysm.rc))(pos)
    out["edges_per_box"] = {"r_list": int(nl_skin[3]) / nr, "rc": int(nl_rc[3]) / nr}
    vg = jax.jit(jax.value_and_grad(sysm.energy, has_aux=True))
    (_, (e, mu, q)), g = vg(pos, nl_skin)
    F = -np.asarray(g)
    (_, (e_rc, _, _)), g_rc = vg(pos, nl_rc)
    out["skin_invariance"] = {"dE_eV": float(np.abs(np.asarray(e) - np.asarray(e_rc)).max()),
                              "dF": float(np.abs(F + np.asarray(g_rc)).max())}
    # wrap all atoms into [0, L): molecules broken, E and F must not change
    pw = jnp.asarray(np.mod(R, L), fdt)
    (_, (e_w, _, _)), g_w = vg(pw, jax.jit(sysm.neighbors)(pw))
    out["wrap_invariance"] = {"dE_eV": float(np.abs(np.asarray(e) - np.asarray(e_w)).max()),
                              "dF": float(np.abs(F + np.asarray(g_w)).max())}
    # whole-box translation by L/3 (wrapped): same check with a different image pattern
    pt = jnp.asarray(np.mod(R + L / 3.0, L), fdt)
    (_, (e_t, _, _)), _ = vg(pt, jax.jit(sysm.neighbors)(pt))
    out["translation_wrap_dE_eV"] = float(np.abs(np.asarray(e) - np.asarray(e_t)).max())
    # finite differences on box 0 (central, h)
    rng = np.random.default_rng(0)
    efun = jax.jit(lambda p: sysm.energy(p, nl_skin)[1][0][0])
    h = 1e-4 if args.x64 else 1e-2
    fd, an = [], []
    for _ in range(40):
        i, a = rng.integers(len(Z)), rng.integers(3)
        dp = np.zeros_like(R); dp[0, i, a] = h
        fd.append(-(float(efun(jnp.asarray(R + dp, fdt))) - float(efun(jnp.asarray(R - dp, fdt)))) / (2 * h))
        an.append(F[0, i, a])
    fd, an = np.asarray(fd), np.asarray(an)
    out["finite_difference"] = {"h_A": h, "n": len(fd), "max_abs_err": float(np.abs(fd - an).max()),
                                "rel_rms_err": float(np.sqrt(np.mean((fd - an) ** 2) / np.mean(an ** 2)))}
    # MD potential vs training potential vs DFT
    vg_t = jax.jit(jax.value_and_grad(sysm.energy_train, has_aux=True))
    (_, (e_tr, _)), g_tr = vg_t(pos, nl_rc)
    F_tr = -np.asarray(g_tr)
    rm = lambda x: float(np.sqrt(np.mean(np.square(x))))
    nw = len(Z) / 3
    de = np.asarray(e) - np.asarray(e_tr)
    out["md_vs_train_potential"] = {
        "forces_rmse_md_vs_dft": rm(F - F_ref), "forces_rmse_train_vs_dft": rm(F_tr - F_ref),
        "forces_rmse_md_vs_train": rm(F - F_tr),
        "energy_md_minus_train_meV_per_h2o_mean": float(de.mean() / nw * 1000),
        "energy_md_minus_train_meV_per_h2o_std": float(de.std() / nw * 1000)}
    out["charges"] = {"q_O_mean": float(np.asarray(q).reshape(nr, -1)[:, Z == 8].mean()),
                      "q_H_mean": float(np.asarray(q).reshape(nr, -1)[:, Z == 1].mean()),
                      "q_box_max_abs": float(np.abs(np.asarray(q).reshape(nr, -1).sum(1)).max())}
    out["cell_dipole_norm_eA"] = np.linalg.norm(np.asarray(mu), axis=-1).round(3).tolist()
    return out


def main():
    args = parse_args()
    import jax
    import jax.numpy as jnp
    jax.config.update("jax_enable_x64", args.x64)
    from ase.data import atomic_masses
    from les_water_eval import build

    fdt = jnp.float64 if args.x64 else jnp.float32
    cast = lambda p: jax.tree_util.tree_map(lambda x: jnp.asarray(x, fdt), p)
    model, params, cfg = build(args.ckpt_dir, args.params)
    params = cast(params)
    rc = float(cfg["cutoff"])
    r_on = rc - 0.5 if args.coulomb_switch_on is None else args.coulomb_switch_on
    Z, R0, F_ref, E_ref, L = load_start(args.data, args.frames)
    nr, n = R0.shape[:2]
    out = args.out_dir
    out.mkdir(parents=True, exist_ok=True)
    print(f"{nr} boxes x {n} atoms, L = {L} Å, cutoff {rc} + skin {args.skin} Å, Coulomb switch {r_on}–{rc} Å, "
          f"x64 {args.x64}", flush=True)

    if args.checks:
        res = run_checks(args, model, params, cfg, Z, R0, F_ref, L, fdt, r_on)
        print(json.dumps(res, indent=1), flush=True)
        (out / f"{args.tag}_checks.json").write_text(json.dumps(res, indent=1))
        return

    amodel = aparams = None
    if args.alpha_ckpt:
        amodel, aparams, _ = build(args.alpha_ckpt, args.alpha_params)
        aparams = cast(aparams)

    se, rb, pe = args.save_every, args.rebuild, args.pos_every
    assert rb % se == 0 and pe % rb == 0
    sysm = System(model, params, cfg, Z, L, nr, fdt, args.skin, r_on, n_edges_hint=0)
    cnt0 = int(jax.jit(lambda p: sysm.neighbors(p, size=8)[3])(jnp.asarray(R0, fdt)))
    sysm = System(model, params, cfg, Z, L, nr, fdt, args.skin, r_on, n_edges_hint=cnt0)
    print(f"edges at r_list: {cnt0 / nr:.0f} per box; budget {sysm.E_max}", flush=True)

    masses = np.array([atomic_masses[z] for z in Z])
    m = jnp.asarray(masses, fdt)[None, :, None]
    kT = KB_EV * args.temperature
    dt = args.dt
    ndof = 3 * n - 3

    def kinetic(v):
        return 0.5 * jnp.sum(m * v * v, axis=(1, 2)) / ACC

    wall = None
    if args.oh_wall is not None:
        r0_w, k_w = float(args.oh_wall[0]), float(args.oh_wall[1])
        # molecule index from the bond graph of the start frame (covalent pairs < 1.65 Å, minimum image; C–O is 1.43 Å)
        R0u = np.asarray(R0[0], np.float64); dmat = R0u[:, None] - R0u[None]; dmat -= L * np.round(dmat / L)
        adj = (np.linalg.norm(dmat, axis=-1) < 1.65) & ~np.eye(n, dtype=bool)
        mol_id = -np.ones(n, int); cur = 0
        for i0 in range(n):
            if mol_id[i0] >= 0: continue
            stack = [i0]; mol_id[i0] = cur
            while stack:
                a = stack.pop()
                for b in np.where(adj[a] & (mol_id < 0))[0]: mol_id[b] = cur; stack.append(b)
            cur += 1
        A = sysm.A
        Zrep = jnp.asarray(np.r_[np.tile(Z, nr), 0].astype(np.int32)); Mrep = jnp.asarray(np.r_[np.tile(mol_id, nr) + np.repeat(np.arange(nr) * cur, n), -1].astype(np.int32))
        print(f"O···H wall: {cur} molecules per box, r0 {r0_w} Å, k {k_w} eV/Å²", flush=True)

        def wall(p, nl):
            dst, src, shift, _ = nl
            pos = jnp.concatenate([p.reshape(-1, 3), jnp.zeros((1, 3), fdt)])
            valid = dst != A - 1
            r = jnp.linalg.norm(jnp.where(valid[:, None], pos[src] - pos[dst] + shift, 1.0), axis=-1)
            mask = valid & (Zrep[dst] == 8) & (Zrep[src] == 1) & (Mrep[dst] != Mrep[src]) & (r < r0_w)
            return k_w * jnp.sum(jnp.where(mask, (r0_w - r) ** 2, 0.0)), jnp.sum(mask)

    def forces(p, nl):
        if wall is None:
            (_, (e, mu, q)), g = jax.value_and_grad(sysm.energy, has_aux=True)(p, nl)
            return -g, e, mu
        def tot(p):
            etot, (e, mu, q) = sysm.energy(p, nl)
            ew, nw_ = wall(p, nl)
            return etot + ew, (e + ew / nr, mu)
        (_, (e, mu)), g = jax.value_and_grad(tot, has_aux=True)(p)
        return -g, e, mu

    def zero_com(v):
        return v - jnp.sum(m * v, axis=1, keepdims=True) / jnp.sum(m)

    def make_chunk(langevin, n_blocks, record):
        c1 = float(np.exp(-args.friction * dt))
        sig = jnp.sqrt((1 - c1 * c1) * kT * ACC / m)

        def vv_step(c, k):
            p, v, f, p0, dmax = c[:5]
            nl = c[5]
            if langevin:
                v = v + 0.5 * dt * ACC * f / m
                p = p + 0.5 * dt * v
                v = c1 * v + sig * jax.random.normal(k, v.shape, fdt)
                p = p + 0.5 * dt * v
            else:
                v = v + 0.5 * dt * ACC * f / m
                p = p + dt * v
            f, e, mu = forces(p, nl)
            v = v + 0.5 * dt * ACC * f / m
            disp = jnp.linalg.norm((p - p0).reshape(-1, 3), axis=-1)
            top2 = jnp.sum(jax.lax.top_k(disp, 2)[0])          # worst pair approach since the rebuild
            return (p, v, f, p0, jnp.maximum(dmax, top2), nl), (e, mu)

        def sub(c, keys):  # se steps, then record
            c, (e, mu) = jax.lax.scan(vv_step, c, keys)
            p, v = c[0], c[1]
            rec = {"epot": e[-1], "ekin": kinetic(v), "mu": mu[-1]}
            if record and amodel is not None:
                mu_a, al = sysm.dipole_polar(p, c[5], amodel.apply, aparams)
                rec["mu_alpha"], rec["alpha"] = mu_a, al
            return c, rec

        def block(c, keys):  # rebuild, then rb steps
            p, v, f = c[:3]
            nl = sysm.neighbors(p)
            c = (p, v, f, p, jnp.zeros((), fdt), nl)
            c, rec = jax.lax.scan(sub, c, keys.reshape(rb // se, se, 2))
            return c[:3], (rec, c[0], c[4], nl[3])

        def chunk(c, key):
            keys = jax.random.split(key, n_blocks * rb).reshape(n_blocks, rb, 2)
            return jax.lax.scan(block, c, keys)

        return jax.jit(chunk)

    rng = np.random.default_rng(args.seed)
    v0 = rng.normal(size=(nr, n, 3)) * np.sqrt(kT / masses[None, :, None] * ACC)
    vel = zero_com(jnp.asarray(v0, fdt))
    pos = jnp.asarray(R0, fdt)
    f0 = jax.jit(lambda p: forces(p, sysm.neighbors(p))[0])(pos)
    carry = (pos, vel, f0)
    key = jax.random.PRNGKey(args.seed)
    blocks_per_chunk = max(int(round(args.chunk_ps * 1000 / dt / rb)), 1)
    steps_per_chunk = blocks_per_chunk * rb
    meta = {"args": {k: str(v) for k, v in vars(args).items()}, "n_atoms": n, "replicas": nr, "L": L.tolist(),
            "coulomb_switch": [r_on, rc], "edge_budget": sysm.E_max, "edges_per_box_start": cnt0 / nr}

    # --- NVT (Langevin BAOAB)
    eq_blocks = int(round(args.equil_ps * 1000 / dt / rb))
    n_eq = int(np.ceil(eq_blocks / blocks_per_chunk))
    eq_fns = {}
    t0 = time.time()
    Teq, eq_dmax, eq_cnt = [], 0.0, 0
    for i in range(n_eq):
        nb = min(blocks_per_chunk, eq_blocks - i * blocks_per_chunk)
        if nb not in eq_fns:
            eq_fns[nb] = make_chunk(True, nb, False)
        key, sk = jax.random.split(key)
        carry, (rec, _, dmax, cnt) = eq_fns[nb](carry, sk)
        Tb = 2 * np.asarray(rec["ekin"]) / (ndof * KB_EV)
        Teq.append(Tb.mean())
        eq_dmax, eq_cnt = max(eq_dmax, float(dmax.max())), max(eq_cnt, int(cnt.max()))
        if i == 0 or (i + 1) % 5 == 0:
            print(f"  NVT {min((i + 1) * blocks_per_chunk, eq_blocks) * rb * dt / 1000:.1f} ps  T {Tb[-1].round(0)}  "
                  f"Epot/H2O {np.asarray(rec['epot'])[-1] / (n / 3)}  {time.time() - t0:.0f}s", flush=True)
    meta.update(nvt_ps=eq_blocks * rb * dt / 1000, nvt_wall_s=time.time() - t0,
                nvt_T_last_ps=float(np.mean(Teq[-max(len(Teq) // 5, 1):])) if Teq else None,
                nvt_max_pair_disp=eq_dmax, nvt_max_edges=eq_cnt)

    # --- NVE
    carry = (carry[0], zero_com(carry[1]), carry[2])
    nve = make_chunk(False, blocks_per_chunk, True)
    n_ch = int(round(args.nve_ps * 1000 / dt / steps_per_chunk))
    recs, poss, dmaxs, cnts = [], [], [], []
    t0 = time.time()
    t_first = None
    def finalize(wall, final):
        rec = {k: np.concatenate([r[k] for r in recs]) for k in recs[0]}
        pos_traj = np.concatenate(poss)
        dmax_all, cnt_all = np.concatenate(dmaxs), np.concatenate(cnts)
        etot = rec["epot"] + rec["ekin"]
        tt = np.arange(len(etot)) * se * dt / 1000  # ps
        slope = np.polyfit(tt, etot - etot[0], 1)[0]  # eV/ps per box
        steps_s = (len(recs) - 1) * steps_per_chunk / max(wall - t_first, 1e-9) if len(recs) > 1 else len(recs) * steps_per_chunk / wall
        T_nve = 2 * rec["ekin"] / (ndof * KB_EV)
        meta.update(nve_ps=float(len(etot) * se * dt / 1000), nve_wall_s=wall, steps_per_s=steps_s,
                    ns_per_day_aggregate=steps_s * nr * dt * 86400 / 1e6,
                    drift_meV_per_ps_per_box=(slope * 1000).tolist(),
                    drift_meV_per_ps_per_h2o_absmean=float(np.abs(slope).mean() * 1000 / (n / 3)),
                    etot_std_meV_per_box=(etot.std(0) * 1000).tolist(),
                    etot_std_after_detrend_meV_per_box=[float(np.std(etot[:, b] - np.polyval(np.polyfit(tt, etot[:, b], 1), tt)) * 1000) for b in range(nr)],
                    T_nve_mean=T_nve.mean(0).tolist(), T_nve_std=T_nve.std(0).tolist(),
                    max_pair_disp_between_rebuilds=float(dmax_all.max()), skin=args.skin,
                    n_rebuild_violations=int((dmax_all > args.skin).sum()), max_edges=int(cnt_all.max()),
                    edge_overflow=bool(cnt_all.max() > sysm.E_max))
        np.savez(out / f"{args.tag}_traj.npz", Z=Z, masses=masses, L=L, dt_fs=dt, save_every=se,
                 pos_every=pe, temperature=args.temperature, pos=pos_traj, frames=np.asarray(args.frames),
                 **{k: v.astype(np.float64 if k in ("epot", "ekin") else np.float32) for k, v in rec.items()})
        (out / f"{args.tag}_meta.json").write_text(json.dumps(meta, indent=1))
        if final:
            print(json.dumps({k: v for k, v in meta.items() if k != "args"}, indent=1), flush=True)

    for i in range(n_ch):
        carry, (rec, p_blk, dmax, cnt) = nve(carry, key)
        rec = jax.tree_util.tree_map(lambda x: np.asarray(x).reshape((-1,) + x.shape[2:]), rec)
        recs.append(rec)
        poss.append(np.asarray(p_blk[::pe // rb], np.float32))
        dmaxs.append(np.asarray(dmax)); cnts.append(np.asarray(cnt))
        if t_first is None:
            t_first = time.time() - t0
        if not np.isfinite(rec["epot"]).all():
            print("non-finite energy, stopping", flush=True)
            break
        if i == 0 or (i + 1) % 5 == 0:
            et = rec["epot"] + rec["ekin"]
            Tn = 2 * rec["ekin"][-1] / (ndof * KB_EV)
            print(f"  NVE {(i + 1) * steps_per_chunk * dt / 1000:.1f} ps  T {Tn.round(0)}  "
                  f"Etot box0 {et[-1, 0]:.4f}  {time.time() - t0:.0f}s", flush=True)
        if (i + 1) % 10 == 0 and i + 1 < n_ch:
            finalize(time.time() - t0, False)  # partial save
    finalize(time.time() - t0, True)
    print(f"wrote {out}/{args.tag}_traj.npz", flush=True)


if __name__ == "__main__":
    main()
