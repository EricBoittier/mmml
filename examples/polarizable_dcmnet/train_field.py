"""Train / fine-tune the field-conditioned (polarisable) DCMnet on dimer dESP labels (single module with all recipe switches).

The monomer model is local: every molecule is evaluated alone, and the partner / environment enters only as a background
potential V and field E at its atoms (``karml.models.dcmnet.polarizable.field_inputs``).  For a dimer A+B

    V_AB(model) = V(A | field of B) + V(B | field of A),   dESP(model) = V_AB(model) - V(A|0) - V(B|0)
    dESP(QM)    = V_AB - V_A - V_B

Recipes (all numbers in docs/polarisable-dcmnet.md):

* F1 (potential+field, partner = zero-field model charges, one shot):  ``--source model --emb_w 0``
* ablations:                      ``--field_terms V|E|VE  --inject first|both  --scf K  --huber DELTA``
* environment-aware (ML/MM):      ``--source cgenff --emb_w 1 --env_file ...``  (CGenFF charges of partner and MM
                                  environment are the field source; needs dimemb_*.npz labels)
* scratch / wide models:          ``--scratch --features 64 --max_degree 2 ...``
* other molecules (e.g. DCM):     element masses via ``--mass``; atoms per molecule from the data (``len(Z) // 2``)

Padding: a pass uses the exact 11-node merged padding graph (``merged_graph``), identical to the 60-node padded model.

Data (no data ships with the repository, see the docs): ``--data`` comma list of directories holding ``dim_*.npz``
(and ``dimemb_*.npz``), ``--mono_data`` / ``--mono_test`` monomer replay sets, optional ``--embed_in`` / ``--embed_out``
embedded-monomer set, ``--env_file`` environment charges.

usage: python train_field.py --start RUN_DIR --data DIR1,DIR2 --mono_data M.npz --mono_test T.npz --out OUT [--epochs 150]
"""

import argparse
import glob
import json
import math
import os
import time
from pathlib import Path

os.environ.setdefault("JAX_DEFAULT_MATMUL_PRECISION", "highest")

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
import optax  # noqa: E402
from flax import serialization  # noqa: E402

from karml.models.dcmnet.polarizable import (  # noqa: E402
    BOHR_PER_ANG,
    ESCALE,
    VSCALE,
    FieldMessagePassingModel,
    field_inputs,
    merged_graph,
    pad_field,
)

HA = 627.5095  # Hartree -> kcal/mol
DEBYE_PER_AU = 2.541746
DEBYE_PER_EANG = 4.80320
DEFAULT_ARCH = dict(
    features=32,
    max_degree=2,
    num_iterations=3,
    num_basis_functions=8,
    cutoff=5.0,
    max_atomic_number=17,
    n_dcm=3,
    include_pseudotensors=False,
    dc_scale=0.175,
)


def load_dimers(dirs, target="gas"):
    """Load ``dim_*.npz`` from comma-separated directories into padded arrays.

    Per-file keys: Z, R (2n atoms: monomer A then B), grid_ang (G,3), scale (G), esp_AB/esp_A/esp_B (G, Hartree/e),
    centroid, frame, E_int_kcal.  With ``target='emb'`` the ESP labels are replaced by those of the matching
    ``dimemb_*.npz`` (dimer-in-environment); dimers without such a file are dropped."""
    dirs = dirs.split(",")
    fs = sorted(sum([glob.glob(str(Path(x) / "dim_*.npz")) for x in dirs], []), key=os.path.basename)
    if not fs:
        raise SystemExit("no dim_*.npz in " + str(dirs))
    L = []
    if target == "emb":
        emb = {os.path.basename(f)[7:]: f for x in dirs for f in glob.glob(str(Path(x) / "dimemb_*.npz"))}
        fs = [f for f in fs if os.path.basename(f)[4:] in emb]
        for f in fs:
            x = dict(np.load(f))
            e = np.load(emb[os.path.basename(f)[4:]])
            assert np.allclose(x["grid_ang"], e["grid_ang"], atol=1e-4)
            for k in ("esp_AB", "esp_A", "esp_B", "E_AB", "E_A", "E_B", "E_int_kcal"):
                x[k] = e[k]
            L.append(x)
    else:
        L = [dict(np.load(f)) for f in fs]
    G = max(len(x["grid_ang"]) for x in L)
    N = len(L)
    na = len(L[0]["Z"])
    out = {
        "R": np.zeros((N, na, 3), np.float32),
        "grid": np.zeros((N, G, 3), np.float32),
        "gm": np.zeros((N, G), bool),
        "scale": np.zeros((N, G), np.float32),
        "VAB": np.zeros((N, G), np.float32),
        "VA": np.zeros((N, G), np.float32),
        "VB": np.zeros((N, G), np.float32),
    }
    for i, x in enumerate(L):
        g = len(x["grid_ang"])
        out["R"][i] = x["R"]
        out["grid"][i, :g] = x["grid_ang"]
        out["gm"][i, :g] = True
        out["scale"][i, :g] = x["scale"]
        out["VAB"][i, :g] = x["esp_AB"]
        out["VA"][i, :g] = x["esp_A"]
        out["VB"][i, :g] = x["esp_B"]
    out["centroid"] = np.array([float(x["centroid"]) for x in L])
    out["frame"] = np.array([int(x["frame"]) for x in L])
    out["Z"] = L[0]["Z"].astype(np.int32)
    return out


def env_arrays(env_files, n_dimers):
    """Environment charges per dimer from npz files (comma list, concatenated in order, dimer order = sorted dim_*.npz).

    Each file: ``mm_off`` (m+1,), ``mm_q`` (K,) e, ``mm_xyz`` (K,3) A, entry k = charges mm_off[k]:mm_off[k+1]."""
    items = []
    for f in env_files.split(","):
        z = np.load(f)
        for k in range(len(z["mm_off"]) - 1):
            a, b = z["mm_off"][k], z["mm_off"][k + 1]
            items.append((z["mm_q"][a:b], z["mm_xyz"][a:b]))
    assert len(items) >= n_dimers, f"env_file has {len(items)} entries for {n_dimers} dimers"
    items = items[:n_dimers]
    K = max(len(q) for q, _ in items)
    EQ = np.zeros((n_dimers, K), np.float32)
    EX = np.zeros((n_dimers, K, 3), np.float32)
    for i, (q, x) in enumerate(items):
        EQ[i, : len(q)] = q
        EX[i, : len(q)] = x
    return EQ, EX


def get_parser():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--start", default="", help="run dir with model_config.json + best_params.msgpack (warm start)")
    ap.add_argument("--data", required=True, help="comma list of dirs with dim_*.npz")
    ap.add_argument(
        "--target", default="gas", choices=["gas", "emb"], help="label set of the plain (no-environment) terms"
    )
    ap.add_argument("--out", required=True)
    ap.add_argument(
        "--mono_data",
        required=True,
        help="monomer replay train set (R, Z, grid, gmask, mmask, esp, dip_au, q_cgenff, q_qm, split)",
    )
    ap.add_argument("--mono_test", required=True)
    ap.add_argument(
        "--mass",
        default="1:1.008,6:12.011,8:15.999,17:35.453",
        help="element:mass list (centre of mass for the frame)",
    )
    ap.add_argument(
        "--env_file",
        default="",
        help="npz list with environment charges (mm_off, mm_q, mm_xyz); needed for --emb_w > 0",
    )
    ap.add_argument(
        "--embed_in",
        default="",
        help="embedded-monomer inputs npz (R, mm_off, mm_q, mm_xyz); enables far-field evaluation",
    )
    ap.add_argument("--embed_out", default="", help="dir with emb_NNNN.npz labels (grid_ang, esp, r_com, dipole_D)")
    ap.add_argument(
        "--conf_file", default="", help="npz with 'frame' of the embedded conformers (train <=45, val 46-55, test >55)"
    )
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--rbs", type=int, default=8)
    ap.add_argument("--epochs", type=int, default=150)
    ap.add_argument("--lr_floor", type=float, default=0.02)
    ap.add_argument("--esp_w", type=float, default=1e4)
    ap.add_argument("--dw", type=float, default=5e4)
    ap.add_argument("--replay_w", type=float, default=1.0)
    ap.add_argument("--dip_w", type=float, default=10.0)
    ap.add_argument("--mono_w", type=float, default=1.0)
    ap.add_argument(
        "--emb_w", type=float, default=0.0, help="weight of the dimer-in-environment terms (0 = gas phase only)"
    )
    ap.add_argument(
        "--source",
        default="model",
        choices=["model", "cgenff"],
        help="field source: zero-field model charges or fixed CGenFF charges",
    )
    ap.add_argument("--sigma", type=float, default=1.0)
    ap.add_argument("--field_terms", default="VE", choices=["VE", "V", "E"])
    ap.add_argument("--inject", default="first", choices=["first", "both"])
    ap.add_argument(
        "--scf",
        type=int,
        default=1,
        help="k>1: recompute fields from polarised charges (source=model only, stop-gradient)",
    )
    ap.add_argument(
        "--huber", type=float, default=0.0, help="pseudo-Huber delta (kcal/mol/e) on dimer ESP residuals; 0 = MSE"
    )
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max_train", type=int, default=0)
    ap.add_argument("--far_every", type=int, default=10)
    ap.add_argument("--far_w", type=float, default=0.0, help="weight of the embedded-monomer term")
    ap.add_argument("--far_bs", type=int, default=8)
    ap.add_argument("--features", type=int, default=0)
    ap.add_argument("--max_degree", type=int, default=0)
    ap.add_argument("--num_iterations", type=int, default=0)
    ap.add_argument("--cutoff", type=float, default=0.0)
    ap.add_argument("--scratch", action="store_true", help="random init (forced by any architecture override)")
    ap.add_argument(
        "--mono_ref",
        type=float,
        default=0.0,
        help="monomer-val reference for the selection penalty (default: value at start; ~1.10 for random init)",
    )
    return ap


def main(argv=None):
    a = get_parser().parse_args(argv)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    MASS = {int(k): float(v) for k, v in (x.split(":") for x in a.mass.split(","))}
    if a.start:
        cfg = json.load(open(Path(a.start) / "model_config.json"))["architecture"]
    else:
        cfg = dict(DEFAULT_ARCH)
    over = {
        k: v
        for k, v in dict(
            features=a.features, max_degree=a.max_degree, num_iterations=a.num_iterations, cutoff=a.cutoff
        ).items()
        if v
    }
    scratch = a.scratch or bool(over) or not a.start
    cfg.update(over)
    nd = cfg["n_dcm"]
    model = FieldMessagePassingModel(**cfg, inject_all=(a.inject == "both"))
    p0 = (
        None
        if scratch
        else jax.tree_util.tree_map(
            jnp.asarray, serialization.msgpack_restore((Path(a.start) / "best_params.msgpack").read_bytes())
        )
    )

    if a.emb_w > 0 and not a.env_file:
        raise SystemExit("--env_file is required for --emb_w > 0")
    D = load_dimers(a.data, a.target)
    use_env_labels = bool(a.env_file)
    if use_env_labels:
        De = load_dimers(a.data, "emb")
        assert len(D["R"]) == len(De["R"]), "emb labels missing for some dimers"
    Z = D["Z"]
    fr = D["frame"]
    n = len(Z) // 2
    Z1 = Z[:n]
    if a.env_file:
        EQ, EX = env_arrays(a.env_file, len(D["R"]))
    else:
        EQ, EX = np.zeros((len(D["R"]), 1), np.float32), np.zeros((len(D["R"]), 1, 3), np.float32)
    # frame-based split: train frames <= 45 or 1000-1049, val 46-55 or 1050-1074, rest test
    split = np.where(
        (fr <= 45) | ((fr >= 1000) & (fr < 1050)),
        0,
        np.where(((fr > 45) & (fr <= 55)) | ((fr >= 1050) & (fr < 1075)), 1, 2),
    )
    idx = {k: np.where(split == k)[0] for k in range(3)}
    if a.max_train:
        idx[0] = idx[0][: a.max_train]
    P, dst, src, EW = merged_graph(n)
    Zp = jnp.zeros(P, jnp.int32).at[:n].set(jnp.asarray(Z1, jnp.int32))
    wm = np.array([MASS[int(z)] for z in Z1], np.float32)
    pf = model.init(
        jax.random.PRNGKey(a.seed),
        atomic_numbers=Zp,
        positions=jnp.zeros((P, 3), jnp.float32),
        dst_idx=dst,
        src_idx=src,
        field_V=jnp.zeros(P),
        field_E=jnp.zeros((P, 3)),
        edge_w=EW,
    )
    params0 = pf if scratch else {"params": {**pf["params"], **p0["params"]}}
    print(
        f"dimers: train {len(idx[0])} val {len(idx[1])} test {len(idx[2])}; env atoms max {EQ.shape[1]}; backend {jax.default_backend()}; "
        f"n_params {sum(x.size for x in jax.tree_util.tree_leaves(params0))}; merged graph {P} nodes",
        flush=True,
    )
    J = {k: jnp.asarray(D[k]) for k in ("R", "grid", "gm", "scale", "VAB", "VA", "VB")}
    if use_env_labels:
        J.update(VABe=jnp.asarray(De["VAB"]), VAe=jnp.asarray(De["VA"]), VBe=jnp.asarray(De["VB"]))
    else:
        J.update(VABe=J["VAB"], VAe=J["VA"], VBe=J["VB"])
    J.update(EQ=jnp.asarray(EQ), EX=jnp.asarray(EX))
    M = np.load(a.mono_data)
    MT = np.load(a.mono_test)
    qcg = jnp.asarray(M["q_cgenff"], jnp.float32)
    zV = jnp.zeros(n, jnp.float32)
    zE = jnp.zeros((n, 3), jnp.float32)
    sg = jax.lax.stop_gradient
    useV = 1.0 if "V" in a.field_terms else 0.0
    useE = 1.0 if "E" in a.field_terms else 0.0

    def post(mono, dipo):
        d = jnp.moveaxis(dipo, -1, -2).reshape(P * nd, 3)
        m = mono.reshape(P * nd)
        NDC = n * nd
        valid = jnp.where(jnp.arange(P * nd) < NDC, 1, 0)
        m = m * valid
        m = (m - m.sum() / NDC) * valid
        return m[: n * nd], d[: n * nd]

    def charges(params, R10, fV, fE, com=True):
        c = (R10 * wm[:, None]).sum(0) / wm.sum() if com else jnp.zeros(3)
        Rp = jnp.zeros((P, 3), jnp.float32).at[:n].set(R10 - c)
        Vp, Ep = pad_field(fV, fE, P)
        mono, dipo = model.apply(
            params, atomic_numbers=Zp, positions=Rp, dst_idx=dst, src_idx=src, field_V=Vp, field_E=Ep, edge_w=EW
        )
        q, p = post(mono, dipo)
        return q, p + c

    def pot(q, p, grid):
        return jnp.sum(q[None] / (jnp.linalg.norm(grid[:, None] - p[None], axis=-1) * BOHR_PER_ANG), axis=-1)

    def FI(Ra, q, p):
        V, E = field_inputs(Ra, q, p, n, a.sigma)
        return V * useV, E * useE

    def dimer_fwd(params, R, grid, eq, ex):
        A, B = R[:n], R[n:]
        qA0, pA0 = charges(params, A, *FI(A, eq, ex))  # monomers in the environment (zero charges = vacuum)
        qB0, pB0 = charges(params, B, *FI(B, eq, ex))
        if a.source == "cgenff":
            sA = (jnp.concatenate([qcg, eq]), jnp.concatenate([A, ex]))
            sB = (jnp.concatenate([qcg, eq]), jnp.concatenate([B, ex]))
        else:
            sA = (jnp.concatenate([sg(qA0), eq]), jnp.concatenate([sg(pA0), ex]))
            sB = (jnp.concatenate([sg(qB0), eq]), jnp.concatenate([sg(pB0), ex]))
        qA1, pA1, qB1, pB1 = qA0, pA0, qB0, pB0
        for _ in range(a.scf):
            qA1, pA1 = charges(params, A, *FI(A, *sB))  # field at A from B (+env), at B from A (+env)
            qB1, pB1 = charges(params, B, *FI(B, *sA))
            if a.source == "model":
                sA = (jnp.concatenate([sg(qA1), eq]), jnp.concatenate([sg(pA1), ex]))
                sB = (jnp.concatenate([sg(qB1), eq]), jnp.concatenate([sg(pB1), ex]))
        return pot(qA1, pA1, grid) + pot(qB1, pB1, grid), pot(qA0, pA0, grid), pot(qB0, pB0, grid)

    dh = a.huber / HA

    def rho(x):
        return x**2 if a.huber <= 0 else 2 * dh**2 * (jnp.sqrt(1 + (x / dh) ** 2) - 1)

    def msk(x, m):
        return jnp.sum(jnp.where(m, x, 0.0)) / jnp.maximum(m.sum(), 1)

    def terms(params, R, grid, gm, VAB, VA, VB, eq, ex):
        mAB, mA, mB = dimer_fwd(params, R, grid, eq, ex)
        dQ = VAB - VA - VB
        dM = mAB - mA - mB
        return msk(rho(mAB - VAB), gm), msk(rho(mA - VA), gm), msk(rho(mB - VB), gm), msk(rho(dM - dQ), gm)

    def mpack(X, q):
        return {
            k: jnp.asarray(v)
            for k, v in dict(
                R=X["R"], grid=X["grid"], gm=X["gmask"], mm=X["mmask"], esp=X["esp"], dip=X["dip_au"], qref=q
            ).items()
        }

    Mfull = mpack(M, np.broadcast_to(M["q_cgenff"], M["q_qm"].shape).astype(np.float32))
    mtr = np.where(M["split"] == 0)[0]
    mva = np.where(M["split"] == 1)[0]
    Mte = mpack(MT, np.broadcast_to(MT["q_cgenff"], MT["q_qm"].shape).astype(np.float32))

    def mono_one(params, R, grid, gm, mm, esp, dip, qref):  # monomer replay without field
        del mm
        q, p = charges(params, R, zV, zE, com=False)
        V = pot(q, p, grid)
        d2 = (V - esp) ** 2
        mse = jnp.sum(jnp.where(gm, d2, 0.0)) / gm.sum()
        mu = jnp.sum(q[:, None] * p, axis=0) * BOHR_PER_ANG
        qa = q.reshape(n, nd).sum(-1)
        return 0.5 * mse, 0.5 * jnp.sum((mu - dip) ** 2), 0.5 * jnp.sum((qa - qref) ** 2) / n, jnp.sqrt(mse) * HA

    FAR = bool(a.embed_in)
    EF = []
    ctr = cva = cte = np.zeros(0, int)
    FB = {k: jnp.zeros((1,)) for k in ("R", "grid", "esp", "dip", "mq", "mx")}
    if FAR:
        E_in = np.load(a.embed_in)
        for j in range(len(E_in["R"])):
            x = np.load(Path(a.embed_out) / f"emb_{j:04d}.npz")
            s0, s1 = E_in["mm_off"][j], E_in["mm_off"][j + 1]
            EF.append(
                dict(
                    R=jnp.asarray(E_in["R"][j], jnp.float32),
                    grid=jnp.asarray(x["grid_ang"], jnp.float32),
                    esp=jnp.asarray(x["esp"], jnp.float32),
                    r=jnp.asarray(x["r_com"], jnp.float32),
                    dip=float(np.linalg.norm(x["dipole_D"])),
                    mq=jnp.asarray(E_in["mm_q"][s0:s1], jnp.float32),
                    mx=jnp.asarray(E_in["mm_xyz"][s0:s1], jnp.float32),
                    dipv=np.asarray(x["dipole_D"]) / DEBYE_PER_AU,
                )
            )
        cf = np.load(a.conf_file)["frame"][: len(EF)]
        ctr = np.where(cf <= 45)[0]
        cva = np.where((cf > 45) & (cf <= 55))[0]
        cte = np.where(cf > 55)[0]
        Kp = max(len(e["mq"]) for e in EF)
        FB = {k: [] for k in ("R", "grid", "esp", "dip", "mq", "mx")}
        for e in EF:
            k = len(e["mq"])
            FB["R"].append(np.asarray(e["R"]))
            FB["grid"].append(np.asarray(e["grid"]))
            FB["esp"].append(np.asarray(e["esp"]))
            FB["dip"].append(e["dipv"])
            FB["mq"].append(np.pad(np.asarray(e["mq"]), (0, Kp - k)))
            FB["mx"].append(np.pad(np.asarray(e["mx"]), ((0, Kp - k), (0, 0))))
        FB = {k: jnp.asarray(np.stack(v), jnp.float32) for k, v in FB.items()}
        print(f"embedded conformers: train {len(ctr)} val {len(cva)} test {len(cte)}; far_w {a.far_w}", flush=True)
    elif a.far_w > 0:
        raise SystemExit("--far_w > 0 needs --embed_in/--embed_out/--conf_file")

    def far_term(params, R, grid, esp, dip_au, mq, mx):
        q, p = charges(params, R, *FI(R, mq, mx))
        c = (R * wm[:, None]).sum(0) / wm.sum()
        V = pot(q, p, grid)
        mu = jnp.sum(q[:, None] * (p - c), axis=0) * BOHR_PER_ANG
        return 0.5 * jnp.mean((V - esp) ** 2), 0.5 * jnp.sum((mu - dip_au) ** 2)

    def loss_fn(params, b, rb, fb):
        e_ab, e_a, e_b, e_d = jax.vmap(
            lambda R, g, m, v1, v2, v3, eq, ex: terms(params, R, g, m, v1, v2, v3, eq * 0.0, ex)
        )(b["R"], b["grid"], b["gm"], b["VAB"], b["VA"], b["VB"], b["EQ"], b["EX"])
        L = a.esp_w * 0.5 * (e_ab + 0.5 * (e_a + e_b)).mean() + a.dw * 0.5 * e_d.mean()
        if a.emb_w > 0:
            f_ab, f_a, f_b, f_d = jax.vmap(
                lambda R, g, m, v1, v2, v3, eq, ex: terms(params, R, g, m, v1, v2, v3, eq, ex)
            )(b["R"], b["grid"], b["gm"], b["VABe"], b["VAe"], b["VBe"], b["EQ"], b["EX"])
            L = L + a.emb_w * (a.esp_w * 0.5 * (f_ab + 0.5 * (f_a + f_b)).mean() + a.dw * 0.5 * f_d.mean())
        l_esp, l_dip, l_mono, _ = jax.vmap(lambda *x: mono_one(params, *x))(
            rb["R"], rb["grid"], rb["gm"], rb["mm"], rb["esp"], rb["dip"], rb["qref"]
        )
        L = L + a.replay_w * (a.esp_w * l_esp.mean() + a.dip_w * l_dip.mean() + a.mono_w * l_mono.mean())
        if a.far_w > 0:
            f_esp, f_dip = jax.vmap(lambda *x: far_term(params, *x))(
                fb["R"], fb["grid"], fb["esp"], fb["dip"], fb["mq"], fb["mx"]
            )
            L = L + a.far_w * (a.esp_w * f_esp.mean() + a.dip_w * f_dip.mean())
        return L

    opt = optax.chain(optax.clip_by_global_norm(1.0), optax.inject_hyperparams(optax.adam)(learning_rate=a.lr))

    @jax.jit
    def step(params, st, b, rb, fb):
        L, g = jax.value_and_grad(loss_fn)(params, b, rb, fb)
        up, st = opt.update(g, st, params)
        return optax.apply_updates(params, up), st, L

    def make_evd(use_env):
        VAB_, VA_, VB_ = ("VABe", "VAe", "VBe") if use_env else ("VAB", "VA", "VB")

        @jax.jit
        def evd(params, b):
            def one(R, grid, gm, sc, VAB, VA, VB, eq, ex):
                mAB, mA, mB = dimer_fwd(params, R, grid, eq * (1.0 if use_env else 0.0), ex)

                def r(x, m):
                    return jnp.sqrt(msk(x**2, m)) * HA

                dQ = VAB - VA - VB
                dM = mAB - mA - mB
                res = [r(mAB - VAB, gm), r(mA - VA, gm), r(mB - VB, gm), r(dM - dQ, gm), r(dQ, gm)]
                for s in (1.4, 1.8, 2.4):  # scale shells of the ESP grid (multiples of the vdW radius)
                    m = gm & (jnp.abs(sc - s) < 0.01)
                    res += [r(dM - dQ, m), r(dQ, m)]
                return jnp.stack(res + [msk(jnp.abs(dM - dQ), gm) * HA, msk(jnp.abs(dQ), gm) * HA])

            return jax.vmap(one)(b["R"], b["grid"], b["gm"], b["scale"], b[VAB_], b[VA_], b[VB_], b["EQ"], b["EX"])

        return evd

    evd_gas, evd_emb = make_evd(False), make_evd(True)

    @jax.jit
    def evm(params, b):
        return jax.vmap(lambda *x: mono_one(params, *x))(
            b["R"], b["grid"], b["gm"], b["mm"], b["esp"], b["dip"], b["qref"]
        )[3]

    def eval_dimers(fn, params, ix, chunk=16):
        return np.concatenate(
            [np.asarray(fn(params, {k: v[ix[s : s + chunk]] for k, v in J.items()})) for s in range(0, len(ix), chunk)]
        )

    def eval_mono(params, X, ix=None, chunk=50):
        m_ = X["R"].shape[0] if ix is None else len(ix)
        ix = np.arange(m_) if ix is None else ix
        return float(
            np.concatenate(
                [np.asarray(evm(params, {k: v[ix[s : s + chunk]] for k, v in X.items()})) for s in range(0, m_, chunk)]
            ).mean()
        )

    cols = [
        "rmse_AB",
        "rmse_A",
        "rmse_B",
        "rmse_dESP",
        "rms_dESP_QM",
        "dESP_s14",
        "QM_s14",
        "dESP_s18",
        "QM_s18",
        "dESP_s24",
        "QM_s24",
        "mae_dESP",
        "mae_QM",
    ]
    mind = np.array([np.linalg.norm(r[:n][:, None] - r[n:][None], axis=-1).min() for r in np.asarray(D["R"])])

    def agg(r):
        return {
            c: (float(r[:, j].mean()) if c.startswith("mae") else float(np.sqrt((r[:, j] ** 2).mean())))
            for j, c in enumerate(cols)
        }

    def summarize(params):
        res = {}
        for tag, fn in (("gas", evd_gas), ("env", evd_emb)):
            for sn, k in (("val", 1), ("test", 2)):
                ix = idx[k]
                if len(ix) == 0:
                    continue
                r = eval_dimers(fn, params, ix)
                c = D["centroid"][ix]
                d = {"n": len(r), **agg(r), "bins": {}}
                for lo, hi in ((3, 4), (4, 5), (5, 6), (6, 8.1)):  # centroid-distance bins (A)
                    m = (c >= lo) & (c < hi)
                    if m.sum():
                        d["bins"][f"{lo}-{min(hi, 8)}"] = dict(n=int(m.sum()), **agg(r[m]))
                m = mind[ix] >= 4.0
                if m.sum():
                    d["no_edge_4A"] = dict(n=int(m.sum()), **agg(r[m]))
                res[f"{tag}_{sn}"] = d
        res["mono_val"] = eval_mono(params, Mfull, mva)
        res["mono_test120"] = eval_mono(params, Mte)
        return res

    @jax.jit
    def far_one(params, R, grid, esp, r, mq, mx):
        q1, p1 = charges(params, R, *FI(R, mq, mx))
        q0, p0_ = charges(params, R, zV, zE)
        c = (R * wm[:, None]).sum(0) / wm.sum()
        res = []
        for q, p in ((q1, p1), (q0, p0_)):
            V = pot(q, p, grid)
            far = r >= 6.0
            sc = jnp.sum(jnp.where(far, V * esp, 0.0)) / jnp.sum(jnp.where(far, V * V, 0.0))
            res += [
                jnp.sqrt(jnp.mean((V - esp) ** 2)) * HA,
                sc,
                jnp.sqrt(jnp.sum(((q[:, None] * (p - c)).sum(0)) ** 2)) * DEBYE_PER_EANG,
                jnp.sqrt(jnp.sum(jnp.where(far, (sc * V - esp) ** 2, 0.0)) / jnp.sum(far)) * HA,
            ]
        return jnp.stack(res)

    def far_eval(params, sel=None):
        if not FAR:
            return {}
        sel = np.arange(len(EF)) if sel is None else sel
        r = np.array(
            [
                np.asarray(
                    far_one(params, EF[j]["R"], EF[j]["grid"], EF[j]["esp"], EF[j]["r"], EF[j]["mq"], EF[j]["mx"])
                )
                for j in sel
            ]
        )
        qd = np.array([EF[j]["dip"] for j in sel])
        return {
            "rmse_field": float(np.sqrt((r[:, 0] ** 2).mean())),
            "scale_field": float(r[:, 1].mean()),
            "dip_field_D": float(r[:, 2].mean()),
            "rmse_far_scaled_field": float(np.sqrt((r[:, 3] ** 2).mean())),
            "rmse_nofield": float(np.sqrt((r[:, 4] ** 2).mean())),
            "scale_nofield": float(r[:, 5].mean()),
            "dip_nofield_D": float(r[:, 6].mean()),
            "dip_QM_D": float(qd.mean()),
            "scale_field_sd": float(r[:, 1].std()),
        }

    r0 = summarize(params0)
    f0 = far_eval(params0)
    json.dump({**r0, "far": f0}, open(out / "start.json", "w"), indent=1)
    print(
        "START gas val/QMrms %.3f/%.3f  env val/QMrms %.3f/%.3f  mono %.3f"
        % (
            r0["gas_val"]["rmse_dESP"],
            r0["gas_val"]["rms_dESP_QM"],
            r0["env_val"]["rmse_dESP"],
            r0["env_val"]["rms_dESP_QM"],
            r0["mono_val"],
        ),
        flush=True,
    )
    base_mv = a.mono_ref if a.mono_ref > 0 else r0["mono_val"]

    def score(r):
        return (
            0.5 * (r["gas_val"]["rmse_dESP"] + r["env_val"]["rmse_dESP"])
            + 0.5 * max(0.0, r["mono_val"] - base_mv - 0.05)
            + 0.1 * 0.5 * (r["gas_val"]["rmse_AB"] + r["env_val"]["rmse_AB"])
        )

    def save(p):
        (out / "best_params.msgpack").write_bytes(
            serialization.msgpack_serialize(jax.tree_util.tree_map(np.asarray, p))
        )

    params = params0
    st = opt.init(params)
    rng = np.random.default_rng(a.seed)
    tr = idx[0]
    log = []
    best = 1e9
    best_ep = 0
    T0 = time.time()
    save(params)
    for ep in range(1, a.epochs + 1):
        lr = a.lr * (a.lr_floor + (1 - a.lr_floor) * 0.5 * (1 + math.cos(math.pi * (ep - 1) / a.epochs)))
        st[1].hyperparams["learning_rate"] = jnp.asarray(lr, jnp.float32)
        t0 = time.time()
        perm = rng.permutation(tr)
        tl = []
        for s in range(0, len(tr) - a.bs + 1, a.bs):
            ix = perm[s : s + a.bs]
            rix = rng.choice(mtr, a.rbs, replace=False)
            fb = {k: v[rng.choice(ctr, a.far_bs, replace=len(ctr) < a.far_bs)] for k, v in FB.items()} if FAR else FB
            params, st, L = step(
                params, st, {k: v[ix] for k, v in J.items()}, {k: v[rix] for k, v in Mfull.items()}, fb
            )
            tl.append(float(L))
        jax.block_until_ready(params)
        tt = time.time() - t0
        sm = summarize(params)
        sc = score(sm)
        rec = dict(
            epoch=ep,
            lr=lr,
            loss=float(np.mean(tl)),
            score=sc,
            t_epoch=tt,
            elapsed_h=(time.time() - T0) / 3600,
            gas_val=sm["gas_val"]["rmse_dESP"],
            gas_test=sm["gas_test"]["rmse_dESP"] if "gas_test" in sm else None,
            env_val=sm["env_val"]["rmse_dESP"],
            env_test=sm["env_test"]["rmse_dESP"] if "env_test" in sm else None,
            mono_val=sm["mono_val"],
        )
        if sc < best - 1e-4:
            best, best_ep = sc, ep
            save(params)
        if FAR and a.far_every and (ep % a.far_every == 0 or ep == 1):
            rec["far"] = far_eval(params)
            rec["far_test"] = far_eval(params, cte)
            rec["far_val"] = far_eval(params, cva)
        rec["best_epoch"] = best_ep
        log.append(rec)
        json.dump(log, open(out / "log.json", "w"), indent=1)
        fa = rec.get("far")
        fas = (
            (" | far scale %.3f dip %.2fD rmse %.3f" % (fa["scale_field"], fa["dip_field_D"], fa["rmse_field"]))
            if fa
            else ""
        )
        f3 = lambda v: "nan" if v is None else f"{v:.3f}"  # noqa: E731
        print(
            f"ep {ep:3d} lr {lr:.1e} gas val {rec['gas_val']:.3f} test {f3(rec['gas_test'])} | env val {rec['env_val']:.3f} test {f3(rec['env_test'])}"
            f" | mono {rec['mono_val']:.3f} | best@{best_ep} | {tt:.1f}s{fas}",
            flush=True,
        )
    pb = jax.tree_util.tree_map(jnp.asarray, serialization.msgpack_restore((out / "best_params.msgpack").read_bytes()))
    rf = summarize(pb)
    rf["far"] = far_eval(pb)
    rf["far_train"] = far_eval(pb, ctr)
    rf["far_val"] = far_eval(pb, cva)
    rf["far_test"] = far_eval(pb, cte)
    rf["best_epoch"] = best_ep
    rf["args"] = vars(a)
    json.dump(rf, open(out / "final.json", "w"), indent=1)
    (out / "model_config.json").write_text(
        json.dumps(
            dict(
                architecture=cfg,
                args=vars(a),
                best_epoch=best_ep,
                field=dict(
                    VSCALE=VSCALE,
                    ESCALE=ESCALE,
                    SIGMA=a.sigma,
                    source=a.source,
                    merged_graph=True,
                    inject_all=(a.inject == "both"),
                    field_terms=a.field_terms,
                ),
            ),
            indent=1,
        )
    )
    print(
        "FINAL",
        {k: round(rf[k]["rmse_dESP"], 3) for k in ("gas_val", "gas_test", "env_val", "env_test") if k in rf},
        "mono",
        round(rf["mono_val"], 3),
        flush=True,
    )


if __name__ == "__main__":
    main()
