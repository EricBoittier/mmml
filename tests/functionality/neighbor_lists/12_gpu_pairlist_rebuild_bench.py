"""CPU vs GPU MM pair-list rebuild through the real update_mm_pairs closure.

ETOH:181 in a 26 A box (mm_switch_on 6, width 1.5, skin 0.25, mm_r_min 4.05, as in
the PyCHARMM MLpot NVE runs). CHARMM PSF/parameters are mocked: only the pair list
is timed. Each timed call is update(positions, force_rebuild=True) until the
padded JAX pair arrays are ready, i.e. what the MLpot callback pays per rebuild.

    MMML_MM_NL_DEVICE is set per mode (cpu, auto); run on a GPU node with
    pip install 'mmml[nl-gpu]':

    python tests/functionality/neighbor_lists/12_gpu_pairlist_rebuild_bench.py \
        path/to/etoh_26A/model_petmin.crd out.json   # N_REP=100 N_WARM=5

Exits nonzero if any frame's GPU pair list differs from CPU, or if auto falls
back to the CPU rebuild (this script is a GPU bench, not a fallback probe).
"""
import json
import os
import statistics
import sys
import time
from unittest.mock import MagicMock, patch
import numpy as np
import jax
import jax.numpy as jnp
jax.config.update("jax_enable_x64", True)
from mmml.interfaces.pycharmmInterface.mm_energy_forces import build_mm_energy_forces_fn

CRD = sys.argv[1]
rows = [l.split() for l in open(CRD) if not l.startswith("*")][1:]
R0 = np.array([[float(x) for x in r[4:7]] for r in rows])
n, APM, L = len(R0), 9, 26.0
n_mono = n // APM
N_WARM, N = int(os.environ.get("N_WARM", 5)), int(os.environ.get("N_REP", 50))


def build():
    fake_psf = MagicMock(); fake_psf.get_charges.return_value = np.zeros(n); fake_psf.get_iac.return_value = np.zeros(n, np.int32)
    fake_param = MagicMock(); fake_param.get_atc.return_value = ["CG321"]
    rtf = MagicMock(); rtf.readlines.return_value = ["ATOM C1 CG321 -0.1\n"]
    prm = MagicMock(); prm.readlines.return_value = ["CG321 0.0 -0.05 1.6 0.0 -0.01 1.9\n"]
    mod = "mmml.interfaces.pycharmmInterface.mm_energy_forces"
    with patch("pycharmm.psf", fake_psf), patch("pycharmm.param", fake_param), patch(f"{mod}.open", side_effect=[rtf, prm]), \
         patch(f"{mod}._get_actual_psf_charges", return_value=np.zeros(n)), patch(f"{mod}.CGENFF_PRM", "/dev/null"), patch(f"{mod}.CGENFF_RTF", "/dev/null"):
        return build_mm_energy_forces_fn(
            R0, total_atoms=n, n_monomers=n_mono, monomer_offsets=np.arange(0, n + 1, APM, dtype=np.int32),
            atoms_per_monomer_list=[APM] * n_mono, lambda_monomer=np.ones(n_mono), ml_switch_width=1.0,
            mm_switch_on=6.0, mm_switch_width=1.5, mm_r_min=4.05, jax_md_skin_distance=0.25,
            pbc_cell=np.diag([L] * 3), use_jax_md_neighbor_list=False, mm_nl_backend="vesin", lr_solver="mic",
            defer_xla_gpu_warmup=True)


def _valid_pair_keys(pidx, pmask):
    pidx, pmask = np.asarray(pidx), np.asarray(pmask)
    keep = pmask > 0
    return pidx[keep, 0].astype(np.int64) * (n + 1) + pidx[keep, 1].astype(np.int64)


rng = np.random.default_rng(0)
frames = [R0 + rng.normal(scale=0.02, size=R0.shape) for _ in range(N_WARM + N)]
res = {}
failures = []
cpu_keys = []
gpu = jax.devices("gpu")[0]
for mode, dev_input in (("cpu", False), ("auto", False), ("auto", True)):
    os.environ["MMML_MM_NL_DEVICE"] = mode
    _, update = build()
    ts = []
    n_mismatch = 0
    for k, R in enumerate(frames):
        pos = jax.device_put(jnp.asarray(R), gpu) if dev_input else R
        jax.block_until_ready(pos)
        t0 = time.perf_counter()
        pidx, pmask = update(pos, force_rebuild=True)
        jax.block_until_ready((pidx, pmask))
        dt = (time.perf_counter() - t0) * 1e3
        if k >= N_WARM:
            ts.append(dt)
        keys = _valid_pair_keys(pidx, pmask)
        if mode == "cpu":
            cpu_keys.append(keys)
        elif not np.array_equal(keys, cpu_keys[k]):
            n_mismatch += 1
            failures.append(
                f"pair mismatch {mode}{'/dev' if dev_input else '/host'} frame {k}: "
                f"n_gpu={keys.size} n_cpu={cpu_keys[k].size}"
            )
    st = update.get_stats()
    key = f"{mode}{'_devinput' if dev_input else ''}"
    gpu_rebuilds = int(st.get("gpu_rebuilds") or 0)
    cpu_rebuilds = int(st.get("cpu_rebuilds") or 0)
    fallbacks = int(st.get("fallbacks") or 0)
    res[key] = dict(
        mean=statistics.mean(ts), std=statistics.stdev(ts), median=statistics.median(ts), n=len(ts),
        gpu_rebuilds=gpu_rebuilds, cpu_rebuilds=cpu_rebuilds, fallbacks=fallbacks,
        capacity=int(pidx.shape[0]), n_valid=int(np.asarray(pmask).sum()),
        n_frames=len(frames), n_mismatch=n_mismatch,
    )
    print(key, {k: (round(v, 3) if isinstance(v, float) else v) for k, v in res[key].items()}, flush=True)
    if mode == "auto":
        if gpu_rebuilds == 0 or cpu_rebuilds > 0 or fallbacks > 0:
            failures.append(
                f"{key}: expected GPU rebuilds only, got gpu={gpu_rebuilds} "
                f"cpu={cpu_rebuilds} fallbacks={fallbacks}"
            )
        res[f"identical_{key}"] = n_mismatch == 0
        print("identical to cpu", key, n_mismatch == 0, f"checked {len(frames)} frames", flush=True)

out_path = sys.argv[2] if len(sys.argv) > 2 else "bench.json"
res["ok"] = not failures
res["failures"] = failures
json.dump(res, open(out_path, "w"), indent=1)
if failures:
    print("BENCH FAILED:", *failures, sep="\n  ", flush=True)
    sys.exit(1)
print("BENCH OK", flush=True)
