"""CPU vs GPU MM pair-list rebuild through the real update_mm_pairs closure.

ETOH:181 in a 26 A box (mm_switch_on 6, width 1.5, skin 0.25, mm_r_min 4.05, as in
the PyCHARMM MLpot NVE runs). CHARMM PSF/parameters are mocked: only the pair list
is timed. Each timed call is update(positions, force_rebuild=True) until the
padded JAX pair arrays are ready, i.e. what the MLpot callback pays per rebuild.

    MMML_MM_NL_DEVICE is set per mode (cpu, auto); run on a GPU node with
    pip install 'mmml[nl-gpu]':

    python tests/functionality/neighbor_lists/12_gpu_pairlist_rebuild_bench.py \
        path/to/etoh_26A/model_petmin.crd out.json   # N_REP=100 N_WARM=5
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

rng = np.random.default_rng(0)
frames = [R0 + rng.normal(scale=0.02, size=R0.shape) for _ in range(N_WARM + N)]
res, pairs = {}, {}
gpu = jax.devices("gpu")[0]
for mode, dev_input in (("cpu", False), ("auto", False), ("auto", True)):
    os.environ["MMML_MM_NL_DEVICE"] = mode
    _, update = build()
    ts = []
    for k, R in enumerate(frames):
        pos = jax.device_put(jnp.asarray(R), gpu) if dev_input else R
        jax.block_until_ready(pos)
        t0 = time.perf_counter()
        pidx, pmask = update(pos, force_rebuild=True)
        jax.block_until_ready((pidx, pmask))
        dt = (time.perf_counter() - t0) * 1e3
        if k >= N_WARM:
            ts.append(dt)
        if k == N_WARM:
            pairs[(mode, dev_input)] = (np.asarray(pidx), np.asarray(pmask))
    st = update.get_stats()
    key = f"{mode}{'_devinput' if dev_input else ''}"
    res[key] = dict(mean=statistics.mean(ts), std=statistics.stdev(ts), median=statistics.median(ts), n=len(ts),
                    gpu_rebuilds=st.get("gpu_rebuilds"), cpu_rebuilds=st.get("cpu_rebuilds"), capacity=int(pidx.shape[0]),
                    n_valid=int(np.asarray(pmask).sum()))
    print(key, {k: (round(v, 3) if isinstance(v, float) else v) for k, v in res[key].items()}, flush=True)
ic, mc = pairs[("cpu", False)]
for k in (("auto", False), ("auto", True)):
    ig, mg = pairs[k]
    same = np.array_equal(mc, mg) and np.array_equal(ic[mc > 0], ig[mg > 0])
    print("identical to cpu", k, same, flush=True)
    res[f"identical_{k[0]}_{'dev' if k[1] else 'host'}"] = bool(same)
json.dump(res, open(sys.argv[2] if len(sys.argv) > 2 else "bench.json", "w"), indent=1)
