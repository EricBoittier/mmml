"""Live CHARMM check: ``SKIPE ELEC IMEL`` is exact once every charge is zero.

All-ML MLpot registration zeroes every CHARMM partial charge and the ``vdw``
energy policy skips VDW/IMNB (JAX computes the intermolecular MM nonbond).
CHARMM then still runs its nonbond kernel (ENBFS8) over the primary and image
pair lists every step, for an ELEC/IMEL that is identically zero.
:func:`skip_redundant_charmm_elec` removes it; this test pins down that the
total energy and forces are unchanged, on a periodic ethanol box with images.

Runs in a fresh Python process so CHARMM starts from a clean state.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from tests.conftest import can_import_pycharmm

pytestmark = [
    pytest.mark.pycharmm,
    pytest.mark.skipif(
        not can_import_pycharmm(),
        reason="pycharmm / libcharmm not available",
    ),
]

_REPO_ROOT = Path(__file__).resolve().parents[3]

_WORKER = r"""
import argparse
import json
import sys

import numpy as np

import mmml.interfaces.pycharmmInterface.import_pycharmm  # noqa: F401
import pycharmm
import pycharmm.coor as coor
import pycharmm.energy as energy
import pycharmm.generate as gen
import pycharmm.read as read

from mmml.interfaces.pycharmmInterface.charmm_levels import charmm_relaxed_bomlev
from mmml.interfaces.pycharmmInterface.mlpot import charmm_energy_policy as cep
from mmml.interfaces.pycharmmInterface.nbonds_config import read_cgenff_toppar

# One ethanol in RTF order C1 O1 HO1 H11 H12 C2 H21 H22 H23.
X = [0.0, -0.5, -1.45, -0.35, -0.35, 1.5, 1.9, 1.9, 1.9]
Y = [0.0, 1.3, 1.2, -0.55, -0.55, 0.0, 1.0, -0.5, -0.5]
Z = [0.0, 0.3, 0.1, 0.9, -0.9, 0.0, 0.0, 0.85, -0.85]
L = 16.0
N_SIDE = 2  # 8 ethanols, 5 A apart, wrapped into a 16 A cube

read_cgenff_toppar()
nmol = N_SIDE ** 3
with charmm_relaxed_bomlev():
    read.sequence_string(" ".join(["ETOH"] * nmol))
    gen.new_segment(seg_name="LIG", setup_ic=True)
xs, ys, zs = [], [], []
rng = np.random.default_rng(3)
for i in range(N_SIDE):
    for j in range(N_SIDE):
        for k in range(N_SIDE):
            off = np.array([i, j, k]) * (L / N_SIDE) - L / 2 + 2.0 + rng.normal(0, 0.2, 3)
            xs += [v + off[0] for v in X]
            ys += [v + off[1] for v in Y]
            zs += [v + off[2] for v in Z]
pos = coor.get_positions()
pos.iloc[:, 0] = xs
pos.iloc[:, 1] = ys
pos.iloc[:, 2] = zs
coor.set_positions(pos)

# Every command uppercase: eval_charmm_script does no case folding.
for cmd in (
    f"CRYSTAL DEFINE CUBIC {L} {L} {L} 90.0 90.0 90.0",
    "CRYSTAL BUILD CUTOFF 9.0 NOPERATIONS 0",
    "IMAGE BYRESID XCEN 0.0 YCEN 0.0 ZCEN 0.0 SELE ALL END",
):
    pycharmm.lingo.charmm_script(cmd)
NBONDS = (
    "NBONDS CUTNB 8.0 CUTIM 8.0 CTOFNB 7.0 CTONNB 6.0 NBXMOD 5 ATOM CDIE "
    "VATOM VSWITCH SWITCH INBFRQ -1 IMGFRQ -1"
)
pycharmm.lingo.charmm_script(NBONDS)

TERMS = ("VDW", "ELEC", "IMNB", "IMEL")


def ener():
    pycharmm.lingo.charmm_script("ENER")
    out = {k: float(energy.get_term_by_name(k)) for k in TERMS}
    out["ENER"] = float(energy.get_total())
    f = coor.get_forces()
    out["grad"] = np.stack([f["dx"], f["dy"], f["dz"]], axis=1).tolist()
    return out


charged = ener()
# All-ML state: MLpot registration zeroes every charge; the vdw policy SKIPEs VDW/IMNB.
pycharmm.psf.set_charge([0.0] * int(pycharmm.psf.get_natom()))
policies = cep.resolve_charmm_energy_term_policies(
    argparse.Namespace(mm_nonbond_mode="jax_mic", charmm_zero_energy_terms=None)
)
cep._skip_policy_terms(policies, verbose=False)
before = ener()
skipped = cep.skip_redundant_charmm_elec(policies, verbose=True)
after = ener()
payload = {
    "charged": charged,
    "before": before,
    "after": after,
    "skipped": skipped,
    "registry": sorted(cep.charmm_skipped_terms()),
}
sys.stdout.flush()
print("\n@@JSON " + json.dumps(payload), flush=True)
"""


def _run_worker() -> dict:
    env = dict(os.environ)
    env.setdefault("MMML_NO_CHARMM_MPI", "1")
    env.setdefault("MMML_NO_MPI_RERUN", "1")
    env.pop("MMML_MLPOT_KEEP_CHARMM_ELEC", None)
    env["PYTHONPATH"] = os.pathsep.join(
        p for p in (str(_REPO_ROOT), env.get("PYTHONPATH", "")) if p
    )
    proc = subprocess.run(
        [sys.executable, "-c", _WORKER],
        capture_output=True,
        text=True,
        env=env,
        cwd=str(_REPO_ROOT),
        timeout=600,
    )
    assert proc.returncode == 0, proc.stderr[-4000:]
    # CHARMM writes its own log lines to stdout; the payload line is tagged.
    tagged = [ln for ln in proc.stdout.splitlines() if ln.startswith("@@JSON ")]
    assert tagged, proc.stdout[-4000:] + proc.stderr[-4000:]
    return json.loads(tagged[-1][len("@@JSON ") :])


def test_skipe_elec_imel_is_exact_for_zero_charges():
    import numpy as np

    out = _run_worker()
    charged, before, after = out["charged"], out["before"], out["after"]

    # Sanity: the periodic setup exercises the Coulomb kernel on primary and images.
    assert abs(charged["ELEC"]) > 1.0e-3
    assert abs(charged["IMEL"]) > 1.0e-6

    assert out["skipped"] == ["ELEC", "IMEL"]
    assert {"VDW", "IMNB", "ELEC", "IMEL"} <= set(out["registry"])

    # Zero charges: CHARMM's ELEC/IMEL are exactly zero, so skipping them is free.
    assert before["ELEC"] == 0.0 and before["IMEL"] == 0.0
    assert after["ENER"] == pytest.approx(before["ENER"], abs=1.0e-10)
    np.testing.assert_allclose(
        np.asarray(after["grad"]), np.asarray(before["grad"]), rtol=0.0, atol=1.0e-10
    )
