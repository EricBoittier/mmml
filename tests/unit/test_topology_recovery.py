"""Unit tests for safe inplace topology recovery."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from mmml.interfaces.pycharmmInterface.mlpot.topology_recovery import (
    TopologyFingerprint,
    allow_psf_delete_reload,
    attach_topology_recovery_state,
    ensure_composition_unchanged,
    load_topology_sidecar,
    save_topology_sidecar,
    topology_fingerprint_path,
)


def test_per_atom_resids_expands_from_residue_table():
    import sys

    from mmml.interfaces.pycharmmInterface.mlpot.topology_recovery import _per_atom_resids

    mock_atom_info = MagicMock()
    mock_atom_info.get_res_ids.return_value = ["1", "1", "2", "2", "3"]
    stub_pycharmm = MagicMock()
    stub_pycharmm.atom_info = mock_atom_info
    with patch.dict(
        sys.modules,
        {
            "pycharmm": stub_pycharmm,
            "pycharmm.atom_info": mock_atom_info,
        },
    ):
        assert _per_atom_resids(5) == (1, 1, 2, 2, 3)


def test_topology_fingerprint_path_sidecar():
    assert topology_fingerprint_path("/tmp/cluster_for_vmd_dcm.psf") == Path(
        "/tmp/cluster_for_vmd_dcm.topology.json"
    )


def test_save_and_load_topology_sidecar(tmp_path):
    fp = TopologyFingerprint(
        natom=3,
        nres=1,
        nseg=1,
        atom_names=("C1", "H1", "H2"),
        resids=(1, 1, 1),
    )
    path = save_topology_sidecar(
        tmp_path / "cluster.topology.json",
        fp,
        pre_mlpot_iblo=[0, 1, 2],
        pre_mlpot_inb=[4, 5],
    )
    loaded = load_topology_sidecar(path)
    assert loaded is not None
    assert loaded.fingerprint == fp
    assert loaded.pre_mlpot_iblo == (0, 1, 2)
    assert loaded.pre_mlpot_inb == (4, 5)
    raw = json.loads(path.read_text(encoding="utf-8"))
    assert raw["natom"] == 3


def test_load_topology_sidecar_rejects_stale_resid_length(tmp_path):
    path = tmp_path / "stale.topology.json"
    path.write_text(
        json.dumps(
            {
                "natom": 1000,
                "nres": 200,
                "nseg": 1,
                "atom_names": ["C"] * 1000,
                "resids": [1] * 100,
            }
        ),
        encoding="utf-8",
    )
    assert load_topology_sidecar(path) is None


def test_attach_topology_recovery_state_sets_ctx_fields(tmp_path):
    sidecar_fp = TopologyFingerprint(
        natom=2,
        nres=1,
        nseg=1,
        atom_names=("C", "H"),
        resids=(1, 1),
    )
    live_fp = TopologyFingerprint(
        natom=2,
        nres=1,
        nseg=1,
        atom_names=("C", "H"),
        resids=(1, 1),
    )
    psf = tmp_path / "cluster_for_vmd_test.psf"
    psf.write_text("* psf\n", encoding="utf-8")
    save_topology_sidecar(
        topology_fingerprint_path(psf),
        sidecar_fp,
        pre_mlpot_iblo=[0, 1],
        pre_mlpot_inb=[2],
    )
    ctx = MagicMock()
    ctx.topology_psf_path = None
    ctx.topology_fingerprint = None
    ctx.pre_mlpot_iblo = None
    ctx.pre_mlpot_inb = None

    with patch(
        "mmml.interfaces.pycharmmInterface.mlpot.topology_recovery.capture_topology_fingerprint_from_charmm",
        return_value=live_fp,
    ):
        attach_topology_recovery_state(ctx, psf)

    assert ctx.topology_psf_path == psf.resolve()
    assert ctx.topology_fingerprint == live_fp
    assert ctx.pre_mlpot_iblo == [0, 1]
    assert ctx.pre_mlpot_inb == [2]


def test_ensure_composition_unchanged_raises_on_mismatch():
    fp = TopologyFingerprint(
        natom=2,
        nres=1,
        nseg=1,
        atom_names=("C", "H"),
        resids=(1, 1),
    )
    live = TopologyFingerprint(
        natom=3,
        nres=1,
        nseg=1,
        atom_names=("C", "H", "O"),
        resids=(1, 1, 2),
    )
    with patch(
        "mmml.interfaces.pycharmmInterface.mlpot.topology_recovery.capture_topology_fingerprint_from_charmm",
        return_value=live,
    ), pytest.raises(RuntimeError, match="MLpot registration"):
        ensure_composition_unchanged(fp, context="test")


def test_allow_psf_delete_reload_env(monkeypatch):
    monkeypatch.delenv("MMML_ALLOW_PSF_DELETE_RELOAD", raising=False)
    assert allow_psf_delete_reload() is False
    monkeypatch.setenv("MMML_ALLOW_PSF_DELETE_RELOAD", "1")
    assert allow_psf_delete_reload() is True


def test_run_bonded_recovery_inplace_never_deletes_psf(tmp_path):
    from mmml.interfaces.pycharmmInterface.mlpot.dynamics import BondedMmMiniConfig
    from mmml.interfaces.pycharmmInterface.mlpot.setup import MlpotContext
    from mmml.interfaces.pycharmmInterface.mlpot.topology_recovery import (
        BondedRecoveryMode,
        run_bonded_recovery_inplace,
    )

    fp = TopologyFingerprint(
        natom=1,
        nres=1,
        nseg=1,
        atom_names=("C",),
        resids=(1,),
    )
    psf = tmp_path / "cluster_for_vmd_x.psf"
    psf.write_text("* psf\n", encoding="utf-8")
    save_topology_sidecar(topology_fingerprint_path(psf), fp)

    ctx = MlpotContext(
        mlpot=MagicMock(),
        pyCModel=MagicMock(),
        params=None,
        model=None,
        topology_fingerprint=fp,
        topology_psf_path=psf,
    )
    cfg = BondedMmMiniConfig(nstep_sd=0, verbose=False)
    mock_py = MagicMock()

    with patch(
        "mmml.interfaces.pycharmmInterface.mlpot.topology_recovery.capture_topology_fingerprint_from_charmm",
        return_value=fp,
    ), patch(
        "mmml.interfaces.pycharmmInterface.mlpot.dynamics._with_mlpot_detached",
        side_effect=lambda _ctx, fn: fn(),
    ), patch(
        "mmml.interfaces.pycharmmInterface.mlpot.topology_recovery._apply_recovery_block",
    ), patch(
        "mmml.interfaces.pycharmmInterface.mlpot.topology_recovery.prepare_rescue_lists_safe",
    ), patch(
        "mmml.interfaces.pycharmmInterface.mlpot.dynamics._import_pycharmm_modules",
        return_value=(mock_py, MagicMock(), MagicMock(), MagicMock()),
    ), patch(
        "mmml.interfaces.pycharmmInterface.mlpot.cli_common.charmm_grms",
        return_value=0.1,
    ), patch(
        "mmml.interfaces.pycharmmInterface.charmm_levels.run_charmm_script_quiet",
    ):
        run_bonded_recovery_inplace(
            ctx,
            BondedRecoveryMode.BONDED_ONLY,
            cfg,
            topology_psf=psf,
        )

    script_calls = [
        str(c)
        for c in mock_py.lingo.charmm_script.call_args_list
        if mock_py.lingo.charmm_script.called
    ]
    joined = " ".join(script_calls)
    assert "DELETE ATOM" not in joined
