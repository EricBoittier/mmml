"""Per-type MM LJ σ/ε scales for hybrid train + MD ATC remapping."""

from __future__ import annotations

import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from mmml.models.mm_lj_scales import (
    MM_LJ_EPSILON_SCALE_KEY,
    MM_LJ_SIGMA_SCALE_KEY,
    apply_mm_lj_scales,
    attach_mm_lj_scales,
    cgenff_type_names_from_prm,
    load_mm_lj_scales_sidecar,
    mm_lj_scales_metadata,
    resolve_md_lj_scales,
    scales_to_atc,
    split_mm_lj_scale_params,
    write_mm_lj_scales_into_hybrid_mm_json,
)


def test_cgenff_type_names_match_load_reference_master_tables():
    """Scale vectors must line up with lattice/MD master LJ tables (incl. ions)."""
    from mmml.data.cgenff_dataset import load_reference

    ref = load_reference()
    names = cgenff_type_names_from_prm()
    assert len(names) == len(ref.sigmas) == len(ref.epsilons)
    assert names == [n for n, _ in sorted(ref.nb_map.items(), key=lambda kv: kv[1])]


def test_apply_unit_scales_identity():
    sig = jnp.array([3.5, 2.1])
    eps = jnp.array([0.08, 0.02])
    s2, e2 = apply_mm_lj_scales(sig, eps, jnp.ones(2), jnp.ones(2))
    np.testing.assert_allclose(s2, sig)
    np.testing.assert_allclose(e2, eps)


def test_apply_scales_and_include_lj_off():
    sig = jnp.array([3.5, 2.1])
    eps = jnp.array([0.08, 0.02])
    s2, e2 = apply_mm_lj_scales(
        sig, eps, jnp.array([1.1, 0.9]), jnp.array([2.0, 0.5]), include_lj=False
    )
    np.testing.assert_allclose(s2, sig * jnp.array([1.1, 0.9]))
    np.testing.assert_allclose(e2, jnp.zeros(2))


def test_attach_and_split_params():
    base = {"params": {"w": jnp.array(1.0)}}
    attached = attach_mm_lj_scales(base, 3)
    assert attached[MM_LJ_SIGMA_SCALE_KEY].shape == (3,)
    assert attached[MM_LJ_EPSILON_SCALE_KEY].shape == (3,)
    model, sig, eps = split_mm_lj_scale_params(attached)
    assert MM_LJ_SIGMA_SCALE_KEY not in model
    assert "params" in model
    np.testing.assert_allclose(sig, jnp.ones(3))
    np.testing.assert_allclose(eps, jnp.ones(3))


def test_scales_to_atc_by_name():
    ep, sig = scales_to_atc(
        ["CG2O1", "HGR52", "DEFAULT"],
        [1.1, 0.9, 1.0],
        [2.0, 0.5, 1.0],
        ["HGR52", "CG2O1", "OTHER"],
    )
    np.testing.assert_allclose(sig, [0.9, 1.1, 1.0])
    np.testing.assert_allclose(ep, [0.5, 2.0, 1.0])


def test_hybrid_mm_json_round_trip(tmp_path: Path):
    path = tmp_path / "hybrid_mm.json"
    write_mm_lj_scales_into_hybrid_mm_json(
        path,
        type_names=["A", "B"],
        sigma_scale=[1.04, 0.96],
        epsilon_scale=[1.5, 0.5],
    )
    raw = json.loads(path.read_text())
    assert raw["learn_mm_lj_scales"] is True
    loaded = load_mm_lj_scales_sidecar(path)
    assert loaded is not None
    np.testing.assert_allclose(loaded["mm_lj_sigma_scale"], [1.04, 0.96])
    ep, sig = resolve_md_lj_scales(
        scales_file=path,
        atc_names=["B", "A"],
    )
    assert ep is not None and sig is not None
    np.testing.assert_allclose(sig, [0.96, 1.04])
    np.testing.assert_allclose(ep, [0.5, 1.5])


def test_metadata_without_scales():
    meta = mm_lj_scales_metadata(learn_mm_lj_scales=False)
    assert meta == {"learn_mm_lj_scales": False}


def test_hybrid_forward_unit_scales_match_baseline():
    from mmml.models.cgenff_mm import cgenff_mm_energy
    from mmml.models.hybrid_energy import hybrid_forward
    from mmml.data.units import KCAL_MOL_TO_EV

    SIG = jnp.array([3.6527, 2.3876])
    EPS = jnp.array([0.0780, 0.0240])
    # Legacy handoff with a short MM onset so intermolecular LJ is active.
    KW = dict(
        mm_switch_on=3.0,
        mm_switch_width=2.0,
        ml_switch_width=1.0,
        complementary_handoff=False,
    )

    def fake_apply(params, *, atomic_numbers, positions, dst_idx, src_idx,
                   batch_segments, batch_size, batch_mask, atom_mask):
        e = jnp.sum(atom_mask) * jnp.asarray(-1.0)
        return {
            "energy": e.reshape(batch_size, 1),
            "forces": jnp.zeros_like(positions),
        }

    n = 4
    # Closest inter-monomer distance ~3.5 A — nonzero LJ under the switch.
    pos = jnp.array(
        [[0.0, 0, 0], [1.0, 0, 0], [3.5, 0, 0], [4.5, 0, 0]], dtype=jnp.float32
    )
    mid = jnp.array([0, 0, 1, 1])
    tidx = jnp.array([0, 1, 0, 1])
    chg = jnp.zeros(n)
    atom_mask = jnp.ones(n, dtype=jnp.float32)
    idx = jnp.arange(n)
    dst, src = jnp.meshgrid(idx, idx, indexing="ij")
    dst, src = dst.reshape(-1), src.reshape(-1)
    keep = (dst != src).astype(jnp.float32)
    batch = {
        "R": pos,
        "Z": jnp.array([6, 1, 6, 1]),
        "mol_id": mid.reshape(1, n),
        "cgenff_type_idx": tidx.reshape(1, n),
        "cgenff_charge": chg.reshape(1, n),
        "atom_mask": atom_mask,
        "batch_mask": keep,
        "dst_idx": dst,
        "src_idx": src,
        "batch_segments": jnp.zeros(n, dtype=jnp.int32),
    }

    # Direct MM check first (charges zero → pure LJ).
    e_mm_base = float(
        KCAL_MOL_TO_EV
        * cgenff_mm_energy(
            pos,
            tidx,
            mid,
            chg,
            SIG,
            EPS,
            mm_switch_on=3.0,
            mm_switch_width=2.0,
            ml_switch_width=1.0,
            complementary_handoff=False,
        )
    )
    assert abs(e_mm_base) > 1e-6

    base = hybrid_forward(fake_apply, {}, batch, 1, SIG, EPS, **KW)
    ones = hybrid_forward(
        fake_apply,
        {},
        batch,
        1,
        SIG,
        EPS,
        learn_mm_lj_scales=True,
        mm_lj_sigma_scale=jnp.ones(2),
        mm_lj_epsilon_scale=jnp.ones(2),
        **KW,
    )
    np.testing.assert_allclose(base["energy"], ones["energy"], rtol=1e-6)
    np.testing.assert_allclose(base["e_mm"], ones["e_mm"], rtol=1e-6)

    scaled = hybrid_forward(
        fake_apply,
        {},
        batch,
        1,
        SIG,
        EPS,
        learn_mm_lj_scales=True,
        mm_lj_sigma_scale=jnp.array([1.2, 0.8]),
        mm_lj_epsilon_scale=jnp.array([1.5, 0.5]),
        **KW,
    )
    assert not np.allclose(base["e_mm"], scaled["e_mm"])


def test_lj_scale_gradients_nonzero():
    from mmml.models.hybrid_energy import hybrid_forward

    SIG = jnp.array([3.6527, 2.3876])
    EPS = jnp.array([0.0780, 0.0240])
    KW = dict(
        mm_switch_on=3.0,
        mm_switch_width=2.0,
        ml_switch_width=1.0,
        complementary_handoff=False,
    )

    def fake_apply(params, *, atomic_numbers, positions, dst_idx, src_idx,
                   batch_segments, batch_size, batch_mask, atom_mask):
        e = jnp.sum(atom_mask) * jnp.asarray(-1.0)
        return {
            "energy": e.reshape(batch_size, 1),
            "forces": jnp.zeros_like(positions),
        }

    n = 4
    pos = jnp.array(
        [[0.0, 0, 0], [1.0, 0, 0], [3.5, 0, 0], [4.5, 0, 0]], dtype=jnp.float32
    )
    mid = jnp.array([0, 0, 1, 1])
    tidx = jnp.array([0, 1, 0, 1])
    chg = jnp.zeros(n)
    atom_mask = jnp.ones(n, dtype=jnp.float32)
    idx = jnp.arange(n)
    dst, src = jnp.meshgrid(idx, idx, indexing="ij")
    dst, src = dst.reshape(-1), src.reshape(-1)
    keep = (dst != src).astype(jnp.float32)
    batch = {
        "R": pos,
        "Z": jnp.array([6, 1, 6, 1]),
        "mol_id": mid.reshape(1, n),
        "cgenff_type_idx": tidx.reshape(1, n),
        "cgenff_charge": chg.reshape(1, n),
        "atom_mask": atom_mask,
        "batch_mask": keep,
        "dst_idx": dst,
        "src_idx": src,
        "batch_segments": jnp.zeros(n, dtype=jnp.int32),
    }

    def loss(scales):
        sig_s, eps_s = scales
        out = hybrid_forward(
            fake_apply,
            {},
            batch,
            1,
            SIG,
            EPS,
            learn_mm_lj_scales=True,
            mm_lj_sigma_scale=sig_s,
            mm_lj_epsilon_scale=eps_s,
            **KW,
        )
        return jnp.sum(out["energy"])

    g_sig, g_eps = jax.grad(loss)((jnp.ones(2), jnp.ones(2)))
    assert np.isfinite(g_sig).all() and np.isfinite(g_eps).all()
    assert float(jnp.sum(jnp.abs(g_sig)) + jnp.sum(jnp.abs(g_eps))) > 0.0


def test_epsilon_only_scale_changes_energy_sigma_only_changes_shape():
    """ε scale multiplies well depth; σ scale shifts the LJ length scale."""
    from mmml.models.cgenff_mm import cgenff_mm_energy
    from mmml.models.mm_lj_scales import apply_mm_lj_scales

    SIG = jnp.array([3.6527, 2.3876])
    EPS = jnp.array([0.0780, 0.0240])
    pos = jnp.array(
        [[0.0, 0, 0], [1.0, 0, 0], [3.5, 0, 0], [4.5, 0, 0]], dtype=jnp.float32
    )
    mid = jnp.array([0, 0, 1, 1])
    tidx = jnp.array([0, 1, 0, 1])
    chg = jnp.zeros(4)
    kw = dict(
        mm_switch_on=3.0,
        mm_switch_width=2.0,
        ml_switch_width=1.0,
        complementary_handoff=False,
    )

    def e_mm(sig, eps):
        return float(
            cgenff_mm_energy(pos, tidx, mid, chg, sig, eps, **kw)
        )

    base = e_mm(SIG, EPS)
    sig2, eps2 = apply_mm_lj_scales(SIG, EPS, jnp.ones(2), jnp.array([2.0, 2.0]))
    doubled_eps = e_mm(sig2, eps2)
    assert doubled_eps == pytest.approx(2.0 * base, rel=1e-5)

    sig3, eps3 = apply_mm_lj_scales(SIG, EPS, jnp.array([1.3, 1.3]), jnp.ones(2))
    shifted_sig = e_mm(sig3, eps3)
    assert shifted_sig != pytest.approx(base, rel=1e-3)


def test_params_leaf_path_through_hybrid_forward():
    """Scales attached on the Optax pytree are picked up when learn flag is on."""
    from mmml.models.hybrid_energy import hybrid_forward
    from mmml.models.mm_lj_scales import attach_mm_lj_scales

    SIG = jnp.array([3.6527, 2.3876])
    EPS = jnp.array([0.0780, 0.0240])
    KW = dict(
        mm_switch_on=3.0,
        mm_switch_width=2.0,
        ml_switch_width=1.0,
        complementary_handoff=False,
        learn_mm_lj_scales=True,
    )

    def fake_apply(params, *, atomic_numbers, positions, dst_idx, src_idx,
                   batch_segments, batch_size, batch_mask, atom_mask):
        assert MM_LJ_SIGMA_SCALE_KEY not in params
        e = jnp.sum(atom_mask) * jnp.asarray(-1.0)
        return {"energy": e.reshape(batch_size, 1), "forces": jnp.zeros_like(positions)}

    n = 4
    pos = jnp.array(
        [[0.0, 0, 0], [1.0, 0, 0], [3.5, 0, 0], [4.5, 0, 0]], dtype=jnp.float32
    )
    mid = jnp.array([0, 0, 1, 1])
    tidx = jnp.array([0, 1, 0, 1])
    atom_mask = jnp.ones(n, dtype=jnp.float32)
    idx = jnp.arange(n)
    dst, src = jnp.meshgrid(idx, idx, indexing="ij")
    dst, src = dst.reshape(-1), src.reshape(-1)
    batch = {
        "R": pos,
        "Z": jnp.array([6, 1, 6, 1]),
        "mol_id": mid.reshape(1, n),
        "cgenff_type_idx": tidx.reshape(1, n),
        "cgenff_charge": jnp.zeros(n).reshape(1, n),
        "atom_mask": atom_mask,
        "batch_mask": (dst != src).astype(jnp.float32),
        "dst_idx": dst,
        "src_idx": src,
        "batch_segments": jnp.zeros(n, dtype=jnp.int32),
    }
    params = attach_mm_lj_scales(
        {"params": {}},
        2,
        sigma_scale=np.array([1.0, 1.0]),
        epsilon_scale=np.array([1.5, 1.5]),
    )
    out = hybrid_forward(fake_apply, params, batch, 1, SIG, EPS, **KW)
    base = hybrid_forward(
        fake_apply,
        {"params": {}},
        batch,
        1,
        SIG,
        EPS,
        learn_mm_lj_scales=True,
        mm_lj_sigma_scale=jnp.ones(2),
        mm_lj_epsilon_scale=jnp.ones(2),
        mm_switch_on=3.0,
        mm_switch_width=2.0,
        ml_switch_width=1.0,
        complementary_handoff=False,
    )
    # 1.5× ε on both types → e_mm ≈ 1.5× (charges zero)
    e_out = float(np.asarray(out["e_mm"]).reshape(-1)[0])
    e_base = float(np.asarray(base["e_mm"]).reshape(-1)[0])
    assert e_out == pytest.approx(1.5 * e_base, rel=1e-5)


def test_resolve_md_lj_scales_from_checkpoint_parent(tmp_path: Path):
    run = tmp_path / "run-abc"
    run.mkdir()
    write_mm_lj_scales_into_hybrid_mm_json(
        run / "hybrid_mm.json",
        type_names=["T0", "T1"],
        sigma_scale=[1.03, 0.97],
        epsilon_scale=[1.2, 0.8],
    )
    ckpt = run / "epoch-1" / "params.json"
    ckpt.parent.mkdir()
    ckpt.write_text("{}", encoding="utf-8")
    ep, sig = resolve_md_lj_scales(checkpoint=ckpt, atc_names=["T1", "T0"])
    assert ep is not None
    np.testing.assert_allclose(sig, [0.97, 1.03])
    np.testing.assert_allclose(ep, [0.8, 1.2])


def test_resolve_md_lj_scales_missing_learn_flag_returns_none(tmp_path: Path):
    path = tmp_path / "hybrid_mm.json"
    path.write_text(
        json.dumps({"hybrid_mm": True, "learn_mm_lj_scales": False}),
        encoding="utf-8",
    )
    ep, sig = resolve_md_lj_scales(scales_file=path, atc_names=["A"])
    assert ep is None and sig is None


def test_load_sidecar_incomplete_raises(tmp_path: Path):
    path = tmp_path / "hybrid_mm.json"
    path.write_text(
        json.dumps({"learn_mm_lj_scales": True, "cgenff_type_names": ["A"]}),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="missing"):
        load_mm_lj_scales_sidecar(path)


def test_scales_to_atc_length_mismatch():
    with pytest.raises(ValueError, match="scale length"):
        scales_to_atc(["A", "B"], [1.0], [1.0, 1.0], ["A"])


def test_hybrid_mm_config_learn_flag_coerce():
    from mmml.models.hybrid_energy import HybridMMConfig

    cfg = HybridMMConfig.coerce(
        {
            "master_sigmas": [3.6, 2.4],
            "master_epsilons": [0.08, 0.02],
            "mm_switch_on": 8.0,
            "mm_switch_width": 5.0,
            "ml_switch_width": 1.5,
            "learn_mm_lj_scales": True,
        }
    )
    assert cfg.learn_mm_lj_scales is True
    assert "learn_mm_lj_scales" in cfg.kwargs()


def test_hybrid_mm_metadata_includes_lj_flag():
    from mmml.models.hybrid_energy import HybridMMConfig
    from mmml.models.mm_charge_mode import hybrid_mm_metadata_dict

    cfg = HybridMMConfig(
        master_sigmas=(3.6, 2.4),
        master_epsilons=(0.08, 0.02),
        mm_switch_on=8.0,
        mm_switch_width=5.0,
        ml_switch_width=1.5,
        learn_mm_lj_scales=True,
    )
    meta = hybrid_mm_metadata_dict(cfg)
    assert meta["learn_mm_lj_scales"] is True
    assert meta["include_lj"] is True
    assert meta["lr_solver"] == "mic"


def test_cli_learn_mm_lj_scales_flag(tmp_path):
    from mmml.cli.make.make_training import _build_hybrid_mm_config, parse_args

    payload = {
        "R": np.zeros((2, 4, 3)),
        "Z": np.ones((2, 4), dtype=int),
        "F": np.zeros((2, 4, 3)),
        "E": np.zeros(2),
        "N": np.full(2, 4),
        "cgenff_type_idx": np.zeros((2, 4), dtype=int),
        "mol_id": np.tile([0, 0, 1, 1], (2, 1)),
        "cgenff_charge": np.zeros((2, 4)),
        "cgenff_master_sigmas": np.array([3.6, 2.4]),
        "cgenff_master_epsilons": np.array([0.078, 0.024]),
    }
    p = tmp_path / "d.npz"
    np.savez(p, **payload)

    args = parse_args(
        ["--data", str(p), "--hybrid-mm", "--learn-mm-lj-scales", "--quiet"]
    )
    assert args.learn_mm_lj_scales is True
    cfg = _build_hybrid_mm_config(args, [str(p)])
    assert cfg["learn_mm_lj_scales"] is True
    assert cfg["include_lj"] is True

    # Ewald forces LJ (and therefore learnable scales) off.
    args_ew = parse_args(
        [
            "--data",
            str(p),
            "--hybrid-mm",
            "--learn-mm-lj-scales",
            "--lr-solver",
            "ewald",
            "--pme-box-length",
            "20",
            "--quiet",
        ]
    )
    cfg_ew = _build_hybrid_mm_config(args_ew, [str(p)])
    assert cfg_ew["include_lj"] is False
    assert cfg_ew["learn_mm_lj_scales"] is False


def test_example_yaml_keys_exist():
    """Checked-in example YAMLs carry the student-facing knobs."""
    import yaml

    root = Path(__file__).resolve().parents[2]
    train = yaml.safe_load(
        (root / "examples/hybrid_mm_charges/train_fixed_lj_scales.yaml").read_text()
    )
    assert train["learn_mm_lj_scales"] is True
    assert train["hybrid_mm"] is True
    assert train.get("lr_solver", "mic") == "mic"
    md = yaml.safe_load(
        (root / "examples/hybrid_mm_charges/md_fixed_lj_scales.yaml").read_text()
    )
    assert "checkpoint" in md["defaults"]
    assert md["defaults"]["include_mm"] is True


def test_md_yaml_stage_times_are_not_silently_the_defaults():
    """Every dynamics stage each run executes must get its length from the YAML.

    ``resolve_stage_ps`` reads ``ps`` only for the prod and nve stages, so a
    pbc_nvt run (mini,heat,equi) that sets ``ps`` alone quietly inherits the
    10 ps / 50 ps stage defaults — 120k steps instead of the intended smoke test.
    """
    from argparse import Namespace

    import yaml

    from mmml.cli.run.md_config import merge_campaign_job_config
    from mmml.interfaces.pycharmmInterface.mlpot.cli_common import (
        resolve_md_stages,
        resolve_stage_ps,
    )

    root = Path(__file__).resolve().parents[2]
    campaign = yaml.safe_load(
        (root / "examples/hybrid_mm_charges/md_fixed_lj_scales.yaml").read_text()
    )
    for run_id in campaign["runs"]:
        merged = merge_campaign_job_config(campaign, run_id)
        ns = Namespace(**merged)
        stages = [s for s in resolve_md_stages(ns) if s != "mini"]
        assert stages, f"{run_id}: no dynamics stages"
        for stage in stages:
            ps = resolve_stage_ps(ns, stage)
            assert ps <= 0.1, (
                f"{run_id}: {stage} runs {ps} ps — set ps_{stage} in the YAML "
                f"(a bare `ps` does not reach this stage)"
            )


def test_docs_page_exists_and_links_examples():
    root = Path(__file__).resolve().parents[2]
    page = (root / "docs/hybrid-mm-lj-scales.md").read_text(encoding="utf-8")
    assert "learn_mm_lj_scales" in page
    assert "train_fixed_lj_scales.yaml" in page
    assert "hybrid-mm-charges.md" in page
    assert "hybrid-mm-dataset-preparation.md" in page
    assert "md-interaction-policies.md" in page
