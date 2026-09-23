"""Tests for the ML/MM nonbonded tuner (mmml.models.mm_nonbonded_tune)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

jax = pytest.importorskip("jax")
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402

from mmml.models.cgenff_mm import cgenff_mm_energy, RMIN_HALF_TO_SIGMA  # noqa: E402
from mmml.models.mm_nonbonded_tune import (  # noqa: E402
    FrameFeatures,
    MonomerNonbonded,
    PriorConfig,
    SwitchConfig,
    TuneParams,
    bootstrap_fit,
    cohesion_budget,
    dimer_features,
    fit_parameters,
    force_blocks,
    frame_features,
    min_image_pairs,
    ml_pair_energy_forces,
    mm_coefficients,
    predict_energy,
    predict_force_rmse,
    summarize_params,
)

REPO = Path(__file__).resolve().parents[2]
ETOH_XYZ = REPO / "examples" / "pet_mad_etoh_pbc" / "etoh.xyz"

# small handoff so a 13 A box holds a single image of every pair
SMALL = SwitchConfig(mm_switch_on=3.5, mm_switch_width=2.0, ml_switch_width=1.0)


def _etoh():
    from ase.io import read

    atoms = read(ETOH_XYZ)
    return np.asarray(atoms.numbers), np.asarray(atoms.positions)


@pytest.fixture(scope="module")
def ff():
    z, r = _etoh()
    return MonomerNonbonded.from_cgenff(z, r)


def _rot(rng):
    q, _ = np.linalg.qr(rng.normal(size=(3, 3)))
    return q * np.sign(np.linalg.det(q))


def _random_box(rng, n_mol=8, L=13.0):
    _, r = _etoh()
    r = r - r.mean(0)
    centers = []
    while len(centers) < n_mol:
        c = rng.uniform(0, L, 3)
        ok = True
        for d in centers:
            dd = c - d
            dd -= L * np.round(dd / L)
            if np.linalg.norm(dd) < 4.2:
                ok = False
                break
        if ok:
            centers.append(c)
    mols = np.stack([r @ _rot(rng).T + c for c in centers])
    return mols, np.eye(3) * L


def test_monomer_types(ff):
    assert ff.residue == "ETOH"
    assert set(ff.type_names) == {"CG331", "CG321", "OG311", "HGA3", "HGA2", "HGP1"}
    assert abs(ff.charges.sum()) < 1e-8
    assert ff.n_classes == 21


def _direct_dimer_energy(ff, sw, R, p: TuneParams):
    """Reference energy through the independent cgenff_mm_energy path."""
    a = ff.n_atoms
    sig = ff.rmin_half * p.rmin_scale * RMIN_HALF_TO_SIGMA
    eps = ff.epsilon * p.eps_scale
    return float(
        cgenff_mm_energy(
            jnp.asarray(R),
            jnp.asarray(np.concatenate([ff.type_of_atom, ff.type_of_atom])),
            jnp.asarray(np.repeat([0, 1], a)),
            jnp.asarray(np.tile(ff.charges * p.charge_scale, 2)),
            jnp.asarray(sig),
            jnp.asarray(eps),
            mm_switch_on=sw.mm_switch_on,
            mm_switch_width=sw.mm_switch_width,
            ml_switch_width=sw.ml_switch_width,
        )
    )


@pytest.mark.parametrize("scaled", [False, True])
def test_features_match_cgenff_mm_energy(ff, scaled):
    rng = np.random.default_rng(1)
    _, r = _etoh()
    sw = SwitchConfig()
    p = TuneParams.cgenff(ff.n_types)
    if scaled:
        p = TuneParams(rng.uniform(0.5, 2.0, ff.n_types), rng.uniform(0.96, 1.04, ff.n_types), 1.1)
    for dist in (4.8, 5.5, 7.0, 9.5):
        R = np.concatenate([r, r @ _rot(rng).T + np.array([dist, 0.3, -0.2])])
        e_feat = float((dimer_features(ff, sw, R[None]) @ np.asarray(mm_coefficients(ff, p)))[0])
        e_ref = _direct_dimer_energy(ff, sw, R, p)
        assert e_feat == pytest.approx(e_ref, rel=1e-8, abs=1e-10)


def test_box_features_equal_sum_of_min_image_dimers(ff):
    rng = np.random.default_rng(2)
    mols, cell = _random_box(rng)
    G, _ = frame_features(ff, SMALL, mols, cell, with_jacobian=False)
    pairs, shifts, _ = min_image_pairs(mols, cell, SMALL.mm_cutoff)
    R = np.stack([np.concatenate([mols[i], mols[j] + s]) for (i, j), s in zip(pairs, shifts)])
    assert np.allclose(G, dimer_features(ff, SMALL, R).sum(0), rtol=1e-9, atol=1e-12)


def test_min_image_pairs_rejects_large_cutoff():
    mols = np.zeros((2, 1, 3))
    mols[1, 0] = [1.0, 0, 0]
    with pytest.raises(ValueError):
        min_image_pairs(mols, np.eye(3) * 10.0, 6.0)


def test_jacobian_gives_forces(ff):
    rng = np.random.default_rng(3)
    mols, cell = _random_box(rng, n_mol=6)
    p = TuneParams(rng.uniform(0.7, 1.4, ff.n_types), np.ones(ff.n_types), 0.95, 0.3, 0.2)
    c = np.asarray(mm_coefficients(ff, p))
    G, J = frame_features(ff, SMALL, mols, cell)
    f_feat = -(J.T @ c)
    k = 7  # finite difference on one coordinate
    h = 1e-5
    x = mols.reshape(-1).copy()
    ep = []
    for sgn in (1, -1):
        y = x.copy()
        y[k] += sgn * h
        g, _ = frame_features(ff, SMALL, y.reshape(mols.shape), cell, with_jacobian=False)
        ep.append(g @ c)
    assert f_feat[k] == pytest.approx(-(ep[0] - ep[1]) / (2 * h), rel=1e-5, abs=1e-6)


def test_ml_pair_switch_forces_are_conservative():
    sw = SMALL
    rng = np.random.default_rng(4)
    mols = rng.normal(size=(3, 2, 3)) * 0.3
    mols[1] += [2.8, 0, 0]
    mols[2] += [0, 3.1, 0]
    pairs = np.array([[0, 1], [0, 2], [1, 2]])
    shifts = np.zeros((3, 3))
    e_pair = np.array([-1.2, 0.7, -0.4])

    def energy(m):
        c = m.mean(1)
        r = np.linalg.norm(c[pairs[:, 1]] - c[pairs[:, 0]], axis=1)
        return ml_pair_energy_forces(sw, 3, 2, pairs, r, shifts, m, e_pair, None)[0]

    c = mols.mean(1)
    r = np.linalg.norm(c[pairs[:, 1]] - c[pairs[:, 0]], axis=1)
    _, F = ml_pair_energy_forces(sw, 3, 2, pairs, r, shifts, mols, e_pair, np.zeros((3, 4, 3)))
    h = 1e-6
    for idx in [(1, 0, 0), (2, 1, 1), (0, 1, 2)]:
        mp, mm = mols.copy(), mols.copy()
        mp[idx] += h
        mm[idx] -= h
        assert F[idx] == pytest.approx(-(energy(mp) - energy(mm)) / (2 * h), rel=1e-5, abs=1e-7)


def _synthetic(ff, truth: TuneParams, n_frames=12, seed=5, forces=True):
    rng = np.random.default_rng(seed)
    G, A, b, f2, e_t, e_ml, grp = [], [], [], [], [], [], []
    c = np.asarray(mm_coefficients(ff, truth))
    for k in range(n_frames):
        mols, cell = _random_box(rng)
        g, J = frame_features(ff, SMALL, mols, cell, with_jacobian=forces)
        e_int = float(g @ c) - 3.0  # "ML" part of -3 kcal/mol
        G.append(g)
        e_t.append(e_int)
        e_ml.append(-3.0)
        grp.append(k % 4)
        if forces:
            f_res = -(J.T @ c)
            Lk, yk, fp = force_blocks(J, f_res)
            A.append(Lk)
            b.append(yk)
            f2.append(fp)
    n = n_frames
    return FrameFeatures(
        G=np.array(G), e_res=np.array(e_t) - np.array(e_ml), n_mol=np.full(n, 8.0),
        group=np.array(grp), L=np.array(A) if forces else None, y=np.array(b) if forces else None,
        f_perp_sq=np.array(f2) if forces else None, n_atoms=np.full(n, 72.0),
        e_teacher=np.array(e_t), e_ml=np.array(e_ml),
    )


def test_fit_recovers_synthetic_parameters(ff):
    truth = TuneParams(np.ones(ff.n_types), np.ones(ff.n_types), 1.15)
    truth.eps_scale[ff.type_names.index("OG311")] = 1.6
    truth.eps_scale[ff.type_names.index("CG331")] = 0.7
    data = _synthetic(ff, truth)
    prior = PriorConfig(tau_eps=5.0, tau_rmin=0.5, tau_charge=5.0, sigma_e=0.01, sigma_f=0.01)
    res = fit_parameters(ff, data, prior)
    assert res.params.charge_scale == pytest.approx(1.15, rel=2e-2)
    assert predict_force_rmse(ff, res.params, data) < 1e-2
    e = predict_energy(ff, res.params, data)
    assert np.allclose(e, data.e_res, atol=1e-2)
    # heavy-atom well depths are identifiable from forces
    k = ff.type_names.index("OG311")
    assert res.params.eps_scale[k] == pytest.approx(1.6, rel=0.05)


def test_prior_pulls_toward_cgenff(ff):
    truth = TuneParams(np.full(ff.n_types, 2.0), np.ones(ff.n_types), 1.0)
    data = _synthetic(ff, truth, n_frames=4, forces=False)
    weak = fit_parameters(ff, data, PriorConfig(tau_eps=10.0, sigma_e=0.01, fit_rmin=False,
                                                fit_charge=False))
    strong = fit_parameters(ff, data, PriorConfig(tau_eps=0.01, sigma_e=10.0, fit_rmin=False,
                                                  fit_charge=False))
    dev_strong = np.mean(np.abs(np.log(strong.params.eps_scale)))
    dev_weak = np.mean(np.abs(np.log(weak.params.eps_scale)))
    assert dev_weak > 0.3
    assert dev_strong < 0.5 * dev_weak


def test_bootstrap_summary_and_budget(ff):
    truth = TuneParams(np.full(ff.n_types, 1.3), np.ones(ff.n_types), 1.05)
    data = _synthetic(ff, truth, n_frames=8, forces=False)
    fits = bootstrap_fit(ff, data, PriorConfig(), n_boot=3, seed=0)
    summ = summarize_params(ff, fits)
    assert len(summ["charge_scale"]["values"]) == 3
    assert summ["charge_scale"]["std"] >= 0.0
    budget = cohesion_budget(ff, TuneParams.cgenff(ff.n_types), data)
    assert budget["ml_pairs"] == pytest.approx(-3.0 / 8.0)
    assert budget["hybrid_total"] == pytest.approx(budget["ml_pairs"] + budget["mm_tail"])
    assert budget["missing"] == pytest.approx(budget["teacher_total"] - budget["hybrid_total"])


def test_underlay_features_only_inside_ml_region(ff):
    _, r = _etoh()
    sw = SwitchConfig()
    far = np.concatenate([r, r + [9.0, 0, 0]])[None]
    near = np.concatenate([r, r + [3.6, 0.5, 0]])[None]
    K = ff.n_classes
    g_far, g_near = dimer_features(ff, sw, far)[0], dimer_features(ff, sw, near)[0]
    assert np.all(g_far[2 * K + 1 :] == 0.0)  # no underlay beyond the ML taper
    assert np.all(g_near[: 2 * K + 1] == 0.0)  # MM tail off below 4.5 A
    assert np.any(g_near[2 * K + 1 :] != 0.0)


def test_label_roundtrip(tmp_path):
    from mmml.distill.box_cohesion import load_labels, save_labels

    frames = []
    for k, n_pairs in enumerate((3, 5)):
        frames.append({
            "group": k, "index": k, "cell": np.eye(3), "mols": np.zeros((4, 2, 3)),
            "z_mol": np.array([8, 1]), "teacher_e_int": -1.0 * k,
            "pairs": np.zeros((n_pairs, 2), dtype=np.int32), "shifts": np.zeros((n_pairs, 3)),
            "r_com": np.arange(n_pairs, dtype=float), "student_e_pair": np.ones(n_pairs),
            "student_f_pair": np.zeros((n_pairs, 4, 3)),
        })
    save_labels(tmp_path / "labels_x.npz", frames, meta={"a": 1})
    back = load_labels(tmp_path / "labels_x.npz")
    assert len(back) == 2
    assert back[1]["pairs"].shape == (5, 2)
    assert back[0]["r_com"].tolist() == [0.0, 1.0, 2.0]
    assert back[1]["teacher_e_int"] == -1.0


def test_sidecar_payload_roundtrips_through_md_loader(ff, tmp_path):
    import json

    from mmml.models.mm_lj_scales import load_mm_lj_scales_sidecar
    from mmml.models.mm_nonbonded_tune import lj_sidecar_payload

    p = TuneParams(np.full(ff.n_types, 1.5), np.full(ff.n_types, 1.01), 1.1)
    path = tmp_path / "hybrid_mm.json"
    path.write_text(json.dumps(lj_sidecar_payload(ff, p)))
    loaded = load_mm_lj_scales_sidecar(path)
    assert loaded is not None
    names = list(loaded["cgenff_type_names"]) if "cgenff_type_names" in loaded else None
    if names is not None:
        k = names.index("OG311")
        assert float(np.asarray(loaded["mm_lj_epsilon_scale"])[k]) == pytest.approx(1.5)


def test_cli_parser():
    from mmml.cli.misc.tune_mm_nonbonded import build_parser

    p = build_parser()
    a = p.parse_args(["fit", "--labels", "x", "--out-json", "y.json", "--handoff-grid", "6:5,5:4"])
    assert a.stage == "fit" and a.mm_switch_on == 6.0
    from mmml.cli.misc.tune_mm_nonbonded import _parse_grid

    assert _parse_grid(a.handoff_grid) == [(6.0, 5.0), (5.0, 4.0)]


def test_force_blocks_identity():
    rng = np.random.default_rng(7)
    J = rng.normal(size=(5, 30)) * np.array([1e6, 1.0, 1e-3, 10.0, 1.0])[:, None]
    f = rng.normal(size=30)
    L, y, fp = force_blocks(J, f)
    for _ in range(3):
        c = rng.normal(size=5)
        direct = float(np.sum((f + J.T @ c) ** 2))
        assert fp + float(np.sum((y + L @ c) ** 2)) == pytest.approx(direct, rel=1e-9)
