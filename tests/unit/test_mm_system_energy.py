"""Unit tests for :mod:`mmml.interfaces.pycharmmInterface.mm_system_energy`."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

from mmml.interfaces.pycharmmInterface.mm_system_energy import (
    COULOMB_KCAL,
    CharmmFswitchCoeffs,
    CharmmNbondSettings,
    CharmmVfswitchCoeffs,
    NonbondedSystemData,
    _pair_lj_epsilon,
    charmm_fshift_elec,
    charmm_fswitch_coeffs,
    charmm_fswitch_elec,
    charmm_switch_factor,
    charmm_vfswitch_coeffs,
    charmm_vfswitch_vdw,
    decompose_nonbonded_pair_energies,
    excluded_pairs_from_psf_bonds,
    excluded_pairs_from_psf_nnb,
    fully_excluded_pairs,
    nonbonded_energy_and_forces,
    one_four_pairs_from_bonds,
    single_pair_mic_nonbonded_energies,
)

_SETTINGS = CharmmNbondSettings(cutnb=14.0, ctonnb=10.0, ctofnb=12.0)


def _ref_fshift_elec(r: float, qq: float, settings: CharmmNbondSettings) -> float:
    r1 = 1.0 / r
    r_sq = r * r
    ch = qq * r1
    return ch * (1.0 + r_sq * (settings.min2of * r1 - settings.ctrof2))


def _ref_vfswitch_vdw(
    r: float,
    a_coef: float,
    b_coef: float,
    settings: CharmmNbondSettings,
    coeffs: CharmmVfswitchCoeffs,
) -> float:
    r1 = 1.0 / r
    r_sq = r * r
    tr2 = r1 * r1
    tr6 = tr2**3
    if r_sq > settings.c2onnb:
        r3 = r1 * tr2
        rjunk6 = tr6 - coeffs.recof6
        rjunk3 = r3 - coeffs.recof3
        cr12 = a_coef * coeffs.ofdif6 * rjunk6
        cr6 = b_coef * coeffs.ofdif3 * rjunk3
        return cr12 * rjunk6 - cr6 * rjunk3
    ca = a_coef * tr6 * tr6
    enevdw = ca - b_coef * tr6
    return enevdw + b_coef * coeffs.onoff3 - a_coef * coeffs.onoff6


def _ref_fswitch_elec(
    r: float,
    qq: float,
    settings: CharmmNbondSettings,
    coeffs: CharmmFswitchCoeffs,
) -> float:
    r_sq = r * r
    r1 = 1.0 / r
    if r_sq > settings.c2onnb:
        return qq * (
            r1 * (
                coeffs.acoef
                - r_sq * (coeffs.bcoef + r_sq * (coeffs.cover3 + coeffs.dover5 * r_sq))
            )
            + coeffs.const
        )
    return qq * (r1 + coeffs.eadd)


def test_charmm_switch_factor_endpoints() -> None:
    settings = _SETTINGS
    below = float(charmm_switch_factor(jnp.asarray(settings.c2onnb * 0.5), settings))
    at_on = float(charmm_switch_factor(jnp.asarray(settings.c2onnb), settings))
    at_off = float(charmm_switch_factor(jnp.asarray(settings.c2ofnb), settings))
    above = float(charmm_switch_factor(jnp.asarray(settings.c2ofnb * 1.1), settings))
    assert below == pytest.approx(1.0)
    assert at_on == pytest.approx(1.0)
    assert at_off == pytest.approx(0.0, abs=1e-6)
    assert above == pytest.approx(0.0)


@pytest.mark.parametrize(
    ("r", "qq"),
    [
        (5.0, 0.25),
        (9.5, -0.12),
        (11.5, 0.33),
    ],
)
def test_charmm_fshift_elec_matches_reference(r: float, qq: float) -> None:
    settings = CharmmNbondSettings(
        cutnb=14.0,
        ctonnb=10.0,
        ctofnb=12.0,
        elec_switch="fshift",
    )
    got = float(charmm_fshift_elec(jnp.asarray(r, dtype=jnp.float64), jnp.asarray(qq, dtype=jnp.float64), settings))
    ref = _ref_fshift_elec(r, qq, settings)
    assert got == pytest.approx(ref, rel=1e-12)


@pytest.mark.parametrize(
    ("r", "qq"),
    [
        (5.0, 0.25),
        (10.5, -0.12),
        (11.8, 0.33),
    ],
)
def test_charmm_fswitch_elec_matches_reference(r: float, qq: float) -> None:
    settings = CharmmNbondSettings(
        cutnb=14.0,
        ctonnb=10.0,
        ctofnb=12.0,
        elec_switch="fswitch",
    )
    coeffs = charmm_fswitch_coeffs(settings)
    got = float(charmm_fswitch_elec(jnp.asarray(r, dtype=jnp.float64), jnp.asarray(qq, dtype=jnp.float64), settings, coeffs))
    ref = _ref_fswitch_elec(r, qq, settings, coeffs)
    assert got == pytest.approx(ref, rel=1e-12)


@pytest.mark.parametrize(
    ("r", "ep", "sig"),
    [
        (4.0, 0.05, 3.2),
        (10.5, 0.08, 3.5),
        (11.9, 0.03, 3.0),
    ],
)
def test_charmm_vfswitch_vdw_matches_reference(r: float, ep: float, sig: float) -> None:
    settings = _SETTINGS
    coeffs = charmm_vfswitch_coeffs(settings)
    a_coef = ep * sig**12
    b_coef = 2.0 * ep * sig**6
    got = float(
        charmm_vfswitch_vdw(
            jnp.asarray(r, dtype=jnp.float64),
            jnp.asarray(a_coef, dtype=jnp.float64),
            jnp.asarray(b_coef, dtype=jnp.float64),
            settings,
            coeffs,
        )
    )
    ref = _ref_vfswitch_vdw(r, a_coef, b_coef, settings, coeffs)
    assert got == pytest.approx(ref, rel=1e-12)


def test_vfswitch_vdw_zero_at_ctofnb() -> None:
    settings = _SETTINGS
    coeffs = charmm_vfswitch_coeffs(settings)
    ep, sig = 0.06, 3.4
    a_coef = ep * sig**12
    b_coef = 2.0 * ep * sig**6
    at_off = float(
        charmm_vfswitch_vdw(
            jnp.asarray(settings.ctofnb, dtype=jnp.float64),
            jnp.asarray(a_coef, dtype=jnp.float64),
            jnp.asarray(b_coef, dtype=jnp.float64),
            settings,
            coeffs,
        )
    )
    assert at_off == pytest.approx(0.0, abs=1e-8)


def test_known_pair_elec_fshift_kcal() -> None:
    """Single-pair cdie force-shift energy at r=8 Å (q=±1 e, eps=1)."""
    settings = CharmmNbondSettings(
        cutnb=14.0,
        ctonnb=10.0,
        ctofnb=12.0,
        elec_switch="fshift",
    )
    r = 8.0
    qq = 1.0
    raw = _ref_fshift_elec(r, qq, settings)
    expected = COULOMB_KCAL * raw
    got = COULOMB_KCAL * float(
        charmm_fshift_elec(jnp.asarray(r, dtype=jnp.float64), jnp.asarray(qq, dtype=jnp.float64), settings)
    )
    assert got == pytest.approx(expected, rel=1e-12)
    assert got == pytest.approx(115.30, rel=1e-3)


def test_known_pair_vdw_vfswitch_kcal() -> None:
    """Single-pair VDW force-switch at r=8 Å (CGENFF-like ep/sig)."""
    settings = _SETTINGS
    coeffs = charmm_vfswitch_coeffs(settings)
    ep, sig = 0.066, 3.512
    a_coef = ep * sig**12
    b_coef = 2.0 * ep * sig**6
    r = 8.0
    expected = _ref_vfswitch_vdw(r, a_coef, b_coef, settings, coeffs)
    got = float(
        charmm_vfswitch_vdw(
            jnp.asarray(r, dtype=jnp.float64),
            jnp.asarray(a_coef, dtype=jnp.float64),
            jnp.asarray(b_coef, dtype=jnp.float64),
            settings,
            coeffs,
        )
    )
    assert got == pytest.approx(expected, rel=1e-12)
    assert got == pytest.approx(-7.98e-4, rel=1e-3)


def test_vfswitch_coeffs_match_fortran_init() -> None:
    settings = _SETTINGS
    coeffs = charmm_vfswitch_coeffs(settings)
    b = settings.ctofnb
    off3 = settings.c2ofnb * b
    off6 = off3 * off3
    assert coeffs.recof6 == pytest.approx(1.0 / off6)
    assert coeffs.recof3 == pytest.approx(1.0 / off3)


def test_fswitch_coeffs_inner_outer_continuity() -> None:
    settings = _SETTINGS
    coeffs = charmm_fswitch_coeffs(settings)
    qq = 0.5
    r_on = settings.ctonnb + 1e-6
    inner = _ref_fswitch_elec(settings.ctonnb - 1e-6, qq, settings, coeffs)
    outer = _ref_fswitch_elec(r_on, qq, settings, coeffs)
    assert inner == pytest.approx(outer, rel=1e-4)


def test_fully_excluded_pairs_from_iblo_inb() -> None:
    # Two atoms, atom 1 excludes atom 2 (CHARMM 1-based INB entry).
    iblo = [1, 2]
    inb = [2]
    pairs = fully_excluded_pairs(iblo, inb, natom=2)
    assert pairs == frozenset({(0, 1)})


def test_fully_excluded_pairs_empty_inb() -> None:
    assert fully_excluded_pairs([0, 0, 0], [], natom=3) == frozenset()


def test_excluded_pairs_from_psf_bonds_chain() -> None:
    bonds = np.asarray([[0, 1], [1, 2], [2, 3]], dtype=np.int32)
    pairs = excluded_pairs_from_psf_bonds(bonds)
    assert (0, 1) in pairs
    assert (1, 2) in pairs
    assert (0, 2) in pairs
    assert (1, 3) in pairs
    assert (0, 3) not in pairs


def test_one_four_pairs_from_bonds_chain() -> None:
    # Linear 4-atom chain: 0-1-2-3 => one 1-4 pair (0,3).
    bonds = np.asarray([[0, 1], [1, 2], [2, 3]], dtype=np.int32)
    pairs = one_four_pairs_from_bonds(bonds, natom=4)
    assert (0, 3) in pairs


def test_charmm_nbond_settings_default_vdw14fac_zero() -> None:
    settings = CharmmNbondSettings(cutnb=14.0, ctonnb=10.0, ctofnb=12.0)
    assert settings.vdw14fac == 0.0
    assert settings.e14fac == 1.0


def test_pair_lj_epsilon_uses_abs_product() -> None:
    ep_i = jnp.asarray([-0.15, 0.0], dtype=jnp.float64)
    ep_j = jnp.asarray([-0.20, 0.1], dtype=jnp.float64)
    got = _pair_lj_epsilon(ep_i, ep_j)
    assert float(got[0]) == pytest.approx(float(np.sqrt(0.03)), rel=1e-12)
    assert float(got[1]) == pytest.approx(0.0)


def test_resolve_nonbonded_excluded_pairs_prefers_psf_iblo_inb() -> None:
    from pathlib import Path

    from mmml.interfaces.pycharmmInterface.cgenff_topology import parse_psf_ext
    from mmml.interfaces.pycharmmInterface.mm_system_energy import (
        excluded_pairs_from_psf_bonds,
        resolve_nonbonded_excluded_pairs,
    )

    psf = Path("tests/functionality/mlpot/output/minimize/mini_full_mlpot.psf")
    if not psf.is_file():
        pytest.skip("mini_full_mlpot.psf fixture missing")
    data = parse_psf_ext(psf)
    resolved = resolve_nonbonded_excluded_pairs(
        psf,
        data.bonds,
        natom=data.n_atoms,
    )
    bond_only = excluded_pairs_from_psf_bonds(data.bonds)
    assert len(resolved) > len(bond_only)
    assert len(resolved) >= 170


def test_excluded_pairs_from_psf_nnb_mini_mlpot_fixture() -> None:
    from pathlib import Path

    from mmml.interfaces.pycharmmInterface.cgenff_topology import parse_psf_ext
    from mmml.interfaces.pycharmmInterface.mm_system_energy import (
        excluded_pairs_from_psf_inb_iblo,
        excluded_pairs_from_psf_nnb,
    )

    psf = Path("tests/functionality/mlpot/output/minimize/mini_full_mlpot.psf")
    if not psf.is_file():
        pytest.skip("mini_full_mlpot.psf fixture missing")
    data = parse_psf_ext(psf)
    assert data.nnb_indices.size == 190
    assert data.iblo_indices.size == data.n_atoms
    pairs = excluded_pairs_from_psf_inb_iblo(
        data.nnb_indices,
        data.iblo_indices,
        data.n_atoms,
    )
    assert (0, 2) in pairs
    assert len(pairs) > 50
    # IBLO/INB must differ from legacy packed mis-parse of the same flat array.
    packed_wrong = excluded_pairs_from_psf_nnb(data.nnb_indices, data.n_atoms)
    assert len(packed_wrong) != len(pairs)


def test_vdw14fac_scales_one_four_lj_only() -> None:
    settings = CharmmNbondSettings(
        cutnb=14.0,
        ctonnb=10.0,
        ctofnb=12.0,
        vdw14fac=0.0,
    )
    coeffs = charmm_vfswitch_coeffs(settings)
    r = 4.0
    ep, sig = 0.1, 3.5
    a = ep * sig**12
    b = 2.0 * ep * sig**6
    full = float(
        charmm_vfswitch_vdw(
            jnp.asarray(r),
            jnp.asarray(a),
            jnp.asarray(b),
            settings,
            coeffs,
        )
    )
    assert full != 0.0
    assert full * settings.vdw14fac == pytest.approx(0.0)


def test_decompose_nonbonded_pair_energies_matches_aggregate() -> None:
    rng = np.random.default_rng(0)
    n = 8
    pos = rng.normal(size=(n, 3)) * 3.0
    cell = np.diag([20.0, 20.0, 20.0])
    bonds = np.asarray([[0, 1], [1, 2], [2, 3]], dtype=np.int32)
    excluded = excluded_pairs_from_psf_bonds(bonds)
    e14 = one_four_pairs_from_bonds(bonds, n) - excluded
    nbond = NonbondedSystemData(
        charges=rng.normal(size=n) * 0.2,
        at_codes=np.zeros(n, dtype=np.int32),
        epsilon=np.abs(rng.normal(size=n) * 0.1) + 0.05,
        rmin=np.abs(rng.normal(size=n)) + 1.5,
        excluded_pairs=excluded,
        e14_pairs=e14,
        psf_path=None,
        psf_bonds=None,
    )
    settings = CharmmNbondSettings(cutnb=12.0, ctonnb=8.0, ctofnb=10.0)
    comp, _ = nonbonded_energy_and_forces(pos, nbond, cell, settings)
    decomp = decompose_nonbonded_pair_energies(pos, nbond, cell, settings)
    totals = decomp.totals()
    assert totals["vdw"] == pytest.approx(float(comp["vdw"]), rel=1e-10, abs=1e-10)
    assert totals["elec"] == pytest.approx(float(comp["elec"]), rel=1e-10, abs=1e-10)


def test_single_pair_mic_nonbonded_energies_jax_grad() -> None:
    from mmml.interfaces.pycharmmInterface.mm_system_energy import (
        single_pair_mic_nonbonded_energies,
    )

    rng = np.random.default_rng(1)
    n = 8
    pos = rng.normal(size=(n, 3)) * 3.0
    cell = np.diag([20.0, 20.0, 20.0])
    bonds = np.asarray([[0, 1], [1, 2], [2, 3]], dtype=np.int32)
    excluded = excluded_pairs_from_psf_bonds(bonds)
    e14 = one_four_pairs_from_bonds(bonds, n) - excluded
    nbond = NonbondedSystemData(
        charges=rng.normal(size=n) * 0.2,
        at_codes=np.zeros(n, dtype=np.int32),
        epsilon=np.abs(rng.normal(size=n) * 0.1) + 0.05,
        rmin=np.abs(rng.normal(size=n)) + 1.5,
        excluded_pairs=excluded,
        e14_pairs=e14,
        psf_path=None,
        psf_bonds=None,
    )
    settings = CharmmNbondSettings(cutnb=12.0, ctonnb=8.0, ctofnb=10.0)

    def _vdw_sum(p):
        v, _ = single_pair_mic_nonbonded_energies(p, 0, 5, nbond, cell, settings)
        return v

    grad = jax.grad(_vdw_sum)(jnp.asarray(pos, dtype=jnp.float64))
    assert grad.shape == pos.shape
    assert jnp.all(jnp.isfinite(grad))


def test_single_pair_dedr_numeric_matches_autodiff() -> None:
    from mmml.interfaces.pycharmmInterface.trialanine_nb_parity import (
        _mic_unit_vector,
        _single_pair_analytic_dedr,
        _single_pair_nb_energies,
    )

    rng = np.random.default_rng(2)
    n = 8
    pos = rng.normal(size=(n, 3)) * 3.0
    cell = np.diag([20.0, 20.0, 20.0])
    bonds = np.asarray([[0, 1], [1, 2], [2, 3]], dtype=np.int32)
    excluded = excluded_pairs_from_psf_bonds(bonds)
    e14 = one_four_pairs_from_bonds(bonds, n) - excluded
    nbond = NonbondedSystemData(
        charges=rng.normal(size=n) * 0.2,
        at_codes=np.zeros(n, dtype=np.int32),
        epsilon=np.abs(rng.normal(size=n) * 0.1) + 0.05,
        rmin=np.abs(rng.normal(size=n)) + 1.5,
        excluded_pairs=excluded,
        e14_pairs=e14,
        psf_path=None,
        psf_bonds=None,
    )
    settings = CharmmNbondSettings(cutnb=12.0, ctonnb=8.0, ctofnb=10.0)
    i, j = 0, 5
    r_hat = _mic_unit_vector(pos, i, j, cell)
    dr = 1e-4
    pos_plus = pos.copy()
    pos_minus = pos.copy()
    pos_plus[i] -= 0.5 * dr * r_hat
    pos_plus[j] += 0.5 * dr * r_hat
    pos_minus[i] += 0.5 * dr * r_hat
    pos_minus[j] -= 0.5 * dr * r_hat
    vdw_p, elec_p = _single_pair_nb_energies(pos_plus, i, j, nbond, cell, settings)
    vdw_m, elec_m = _single_pair_nb_energies(pos_minus, i, j, nbond, cell, settings)
    dedr_v_num = (vdw_p - vdw_m) / (2.0 * dr)
    dedr_e_num = (elec_p - elec_m) / (2.0 * dr)
    dedr_v_ana, dedr_e_ana = _single_pair_analytic_dedr(
        pos, i, j, nbond, cell, settings, r_hat
    )
    assert dedr_v_ana == pytest.approx(dedr_v_num, rel=1e-4, abs=1e-3)
    assert dedr_e_ana == pytest.approx(dedr_e_num, rel=1e-4, abs=1e-3)


def test_decompose_nonbonded_pair_energies_jax_grad_single_pair() -> None:
    """Switch audit uses ``jax.grad`` through per-pair decompose — must not host-convert tracers."""
    pos = jnp.asarray([[0.0, 0.0, 0.0], [4.0, 0.0, 0.0]], dtype=jnp.float64)
    cell = jnp.diag(jnp.asarray([20.0, 20.0, 20.0], dtype=jnp.float64))
    nbond = NonbondedSystemData(
        charges=jnp.asarray([0.2, -0.2], dtype=jnp.float64),
        at_codes=jnp.zeros(2, dtype=jnp.int32),
        epsilon=jnp.asarray([0.1, 0.1], dtype=jnp.float64),
        rmin=jnp.asarray([1.8, 1.8], dtype=jnp.float64),
        excluded_pairs=frozenset(),
        e14_pairs=frozenset(),
        psf_path=None,
        psf_bonds=None,
    )
    settings = CharmmNbondSettings(cutnb=12.0, ctonnb=8.0, ctofnb=10.0)
    pi = np.asarray([0], dtype=np.int32)
    pj = np.asarray([1], dtype=np.int32)

    def vdw_sum(p: jnp.ndarray) -> jnp.ndarray:
        d = decompose_nonbonded_pair_energies(
            p, nbond, cell, settings, pair_i=pi, pair_j=pj
        )
        return jnp.sum(d.vdw_kcal)

    grad = jax.grad(vdw_sum)(pos)
    assert grad.shape == pos.shape
    assert float(jnp.linalg.norm(grad)) > 0.0
