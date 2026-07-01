"""Unit tests for :mod:`mmml.interfaces.pycharmmInterface.mm_system_energy`."""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from mmml.interfaces.pycharmmInterface.mm_system_energy import (
    COULOMB_KCAL,
    CharmmFswitchCoeffs,
    CharmmNbondSettings,
    CharmmVfswitchCoeffs,
    charmm_fshift_elec,
    charmm_fswitch_coeffs,
    charmm_fswitch_elec,
    charmm_switch_factor,
    charmm_vfswitch_coeffs,
    charmm_vfswitch_vdw,
    excluded_pairs_from_psf_bonds,
    fully_excluded_pairs,
    one_four_pairs_from_bonds,
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
    got = float(charmm_fshift_elec(jnp.asarray(r), jnp.asarray(qq), settings))
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
    got = float(charmm_fswitch_elec(jnp.asarray(r), jnp.asarray(qq), settings, coeffs))
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
            jnp.asarray(r),
            jnp.asarray(a_coef),
            jnp.asarray(b_coef),
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
            jnp.asarray(settings.ctofnb),
            jnp.asarray(a_coef),
            jnp.asarray(b_coef),
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
    got = COULOMB_KCAL * float(charmm_fshift_elec(jnp.asarray(r), jnp.asarray(qq), settings))
    assert got == pytest.approx(expected, rel=1e-12)
    assert got == pytest.approx(-28.019, rel=1e-3)


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
            jnp.asarray(r),
            jnp.asarray(a_coef),
            jnp.asarray(b_coef),
            settings,
            coeffs,
        )
    )
    assert got == pytest.approx(expected, rel=1e-12)
    assert got == pytest.approx(-0.00184, rel=1e-3)


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
