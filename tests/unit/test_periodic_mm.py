"""Tests for periodic external MM mode."""

from __future__ import annotations

import argparse
from unittest import mock

import numpy as np
import pytest

from mmml.interfaces.pycharmmInterface.mlpot.periodic_mm import (
    assert_periodic_mm_box_side,
    build_periodic_mm_config,
    min_cubic_box_for_periodic_mm,
    resolve_mm_nonbond_mode,
    validate_periodic_mm_args,
)


def test_resolve_mm_nonbond_mode_aliases():
    args = argparse.Namespace(mm_nonbond_mode="periodic")
    assert resolve_mm_nonbond_mode(args) == "periodic_external"


def test_validate_periodic_mm_requires_pbc_setup():
    args = argparse.Namespace(
        mm_nonbond_mode="periodic_external",
        free_space=False,
        setup="free_nve",
        mlpot_pbc=False,
        box_size=30.0,
        lr_solver="scafacos",
    )
    with pytest.raises(ValueError, match="ML MIC PBC"):
        validate_periodic_mm_args(
            args,
            charmm_pbc=True,
            mlpot_pbc=False,
            box_side_A=30.0,
        )


def test_validate_periodic_mm_requires_box():
    args = argparse.Namespace(
        mm_nonbond_mode="periodic_external",
        free_space=False,
        setup="pbc_nvt",
        mlpot_pbc=True,
        box_size=None,
        lr_solver="scafacos",
    )
    with pytest.raises(ValueError, match="positive cubic box"):
        validate_periodic_mm_args(
            args,
            charmm_pbc=True,
            mlpot_pbc=True,
            box_side_A=None,
        )


def test_build_periodic_mm_config_requires_external_coulomb_solver():
    args = argparse.Namespace(
        mm_nonbond_mode="periodic_external",
        lr_solver="mic",
    )
    with mock.patch(
        "mmml.interfaces.pycharmmInterface.long_range_backend.pick_lr_solver",
        return_value="mic",
    ):
        with pytest.raises(ValueError, match="nvalchemiops_pme"):
            build_periodic_mm_config(args)


def test_build_periodic_mm_config_jax_pme():
    args = argparse.Namespace(
        mm_nonbond_mode="periodic_external",
        lr_solver="jax_pme",
        jax_pme_method="pme",
    )
    with mock.patch(
        "mmml.interfaces.pycharmmInterface.long_range_backend.pick_lr_solver",
        return_value="jax_pme",
    ):
        cfg = build_periodic_mm_config(args)
    assert cfg is not None
    assert cfg.uses_jax_pme
    assert cfg.jax_pme_method == "pme"


def test_build_periodic_mm_config_nvalchemiops_pme():
    args = argparse.Namespace(
        mm_nonbond_mode="periodic_external",
        lr_solver="nvalchemiops_pme",
        jax_pme_method=None,
    )
    with mock.patch(
        "mmml.interfaces.pycharmmInterface.long_range_backend.pick_lr_solver",
        return_value="nvalchemiops_pme",
    ):
        cfg = build_periodic_mm_config(args)
    assert cfg is not None
    assert cfg.uses_nvalchemiops_pme


def test_assert_periodic_mm_box_side_too_small():
    with pytest.raises(ValueError, match="too small"):
        assert_periodic_mm_box_side(12.0, cluster_extent_A=8.0)


def test_min_cubic_box_scales_with_extent():
    need = min_cubic_box_for_periodic_mm(box_side_A=40.0, cluster_extent_A=10.0)
    assert need >= 15.0


def test_build_periodic_mm_config_charmm_vdw_flag():
    args = argparse.Namespace(
        mm_nonbond_mode="periodic_external",
        lr_solver="scafacos",
        periodic_charmm_vdw=False,
    )
    with mock.patch(
        "mmml.interfaces.pycharmmInterface.long_range_backend.pick_lr_solver",
        return_value="scafacos",
    ):
        cfg = build_periodic_mm_config(args)
    assert cfg is not None
    assert cfg.charmm_vdw is False


def test_resolve_periodic_charmm_vdw_off_when_no_include_mm():
    from mmml.interfaces.pycharmmInterface.mlpot.periodic_mm import (
        resolve_periodic_charmm_vdw,
    )

    args = argparse.Namespace(include_mm=False, periodic_charmm_vdw=True)
    assert resolve_periodic_charmm_vdw(args) is False

    args_explicit = argparse.Namespace(
        include_mm=False,
        periodic_charmm_vdw=True,
        _cli_explicit={"periodic_charmm_vdw"},
    )
    assert resolve_periodic_charmm_vdw(args_explicit) is True


def test_periodic_mm_status_line_no_vdw():
    from mmml.interfaces.pycharmmInterface.mlpot.periodic_mm import (
        PeriodicMmConfig,
        periodic_mm_status_line,
    )

    cfg = PeriodicMmConfig(
        lr_solver="scafacos",
        scafacos_method="p2nfft",
        charmm_vdw=False,
    )
    line = periodic_mm_status_line(cfg, box_side_A=40.0)
    assert "CHARMM VDW off" in line


def test_periodic_mm_external_adds_coulomb():
    from mmml.interfaces.pycharmmInterface.mlpot.periodic_mm import PeriodicMmConfig
    from mmml.interfaces.pycharmmInterface.mlpot.periodic_mm_external import (
        add_periodic_coulomb_to_callback,
    )

    pos = np.zeros((2, 3))
    cfg = PeriodicMmConfig(lr_solver="scafacos", scafacos_method="p2nfft")
    forces = np.zeros((2, 3))
    with mock.patch(
        "mmml.interfaces.pycharmmInterface.mlpot.periodic_mm_external.read_psf_charges",
        return_value=np.array([1.0, -1.0]),
    ), mock.patch(
        "mmml.interfaces.pycharmmInterface.mlpot.periodic_mm_external.compute_periodic_coulomb_kcalmol",
        return_value=(2.5, np.ones((2, 3))),
    ):
        e, f = add_periodic_coulomb_to_callback(
            pos,
            box_side_A=30.0,
            cfg=cfg,
            energy_kcal=1.0,
            forces_kcal=forces,
        )
    assert e == pytest.approx(3.5)
    assert np.allclose(f, 1.0)
