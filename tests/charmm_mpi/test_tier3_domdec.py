"""Tier 3 DOMDEC + MLpot survey tests."""

from __future__ import annotations

from mmml.interfaces.pycharmmInterface.mlpot.mpi_spatial.domdec_info import survey_domdec_api
from mmml.interfaces.pycharmmInterface.mlpot.tier3_domdec_validate import (
    render_tier3_report,
    validate_tier3_domdec_env,
)


def test_survey_domdec_api_reports_missing_pycharmm_maps():
    survey = survey_domdec_api()
    assert survey.charmm_fortran_domdec is True
    assert survey.pycharmm_local_atom_api is False
    assert survey.pycharmm_ghost_atom_api is False


def test_validate_tier3_domdec_blocked_by_default():
    report = validate_tier3_domdec_env(strict=False)
    assert report.blocked is True
    assert report.ok is True
    assert any("blocked" in e.lower() for e in report.errors)


def test_validate_tier3_domdec_strict_fails():
    report = validate_tier3_domdec_env(strict=True)
    assert report.ok is False


def test_render_tier3_report_mentions_tier2_fallback():
    report = validate_tier3_domdec_env()
    text = render_tier3_report(report)
    assert "Tier 2" in text
    assert "SPATIAL_MPI_DOMDEC" in text or "blocked" in text.lower()


def test_mpi_check_tier3_flag():
    from unittest import mock

    from mmml.cli.run.mpi_check import main

    with mock.patch(
        "mmml.interfaces.pycharmmInterface.charmm_mpi.charmm_lib_available",
        return_value=True,
    ), mock.patch(
        "mmml.interfaces.pycharmmInterface.charmm_mpi._charmm_lib_path",
        return_value=mock.Mock(),
    ), mock.patch(
        "mmml.interfaces.pycharmmInterface.charmm_mpi.charmm_lib_links_mpi",
        return_value=False,
    ), mock.patch(
        "mmml.interfaces.pycharmmInterface.charmm_mpi._under_mpirun",
        return_value=False,
    ), mock.patch(
        "mmml.interfaces.pycharmmInterface.mlpot.mpi_bridge.mpi_rank_size",
        return_value=(0, 1),
    ), mock.patch(
        "mmml.interfaces.pycharmmInterface.charmm_mpi._mpi4py_available",
        return_value=False,
    ), mock.patch(
        "mmml.interfaces.pycharmmInterface.mlpot.spatial_mpi_policy.spatial_mpi_enabled",
        return_value=False,
    ):
        assert main(["--tier3"]) == 0
