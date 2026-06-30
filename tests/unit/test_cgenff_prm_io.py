"""Tests for CGENFF parameter read bomlev helpers."""

from __future__ import annotations

import inspect

from mmml.interfaces.pycharmmInterface.nbonds_config import (
    CGENFF_PRM_BOMLEV,
    ic_prm_fill,
    read_cgenff_prm,
)


def test_cgenff_prm_bomlev_constant():
    assert CGENFF_PRM_BOMLEV == -5


def test_ic_prm_fill_uses_relaxed_bomlev():
    src = inspect.getsource(ic_prm_fill)
    assert "CGENFF_PRM_BOMLEV" in src
    assert "ic.prm_fill" in src


def test_read_cgenff_prm_defaults_to_cgenff_bomlev():
    src = inspect.getsource(read_cgenff_prm)
    assert "CGENFF_PRM_BOMLEV" in src


def test_ic_prm_fill_calls_ic_under_bomlev():
    src = inspect.getsource(ic_prm_fill)
    assert "charmm_relaxed_bomlev(CGENFF_PRM_BOMLEV)" in src.replace(" ", "")
