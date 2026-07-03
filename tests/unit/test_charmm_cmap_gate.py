"""Unit tests for JAX CMAP gating vs PyCHARMM reference."""

from __future__ import annotations

import pytest

from mmml.interfaces.pycharmmInterface.cgenff_bonded_reference import charmm_cmap_is_active


def test_charmm_cmap_is_active_false_for_zero():
    assert not charmm_cmap_is_active({"cmap": 0.0, "bond": 1.0})


def test_charmm_cmap_is_active_true_for_nonzero():
    assert charmm_cmap_is_active({"cmap": 0.5})
