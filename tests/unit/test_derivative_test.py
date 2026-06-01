from __future__ import annotations

import pytest

from mmml.interfaces.pycharmmInterface.mlpot.derivative_test import (
    TestFirstConfig,
    build_test_first_script,
    selection_clause_for_test_first,
)
from mmml.interfaces.pycharmmInterface.mlpot.cli_common import (
    argparse,
    resolve_test_first_config,
)


def test_selection_all():
    assert selection_clause_for_test_first(()) == " SELE ALL END"


def test_selection_resids():
    assert selection_clause_for_test_first([1, 3]) == " SELE RESId 1 3 END"


def test_build_script():
    cfg = TestFirstConfig(tol=0.01, step=2e-4, resids=(2,))
    assert build_test_first_script(cfg) == "TEST FIRSt TOL 0.01 STEP 0.0002 SELE RESId 2 END"


def test_resolve_disabled():
    args = argparse.Namespace(test_first=False)
    assert resolve_test_first_config(args) is None


def test_resolve_enabled():
    args = argparse.Namespace(
        test_first=True,
        test_first_tol=0.1,
        test_first_step=1e-3,
        test_first_resids="1,2",
        test_first_charmm=True,
        test_first_update_nbonds=True,
        quiet=True,
    )
    cfg = resolve_test_first_config(args)
    assert cfg is not None
    assert cfg.tol == pytest.approx(0.1)
    assert cfg.step == pytest.approx(1e-3)
    assert cfg.resids == (1, 2)
    assert cfg.verbose is False
    assert cfg.charmm_lingo is True
    assert cfg.update_nonbonds is True


def test_resolve_charmm_off_by_default():
    args = argparse.Namespace(test_first=True, quiet=False)
    cfg = resolve_test_first_config(args)
    assert cfg is not None
    assert cfg.charmm_lingo is False
