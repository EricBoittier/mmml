"""CGENFF pre-minimize must not clear CHARMM crystal when use_pbc is set."""

from mmml.interfaces.pycharmmInterface.mlpot.dynamics import CharmmMmMinimizeConfig


def test_charmm_mm_minimize_config_accepts_use_pbc() -> None:
    cfg = CharmmMmMinimizeConfig(nstep_sd=10, use_pbc=True)
    assert cfg.use_pbc is True
