"""Unit tests for NPT Hoover dynamics kwargs (no CHARMM runtime required)."""

from __future__ import annotations

import numbers
from pathlib import Path

from mmml.interfaces.pycharmmInterface.mlpot.dynamics import (
    _apply_npt_cpt_kwargs,
    build_heat_dynamics,
    build_cpt_equilibration_dynamics,
    build_cpt_production_dynamics,
    build_nvt_equilibration_dynamics,
    final_npt_segment_restart,
    npt_restart_chain,
)


def _script_string(**kwargs) -> str:
    opts = []
    for k, v in kwargs.items():
        if isinstance(v, bool):
            if v:
                opts.append(f"{k} -")
        elif isinstance(v, (numbers.Number, str)):
            opts.append(f"{k} {v} -")
    script = "dynamics " + " ".join(opts)
    return script.strip(" -").lower()


def test_heat_uses_reference_ramp_without_equilibration_rescale():
    kw = build_heat_dynamics(temp=300.0)

    assert kw["ihtfrq"] == 10
    assert kw["TEMINC"] == 5
    assert kw["ieqfrq"] == 0
    assert kw["iasors"] == 1
    assert kw["iasvel"] == 1
    assert kw["iscvel"] == 0
    assert kw["ichecw"] == 0
    assert kw["firstt"] == 60.0
    assert kw["finalt"] == 300.0
    assert kw["tbath"] == 300.0


def test_heat_free_space_disables_image_update_frequencies():
    kw = build_heat_dynamics(temp=300.0, use_pbc=False)

    assert kw["imgfrq"] == 0
    assert kw["ihbfrq"] == 0
    assert kw["ilbfrq"] == 0


def test_equi_hoover_default_uses_mass_formula_and_disables_rescaling():
    kw = build_cpt_equilibration_dynamics(
        temp=300.0, pmass=16, tmass=160, pref=1.0
    )
    script = _script_string(**kw)

    assert kw["ihtfrq"] == 0
    assert kw["ieqfrq"] == 0
    assert kw["hoover reft"] == 300.0
    assert kw["tmass"] == 160
    assert kw["pmass"] == 16
    assert kw["pgamma"] == 5
    assert kw["pint pconst pref"] == 1.0
    assert kw["firstt"] == 300.0
    assert "hoover reft" in script
    assert "pint pconst pref 1.0" in script


def test_prod_default_matches_equi_barostat_pgamma():
    kw = build_cpt_production_dynamics(
        temp=298.15, pmass=33, tmass=330, pref=1.0
    )

    assert kw["pgamma"] == 5
    assert kw["pint pconst pref"] == 1.0
    assert kw["hoover reft"] == 298.15
    assert "firstt" not in kw


def test_pgamma_zero_disables_barostat_coupling_when_requested():
    kw = build_cpt_production_dynamics(pgamma=0, pmass=10, tmass=100)
    assert kw["pgamma"] == 0


def test_custom_npt_pressure():
    kw = build_cpt_equilibration_dynamics(pref=2.5, pmass=10, tmass=100)
    assert kw["pint pconst pref"] == 2.5


def test_equi_later_segment_omits_firstt():
    kw = build_cpt_equilibration_dynamics(
        temp=300.0, pmass=10, tmass=100, include_firstt=False
    )
    assert "firstt" not in kw


def test_nvt_equilibration_omits_cpt_and_crystal_keywords():
    kw = build_nvt_equilibration_dynamics(temp=300.0, tmass=160)
    script = _script_string(**kw)

    assert kw["hoover reft"] == 300.0
    assert kw["tmass"] == 160
    assert kw["imgfrq"] == 0
    assert "cpt" not in kw
    assert "pint pconst pref" not in kw
    assert "cpt" not in script
    assert "hoover reft" in script


def test_berendsen_thermostat_option():
    kw: dict = {}
    _apply_npt_cpt_kwargs(
        kw,
        temp=300.0,
        thermostat="berendsen",
        pref=1.0,
        pmass=10,
        tmass=100,
        pgamma=5,
    )
    assert kw["tcons"] is True
    assert kw["tcoupling"] == 5.0
    assert kw["treference"] == 300.0
    assert "hoover reft" not in kw


def test_npt_restart_chain_and_final_restart(tmp_path: Path):
    chain = npt_restart_chain(
        tmp_path,
        n_segments=3,
        prefix="equi_x",
        initial_restart=tmp_path / "nve_x.res",
    )
    assert len(chain) == 3
    assert chain[0].restart_read == tmp_path / "nve_x.res"
    assert chain[0].restart_write == tmp_path / "equi_x.0.res"
    assert chain[2].restart_write == tmp_path / "equi_x.2.res"

    final = final_npt_segment_restart(tmp_path, "equi_x", 3)
    assert final == tmp_path / "equi_x.2.res"
    assert final_npt_segment_restart(tmp_path, "equi_x", 1) == tmp_path / "equi_x.res"
