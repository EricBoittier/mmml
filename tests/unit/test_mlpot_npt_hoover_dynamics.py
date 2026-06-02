"""Unit tests for NPT Hoover dynamics kwargs (no CHARMM runtime required)."""

from __future__ import annotations

import numbers

from mmml.interfaces.pycharmmInterface.mlpot.dynamics import (
    _apply_npt_cpt_kwargs,
    build_cpt_equilibration_dynamics,
    build_cpt_production_dynamics,
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


def test_equi_hoover_default_uses_mass_formula_and_disables_rescaling():
    kw = build_cpt_equilibration_dynamics(temp=300.0, pmass=16, tmass=160)
    script = _script_string(**kw)

    assert kw["ihtfrq"] == 0
    assert kw["ieqfrq"] == 0
    assert kw["hoover reft"] == 300.0
    assert kw["tmass"] == 160
    assert kw["pmass"] == 16
    assert kw["pgamma"] == 5
    assert kw["firstt"] == 300.0
    assert "hoover reft" in script
    assert "ihtfrq 0" in script
    assert "ieqfrq 0" in script
    assert "cpt" in script


def test_prod_hoover_default_uses_pgamma_zero():
    kw = build_cpt_production_dynamics(temp=298.15, pmass=33, tmass=330)
    script = _script_string(**kw)

    assert kw["pgamma"] == 0
    assert kw["hoover reft"] == 298.15
    assert kw["tmass"] == 330
    assert "firstt" not in kw
    assert "hoover reft" in script
    assert "pgamma 0" in script


def test_berendsen_thermostat_option():
    kw: dict = {}
    _apply_npt_cpt_kwargs(
        kw,
        temp=300.0,
        thermostat="berendsen",
        pmass=10,
        tmass=100,
        pgamma=5,
    )
    assert kw["tcons"] is True
    assert kw["tcoupling"] == 5.0
    assert kw["treference"] == 300.0
    assert "hoover reft" not in kw
