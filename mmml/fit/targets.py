"""Experimental liquid-state targets (density, enthalpy of vaporisation).

Reads the cited reference JSON collected for the ML/MM liquid fits (acetone,
dichloromethane) and turns it into :class:`StatePointTarget` records used by
:mod:`mmml.fit.loss`. Units: T in K, P in atm, density in g/cm^3, dHvap in
kJ/mol (sigmas in the same units).

Acetone density: DIPPR equation 105, ``rho = A / B^(1 + (1 - T/C)^D)`` in kg/m^3
(saturated liquid; at 1 atm the difference is far below the default sigma for
T well below the critical point). DCM has dHvap points only (no cited density).

dHvap sigmas are per source (:func:`dhvap_point_sigma`): the acetone points are
mutually inconsistent beyond 0.5 kJ/mol (non-monotonic in T; the 1926 value at
293 K sits 0.8 kJ/mol above the Majer-Svoboda correlation), so values derived
from vapour-pressure fits get ``SIGMA_DHVAP_VAPOR_PRESSURE_KJ_MOL`` and
pre-1950 values ``SIGMA_DHVAP_PRE1950_KJ_MOL``. Points at the normal boiling
point are left out of the default temperature list (an NPT liquid box at T_b
and 1 atm is at best metastable).
"""

from __future__ import annotations

import copy
import json
import os
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

REFERENCE_JSON_ENV = "MMML_EXP_REFERENCE_JSON"
DEFAULT_REFERENCE_JSON = Path(
    "/mmhome/boittier/home/mmml-pet-run/scratch/pet_200K/artifacts/experimental_reference_aco_dcm.json"
)

# Cited subset used when the machine-local JSON is absent (acetone DIPPR-105
# from DDB; NIST WebBook dHvap for acetone and DCM). Enough for CLI dry-runs.
BUNDLED_REFERENCE: dict[str, Any] = {
    "ACO": {
        "T_boil_K": {"value": 329.3, "source": "NIST"},
        "dHvap_kJ_mol": [
            {"T_K": 298.15, "value": 31.27, "source": "Majer and Svoboda, 1985"},
            {
                "T_K": 228.0,
                "value": 32.9,
                "source": "Stephenson and Malanowski, 1987; from 178-243 K vapor pressure",
            },
            {
                "T_K": 293.0,
                "value": 32.1,
                "source": "Felsing and Durban, 1926 (via NIST WebBook)",
            },
            {"T_K": 329.3, "value": 29.1, "source": "Majer and Svoboda, 1985"},
        ],
        "density_DIPPR105": {
            "A": 57.6214,
            "B": 0.233955,
            "C": 507.803,
            "D": 0.254167,
            "range_K": [183, 507],
            "source": "DDBST DIPPR105",
            "check_kg_m3": {
                "195.96": 888.763,
                "202.44": 882.713,
                "293.16": 791.24,
                "299.64": 784.105,
            },
        },
    },
    "DCM": {
        "dHvap_kJ_mol": [
            {"T_K": 298.15, "value": 29.03, "unc": 0.08, "source": "Manion, 2002"},
            {"T_K": 313.0, "value": 28.06, "source": "Majer and Svoboda, 1985"},
            {"T_K": [186, 312], "value": 29.4, "source": "Perry, 1926"},
        ],
    },
}

DEFAULT_REL_SIGMA_DENSITY = 0.005  # 0.5 % of rho
DEFAULT_SIGMA_DHVAP_KJ_MOL = 0.5
# Per-source dHvap sigma floors (kJ/mol), see dhvap_point_sigma.
SIGMA_DHVAP_VAPOR_PRESSURE_KJ_MOL = 1.5
SIGMA_DHVAP_PRE1950_KJ_MOL = 1.0
_VAPOR_PRESSURE_RE = re.compile(r"vapou?r pressure|\bfrom \d+(\.\d+)?\s*-\s*\d+(\.\d+)?\s*K", re.IGNORECASE)
_YEAR_RE = re.compile(r"\b(1[89]\d\d|20\d\d)\b")

# g/mol, for converting box density <-> volume per molecule.
MOLAR_MASS_G_MOL = {"ACO": 58.080, "DCM": 84.932}


@dataclass(frozen=True)
class DhvapPoint:
    """One experimental dHvap value (kJ/mol) at ``T_K`` (or over a T range)."""

    T_K: float | None  # None when the source only gives a range
    value_kJ_mol: float
    source: str
    unc_kJ_mol: float | None = None
    T_range_K: tuple[float, float] | None = None


@dataclass(frozen=True)
class StatePointTarget:
    """Experimental observables at one (T, P); ``None`` = not fitted there."""

    molecule: str
    T_K: float
    P_atm: float
    rho_g_cm3: float | None
    dhvap_kJ_mol: float | None
    sigma_rho_g_cm3: float | None
    sigma_dhvap_kJ_mol: float | None
    rho_source: str = ""
    dhvap_source: str = ""
    # Closest experimental dHvap (kJ/mol) even when not fitted here; used only
    # for the unit sanity check in mmml.fit.loss.check_reference_consistency.
    nominal_dhvap_kJ_mol: float | None = None


def load_reference(path: str | os.PathLike | None = None) -> dict[str, Any]:
    """Load the reference JSON (``path`` > ``$MMML_EXP_REFERENCE_JSON`` > default).

    Falls back to :data:`BUNDLED_REFERENCE` when the machine-local file is
    missing, so ``mmml fit-liquid`` works without the original scratch path.
    """
    explicit = path if path is not None else os.environ.get(REFERENCE_JSON_ENV)
    candidates = []
    if explicit:
        candidates.append(Path(explicit))
    else:
        candidates.append(DEFAULT_REFERENCE_JSON)
    for p in candidates:
        if p.is_file():
            with open(p) as fh:
                return json.load(fh)
    if explicit:
        raise FileNotFoundError(f"experimental reference JSON not found: {explicit}")
    return copy.deepcopy(BUNDLED_REFERENCE)


def dippr105_density_kg_m3(T_K: float | np.ndarray, A: float, B: float, C: float, D: float) -> np.ndarray:
    """DIPPR-105 liquid density ``A / B^(1 + (1 - T/C)^D)`` in kg/m^3."""
    T = np.asarray(T_K, dtype=float)
    return A / B ** (1.0 + (1.0 - T / C) ** D)


def _mol(ref: Mapping[str, Any], molecule: str) -> Mapping[str, Any]:
    key = molecule.upper()
    if key not in ref:
        raise KeyError(f"molecule {molecule!r} not in reference (have {sorted(ref)})")
    return ref[key]


def has_density(ref: Mapping[str, Any], molecule: str) -> bool:
    return "density_DIPPR105" in _mol(ref, molecule)


def density_g_cm3(
    ref: Mapping[str, Any], molecule: str, T_K: float | np.ndarray, *, strict: bool = True
) -> np.ndarray:
    """Experimental liquid density (g/cm^3) from the molecule's DIPPR-105 fit.

    ``strict`` raises outside the fit's validity range.
    """
    mol = _mol(ref, molecule)
    if "density_DIPPR105" not in mol:
        raise KeyError(f"no density correlation for {molecule!r}")
    d = mol["density_DIPPR105"]
    T = np.asarray(T_K, dtype=float)
    lo, hi = d.get("range_K", (-np.inf, np.inf))
    if strict and (np.any(T < lo) or np.any(T > hi)):
        raise ValueError(f"T={T} K outside DIPPR105 range [{lo}, {hi}] K for {molecule}")
    return dippr105_density_kg_m3(T, d["A"], d["B"], d["C"], d["D"]) / 1000.0


def density_check_max_rel_error(ref: Mapping[str, Any], molecule: str) -> float:
    """Max relative deviation of the DIPPR-105 fit from the stored check values."""
    d = _mol(ref, molecule)["density_DIPPR105"]
    checks = d.get("check_kg_m3", {})
    if not checks:
        raise ValueError(f"no check values stored for {molecule}")
    T = np.array([float(t) for t in checks])
    want = np.array([float(v) for v in checks.values()])
    got = dippr105_density_kg_m3(T, d["A"], d["B"], d["C"], d["D"])
    return float(np.max(np.abs(got - want) / want))


def dhvap_points(ref: Mapping[str, Any], molecule: str) -> list[DhvapPoint]:
    """All dHvap entries for ``molecule`` (range-only entries get ``T_K=None``)."""
    out = []
    for e in _mol(ref, molecule).get("dHvap_kJ_mol", []):
        T = e["T_K"]
        unc = e.get("unc")
        if isinstance(T, (list, tuple)):
            out.append(
                DhvapPoint(
                    None,
                    float(e["value"]),
                    str(e["source"]),
                    None if unc is None else float(unc),
                    (float(T[0]), float(T[1])),
                )
            )
        else:
            out.append(DhvapPoint(float(T), float(e["value"]), str(e["source"]), None if unc is None else float(unc)))
    return out


def dhvap_point_sigma(p: DhvapPoint, default_kJ_mol: float = DEFAULT_SIGMA_DHVAP_KJ_MOL) -> float:
    """sigma (kJ/mol) for one dHvap point: the largest of ``default``, the stated
    uncertainty, and the source-class floor (vapour-pressure derived:
    ``SIGMA_DHVAP_VAPOR_PRESSURE_KJ_MOL``; published before 1950:
    ``SIGMA_DHVAP_PRE1950_KJ_MOL``)."""
    sig = max(default_kJ_mol, p.unc_kJ_mol or 0.0)
    if _VAPOR_PRESSURE_RE.search(p.source):
        sig = max(sig, SIGMA_DHVAP_VAPOR_PRESSURE_KJ_MOL)
    years = [int(y) for y in _YEAR_RE.findall(p.source)]
    if years and min(years) < 1950:
        sig = max(sig, SIGMA_DHVAP_PRE1950_KJ_MOL)
    return sig


def boiling_point_K(ref: Mapping[str, Any], molecule: str) -> float | None:
    """Normal boiling point (K) if the reference gives one."""
    tb = _mol(ref, molecule).get("T_boil_K")
    if tb is None:
        return None
    return float(tb["value"] if isinstance(tb, Mapping) else tb)


def _density_source(ref: Mapping[str, Any], molecule: str) -> str:
    d = _mol(ref, molecule)["density_DIPPR105"]
    return f"DIPPR105: {d.get('source', '')}".strip()


def build_state_point_targets(
    molecule: str,
    temperatures_K: Sequence[float] | None = None,
    *,
    P_atm: float = 1.0,
    ref: Mapping[str, Any] | None = None,
    rel_sigma_rho: float = DEFAULT_REL_SIGMA_DENSITY,
    sigma_dhvap_kJ_mol: float = DEFAULT_SIGMA_DHVAP_KJ_MOL,
    dhvap_match_tol_K: float = 1.0,
    fit_density: bool = True,
    fit_dhvap: bool = True,
    per_source_sigma: bool = True,
    exclude_boiling_point: bool = True,
) -> list[StatePointTarget]:
    """State-point targets for ``molecule`` ("ACO" or "DCM").

    ``temperatures_K=None`` uses every dHvap point with a single temperature,
    minus (``exclude_boiling_point``) those within ``dhvap_match_tol_K`` of or
    above the normal boiling point. Otherwise each requested T gets the dHvap
    point within ``dhvap_match_tol_K`` (closest wins; ``None`` if none) and,
    when a density correlation exists and covers T, the DIPPR-105 density.
    dHvap sigma is ``max(sigma_dhvap_kJ_mol, stated unc)``, raised to the
    per-source floor of :func:`dhvap_point_sigma` when ``per_source_sigma``;
    density sigma is ``rel_sigma_rho * rho``. Every target carries
    ``nominal_dhvap_kJ_mol`` (closest point in T) for unit checks. State
    points with neither observable are dropped.
    """
    ref = load_reference() if ref is None else ref
    mol = _mol(ref, molecule)
    all_points = dhvap_points(ref, molecule)
    points = [p for p in all_points if p.T_K is not None]
    if temperatures_K is None:
        t_boil = boiling_point_K(ref, molecule) if exclude_boiling_point else None
        temperatures_K = sorted({p.T_K for p in points if t_boil is None or p.T_K < t_boil - dhvap_match_tol_K})

    def _t_dist(p: DhvapPoint, T: float) -> float:
        if p.T_K is not None:
            return abs(p.T_K - T)
        lo, hi = p.T_range_K
        return 0.0 if lo <= T <= hi else min(abs(lo - T), abs(hi - T))

    dens = mol.get("density_DIPPR105")
    targets = []
    for T in temperatures_K:
        T = float(T)
        rho = sig_rho = None
        rho_src = ""
        if fit_density and dens is not None:
            lo, hi = dens.get("range_K", (-np.inf, np.inf))
            if lo <= T <= hi:
                rho = float(density_g_cm3(ref, molecule, T))
                sig_rho = rel_sigma_rho * rho
                rho_src = _density_source(ref, molecule)
        dh = sig_dh = None
        dh_src = ""
        if fit_dhvap and points:
            best = min(points, key=lambda p: abs(p.T_K - T))
            if abs(best.T_K - T) <= dhvap_match_tol_K:
                dh = best.value_kJ_mol
                if per_source_sigma:
                    sig_dh = dhvap_point_sigma(best, sigma_dhvap_kJ_mol)
                else:
                    sig_dh = max(sigma_dhvap_kJ_mol, best.unc_kJ_mol or 0.0)
                dh_src = best.source
        if rho is None and dh is None:
            continue
        nominal = min(all_points, key=lambda p: _t_dist(p, T)).value_kJ_mol if all_points else None
        targets.append(
            StatePointTarget(molecule.upper(), T, float(P_atm), rho, dh, sig_rho, sig_dh, rho_src, dh_src, nominal)
        )
    return targets
