"""Where a condensed-phase reference exists for each DES/CGenFF species — and where it does not.

The DES dimer set covers 94 CGenFF residues. Only a minority of them have a
liquid-phase reference at 298 K / 1 atm, so "run every species and compare the
density" is not a plan. This module says, per species, *what state point is
physically meaningful* and *whether a reference number can be obtained*.

Four cases, and the distinction matters more than any single number:

``NIST_EOS``
    A NIST/REFPROP reference equation of state exists, so density is available
    at **any** (T, P) to reference accuracy — not just at one tabulated point.
    Fourteen of our species are in this class. These are the cheap ones, and
    they are the ones that make a low-density state point practical: argon's
    saturated liquid runs 1.379 g/cm3 at 90 K down to 0.680 at 150 K (Tc =
    150.86 K), every point of it referenced.

``LIQUID``
    Liquid at 298 K / 1 atm but no reference EOS; a single tabulated density
    (and sometimes dHvap) has to be looked up per species.

``GAS``
    A gas at 298 K / 1 atm. A "liquid box" at 298 K is then a *gas* box and its
    density is meaningless — this is the AMM1 mistake, generalised. Run below
    the normal boiling point, or above it at elevated pressure.

``SOLID``
    Solid at 298 K / 1 atm. Requires running above the melting point.

``ION``
    Not a molecular solvent. There is **no pure-liquid reference at all** — a
    box of chloride ions is not a physical system. These can only be validated
    in solution (hydration free energy, or a salt solution at known molality),
    which is a different and much more expensive experiment.

Numbers here are only populated where they were **verified against a cited
source**. Everything else is ``None`` on purpose: a plausible-looking density
recalled from memory is exactly the kind of self-consistent wrong constant that
hides forever, so the table refuses to carry one. Use
``scripts/fetch_nist_saturation.py`` to fill the NIST_EOS rows from the source.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class Phase(str, Enum):
    NIST_EOS = "nist_eos"
    LIQUID = "liquid"
    GAS = "gas"
    SOLID = "solid"
    ION = "ion"


@dataclass(frozen=True)
class StatePoint:
    """A (T, P) at which a reference value is or could be known."""

    T_K: float
    P_atm: float = 1.0
    density_g_cm3: float | None = None
    hvap_kJ_mol: float | None = None
    source: str = ""
    verified: bool = False

    def __post_init__(self) -> None:
        if self.density_g_cm3 is not None and not self.verified:
            raise ValueError(
                f"{self!r}: a density may not be recorded without verified=True. "
                "Populate it from a cited source, or leave it None."
            )


@dataclass(frozen=True)
class Species:
    resname: str
    name: str
    phase_298: Phase
    frames: int
    # Normal boiling / melting point, where that is what makes 298 K wrong.
    nbp_K: float | None = None
    mp_K: float | None = None
    nist_id: str | None = None
    states: tuple[StatePoint, ...] = ()
    note: str = ""


# Verified this session against the NIST Chemistry WebBook saturation table
# (ID C7440371, SatP). The dHvap column of that same fetch came back
# non-monotonic in T, which is unphysical -- it must fall to zero at Tc -- so it
# was discarded and only the densities are carried here.
_ARGON_SAT = (
    StatePoint(90.0, 1.3351, 1.3786, source="NIST WebBook SatP C7440371", verified=True),
    StatePoint(100.0, 3.2377, 1.3137, source="NIST WebBook SatP C7440371", verified=True),
    StatePoint(120.0, 12.130, 1.1628, source="NIST WebBook SatP C7440371", verified=True),
    StatePoint(140.0, 31.682, 0.94371, source="NIST WebBook SatP C7440371", verified=True),
    StatePoint(150.0, 47.346, 0.68043, source="NIST WebBook SatP C7440371", verified=True),
)

SPECIES: tuple[Species, ...] = (
    # ---- NIST reference EOS: density at any (T, P) -------------------------
    Species("TIP3", "water", Phase.NIST_EOS, 4518, nist_id="C7732185"),
    Species("METH", "methane", Phase.NIST_EOS, 662, nbp_K=111.7, nist_id="C74828",
            note="gas at 298 K; NIST EOS covers the whole liquid range"),
    Species("AMM1", "ammonia", Phase.NIST_EOS, 644, nbp_K=239.82, nist_id="C7664417",
            note="gas at 298 K -- a 298 K 'liquid ammonia' box is a gas"),
    Species("ETHE", "ethene", Phase.NIST_EOS, 435, nbp_K=169.4, nist_id="C74851"),
    Species("MEOH", "methanol", Phase.NIST_EOS, 340, nist_id="C67561"),
    Species("AR1", "argon", Phase.NIST_EOS, 252, nbp_K=87.28, nist_id="C7440371",
            states=_ARGON_SAT,
            note="Tc 150.86 K; saturated liquid spans 1.379 -> 0.680 g/cm3"),
    Species("HE1", "helium", Phase.NIST_EOS, 200, nbp_K=4.22, nist_id="C7440597",
            note="quantum fluid -- classical MD is not meaningful here"),
    Species("BENZ", "benzene", Phase.NIST_EOS, 187, nist_id="C71432"),
    Species("NE1", "neon", Phase.NIST_EOS, 182, nbp_K=27.10, nist_id="C7440019",
            note="light enough that nuclear quantum effects are non-negligible"),
    Species("ETHA", "ethane", Phase.NIST_EOS, 180, nbp_K=184.6, nist_id="C74840"),
    Species("BUTA", "butane", Phase.NIST_EOS, 177, nbp_K=272.7, nist_id="C106978",
            note="marginal: NBP is 272.7 K, so 298 K / 1 atm is a gas"),
    Species("KR1", "krypton", Phase.NIST_EOS, 171, nbp_K=119.74, nist_id="C7439909"),
    Species("XE1", "xenon", Phase.NIST_EOS, 165, nbp_K=165.05, nist_id="C7440633"),
    Species("PRPA", "propane", Phase.NIST_EOS, 133, nbp_K=231.0, nist_id="C74986"),
    Species("TOLU", "toluene", Phase.NIST_EOS, 58, nist_id="C108883"),
    # ---- Liquid at 298 K, single-point lookup needed -----------------------
    Species("FORH", "formic acid", Phase.LIQUID, 344),
    Species("ETOH", "ethanol", Phase.LIQUID, 207),
    Species("ACEH", "acetic acid", Phase.LIQUID, 205),
    Species("ACO", "acetone", Phase.LIQUID, 204),
    Species("ETSH", "ethanethiol", Phase.LIQUID, 204),
    Species("FORM", "formamide", Phase.LIQUID, 190),
    Species("DMDS", "dimethyl disulfide", Phase.LIQUID, 174),
    Species("PYRL", "pyrrole", Phase.LIQUID, 123),
    Species("PYR1", "pyridine", Phase.LIQUID, 117),
    Species("PRO2", "2-propanol", Phase.LIQUID, 117),
    Species("MAS", "methyl acetate", Phase.LIQUID, 115),
    Species("EMS", "ethyl methyl sulfide", Phase.LIQUID, 115),
    Species("DETE", "diethyl ether", Phase.LIQUID, 110),
    Species("MIMI", "1-methylimidazole", Phase.LIQUID, 109),
    Species("CPEN", "cyclopentane", Phase.LIQUID, 106),
    Species("PRLD", "pyrrolidine", Phase.LIQUID, 104),
    Species("THF", "tetrahydrofuran", Phase.LIQUID, 103),
    Species("PRAM", "propylamine", Phase.LIQUID, 81),
    Species("ETAC", "ethyl acetate", Phase.LIQUID, 79),
    Species("PENT", "pentane", Phase.LIQUID, 74),
    Species("HEXA", "hexane", Phase.LIQUID, 61),
    Species("ACN", "acetonitrile", Phase.LIQUID, 45),
    Species("DCM", "dichloromethane", Phase.LIQUID, 42),
    # ---- Gas at 298 K / 1 atm ---------------------------------------------
    Species("MAM1", "methylamine", Phase.GAS, 194, nbp_K=266.8),
    Species("MESH", "methanethiol", Phase.GAS, 192, nbp_K=279.1),
    Species("DMAM", "dimethylamine", Phase.GAS, 129, nbp_K=280.0),
    Species("TMAM", "trimethylamine", Phase.GAS, 113, nbp_K=276.0),
    # ---- Solid at 298 K / 1 atm -------------------------------------------
    Species("ACEM", "acetamide", Phase.SOLID, 205, mp_K=353.0),
    Species("IMIA", "imidazole", Phase.SOLID, 204, mp_K=362.0),
    Species("PHEN", "phenol", Phase.SOLID, 176, mp_K=314.0),
    # ---- Ions: no pure-liquid reference exists ----------------------------
    Species("CLA", "chloride", Phase.ION, 241),
    Species("NH4", "ammonium", Phase.ION, 162),
    Species("FORA", "formate", Phase.ION, 158),
    Species("ACET", "acetate", Phase.ION, 115),
    Species("MAMM", "methylammonium", Phase.ION, 109),
    Species("POT", "potassium", Phase.ION, 107),
    Species("MGUA", "methylguanidinium", Phase.ION, 106),
    Species("IMIM", "imidazolium", Phase.ION, 103),
    Species("SOD", "sodium", Phase.ION, 95),
    Species("LIT", "lithium", Phase.ION, 83),
)

BY_RESNAME = {s.resname: s for s in SPECIES}


def runnable_at(T_K: float, P_atm: float = 1.0) -> tuple[list[Species], list[Species]]:
    """Species confirmed liquid at this state point, and those we cannot confirm.

    Returns ``(confirmed, unknown)``. The split exists because most rows carry no
    melting point: only the species that are *solid* at 298 K needed one for the
    298 K classification. Without ``mp_K`` there is no way to tell a liquid from
    a frozen solid below ambient, and an earlier version of this function
    silently reported water and benzene as runnable at 90 K. A species is
    therefore only *confirmed* when the state point can actually be checked
    against known transitions -- everything else is returned as ``unknown`` for
    the caller to look up, never quietly included.
    """
    confirmed: list[Species] = []
    unknown: list[Species] = []
    for s in SPECIES:
        if s.phase_298 is Phase.ION:
            continue
        ambient = P_atm <= 1.5
        # At 298 K / 1 atm the phase_298 classification is itself the answer.
        if abs(T_K - 298.15) < 5.0 and ambient:
            if s.phase_298 in (Phase.LIQUID, Phase.NIST_EOS):
                (confirmed if s.nbp_K is None or s.nbp_K > T_K else unknown).append(s)
            continue
        if s.nbp_K is not None and T_K > s.nbp_K and ambient:
            continue  # a gas here: excluded outright, not "unknown"
        if s.mp_K is not None and T_K < s.mp_K:
            continue  # a solid here
        if s.mp_K is None and T_K < 298.15:
            unknown.append(s)  # cannot rule out that it has frozen
            continue
        confirmed.append(s)
    return confirmed, unknown


def summary() -> str:
    from collections import Counter

    c = Counter(s.phase_298 for s in SPECIES)
    n_ref = sum(1 for s in SPECIES for st in s.states if st.verified)
    lines = [
        f"{len(SPECIES)} species classified of 94 residues in the DES set",
        *(f"  {p.value:9s} {c[p]:3d}" for p in Phase if c[p]),
        f"  verified reference state points: {n_ref}",
    ]
    return "\n".join(lines)


if __name__ == "__main__":
    print(summary())
