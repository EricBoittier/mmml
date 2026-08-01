"""Repo-wide guard against the unit-constant bug class.

Ten unit-constant bugs were found in one audit. Every one of them shared a
shape: a conversion factor written as a *module-local literal*, used on both
the write and the read side, so the value round-tripped perfectly and no
internal check disagreed. Only physics disagreed.

``mmml/data/units.py`` is anchored against CODATA by
``tests/unit/test_units_conversions.py``, but that is not where the bugs were.
They were in the ~50 modules that keep their own copy of a factor:

* ``_AMU_ANG_PS2_TO_KCALMOL = 1.036427e-3`` in ``charmm_ase_velocities`` -- the
  mantissa of the *eV* conversion with the wrong exponent. Factor 2.306 out, so
  every equilibration run through it ran at ~690 K while asking for 300 K.
* ``1.88873`` in ``dcmnet/loss.py`` -- transposed digits of the Angstrom->bohr
  factor 1.8897261, a 5.3e-4 relative error, reported as Debye.

Neither is visible to a test of the module that holds it. Both are visible the
moment you compare the literal to a number computed somewhere else, which is
what this file does:

1. Any module-level constant whose name matches one in ``mmml.data.units`` must
   agree with the canonical value (``test_shadowing_definitions_*``). The tree
   holds ~33 such copies; the worst legitimate rounding among them is 1.5e-5,
   so ``_TOL`` sits at 1e-4 -- loose enough for rounded literals, tight enough
   that the 5.3e-4 transposition above fails.
2. Named conversions that have no canonical twin are anchored one by one
   against SI/CODATA values spelled out below (``test_anchored_*``).
3. The AKMA velocity constant is additionally cross-checked against the eV-side
   constant for the same physical quantity, which is precisely the pair that
   drifted (``test_akma_and_ev_velocity_constants_describe_the_same_physics``).

Values are read from source with ``ast``, not by importing, so the guard covers
modules whose imports need JAX/PySCF/CHARMM and cannot be imported in CI.
"""

from __future__ import annotations

import ast
from pathlib import Path
from typing import Iterator, NamedTuple

import pytest

from mmml.data import units as units_module

MMML_ROOT = Path(units_module.__file__).resolve().parent.parent
CANONICAL_SOURCE = MMML_ROOT / "data" / "units.py"

# Rounded literals in the tree deviate by at most ~1.5e-5 from canonical; the
# historical transposition typo was 5.3e-4. Anything in between is a rounding
# choice, anything above is a mistake.
_TOL = 1e-4

# --- CODATA 2018 / SI definitions, spelled out so they are independent -------
_BOHR_A = 0.529177210903
_HARTREE_EV = 27.211386245988
_HARTREE_KCAL = 627.5094740631
_E_C = 1.602176634e-19  # elementary charge, C (exact)
_N_A = 6.02214076e23  # Avogadro, 1/mol (exact)
_AMU_KG = 1.66053906660e-27
_CAL_J = 4.184  # thermochemical calorie (exact)
_C_CM_S = 29979245800.0  # speed of light, cm/s (exact)
_EPS0 = 8.8541878128e-12  # vacuum permittivity, F/m

_EV_KCAL = _HARTREE_KCAL / _HARTREE_EV
# e^2 / (4 pi eps0) at 1 Angstrom, in eV.
_COULOMB_EV_A = _E_C / (4.0 * 3.141592653589793 * _EPS0 * 1e-10)


class Definition(NamedTuple):
    """A module-level constant assignment found in the package source."""

    path: Path
    lineno: int
    name: str
    value: float

    @property
    def where(self) -> str:
        return f"{self.path.relative_to(MMML_ROOT.parent)}:{self.lineno}"


def _literal_value(node: ast.expr) -> float | None:
    """Value of a numeric literal or a literal-only arithmetic expression."""
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return float(node.value)
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        inner = _literal_value(node.operand)
        return None if inner is None else -inner
    if isinstance(node, ast.BinOp):
        left = _literal_value(node.left)
        right = _literal_value(node.right)
        if left is None or right is None:
            return None
        try:
            if isinstance(node.op, ast.Div):
                return left / right
            if isinstance(node.op, ast.Mult):
                return left * right
            if isinstance(node.op, ast.Add):
                return left + right
            if isinstance(node.op, ast.Sub):
                return left - right
            if isinstance(node.op, ast.Pow):
                return float(left**right)
        except (ZeroDivisionError, OverflowError, ValueError):
            return None
    return None


def _iter_definitions(path: Path) -> Iterator[Definition]:
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"))
    except (SyntaxError, UnicodeDecodeError):  # pragma: no cover - source is valid
        return
    for node in tree.body:
        if not isinstance(node, (ast.Assign, ast.AnnAssign)):
            continue
        targets = node.targets if isinstance(node, ast.Assign) else [node.target]
        if node.value is None:
            continue
        value = _literal_value(node.value)
        if value is None:
            continue
        for target in targets:
            if isinstance(target, ast.Name):
                yield Definition(path, node.lineno, target.id, value)


def _package_files() -> list[Path]:
    return sorted(p for p in MMML_ROOT.rglob("*.py") if p.resolve() != CANONICAL_SOURCE)


_CANONICAL: dict[str, float] = {
    name: value
    for name, value in vars(units_module).items()
    if name.isupper() and isinstance(value, float)
}


def _shadowing_definitions() -> list[Definition]:
    """Module-level literals that reuse a canonical constant's name."""
    return [
        d
        for path in _package_files()
        for d in _iter_definitions(path)
        if d.name.upper() in _CANONICAL
    ]


_SHADOWING = _shadowing_definitions()


# --- the scanner itself ------------------------------------------------------


def test_the_scanner_finds_the_definitions_it_is_supposed_to_check():
    """A guard that silently scans nothing passes forever.

    If an import rewrite legitimately removes every duplicate, lower this
    number -- do not delete the assertion.
    """
    assert len(_SHADOWING) >= 25
    assert len({d.path for d in _SHADOWING}) >= 15


def test_the_scanner_reads_expressions_not_just_bare_numbers():
    """``1.0 / 418.4`` and ``2.0 ** (1.0 / 6.0)`` must both be seen."""
    assert _literal_value(ast.parse("1.0 / 418.4", mode="eval").body) == pytest.approx(
        1.0 / 418.4
    )
    assert _literal_value(
        ast.parse("2.0 / 2.0 ** (1.0 / 6.0)", mode="eval").body
    ) == pytest.approx(2.0 / 2.0 ** (1.0 / 6.0))
    assert _literal_value(ast.parse("-3", mode="eval").body) == -3.0
    assert _literal_value(ast.parse("some_name", mode="eval").body) is None


def test_the_tolerance_would_have_caught_the_historical_typos():
    """``_TOL`` is only useful if it separates rounding from mistakes."""
    # dcmnet/loss.py: 1.88873 written for 1.8897261 (transposed digits).
    assert abs(1.88873 - (1.0 / _BOHR_A)) / (1.0 / _BOHR_A) > _TOL
    # charmm_ase_velocities.py: eV mantissa, wrong exponent.
    assert abs(1.036427e-3 - 1.0 / 418.4) / (1.0 / 418.4) > _TOL
    # ...while the rounded literals actually in the tree are inside it.
    assert abs(627.509 - _HARTREE_KCAL) / _HARTREE_KCAL < _TOL
    assert abs(0.529177 - _BOHR_A) / _BOHR_A < _TOL


# --- duplicated canonical constants -----------------------------------------


@pytest.mark.parametrize(
    "definition", _SHADOWING, ids=lambda d: f"{d.path.stem}:{d.name}"
)
def test_shadowing_definitions_agree_with_mmml_data_units(definition: Definition):
    """A local copy of a canonical constant must not drift from it.

    These copies are where the bugs lived. ``mmml/data/units.py`` was right
    the whole time.
    """
    canonical = _CANONICAL[definition.name.upper()]
    assert definition.value == pytest.approx(canonical, rel=_TOL), (
        f"{definition.where}: {definition.name} = {definition.value!r} but "
        f"mmml.data.units.{definition.name.upper()} = {canonical!r}. Import the "
        f"canonical constant instead of restating it."
    )


def test_canonical_constants_are_internally_derived_not_restated():
    """Related factors are defined from each other so they cannot disagree."""
    assert units_module.EBOHR_TO_DEBYE == pytest.approx(
        units_module.EANGSTROM_TO_DEBYE / units_module.ANGSTROM_TO_BOHR, rel=1e-12
    )
    assert units_module.HARTREE_BOHR_TO_EV_ANGSTROM == pytest.approx(
        units_module.HARTREE_TO_EV / units_module.BOHR_TO_ANGSTROM, rel=1e-12
    )
    for forward, backward in (
        (units_module.HARTREE_TO_EV, units_module.EV_TO_HARTREE),
        (units_module.EV_TO_KCAL_MOL, units_module.KCAL_MOL_TO_EV),
        (units_module.EANGSTROM_TO_DEBYE, units_module.DEBYE_TO_EANGSTROM),
        (units_module.EBOHR_TO_DEBYE, units_module.DEBYE_TO_EBOHR),
    ):
        assert forward * backward == pytest.approx(1.0, rel=1e-12)


# --- conversions with no canonical twin, anchored one by one ----------------

# (module, constant) -> value derived here from SI/CODATA. Every entry is a
# factor that exists only in that module, so nothing else in the repo would
# disagree with it if it were wrong.
_ANCHORED: dict[tuple[str, str], float] = {
    # 1 kcal/mol == 418.4 amu*A^2/ps^2, exact by the AKMA definition. The bug.
    (
        "interfaces/pycharmmInterface/mlpot/charmm_ase_velocities.py",
        "_AMU_ANG_PS2_TO_KCALMOL",
    ): _AMU_KG * (1e-10 / 1e-12) ** 2 * _N_A / (_CAL_J * 1000.0),
    # Same physical quantity on the eV side: 1 amu*(A/fs)^2 in eV.
    ("models/efield/jax_md.py", "AMU_TO_EV_FS2_ANG2"): _AMU_KG
    * (1e-10 / 1e-15) ** 2
    / _E_C,
    ("cli/run/lambda_dynamics.py", "_EV_TO_KCAL"): _EV_KCAL,
    ("umbrella/mbar.py", "_EV_TO_KCAL"): _EV_KCAL,
    (
        "interfaces/pycharmmInterface/mlpot/grms_thresholds.py",
        "_EV_A_TO_KCALMOL_A",
    ): _EV_KCAL,
    ("data/rmd17.py", "KCAL_TO_EV"): 1.0 / _EV_KCAL,
    ("md/energy/terms/zbl.py", "_BOHR_TO_ANGSTROM"): _BOHR_A,
    # 1 GPa * A^3 in kcal/mol.
    ("analysis/lattice_energy.py", "GPA_A3_TO_KCAL_MOL"): 1e9
    * 1e-30
    * _N_A
    / (_CAL_J * 1000.0),
    ("analysis/lattice_energy.py", "KCAL_MOL_TO_KJ_MOL"): _CAL_J,
    # Atomic unit of electric field in V/Angstrom: E_h / (e * a_0).
    ("models/multipoles/electrostatics.py", "AU_FIELD_TO_V_PER_ANGSTROM"): _HARTREE_EV
    / _BOHR_A,
    # sigma = 2 * Rmin/2 / 2^(1/6).
    ("models/cgenff_mm.py", "RMIN_HALF_TO_SIGMA"): 2.0 / 2.0 ** (1.0 / 6.0),
    ("mode_check/kick.py", "_FS_INV_TO_CM_INV"): 1e15 / _C_CM_S,
    # Coulomb prefactor. The PhysNet one is halved because the pair sum runs
    # over ordered pairs; docs/UNITS_SUMMARY.md used to list that factor of two
    # as an unresolved question.
    (
        "models/physnetjax/physnetjax/models/mpnn_kernels.py",
        "COULOMB_PAIR_FACTOR_EV_A",
    ): _COULOMB_EV_A / 2.0,
    (
        "models/physnetjax/physnetjax/models/zbl.py",
        "COULOMB_EV_ANGSTROM",
    ): _COULOMB_EV_A,
    ("md/energy/terms/zbl.py", "_COULOMB_EV_ANGSTROM"): _COULOMB_EV_A,
    ("spectra/spectra_md.py", "FS_INV_TO_CM_INV"): 1e15 / _C_CM_S,
    ("mode_check/forces.py", "_EV_TO_J"): _E_C,
    ("mode_check/forces.py", "_A_TO_M"): 1e-10,
    ("mode_check/forces.py", "_AMU_TO_KG"): _AMU_KG,
    ("models/efield/calc_spectra.py", "EV_TO_J"): _E_C,
    ("models/efield/calc_spectra.py", "ANG_TO_M"): 1e-10,
    ("models/efield/calc_spectra.py", "AMU_TO_KG"): _AMU_KG,
}


def _read_constant(relative_path: str, name: str) -> float:
    path = MMML_ROOT / relative_path
    assert path.is_file(), (
        f"{relative_path} is gone. If it moved, move its entry in _ANCHORED "
        f"with it rather than dropping the anchor."
    )
    for definition in _iter_definitions(path):
        if definition.name == name:
            return definition.value
    raise AssertionError(
        f"{relative_path} no longer defines a literal {name}. If it now imports "
        f"the value, delete the _ANCHORED entry; if it was renamed, rename the "
        f"entry too."
    )


@pytest.mark.parametrize(
    ("location", "expected"),
    sorted(_ANCHORED.items()),
    ids=[
        f"{path.rsplit('/', 1)[-1]}:{name}" for path, name in sorted(_ANCHORED)
    ],
)
def test_anchored_conversions_match_an_independent_derivation(
    location: tuple[str, str], expected: float
):
    relative_path, name = location
    value = _read_constant(relative_path, name)
    assert value == pytest.approx(expected, rel=_TOL), (
        f"{relative_path}:{name} = {value!r}, but the SI/CODATA value is "
        f"{expected!r} (relative error {abs(value - expected) / abs(expected):.2e})."
    )


def test_akma_and_ev_velocity_constants_describe_the_same_physics():
    """The exact cross-check the original bug failed.

    ``_AMU_ANG_PS2_TO_KCALMOL`` (kcal/mol per amu*(A/ps)^2) and
    ``AMU_TO_EV_FS2_ANG2`` (eV per amu*(A/fs)^2) are the same quantity in
    different units, related by (1 ps / 1 fs)^-2 = 1e-6 and eV -> kcal/mol.
    The wrong value 1.036427e-3 is what you get by reaching for the eV
    constant's digits, and it fails this identity by a factor of 2.3.
    """
    akma = _read_constant(
        "interfaces/pycharmmInterface/mlpot/charmm_ase_velocities.py",
        "_AMU_ANG_PS2_TO_KCALMOL",
    )
    ev_side = _read_constant("models/efield/jax_md.py", "AMU_TO_EV_FS2_ANG2")

    assert akma == pytest.approx(ev_side * 1e-6 * _EV_KCAL, rel=_TOL)
    assert 1.036427e-3 != pytest.approx(ev_side * 1e-6 * _EV_KCAL, rel=_TOL)


def test_the_akma_constant_is_the_exact_charmm_definition():
    akma = _read_constant(
        "interfaces/pycharmmInterface/mlpot/charmm_ase_velocities.py",
        "_AMU_ANG_PS2_TO_KCALMOL",
    )
    assert akma == pytest.approx(1.0 / 418.4, rel=1e-12)
