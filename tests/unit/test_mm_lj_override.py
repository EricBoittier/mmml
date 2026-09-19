"""Per-atom LJ override of the dynamic MM path (``update_mm_pairs.energy_with_lj``).

Builds ``build_mm_energy_forces_fn`` on a tiny synthetic PBC system with the
PyCHARMM PSF/param modules and the pair-list backend stubbed out (no
libcharmm.so), then checks the override against the production MM energy, an
independent numpy reference (switch written out from its documented formula),
epsilon linearity and a finite-difference gradient through
:func:`mmml.fit.lj_theta.per_atom_lj`. Also pins the production ``_sharpstep``
switch and guards against the double ``- 1`` on ``psf.get_iac()`` (already
0-based) in CLI callers. Energies in kcal/mol.
"""

from __future__ import annotations

import re
import sys
from itertools import combinations
from pathlib import Path
from unittest import mock

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from mmml.fit.lj_theta import LjTypeMap, init_theta, per_atom_lj

ML_SWITCH_WIDTH_A = 1.5
MM_SWITCH_ON_A = 6.0
MM_SWITCH_WIDTH_A = 5.0
BOX_A = 30.0
COULOMB_KCAL_A = 3.32063711e2  # mm_energy_forces.coulombs_constant

# CGenFF-style table: type -> (epsilon <= 0 kcal/mol, Rmin/2 A)
ATC = ["XA", "XB", "XH"]
LJ_TABLE = {"XA": (-0.110, 2.00), "XB": (-0.050, 1.80), "XH": (-0.030, 1.30)}
MONOMER_TYPES = [0, 1, 2]  # 3-atom monomer: XA, XB, XH
MONOMER_CHARGES = [0.25, -0.35, 0.10]
N_MONOMERS = 4
ATOMS_PER_MONOMER = 3
REPO_ROOT = Path(__file__).resolve().parents[2]

# Switch exponents as documented for the complementary handoff (cutoffs.py):
# ML->MM handoff gamma 1, MM taper gamma 3. Hard-coded on purpose (not imported).
REF_GAMMA_ON = 1.0
REF_GAMMA_OFF = 3.0


def _ref_sharpstep(r: float, x0: float, x1: float, gamma: float) -> float:
    """Smootherstep 6s^5 - 15s^4 + 10s^3 of s = clip((r - x0)/(x1 - x0), 0, 1)**gamma."""
    s = min(max((r - x0) / (x1 - x0), 0.0), 1.0) ** gamma
    return 6.0 * s**5 - 15.0 * s**4 + 10.0 * s**3


def _geometry() -> np.ndarray:
    """4 triatomics with COM separations spread over the MM switch window."""
    rng = np.random.default_rng(7)
    centers = np.array(
        [[5.0, 5.0, 5.0], [10.5, 5.0, 5.0], [5.0, 12.0, 5.0], [28.0, 4.0, 6.0]]
    )  # monomer 3 sits across the x boundary from monomer 0 (MIC ~7 A)
    local = np.array([[0.0, 0.0, 0.0], [1.2, 0.0, 0.0], [-0.5, 0.9, 0.0]])
    pos = []
    for c in centers:
        rot, _ = np.linalg.qr(rng.normal(size=(3, 3)))
        pos.append(c + local @ rot.T)
    return np.concatenate(pos, axis=0)


def _all_inter_pairs() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    i_list, j_list = [], []
    for a, b in combinations(range(N_MONOMERS * ATOMS_PER_MONOMER), 2):
        if a // ATOMS_PER_MONOMER != b // ATOMS_PER_MONOMER:
            i_list.append(a)
            j_list.append(b)
    pi = np.asarray(i_list, dtype=np.int32)
    pj = np.asarray(j_list, dtype=np.int32)
    return pi, pj, np.ones(len(pi), dtype=np.float64)


@pytest.fixture(scope="module")
def x64():
    prev = bool(jax.config.read("jax_enable_x64"))
    jax.config.update("jax_enable_x64", True)
    yield
    jax.config.update("jax_enable_x64", prev)


@pytest.fixture(scope="module")
def mm_system(x64, tmp_path_factory):
    """(mm_fn, update_mm_pairs, positions, pair_idx, pair_mask, charges)."""
    import mmml.interfaces.pycharmmInterface.mm_energy_forces as mef

    tmp: Path = tmp_path_factory.mktemp("cgenff")
    rtf = tmp / "toy.rtf"
    rtf.write_text("".join(f"ATOM X{k} {t} {q}\n" for k, (t, q) in enumerate(zip(ATC, MONOMER_CHARGES))))
    prm = tmp / "toy.prm"
    # CGenFF parser keeps only lines with >4 tokens: the trailing comment matters.
    prm.write_text("".join(f"{t} 0.0 {e} {r} ! toy\n" for t, (e, r) in LJ_TABLE.items()))

    n_atoms = N_MONOMERS * ATOMS_PER_MONOMER
    charges = np.tile(MONOMER_CHARGES, N_MONOMERS).astype(np.float64)
    iac = np.tile(MONOMER_TYPES, N_MONOMERS).astype(np.int32)
    positions = _geometry()
    pi, pj, pm = _all_inter_pairs()

    fake_psf = mock.MagicMock()
    fake_psf.get_charges.return_value = charges
    fake_psf.get_iac.return_value = iac  # 0-based, as the venv pycharmm returns it
    fake_param = mock.MagicMock()
    fake_param.get_atc.return_value = list(ATC)
    fake_pycharmm = mock.MagicMock()
    fake_pycharmm.psf = fake_psf
    fake_pycharmm.param = fake_param
    modules = {"pycharmm": fake_pycharmm, "pycharmm.psf": fake_psf, "pycharmm.param": fake_param}
    backend_out = (pi, pj, pm, len(pi), len(pi), "cell_list")

    with (
        mock.patch.dict(sys.modules, modules),
        mock.patch.multiple(
            mef,
            CGENFF_PRM=str(prm),
            CGENFF_RTF=str(rtf),
            have_jax_md=mock.MagicMock(return_value=False),
            have_vesin=mock.MagicMock(return_value=False),
            _cell_list_pairs=object(),
            build_mm_pairs_with_backend=mock.MagicMock(return_value=backend_out),
            resolve_mm_nl_backend=mock.MagicMock(return_value="cell_list"),
            pick_static_rebuild_backend=mock.MagicMock(return_value="cell_list"),
            _get_actual_psf_charges=mock.MagicMock(return_value=charges),
        ),
    ):
        mm_fn, update_mm_pairs = mef.build_mm_energy_forces_fn(
            positions,
            total_atoms=n_atoms,
            n_monomers=N_MONOMERS,
            monomer_offsets=np.arange(N_MONOMERS + 1) * ATOMS_PER_MONOMER,
            atoms_per_monomer_list=[ATOMS_PER_MONOMER] * N_MONOMERS,
            lambda_monomer=np.ones(N_MONOMERS),
            ml_switch_width=ML_SWITCH_WIDTH_A,
            mm_switch_on=MM_SWITCH_ON_A,
            mm_switch_width=MM_SWITCH_WIDTH_A,
            pbc_cell=np.diag([BOX_A] * 3),
            use_jax_md_neighbor_list=False,
            mm_nl_backend="cell_list",
            lr_solver="mic",
            ml_compute_dtype="float64",
            defer_xla_gpu_warmup=True,
        )
    pair_idx = jnp.stack([jnp.asarray(pi), jnp.asarray(pj)], axis=1)
    return mm_fn, update_mm_pairs, positions, pair_idx, jnp.asarray(pm), charges


def _energy(mm_system, **kw) -> float:
    _, up, pos, pair_idx, pair_mask, _ = mm_system
    cell = jnp.eye(3) * BOX_A
    return float(up.energy_with_lj(jnp.asarray(pos), pair_idx, pair_mask, cell, **kw))


def _numpy_reference(positions, charges, rmins, epsilons, atom_cut: float | None = None) -> float:
    """Independent CHARMM LJ (Rmin/2, eps<=0) + Coulomb with the COM handoff switch.

    ``atom_cut`` (A) drops atom pairs at r >= atom_cut, like a hard pair-list radius.
    """

    def mic(d):
        return d - BOX_A * np.round(d / BOX_A)

    mono = np.arange(len(positions)) // ATOMS_PER_MONOMER
    com = np.stack([positions[mono == m].mean(0) for m in range(N_MONOMERS)])
    total = 0.0
    for a, b in combinations(range(len(positions)), 2):
        if mono[a] == mono[b]:
            continue
        r_com = np.linalg.norm(mic(com[mono[b]] - com[mono[a]]))
        handoff = _ref_sharpstep(r_com, MM_SWITCH_ON_A - ML_SWITCH_WIDTH_A, MM_SWITCH_ON_A, REF_GAMMA_ON)
        taper = 1.0 - _ref_sharpstep(r_com, MM_SWITCH_ON_A, MM_SWITCH_ON_A + MM_SWITCH_WIDTH_A, REF_GAMMA_OFF)
        r = np.linalg.norm(mic(positions[b] - positions[a]))
        if atom_cut is not None and r >= atom_cut:
            continue
        rm = rmins[a] + rmins[b]
        eps = np.sqrt(epsilons[a] * epsilons[b])
        x6 = (rm / r) ** 6
        e_pair = eps * (x6**2 - 2.0 * x6) + COULOMB_KCAL_A * charges[a] * charges[b] / r
        total += handoff * taper * e_pair
    return total


def test_exposed_base_arrays_match_psf_types(mm_system):
    _, up, *_ = mm_system
    assert list(up.atc_names) == ATC
    np.testing.assert_array_equal(up.at_codes, np.tile(MONOMER_TYPES, N_MONOMERS))
    exp_eps = [LJ_TABLE[ATC[c]][0] for c in up.at_codes]
    exp_rm = [LJ_TABLE[ATC[c]][1] for c in up.at_codes]
    np.testing.assert_allclose(np.asarray(up.lj_epsilons), exp_eps)
    np.testing.assert_allclose(np.asarray(up.lj_rmins), exp_rm)
    assert np.all(np.asarray(up.lj_epsilons) <= 0.0)


def test_none_override_equals_base_arrays_equals_production(mm_system):
    mm_fn, up, pos, pair_idx, pair_mask, _ = mm_system
    e_none = _energy(mm_system)
    e_base = _energy(mm_system, lj_rmins=up.lj_rmins, lj_epsilons=up.lj_epsilons)
    e_prod, _forces = mm_fn(jnp.asarray(pos), pair_idx, pair_mask, box_override=jnp.eye(3) * BOX_A)
    assert abs(e_none) > 1e-3  # geometry actually sits inside the MM window
    assert e_base == pytest.approx(e_none, abs=1e-10)
    assert float(e_prod) == pytest.approx(e_none, abs=1e-9)


def test_matches_numpy_reference_with_override(mm_system):
    _, up, pos, _, _, charges = mm_system
    rng = np.random.default_rng(3)
    rm = np.asarray(up.lj_rmins) * rng.uniform(0.9, 1.1, size=pos.shape[0])
    ep = np.asarray(up.lj_epsilons) * rng.uniform(0.5, 1.5, size=pos.shape[0])
    ref0 = _numpy_reference(pos, charges, np.asarray(up.lj_rmins), np.asarray(up.lj_epsilons))
    ref1 = _numpy_reference(pos, charges, rm, ep)
    assert _energy(mm_system) == pytest.approx(ref0, rel=1e-9, abs=1e-9)
    got = _energy(mm_system, lj_rmins=jnp.asarray(rm), lj_epsilons=jnp.asarray(ep))
    assert got == pytest.approx(ref1, rel=1e-9, abs=1e-9)
    assert abs(ref1 - ref0) > 1e-4  # the override changed something


def test_uniform_epsilon_scale_is_linear_in_lj_part(mm_system):
    _, up, *_ = mm_system
    e0 = _energy(mm_system, lj_epsilons=jnp.zeros_like(up.lj_epsilons))  # Coulomb only
    e1 = _energy(mm_system)
    for s in (0.5, 1.7, 3.0):
        es = _energy(mm_system, lj_epsilons=s * up.lj_epsilons)
        assert es - e0 == pytest.approx(s * (e1 - e0), rel=1e-10, abs=1e-12)


def test_theta_gradient_matches_central_differences(mm_system):
    _, up, pos, pair_idx, pair_mask, _ = mm_system
    type_map = LjTypeMap.from_atc(up.at_codes, up.atc_names, fit_types=["XA", "XB", "XH"])
    cell = jnp.eye(3) * BOX_A
    positions = jnp.asarray(pos)

    def energy(theta):
        rm, ep = per_atom_lj(theta, type_map, up.lj_rmins, up.lj_epsilons)
        return up.energy_with_lj(positions, pair_idx, pair_mask, cell, lj_rmins=rm, lj_epsilons=ep)

    theta = init_theta(type_map)
    theta = {"log_eps": theta["log_eps"] + 0.1, "log_sig": theta["log_sig"] - 0.02}
    grad = jax.grad(energy)(theta)
    h = 1e-5
    for key in ("log_eps", "log_sig"):
        g = np.asarray(grad[key])
        assert np.all(np.isfinite(g))
        for k in range(g.size):
            tp = {**theta, key: theta[key].at[k].add(h)}
            tm = {**theta, key: theta[key].at[k].add(-h)}
            fd = (float(energy(tp)) - float(energy(tm))) / (2 * h)
            assert g[k] == pytest.approx(fd, rel=1e-6, abs=1e-9), (key, k)


def test_unfitted_type_keeps_base_parameters(mm_system):
    _, up, *_ = mm_system
    type_map = LjTypeMap.from_atc(up.at_codes, up.atc_names, fit_types=["XA"])
    theta = {"log_eps": jnp.array([0.3]), "log_sig": jnp.array([0.05])}
    rm, ep = per_atom_lj(theta, type_map, up.lj_rmins, up.lj_epsilons)
    is_xa = np.asarray(up.at_codes) == 0
    np.testing.assert_allclose(np.asarray(ep)[~is_xa], np.asarray(up.lj_epsilons)[~is_xa])
    np.testing.assert_allclose(np.asarray(rm)[~is_xa], np.asarray(up.lj_rmins)[~is_xa])
    np.testing.assert_allclose(np.asarray(ep)[is_xa], np.exp(0.3) * np.asarray(up.lj_epsilons)[is_xa])
    assert _energy(mm_system, lj_rmins=rm, lj_epsilons=ep) != pytest.approx(_energy(mm_system))


@pytest.mark.xfail(
    strict=True,
    reason="sqrt(eps_i*eps_j) combining rule has an infinite derivative at eps=0: "
    "a zero-epsilon atom (e.g. a dummy/unparsed type) makes the whole LJ gradient NaN",
)
def test_gradient_finite_with_zero_epsilon_atom(mm_system):
    _, up, pos, pair_idx, pair_mask, _ = mm_system
    cell = jnp.eye(3) * BOX_A
    base_ep = up.lj_epsilons.at[2].set(0.0)  # one XH atom without LJ

    def energy(log_s):
        return up.energy_with_lj(jnp.asarray(pos), pair_idx, pair_mask, cell, lj_epsilons=base_ep * jnp.exp(log_s))

    assert np.isfinite(float(jax.grad(energy)(0.0)))


def test_reference_geometry_samples_both_switch_windows():
    """The numpy reference only tests the switch shape where COM pairs actually fall."""
    pos = _geometry()
    com = pos.reshape(N_MONOMERS, ATOMS_PER_MONOMER, 3).mean(1)
    r = []
    for a, b in combinations(range(N_MONOMERS), 2):
        d = com[b] - com[a]
        r.append(np.linalg.norm(d - BOX_A * np.round(d / BOX_A)))
    r = np.asarray(r)
    lo, on, off = MM_SWITCH_ON_A - ML_SWITCH_WIDTH_A, MM_SWITCH_ON_A, MM_SWITCH_ON_A + MM_SWITCH_WIDTH_A
    assert np.any((r > lo) & (r < on)), r  # inside the ML->MM handoff
    assert np.any((r > on) & (r < off)), r  # inside the MM taper


@pytest.mark.parametrize("gamma", [1.0, 3.0])
def test_production_sharpstep_matches_documented_formula(x64, gamma):
    from mmml.interfaces.pycharmmInterface.calculator_utils import _sharpstep

    x0, x1 = 6.0, 11.0
    r = np.linspace(4.0, 13.0, 181)
    got = np.asarray(_sharpstep(jnp.asarray(r), x0, x1, gamma=gamma))
    ref = np.array([_ref_sharpstep(float(v), x0, x1, gamma) for v in r])
    np.testing.assert_allclose(got, ref, rtol=0, atol=1e-14)
    # endpoints, and the midpoint of the gamma=1 smootherstep
    assert float(_sharpstep(jnp.asarray(x0), x0, x1, gamma=gamma)) == 0.0
    assert float(_sharpstep(jnp.asarray(x1), x0, x1, gamma=gamma)) == pytest.approx(1.0, abs=1e-15)
    if gamma == 1.0:
        assert float(_sharpstep(jnp.asarray(8.5), x0, x1, gamma=1.0)) == pytest.approx(0.5)
    # C1 at the window edges: one-sided derivatives vanish, and it is monotone inside
    dfn = jax.grad(lambda x: _sharpstep(x, x0, x1, gamma=gamma))
    for edge in (x0 + 1e-6, x1 - 1e-6, x0 - 0.1, x1 + 0.1):
        assert abs(float(dfn(jnp.asarray(edge)))) < 1e-5, edge
    assert np.all(np.diff(got) >= -1e-15)


def test_switch_gamma_constants_are_documented_values():
    from mmml.interfaces.pycharmmInterface.cutoffs import GAMMA_OFF, GAMMA_ON

    assert (GAMMA_ON, GAMMA_OFF) == (REF_GAMMA_ON, REF_GAMMA_OFF)


# ``pycharmm.psf.get_iac()`` already returns 0-based codes (it applies ``i - 1``);
# subtracting 1 again maps every atom to the previous CGenFF type (OG2D3 -> OG2D2 ...).
# The md-system jaxmd runner is unaffected (no at_codes_override); these callers are.
_IAC_DOUBLE_OFFSET = re.compile(r"get_iac\(\)[^\n]*\)\s*-\s*1\b")
_IAC_CALLERS = [
    "mmml/cli/run/md_pbc_suite/ase.py",
    "mmml/cli/run/md_evaluate_npz.py",
    "mmml/cli/run/lambda_jaxmd.py",
    "mmml/cli/run/lambda_dynamics.py",
]


def test_default_mm_path_uses_get_iac_unshifted(mm_system):
    """Documentation only: the builder passes the (mocked, 0-based) get_iac through.

    The mock is 0-based by construction, so this does not prove the real pycharmm is;
    :func:`test_pycharmm_get_iac_is_zero_based` checks that against the source.
    """
    _, up, *_ = mm_system
    names = [up.atc_names[c] for c in up.at_codes]
    assert names == [ATC[t] for t in np.tile(MONOMER_TYPES, N_MONOMERS)]


def _pycharmm_psf_source() -> Path | None:
    """``pycharmm/psf.py`` without importing pycharmm (which needs libcharmm.so)."""
    import importlib.util
    import os

    try:
        spec = importlib.util.find_spec("pycharmm")
    except (ImportError, ValueError):
        spec = None
    candidates = []
    if spec is not None and spec.submodule_search_locations:
        candidates += [Path(d) / "psf.py" for d in spec.submodule_search_locations]
    if os.environ.get("CHARMM_HOME"):
        candidates.append(Path(os.environ["CHARMM_HOME"]) / "tool/pycharmm/pycharmm/psf.py")
    return next((c for c in candidates if c.is_file()), None)


def test_pycharmm_get_iac_is_zero_based():
    """The real ``pycharmm.psf.get_iac`` subtracts 1 from CHARMM's 1-based IAC itself."""
    src_path = _pycharmm_psf_source()
    if src_path is None:
        pytest.skip("pycharmm source (psf.py) not available")
    src = src_path.read_text()
    m = re.search(r"def get_iac\(\):(.*?)(?=\ndef |\Z)", src, flags=re.S)
    assert m, f"get_iac not found in {src_path}"
    body = m.group(1)
    assert re.search(r"\[\s*i\s*-\s*1\s+for\s+i\s+in\s+iac\s*\]", body), (
        f"{src_path}: get_iac no longer shifts to 0-based; revisit the at_codes callers"
    )


def test_double_offset_pattern():
    assert _IAC_DOUBLE_OFFSET.search("at_codes = np.asarray(psf.get_iac(), dtype=int) - 1")
    assert not _IAC_DOUBLE_OFFSET.search("at_codes = np.asarray(psf.get_iac(), dtype=int)")
    assert not _IAC_DOUBLE_OFFSET.search("iac = np.array(psf.get_iac()) - 10")


@pytest.mark.parametrize("rel_path", _IAC_CALLERS)
@pytest.mark.xfail(
    strict=True,
    raises=AssertionError,
    reason="known bug: caller subtracts 1 from the already 0-based psf.get_iac(); "
    "drop the '- 1' there and remove this xfail",
)
def test_callers_do_not_double_shift_get_iac(rel_path):
    src = (REPO_ROOT / rel_path).read_text()
    hits = [ln.strip() for ln in src.splitlines() if _IAC_DOUBLE_OFFSET.search(ln)]
    assert not hits, f"{rel_path}: {hits}"


# Pair-list radius dependence (pre-existing, mm_energy_forces): the MM switch acts on
# dimer COM distance, but atom pairs are dropped hard at the pair-list radius
# (mm_switch_on + mm_switch_width + skin). A dimer with COM inside the taper can have
# atom pairs outside the list, so the energy depends on the list radius / Verlet age.
# For reweighting, re-evaluate reference and theta energies with one fixed list policy.


def _atom_and_com_distances() -> tuple[np.ndarray, np.ndarray]:
    pos = _geometry()
    pi, pj, _ = _all_inter_pairs()
    d = pos[pj] - pos[pi]
    d -= BOX_A * np.round(d / BOX_A)
    com = pos.reshape(N_MONOMERS, ATOMS_PER_MONOMER, 3).mean(1)
    dc = com[pj // ATOMS_PER_MONOMER] - com[pi // ATOMS_PER_MONOMER]
    dc -= BOX_A * np.round(dc / BOX_A)
    return np.linalg.norm(d, axis=1), np.linalg.norm(dc, axis=1)


def _production_list_radius_A() -> float:
    from mmml.interfaces.pycharmmInterface.mm_energy_forces import (
        DEFAULT_JAX_MD_SKIN_DISTANCE_A,
        resolve_mm_pair_list_cutoff_A,
    )

    return resolve_mm_pair_list_cutoff_A(MM_SWITCH_ON_A, MM_SWITCH_WIDTH_A, DEFAULT_JAX_MD_SKIN_DISTANCE_A)


def test_geometry_has_weighted_pairs_beyond_list_radius():
    """Precondition: a dimer inside the taper has atom pairs outside the pair list."""
    r_atom, r_com = _atom_and_com_distances()
    rc = _production_list_radius_A()
    cut_but_weighted = (r_atom >= rc) & (r_com < MM_SWITCH_ON_A + MM_SWITCH_WIDTH_A)
    assert np.any(cut_but_weighted), (rc, r_atom.max())


def test_hard_atom_cut_matches_numpy_reference_and_shifts_energy(mm_system):
    """Masking pairs beyond the list radius == numpy reference with a hard atom cut != full."""
    _, up, pos, pair_idx, pair_mask, charges = mm_system
    r_atom, _ = _atom_and_com_distances()
    rc = _production_list_radius_A()
    listed = jnp.asarray((r_atom < rc).astype(np.float64))
    cell = jnp.eye(3) * BOX_A
    e_list = float(up.energy_with_lj(jnp.asarray(pos), pair_idx, pair_mask * listed, cell))
    rm, ep = np.asarray(up.lj_rmins), np.asarray(up.lj_epsilons)
    ref_cut = _numpy_reference(pos, charges, rm, ep, atom_cut=rc)
    ref_all = _numpy_reference(pos, charges, rm, ep)
    assert e_list == pytest.approx(ref_cut, rel=1e-9, abs=1e-9)
    assert abs(ref_all - ref_cut) > 1e-6  # the hard cut changes the switched energy


@pytest.mark.xfail(
    strict=True,
    reason="pre-existing: atom pairs are cut hard at the pair-list radius while the COM "
    "switch still weights their dimer, so E depends on the list radius (and Verlet age)",
)
def test_energy_independent_of_pair_list_radius(mm_system):
    _, up, pos, pair_idx, pair_mask, _ = mm_system
    r_atom, _ = _atom_and_com_distances()
    cell = jnp.eye(3) * BOX_A
    listed = jnp.asarray((r_atom < _production_list_radius_A()).astype(np.float64))
    e_full = float(up.energy_with_lj(jnp.asarray(pos), pair_idx, pair_mask, cell))
    e_list = float(up.energy_with_lj(jnp.asarray(pos), pair_idx, pair_mask * listed, cell))
    assert e_list == pytest.approx(e_full, rel=1e-12, abs=1e-12)
