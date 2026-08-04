"""The static complete pair list, and its equivalence to the rebuilt one.

``UmbrellaConfig.static_pairs`` defaults to on, so every hybrid umbrella run
takes this path. The default is justified by one claim: pairs beyond ``ctofnb``
contribute exactly zero through the switching function, so enumerating *every*
intermolecular pair gives the same energy as a cutoff list while skipping the
host rebuild. ``test_matches_rebuilt_neighbor_list_at_the_production_cutoff``
is that claim, and it is the reason these tests exist in CI rather than only in
``scripts/bench_static_vs_neighbor_pairs.py``.
"""

from __future__ import annotations

import numpy as np
import pytest

jax = pytest.importorskip("jax")
import jax.numpy as jnp  # noqa: E402

from mmml.md.static_pairs import make_static_pair_fn, static_pair_count  # noqa: E402
from mmml.md.system import FFParams, MolecularSystem  # noqa: E402

CTOFNB = 12.0
CTONNB = 10.0


def _diatomic_system(n_mol: int = 4, spacing: float = 3.0, box_side: float = 30.0):
    """``n_mol`` two-atom molecules on a line, with intramolecular exclusions."""
    n = 2 * n_mol
    pos = np.zeros((n, 3), dtype=np.float64)
    for m in range(n_mol):
        pos[2 * m] = [m * spacing, 0.0, 0.0]
        pos[2 * m + 1] = [m * spacing + 1.0, 0.0, 0.0]
    charges = np.tile([0.4, -0.4], n_mol)
    ff = FFParams(
        charges=charges,
        epsilon=np.full(n, 0.1),
        rmin_half=np.full(n, 1.5),
        at_codes=np.tile([0, 1], n_mol).astype(np.int32),
        exclusions=np.array([[2 * m, 2 * m + 1] for m in range(n_mol)], dtype=np.int32),
        e14_pairs=np.empty((0, 2), dtype=np.int32),
    )
    return MolecularSystem(
        R=pos,
        Z=np.tile([8, 1], n_mol).astype(np.int32),
        box=np.diag([box_side] * 3),
        mol_id=np.repeat(np.arange(n_mol), 2).astype(np.int32),
        ff_params=ff,
    )


def _water_box(n_mol: int, density_kg_m3: float = 997.0, seed: int = 0):
    """TIP3P water on a jittered lattice at experimental density."""
    rng = np.random.default_rng(seed)
    volume = n_mol * 18.01528 / 6.02214076e23 / (density_kg_m3 * 1e-3) * 1e24
    side = float(volume ** (1.0 / 3.0))
    per_side = int(np.ceil(n_mol ** (1.0 / 3.0)))
    step = side / per_side
    sites = np.array(
        [(x, y, z) for x in range(per_side) for y in range(per_side) for z in range(per_side)],
        dtype=np.float64,
    )[:n_mol] * step
    sites += rng.uniform(-0.15, 0.15, size=sites.shape) * step

    r_oh, ang = 0.9572, np.deg2rad(104.52)
    local = np.array([[0.0, 0.0, 0.0], [r_oh, 0.0, 0.0],
                      [r_oh * np.cos(ang), r_oh * np.sin(ang), 0.0]])
    pos = np.empty((3 * n_mol, 3))
    for m in range(n_mol):
        q, r = np.linalg.qr(rng.normal(size=(3, 3)))
        pos[3 * m:3 * m + 3] = local @ (q * np.sign(np.diag(r))).T + sites[m]

    ff = FFParams(
        charges=np.tile([-0.834, 0.417, 0.417], n_mol),
        epsilon=np.tile([0.1521, 0.046, 0.046], n_mol),
        rmin_half=np.tile([1.7682, 0.2245, 0.2245], n_mol),
        at_codes=np.tile([0, 1, 1], n_mol).astype(np.int32),
        exclusions=np.concatenate([
            np.array([[3 * m, 3 * m + 1], [3 * m, 3 * m + 2], [3 * m + 1, 3 * m + 2]])
            for m in range(n_mol)
        ]).astype(np.int32),
        e14_pairs=np.empty((0, 2), dtype=np.int32),
    )
    system = MolecularSystem(
        R=pos, Z=np.tile([8, 1, 1], n_mol).astype(np.int32),
        box=np.diag([side] * 3),
        mol_id=np.repeat(np.arange(n_mol), 3).astype(np.int32),
        ff_params=ff,
    )
    return system, side


def _mm_energy_and_forces(system, pairs, ctofnb: float = CTOFNB):
    from mmml.interfaces.pycharmmInterface.mm_system_energy import CharmmNbondSettings
    from mmml.md.energy import EnergyContext
    from mmml.md.energy.terms import MMNonbondedTerm

    settings = CharmmNbondSettings(cutnb=ctofnb, ctonnb=CTONNB, ctofnb=ctofnb)
    fn = MMNonbondedTerm(settings).make(system, EnergyContext()).jax_energy_fn

    def e(R):
        return fn(R, pair_i=jnp.asarray(pairs["pair_i"]),
                  pair_j=jnp.asarray(pairs["pair_j"]),
                  pair_mask=jnp.asarray(pairs["pair_mask"]))

    R = jnp.asarray(system.R)
    return float(e(R)), np.asarray(jax.grad(e)(R))


# --------------------------------------------------------------- enumeration


def test_enumerates_every_intermolecular_pair_and_no_intramolecular_one():
    system = _diatomic_system(n_mol=4)
    fn = make_static_pair_fn(system, verbose=False)
    pairs = fn(None, None)
    i = np.asarray(pairs["pair_i"])
    j = np.asarray(pairs["pair_j"])

    # 4 molecules x 2 atoms: every cross-molecule atom pair, C(4,2) * 2 * 2 = 24.
    assert i.shape[0] == 24
    mol = np.asarray(system.mol_id)
    assert np.all(mol[i] != mol[j]), "an intramolecular pair reached the list"
    # No duplicates, and canonical ordering (i < j).
    assert np.all(i < j)
    assert len({(int(a), int(b)) for a, b in zip(i, j)}) == i.shape[0]


def test_static_pair_count_agrees_with_the_list_it_describes():
    system = _diatomic_system(n_mol=5)
    assert static_pair_count(system) == make_static_pair_fn(system, verbose=False).n_pairs


def test_intermolecular_exclusions_are_honoured():
    """Exclusions are intramolecular in practice, but the filter must still run."""
    system = _diatomic_system(n_mol=3)
    baseline = static_pair_count(system)

    # Exclude one genuinely intermolecular pair (atom 0 of mol 0, atom 0 of mol 1).
    ff = system.ff_params
    system2 = MolecularSystem(
        R=system.R, Z=system.Z, box=system.box, mol_id=system.mol_id,
        ff_params=FFParams(
            charges=ff.charges, epsilon=ff.epsilon, rmin_half=ff.rmin_half,
            at_codes=ff.at_codes,
            exclusions=np.vstack([ff.exclusions, np.array([[0, 2]])]).astype(np.int32),
            e14_pairs=ff.e14_pairs,
        ),
    )
    assert static_pair_count(system2) == baseline - 1

    pairs = make_static_pair_fn(system2, verbose=False)(None, None)
    got = {(int(a), int(b)) for a, b in zip(np.asarray(pairs["pair_i"]),
                                            np.asarray(pairs["pair_j"]))}
    assert (0, 2) not in got


def test_mask_is_all_live_because_the_list_is_not_padded():
    system = _diatomic_system(n_mol=4)
    pairs = make_static_pair_fn(system, verbose=False)(None, None)
    mask = np.asarray(pairs["pair_mask"])
    assert mask.shape[0] == np.asarray(pairs["pair_i"]).shape[0]
    assert np.all(mask == 1)


# ------------------------------------------------------------------ contract


def test_is_static_across_positions_and_boxes():
    """The whole point: the list never depends on where the atoms are."""
    system = _diatomic_system(n_mol=4)
    fn = make_static_pair_fn(system, verbose=False)
    first = fn(system.R, system.box)
    moved = fn(system.R + 37.0, np.diag([100.0, 100.0, 100.0]))
    for key in ("pair_i", "pair_j", "pair_mask"):
        np.testing.assert_array_equal(np.asarray(first[key]), np.asarray(moved[key]))


def test_marked_device_native_so_the_driver_skips_the_host_round_trip():
    fn = make_static_pair_fn(_diatomic_system(), verbose=False)
    assert getattr(fn, "device_native", False) is True


def test_verbose_reports_the_pair_count_and_the_upload_size(capsys):
    """The one-off log line is how a run records which pair path it took."""
    system = _diatomic_system(n_mol=4)
    make_static_pair_fn(system, verbose=True)
    out = capsys.readouterr().out
    assert "static pair list" in out
    assert str(static_pair_count(system)) in out
    assert "MB" in out


def test_lambda_and_elec_scale_are_absent_unless_requested():
    fn = make_static_pair_fn(_diatomic_system(), verbose=False)
    payload = fn(None, None)
    assert "lambda_t" not in payload
    assert "elec_scale" not in payload


def test_lambda_updates_value_without_changing_shape_or_dtype():
    """Reusing the compiled graph across windows is the reason this exists.

    A window centre baked in as a Python float forces a fresh XLA compilation
    per window; a device scalar of unchanging shape and dtype does not.
    """
    fn = make_static_pair_fn(_diatomic_system(), verbose=False, with_lambda=True)
    before = fn(None, None)["lambda_t"]
    fn.set_lambda(1.75)
    after = fn(None, None)["lambda_t"]

    assert float(before) == 0.0
    assert float(after) == pytest.approx(1.75)
    assert before.shape == after.shape
    assert before.dtype == after.dtype

    assert float(fn(None, None)["elec_scale"]) == 1.0
    fn.set_elec_scale(0.25)
    assert float(fn(None, None)["elec_scale"]) == pytest.approx(0.25)


# ------------------------------------------------------------------- parity


def test_matches_rebuilt_neighbor_list_at_the_production_cutoff():
    """The claim the ``static_pairs=True`` default rests on.

    Built at ``ctofnb``, the cutoff list and the complete list must give the
    same energy *and* the same forces: the switching function has already
    zeroed everything the cutoff list drops.
    """
    from mmml.md.neighbors import make_intermolecular_neighbor_fn

    system, side = _water_box(500)
    assert side > 2 * CTOFNB, "box must exceed twice the cutoff for a fair test"

    static = make_static_pair_fn(system, verbose=False)(None, None)
    rebuilt = make_intermolecular_neighbor_fn(system, cutoff_A=CTOFNB)(
        np.asarray(system.R), np.asarray(system.box)
    )
    # The cutoff genuinely prunes here, or the comparison proves nothing.
    assert int(np.asarray(rebuilt["pair_mask"]).sum()) < static["pair_i"].shape[0]

    e_static, f_static = _mm_energy_and_forces(system, static)
    e_rebuilt, f_rebuilt = _mm_energy_and_forces(system, rebuilt)

    assert e_static == pytest.approx(e_rebuilt, abs=1e-9)
    np.testing.assert_allclose(f_static, f_rebuilt, atol=1e-9)


def test_a_list_built_below_ctofnb_does_not_match_and_so_the_parity_test_bites():
    """Mutation check on the test above.

    If the parity assertion passed for any cutoff, it would be measuring
    nothing. Truncating inside the switching region must change the answer.
    """
    from mmml.md.neighbors import make_intermolecular_neighbor_fn

    system, _ = _water_box(500)
    static = make_static_pair_fn(system, verbose=False)(None, None)
    truncated = make_intermolecular_neighbor_fn(system, cutoff_A=8.0)(
        np.asarray(system.R), np.asarray(system.box)
    )

    e_static, _ = _mm_energy_and_forces(system, static)
    e_trunc, _ = _mm_energy_and_forces(system, truncated)
    assert abs(e_static - e_trunc) > 1e-3, (
        "truncating below ctofnb changed nothing; the parity test is vacuous"
    )
