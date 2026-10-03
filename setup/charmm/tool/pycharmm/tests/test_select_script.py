"""Tests for the persistent-selection store API on SelectAtoms.

Build alanine dipeptide, then exercise:
  - select.store_selection / select.find / select.delete_stored_selection
  - SelectAtoms.store(name) and .unstore()
  - select.get_stored_names / get_num_stored / get_max_name
  - SelectAtoms used as a context manager (auto-store/unstore)
  - SelectAtoms accessor metadata (atom_indexes, chem_types, res_*, etc.)
"""

import pytest

from pycharmm import (
    NonBondedScript,
    SelectAtoms,
    coor,
    gen,
    ic,
    read,
    select,
    settings,
    write,
)
from pycharmm.lingo import charmm_script


@pytest.fixture(scope="module")
def alanine_dipeptide_built():
    """Standard alanine-dipeptide build for selection-store tests."""
    read.rtf("data/top_all36_prot.rtf")
    read.prm("data/par_all36_prot.prm", flex=True)

    read.rtf("data/water_ions.rtf", append=True)
    read.prm("data/water_ions.prm", append=True, flex=True)

    old_warn = settings.set_warn_level(-1)
    old_bomb = settings.set_bomb_level(-1)
    read.prm("data/sodium_oxygen_nbfixes.prm", append=True, flex=True)
    settings.set_warn_level(old_warn)
    settings.set_bomb_level(old_bomb)

    read.sequence_string("ALA")
    gen.new_segment("ADP", "ACE", "CT3", setup_ic=True)
    ic.prm_fill(False)
    ic.seed(1, "CAY", 1, "CY", 1, "N")
    ic.build()
    coor.orient()

    NonBondedScript(
        cutnb=18.0,
        ctonnb=15.0,
        ctofnb=13.0,
        eps=1.0,
        cdie=True,
        atom=True,
        vatom=True,
        fswitch=True,
        vfswitch=True,
    ).run()


def test_store_named_selection(alanine_dipeptide_built):
    """select.store_selection registers a name retrievable via get_stored_names."""
    charmm_script("define STUFF select all end")
    select.store_selection("HYD", select.hydrogen())
    select.find("HYD")

    names = select.get_stored_names()
    assert "HYD" in names
    assert "STUFF" in names


def test_select_store_and_unstore(alanine_dipeptide_built):
    """SelectAtoms.store() / .unstore() round-trip."""
    h_atoms = SelectAtoms().all_hydrogen_atoms()
    name = h_atoms.store()
    assert name in select.get_stored_names()
    h_atoms.unstore()
    assert name not in select.get_stored_names()


def test_select_context_manager(alanine_dipeptide_built):
    """`with SelectAtoms(...)` auto-stores on enter and auto-unstores on exit."""
    pre_count = select.get_num_stored()
    with SelectAtoms(hydrogens=True) as sel:
        name = sel.get_stored_name()
        assert name in select.get_stored_names()
        assert select.get_num_stored() == pre_count + 1
    # after context exit
    assert name not in select.get_stored_names()
    assert select.get_num_stored() == pre_count


def test_select_metadata_accessors(alanine_dipeptide_built):
    """SelectAtoms exposes per-atom metadata as parallel arrays."""
    h_atoms = SelectAtoms().all_hydrogen_atoms()
    n = sum(list(h_atoms))
    assert n > 0
    assert len(h_atoms.get_atom_indexes()) == n
    assert len(h_atoms.get_chem_types()) == n
    assert len(h_atoms.get_res_indexes()) == n
    assert len(h_atoms.get_res_names()) == n
    assert len(h_atoms.get_res_ids()) == n
    assert len(h_atoms.get_seg_indexes()) == n
    assert len(h_atoms.get_seg_ids()) == n
    assert len(h_atoms.get_atom_types()) == n


def test_write_pdb_with_selection(alanine_dipeptide_built, tmp_path):
    """write.coor_pdb honors a hydrogens-only SelectAtoms."""
    out = tmp_path / "testy.pdb"
    write.coor_pdb(str(out), selection=SelectAtoms(hydrogens=True))
    assert out.is_file()
    assert out.stat().st_size > 0


def test_delete_stored_selection(alanine_dipeptide_built):
    """select.delete_stored_selection removes a previously stored name."""
    select.store_selection("TODELETE", select.hydrogen())
    assert "TODELETE" in select.get_stored_names()
    select.delete_stored_selection("TODELETE")
    assert "TODELETE" not in select.get_stored_names()


def test_stored_selection_reused_not_restored(alanine_dipeptide_built, monkeypatch):
    """A pre-stored selection is reused by name in a CommandScript.

    Regression guard: when the caller has already ``store()``-ed a
    selection, ``CommandScript.run()`` must refer to it by its own name
    and must NOT rebuild/store/delete a throwaway copy on every call.
    This is what lets a selection be stored once outside a trajectory
    loop and reused each frame (github dev issue: slowness / retrieval
    errors when a selection is re-saved from every CommandScript).
    """
    import pycharmm.lingo
    import pycharmm.select as _select
    from pycharmm.script import CommandScript

    sel = SelectAtoms(hydrogens=True)
    sel.store("REUSED")
    assert "REUSED" in _select.get_stored_names()

    captured = []
    monkeypatch.setattr(pycharmm.lingo, "charmm_script", lambda s, **k: captured.append(s))
    stored_during_run = []
    monkeypatch.setattr(_select, "store_selection", lambda name, s: stored_during_run.append(name))

    CommandScript("coor", orient=True, selection=sel).run()

    # Command referred to the caller's stored name, not a random throwaway.
    assert any("sele REUSED end" in s for s in captured)
    # No throwaway selection was stored while running the command.
    assert stored_during_run == []
    # The caller's selection is left intact for the next loop iteration.
    assert sel.is_stored()
    assert "REUSED" in _select.get_stored_names()
    sel.unstore()


def test_selection_applied_regardless_of_atom_count(alanine_dipeptide_built, monkeypatch):
    """A passed selection is honored by identity, not by atom count.

    Regression guard for the truthiness edge case: ``SelectAtoms.__len__``
    returns the system atom count, so a plain ``if self.selection:`` would
    wrongly treat a genuine selection as absent when the system has zero
    atoms. ``run()`` must branch on whether a selection was passed
    (``is not None``), not on how many atoms it holds.
    """
    import pycharmm.lingo
    from pycharmm.script import CommandScript

    # A real, stored define so the reuse existence-check succeeds; the
    # wrapper reports len()==0 to mimic a zero-atom system.
    backing = SelectAtoms(hydrogens=True)
    backing.store("ZEROSEL")

    class _ZeroLenSelection:
        def __len__(self):
            return 0

        def is_stored(self):
            return True

        def get_stored_name(self):
            return "ZEROSEL"

    captured = []
    monkeypatch.setattr(pycharmm.lingo, "charmm_script", lambda s, **k: captured.append(s))

    CommandScript("coor", orient=True, selection=_ZeroLenSelection()).run()

    # The selection is applied even though len(selection) == 0.
    assert any("sele ZEROSEL end" in s for s in captured)
    backing.unstore()


def test_reuse_falls_back_when_define_cleared(alanine_dipeptide_built, monkeypatch):
    """A stale is_stored() flag must not produce a dangling selection name.

    If a stored selection's CHARMM define is cleared out from under the
    Python object (structure reload, reset, explicit delete), run() must
    fall back to re-storing a throwaway copy rather than emitting
    ``sele <name> end`` for a name CHARMM no longer knows -- otherwise the
    command errors, which is the failure class behind the original issue.
    """
    import pycharmm.lingo
    import pycharmm.select as _select
    from pycharmm.script import CommandScript

    pre = _select.get_num_stored()
    sel = SelectAtoms(hydrogens=True)
    sel.store("GONE")
    _select.delete_stored_selection("GONE")  # define vanishes
    assert sel.is_stored()  # Python flag now stale
    assert "GONE" not in _select.get_stored_names()

    captured = []
    monkeypatch.setattr(pycharmm.lingo, "charmm_script", lambda s, **k: captured.append(s))

    CommandScript("coor", orient=True, selection=sel).run()

    sele_line = next(s for s in captured if "sele " in s)
    assert "GONE" not in sele_line  # no dangling reference
    # the throwaway copy was cleaned up: no net change in stored count
    assert _select.get_num_stored() == pre


def test_reuse_with_lowercase_stored_name(alanine_dipeptide_built, monkeypatch):
    """A selection stored under a lower/mixed-case name is still reused.

    store() registers names upper-cased in CHARMM, so the reuse existence
    check upper-cases before probing; otherwise a lowercase name would
    silently miss and fall back to the slow re-store path.
    """
    import pycharmm.lingo
    import pycharmm.select as _select
    from pycharmm.script import CommandScript

    sel = SelectAtoms(hydrogens=True)
    sel.store("lower")  # CHARMM registers LOWER
    assert "LOWER" in _select.get_stored_names()

    stored_during_run = []
    monkeypatch.setattr(_select, "store_selection", lambda name, s: stored_during_run.append(name))
    captured = []
    monkeypatch.setattr(pycharmm.lingo, "charmm_script", lambda s, **k: captured.append(s))

    CommandScript("coor", orient=True, selection=sel).run()

    assert stored_during_run == []  # reused, not re-stored
    assert any("sele lower end" in s for s in captured)
    sel.unstore()


def test_reuse_matches_legacy_end_to_end(alanine_dipeptide_built, tmp_path):
    """End-to-end: a reused stored selection drives a real CHARMM command
    identically to the same selection passed unstored.

    No charmm_script stubbing -- this exercises the reuse path through an
    actual WRITE COOR PDB and compares its output against the legacy
    temp-store path.
    """

    def atom_lines(path):
        return [ln for ln in path.read_text().splitlines() if ln.startswith(("ATOM", "HETATM"))]

    legacy = tmp_path / "legacy.pdb"
    write.coor_pdb(str(legacy), selection=SelectAtoms(hydrogens=True))

    bb = SelectAtoms(hydrogens=True)
    bb.store("BBW")
    reused = tmp_path / "reused.pdb"
    write.coor_pdb(str(reused), selection=bb)

    legacy_atoms = atom_lines(legacy)
    reused_atoms = atom_lines(reused)
    assert len(reused_atoms) > 0
    assert reused_atoms == legacy_atoms  # identical selection
    assert bb.is_stored()  # reuse leaves it intact
    bb.unstore()


def test_stored_selection_freed_on_gc(alanine_dipeptide_built):
    """A stored selection dropped without unstore() frees its CHARMM table
    slot once it is collected (prevents the stored-selection table from
    leaking over long runs). Deletion is deferred out of the finalizer, so
    it is applied at the next drain / table op, not inside GC."""
    import gc

    import pycharmm.select as _select
    import pycharmm.select_atoms as _sa

    assert "GCSEL" not in _select.get_stored_names()
    sel = SelectAtoms(hydrogens=True)
    sel.store("GCSEL")
    assert "GCSEL" in _select.get_stored_names()

    del sel
    gc.collect()
    _sa._drain_pending_unstores()
    assert "GCSEL" not in _select.get_stored_names()


def test_explicit_unstore_cancels_gc_finalizer(alanine_dipeptide_built):
    """After an explicit unstore(), collecting the object must not delete a
    same-named selection stored later by a different object -- even when the
    deferred-unstore queue is drained."""
    import gc

    import pycharmm.select as _select
    import pycharmm.select_atoms as _sa

    sel = SelectAtoms(hydrogens=True)
    sel.store("GCSEL2")
    sel.unstore()
    assert "GCSEL2" not in _select.get_stored_names()

    keeper = SelectAtoms(hydrogens=True)
    keeper.store("GCSEL2")  # reuse the name
    del sel
    gc.collect()
    _sa._drain_pending_unstores()
    assert "GCSEL2" in _select.get_stored_names()  # keeper's define survives
    keeper.unstore()


def test_batched_selection_equals_or_of_singles(alanine_dipeptide_built):
    """Batched set-membership selectors equal OR-ing the singular selectors,
    and a single-element list equals the singular selector."""
    import numpy as np

    import pycharmm.select as _select

    def or_singles(single_func, values):
        acc = None
        for v in values:
            s = np.asarray(single_func(v), dtype=bool)
            acc = s if acc is None else (acc | s)
        return acc

    # (batched, singular, values)
    cases = [
        (_select.by_atom_types, _select.by_atom_type, ["N", "CA", "C", "O"]),
        (_select.by_residue_names, _select.by_residue_name, ["ALA", "GLY"]),
        (_select.by_chem_types, _select.by_chem_type, ["CT3", "C", "O"]),
        (_select.by_segment_ids, _select.by_segment_id, ["ADP", "NOPE"]),
        (_select.by_residue_ids, _select.by_residue_id, ["1", "2"]),
    ]
    for batch_func, single_func, values in cases:
        batched = np.asarray(batch_func(values), dtype=bool)
        expected = or_singles(single_func, values)
        assert np.array_equal(batched, expected), f"batched != OR-of-singles for {values}"
        # a single-element list must match the singular selector exactly
        one = np.asarray(batch_func([values[0]]), dtype=bool)
        assert np.array_equal(one, np.asarray(single_func(values[0]), dtype=bool)), values[0]

    # the backbone atom-name selection must actually hit atoms here
    assert np.asarray(_select.by_atom_types(["N", "CA", "C", "O"]), dtype=bool).sum() > 0
