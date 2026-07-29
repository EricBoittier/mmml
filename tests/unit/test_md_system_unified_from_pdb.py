"""``--jaxmd-unified`` must honour ``--from-pdb`` instead of the packmol builder.

Regression: ``runconfig_from_md_system_args`` defaulted ``SystemSpec.builder``
to ``"packmol"`` whenever ``--builder`` was unset, and ``run_unified_jaxmd``
called the packmol builder unconditionally. A config carrying only
``from_pdb:`` (a prebuilt make-box cell) therefore reached the composition
builder and died with "packmol builder requires SystemSpec.composition".
"""

from __future__ import annotations

from argparse import Namespace

import pytest

BASE = dict(
    setup="pbc_nve",
    dt_fs=0.5,
    ps=0.05,
    temperature=300,
    seed=0,
    include_mm=True,
    mm_switch_on=8.0,
    checkpoint="examples/m/kl.json",
    output_dir="artifacts/x",
    backend="jaxmd",
    jaxmd_unified=True,
    interaction_policy=None,
    n_molecules=None,
)


def _args(**extra) -> Namespace:
    return Namespace(**BASE, **extra)


def _from_pdb_args(pdb: str = "artifacts/boxes/tip3/model.pdb") -> Namespace:
    return _args(from_pdb=pdb, composition=None, builder=None, box_size=None)


def _composition_args() -> Namespace:
    return _args(from_pdb=None, composition="TIP3:100", builder=None, box_size=30.0)


def test_from_pdb_selects_the_from_pdb_builder() -> None:
    from mmml.md.lowering import runconfig_from_md_system_args

    spec = runconfig_from_md_system_args(_from_pdb_args()).system
    assert spec.builder == "from_pdb"
    assert str(spec.template_pdb) == "artifacts/boxes/tip3/model.pdb"
    assert spec.composition is None


def test_composition_still_selects_packmol() -> None:
    from mmml.md.lowering import runconfig_from_md_system_args

    spec = runconfig_from_md_system_args(_composition_args()).system
    assert spec.builder == "packmol"
    assert spec.template_pdb is None


def test_explicit_builder_still_wins() -> None:
    from mmml.md.lowering import runconfig_from_md_system_args

    args = _args(from_pdb="x.pdb", composition=None, builder="pyxtal", box_size=30.0)
    assert runconfig_from_md_system_args(args).system.builder == "pyxtal"


def test_guard_accepts_from_pdb() -> None:
    from mmml.cli.run.md_system_unified import check_md_system_args_supported

    check_md_system_args_supported(_from_pdb_args())  # must not raise


def test_guard_rejects_neither_from_pdb_nor_composition() -> None:
    """Fail on the missing input, not deep inside the packmol builder."""
    from mmml.cli.run.md_system_unified import check_md_system_args_supported

    args = _args(from_pdb=None, composition=None, builder=None, box_size=30.0)
    with pytest.raises(ValueError, match="--from-pdb"):
        check_md_system_args_supported(args)


def test_guard_rejects_unsupported_builder() -> None:
    from mmml.cli.run.md_system_unified import check_md_system_args_supported

    args = _args(from_pdb=None, composition="TIP3:10", builder="pyxtal", box_size=30.0)
    with pytest.raises(NotImplementedError, match="pyxtal"):
        check_md_system_args_supported(args)


def test_run_unified_jaxmd_routes_on_builder(monkeypatch) -> None:
    """The dispatch, not just the spec: from_pdb must not reach packmol."""
    import mmml.cli.run.md_system_unified as unified
    import mmml.interfaces.pycharmmInterface.import_pycharmm as import_pycharmm

    monkeypatch.setattr(import_pycharmm, "ensure_pycharmm_loaded", lambda: True)
    routed: list[str] = []

    def _stop(name):
        def _fn(*_a, **_k):
            routed.append(name)
            raise SystemExit(name)

        return _fn

    monkeypatch.setattr(unified, "build_from_pdb_system_with_ffparams", _stop("from_pdb"))
    monkeypatch.setattr(unified, "build_packmol_system_with_ffparams", _stop("packmol"))

    for args, expected in ((_from_pdb_args(), "from_pdb"), (_composition_args(), "packmol")):
        routed.clear()
        with pytest.raises(SystemExit):
            unified.run_unified_jaxmd(args)
        assert routed == [expected]


def test_builder_stays_from_pdb_after_the_alias_sets_composition() -> None:
    """``apply_from_pdb_alias`` sets composition=<pdb path>; that must not
    re-route the run to the packmol composition builder."""
    from mmml.interfaces.pycharmmInterface.mlpot.composition_spec import (
        apply_from_pdb_alias,
    )
    from mmml.md.lowering import runconfig_from_md_system_args

    args = _from_pdb_args()
    args.from_psf = None
    args.from_crd = None
    apply_from_pdb_alias(args)
    assert str(args.composition).endswith("model.pdb")
    assert runconfig_from_md_system_args(args).system.builder == "from_pdb"


def test_from_pdb_builder_reads_box_back_off_args() -> None:
    """The box is resolved during the load, so the pre-load spec cannot supply it."""
    import inspect

    from mmml.cli.run.md_system_unified import build_from_pdb_system_with_ffparams

    src = inspect.getsource(build_from_pdb_system_with_ffparams)
    assert 'getattr(args, "box_size", None) or spec.box_size' in src
    # Must consume the per-residue split the loader derived from the PSF.
    assert "_cluster_atoms_per_list" in src
    assert "_cluster_residue_labels" in src
