"""Regression tests for the pycharmm package export surface (issue #24).

``from pycharmm import *`` must expose the newer submodules ``block``,
``blade`` and ``restraints`` (added after the original set), and more
generally every ``pycharmm/*.py`` submodule -- otherwise a module added
to the package directory but forgotten in ``__init__.py`` silently
disappears from the star-import namespace, which is exactly what was
reported in issue #24.

These are pure-Python tests: they exercise the package's import machinery
and do not require a built CHARMM shared library.
"""

import os
import types

import pycharmm


def _star_namespace():
    """Return the names bound by ``from pycharmm import *``."""
    ns = {}
    exec("from pycharmm import *", ns)
    ns.pop("__builtins__", None)
    return ns


def test_star_exposes_block_blade_restraints():
    """The three submodules named in issue #24 are exposed by ``import *``."""
    ns = _star_namespace()
    for name in ("block", "blade", "restraints"):
        assert name in ns, f"'from pycharmm import *' did not expose {name!r}"
        assert isinstance(ns[name], types.ModuleType), (
            f"{name!r} is exposed but is not a module: {ns[name]!r}"
        )


# ``dimens`` is intentionally shadowed: the name ``dimens`` is bound to the
# pre-initialization Dimens singleton object, not the module. It is still
# reachable, just not as a module object, so it is exempt from the
# "resolves to a module" check below.
_SINGLETON_NAMES = {"dimens"}


def test_star_exposes_every_submodule():
    """Every ``pycharmm/*.py`` submodule is reachable as a module after ``import *``.

    Catches the case where a new submodule file is dropped into the
    package directory but never wired into ``__init__.py`` -- the
    original defect behind issue #24 -- and also the weaker case where a
    submodule is only pulled in via ``from .X import <symbol>`` so the
    module name itself is not uniformly available as ``pycharmm.X``.
    """
    pkg_dir = os.path.dirname(pycharmm.__file__)
    submodules = sorted(
        f[:-3] for f in os.listdir(pkg_dir) if f.endswith(".py") and not f.startswith("_")
    )
    ns = _star_namespace()

    missing = [mod for mod in submodules if mod not in ns]
    assert not missing, (
        "submodule(s) present in the package directory but not exposed by "
        f"'from pycharmm import *': {missing}. Add the corresponding "
        "`import pycharmm.<name> as <name>` line to pycharmm/__init__.py."
    )

    not_a_module = [
        mod
        for mod in submodules
        if mod not in _SINGLETON_NAMES and not isinstance(ns[mod], types.ModuleType)
    ]
    assert not not_a_module, (
        "submodule name(s) exposed by `import *` but not bound to the module "
        f"object: {not_a_module}. Add an explicit "
        "`import pycharmm.<name> as <name>` line to pycharmm/__init__.py so "
        "the module is reachable as `pycharmm.<name>`."
    )


def test_star_exposes_every_subpackage():
    """Every ``pycharmm/<name>/`` subpackage is reachable as a module too.

    The check above globs ``pycharmm/*.py``, so it cannot see a subpackage (a
    directory with an ``__init__.py``). Without this, a subpackage dropped into
    the package directory but never wired into ``__init__.py`` would vanish from
    the star-import namespace exactly as in issue #24.
    """
    pkg_dir = os.path.dirname(pycharmm.__file__)
    subpackages = sorted(
        entry
        for entry in os.listdir(pkg_dir)
        if not entry.startswith(("_", "."))
        and os.path.isfile(os.path.join(pkg_dir, entry, "__init__.py"))
    )
    ns = _star_namespace()

    missing = [pkg for pkg in subpackages if pkg not in ns]
    assert not missing, (
        "subpackage(s) present in the package directory but not exposed by "
        f"'from pycharmm import *': {missing}. Add the corresponding "
        "`import pycharmm.<name> as <name>` line to pycharmm/__init__.py "
        "(and the name to __all__)."
    )

    not_a_module = [pkg for pkg in subpackages if not isinstance(ns[pkg], types.ModuleType)]
    assert not not_a_module, (
        "subpackage name(s) exposed by `import *` but not bound to the module "
        f"object: {not_a_module}."
    )
