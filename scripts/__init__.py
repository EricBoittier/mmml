"""Repo-local helper scripts, importable as ``scripts.<module>`` from tests.

This file exists so ``scripts`` is a *regular* package rather than an implicit
namespace package. Several workflows under ``workflows/*/scripts/`` are put on
``sys.path`` by their own tests, and a namespace ``scripts`` recomputes its
``__path__`` from every matching ``sys.path`` entry — which made
``import scripts.check_evidence_registry`` resolve unpredictably depending on
collection order, so tests errored under a subset run (``pytest tests/unit/``)
while passing under ``pytest tests/``.

Not shipped: ``[tool.setuptools.packages.find]`` only includes ``mmml*`` and
``pycharmm*``.
"""
