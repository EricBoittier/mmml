"""Unit tests for unified ``--resume`` flag helpers (no PyCHARMM import)."""

from __future__ import annotations

from argparse import Namespace

from mmml.cli.run.md_config import resume_requested


def test_resume_requested_cli_and_mapping() -> None:
    assert not resume_requested(Namespace(resume=False, resume_campaign=False))
    assert resume_requested(Namespace(resume=True, resume_campaign=False))
    assert resume_requested(Namespace(resume=False, resume_campaign=True))
    assert resume_requested(mapping={"resume": True})
    assert resume_requested(mapping={"resume_campaign": True})
    assert not resume_requested(mapping={"resume": False})
