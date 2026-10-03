"""Tests for pycharmm.charmm_file — CharmmFile open/close/context manager."""

import os
import tempfile

import pytest

from pycharmm import CharmmFile


@pytest.fixture
def tmp_file():
    """Create a temporary file for testing."""
    fd, path = tempfile.mkstemp(suffix=".dat")
    os.close(fd)
    yield path
    if os.path.exists(path):
        os.unlink(path)


class TestCharmmFile:
    def test_open_close(self, tmp_file):
        f = CharmmFile(file_name=tmp_file, file_unit=-1, read_only=True, formatted=True)
        assert f.is_open
        assert f.file_unit > 0
        f.close()
        assert not f.is_open

    def test_context_manager(self, tmp_file):
        with CharmmFile(file_name=tmp_file, file_unit=-1, read_only=True, formatted=True) as f:
            assert f.is_open
            unit = f.file_unit
            assert unit > 0
        assert not f.is_open

    def test_reopen_read_only(self, tmp_file):
        f = CharmmFile(file_name=tmp_file, file_unit=-1, read_only=False, formatted=True)
        assert not f.read_only
        f.close()
        assert not f.is_open

        f.open(read_only=True)
        assert f.is_open
        assert f.read_only
        f.close()

    def test_open_when_already_open(self, tmp_file):
        f = CharmmFile(file_name=tmp_file, file_unit=-1, read_only=True, formatted=True)
        assert f.is_open
        # open() on an already-open file should be a no-op
        result = f.open()
        assert result is True
        f.close()

    def test_context_manager_on_closed_file(self, tmp_file):
        f = CharmmFile(file_name=tmp_file, file_unit=-1, read_only=True, formatted=True)
        f.close()
        assert not f.is_open
        # __enter__ should reopen
        with f:
            assert f.is_open
        assert not f.is_open
