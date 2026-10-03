"""Tests for pycharmm.lingo — get_energy_value and get_charmm_variable."""

import warnings

import pytest

import pycharmm.lingo as lingo


class TestGetEnergyValue:
    def test_known_parameter(self):
        # PI is always available as a substitution parameter
        val = lingo.get_energy_value("PI")
        assert abs(val - 3.141592653589793) < 1e-10

    def test_missing_raises_keyerror(self):
        with pytest.raises(KeyError, match="NOSUCHPARAM"):
            lingo.get_energy_value("NOSUCHPARAM")

    def test_missing_with_default(self):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            val = lingo.get_energy_value("NOSUCHPARAM", default=42)
            assert val == 42
            assert len(w) == 1
            assert "not found" in str(w[0].message)

    def test_default_none(self):
        # Passing default=None should return None, not raise
        val = lingo.get_energy_value("NOSUCHPARAM", default=None)
        assert val is None


class TestGetCharmmVariable:
    def test_set_and_get_int(self):
        lingo.set_charmm_variable("TSTINT", 7)
        val = lingo.get_charmm_variable("TSTINT")
        assert val == 7

    def test_set_and_get_float(self):
        lingo.set_charmm_variable("TSTFLT", "3.14")
        val = lingo.get_charmm_variable("TSTFLT")
        assert abs(val - 3.14) < 1e-6

    def test_set_and_get_string(self):
        lingo.set_charmm_variable("TSTSTR", "hello")
        val = lingo.get_charmm_variable("TSTSTR")
        assert val == b"hello" or val == "hello"

    def test_missing_raises_keyerror(self):
        with pytest.raises(KeyError, match="NOSUCHVAR"):
            lingo.get_charmm_variable("NOSUCHVAR")

    def test_missing_with_default(self):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            val = lingo.get_charmm_variable("NOSUCHVAR", default="fallback")
            assert val == "fallback"
            assert len(w) == 1
            assert "not found" in str(w[0].message)
