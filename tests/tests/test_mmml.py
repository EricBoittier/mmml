"""
Unit and regression test for the mmml package.
"""

# Import package, test suite, and other packages as needed
import sys

import pytest

import mmml


def test_mmml_imported():
    """Sample test, will always pass so long as import statement worked."""
    assert "mmml" in sys.modules
