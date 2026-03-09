"""
Pytest configuration and fixtures for MMML tests.
"""

import pytest
import numpy as np
from pathlib import Path
import tempfile
import shutil


@pytest.fixture
def co2_xml_file():
    """Path to CO2 test XML file."""
    xml_path = Path(__file__).parent.parent / 'mmml' / 'parse_molpro' / 'co2.xml'
    if not xml_path.exists():
        pytest.skip(f"CO2 XML file not found at {xml_path}")
    return str(xml_path)


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test outputs."""
    temp_path = tempfile.mkdtemp(prefix='mmml_test_')
    yield Path(temp_path)
    # Cleanup
    shutil.rmtree(temp_path, ignore_errors=True)


@pytest.fixture
def sample_npz_data():
    """Create sample NPZ data for testing."""
    data = {
        'R': np.random.randn(10, 5, 3),
        'Z': np.array([[6, 1, 1, 1, 1]] * 10, dtype=np.int32),
        'E': np.random.randn(10),
        'N': np.array([5] * 10, dtype=np.int32),
        'F': np.random.randn(10, 5, 3),
        'D': np.random.randn(10, 3),
    }
    return data


@pytest.fixture
def sample_npz_file(temp_dir, sample_npz_data):
    """Create a sample NPZ file for testing."""
    npz_path = temp_dir / 'test_data.npz'
    np.savez_compressed(npz_path, **sample_npz_data)
    return npz_path


@pytest.fixture
def expected_co2_properties():
    """Expected properties from CO2 XML file."""
    return {
        'n_atoms': 3,
        'elements': [6, 8, 8],  # C, O, O
        'has_energy': True,
        'has_forces': True,
        'has_dipole': True,
        'has_orbitals': True,
        'n_variables': 260,  # Molpro variables
    }

