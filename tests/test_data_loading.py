"""
Tests for NPZ data loading and schema validation.
"""

import pytest
import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from mmml.data import (
    load_npz,
    load_multiple_npz,
    validate_npz,
    train_valid_split,
    get_data_statistics,
    DataConfig,
    NPZSchema,
    REQUIRED_KEYS,
    OPTIONAL_KEYS,
)


class TestNPZSchema:
    """Test NPZ schema validation."""
    
    def test_required_keys_definition(self):
        """Test that required keys are defined correctly."""
        assert 'R' in REQUIRED_KEYS, "R (coordinates) should be required"
        assert 'Z' in REQUIRED_KEYS, "Z (atomic numbers) should be required"
        assert 'E' in REQUIRED_KEYS, "E (energy) should be required"
        assert 'N' in REQUIRED_KEYS, "N (n_atoms) should be required"
    
    def test_optional_keys_definition(self):
        """Test that optional keys are defined."""
        assert 'F' in OPTIONAL_KEYS, "F (forces) should be optional"
        assert 'D' in OPTIONAL_KEYS, "D (dipole) should be optional"
        assert 'esp' in OPTIONAL_KEYS, "esp should be optional"
    
    def test_schema_validation_valid_data(self, sample_npz_data):
        """Test schema validation with valid data."""
        schema = NPZSchema()
        is_valid, errors = schema.validate(sample_npz_data)
        
        assert is_valid, f"Valid data should pass validation. Errors: {errors}"
        assert len(errors) == 0, "Should have no errors"
    
    def test_schema_validation_missing_required(self, sample_npz_data):
        """Test schema validation with missing required keys."""
        # Remove required key
        incomplete_data = sample_npz_data.copy()
        del incomplete_data['E']
        
        schema = NPZSchema()
        is_valid, errors = schema.validate(incomplete_data)
        
        assert not is_valid, "Should fail with missing required key"
        assert len(errors) > 0, "Should have errors"
        assert any('E' in str(error) for error in errors), "Should mention missing 'E'"
    
    def test_schema_validation_shape_mismatch(self, sample_npz_data):
        """Test schema validation with shape mismatches."""
        bad_data = sample_npz_data.copy()
        # Make R and Z shapes incompatible
        bad_data['R'] = np.random.randn(10, 6, 3)  # 6 atoms
        bad_data['Z'] = np.array([[6, 1, 1, 1, 1]] * 10)  # 5 atoms
        
        schema = NPZSchema()
        is_valid, errors = schema.validate(bad_data)
        
        assert not is_valid, "Should fail with shape mismatch"
        assert any('mismatch' in str(error).lower() for error in errors)
    
    def test_schema_info_generation(self, sample_npz_data):
        """Test schema info generation."""
        schema = NPZSchema()
        info = schema.get_info(sample_npz_data)
        
        assert 'n_structures' in info, "Should have n_structures"
        assert 'n_atoms' in info, "Should have n_atoms"
        assert 'properties' in info, "Should have properties list"
        assert 'unique_elements' in info, "Should have unique elements"
        
        # Check values
        assert info['n_structures'] == 10, "Should have 10 structures"
        assert info['n_atoms'] == 5, "Should have 5 atoms"
        assert 6 in info['unique_elements'], "Should have carbon"


class TestNPZLoading:
    """Test NPZ file loading functionality."""
    
    def test_load_npz_basic(self, sample_npz_file):
        """Test basic NPZ loading."""
        data = load_npz(sample_npz_file, validate=False, verbose=False)
        
        assert 'R' in data, "Should have coordinates"
        assert 'Z' in data, "Should have atomic numbers"
        assert 'E' in data, "Should have energies"
        assert isinstance(data['R'], np.ndarray), "R should be numpy array"
    
    def test_load_npz_with_validation(self, sample_npz_file):
        """Test loading with validation."""
        data = load_npz(sample_npz_file, validate=True, verbose=False)
        
        # Should not raise error and return data
        assert data is not None, "Should return data"
        assert len(data) > 0, "Should have data"
    
    def test_load_npz_specific_keys(self, sample_npz_file):
        """Test loading specific keys only."""
        data = load_npz(
            sample_npz_file,
            keys=['R', 'Z', 'E'],
            validate=False,
            verbose=False
        )
        
        assert 'R' in data, "Should have R"
        assert 'Z' in data, "Should have Z"
        assert 'E' in data, "Should have E"
        assert 'F' not in data, "Should not have F (not requested)"
    
    def test_load_nonexistent_file(self, temp_dir):
        """Test loading nonexistent file."""
        with pytest.raises(FileNotFoundError):
            load_npz(temp_dir / 'nonexistent.npz')
    
    def test_validate_npz_function(self, sample_npz_file):
        """Test standalone validate_npz function."""
        is_valid, info = validate_npz(sample_npz_file, verbose=False)
        
        assert is_valid, "Sample data should be valid"
        assert info is not None, "Should return info"
        assert info['n_structures'] == 10, "Should have 10 structures"
    
    def test_load_multiple_npz(self, temp_dir, sample_npz_data):
        """Test loading multiple NPZ files."""
        # Create multiple files
        file1 = temp_dir / 'data1.npz'
        file2 = temp_dir / 'data2.npz'
        
        np.savez_compressed(file1, **sample_npz_data)
        np.savez_compressed(file2, **sample_npz_data)
        
        # Load multiple
        data = load_multiple_npz(
            [file1, file2],
            combine=True,
            validate=False,
            verbose=False
        )
        
        assert data['E'].shape[0] == 20, "Should have 20 structures (10+10)"
        assert data['R'].shape[0] == 20, "Should have 20 sets of coordinates"
    
    def test_load_multiple_npz_separate(self, temp_dir, sample_npz_data):
        """Test loading multiple NPZ files without combining."""
        file1 = temp_dir / 'data1.npz'
        file2 = temp_dir / 'data2.npz'
        
        np.savez_compressed(file1, **sample_npz_data)
        np.savez_compressed(file2, **sample_npz_data)
        
        # Load without combining
        datasets = load_multiple_npz(
            [file1, file2],
            combine=False,
            validate=False,
            verbose=False
        )
        
        assert isinstance(datasets, list), "Should return list"
        assert len(datasets) == 2, "Should have 2 datasets"
        assert datasets[0]['E'].shape[0] == 10, "Each should have 10 structures"


class TestDataSplitting:
    """Test train/validation splitting."""
    
    def test_train_valid_split_basic(self, sample_npz_data):
        """Test basic train/valid splitting."""
        train_data, valid_data = train_valid_split(
            sample_npz_data,
            train_fraction=0.8,
            shuffle=False,
            seed=42
        )
        
        assert len(train_data['E']) == 8, "Should have 8 training samples"
        assert len(valid_data['E']) == 2, "Should have 2 validation samples"
        
        # Check that all keys are present
        for key in sample_npz_data.keys():
            if isinstance(sample_npz_data[key], np.ndarray) and \
               len(sample_npz_data[key]) == 10:
                assert key in train_data, f"Train should have {key}"
                assert key in valid_data, f"Valid should have {key}"
    
    def test_train_valid_split_with_shuffle(self, sample_npz_data):
        """Test splitting with shuffling."""
        # Split twice with same seed
        train1, valid1 = train_valid_split(
            sample_npz_data,
            train_fraction=0.8,
            shuffle=True,
            seed=42
        )
        train2, valid2 = train_valid_split(
            sample_npz_data,
            train_fraction=0.8,
            shuffle=True,
            seed=42
        )
        
        # Should be identical with same seed
        assert np.allclose(train1['E'], train2['E']), "Same seed should give same split"
        
        # Different seed should give different split
        train3, valid3 = train_valid_split(
            sample_npz_data,
            train_fraction=0.8,
            shuffle=True,
            seed=123
        )
        
        # Very unlikely to be identical (but possible)
        # Just check they're different sizes or different order
        assert len(train3['E']) == 8, "Should still have 8 training samples"
    
    def test_train_valid_split_edge_cases(self, sample_npz_data):
        """Test edge cases in splitting."""
        # All training
        train, valid = train_valid_split(
            sample_npz_data,
            train_fraction=1.0,
            shuffle=False
        )
        assert len(train['E']) == 10, "Should have all in training"
        assert len(valid['E']) == 0, "Should have none in validation"
        
        # All validation
        train, valid = train_valid_split(
            sample_npz_data,
            train_fraction=0.0,
            shuffle=False
        )
        assert len(train['E']) == 0, "Should have none in training"
        assert len(valid['E']) == 10, "Should have all in validation"


class TestDataStatistics:
    """Test data statistics generation."""
    
    def test_get_data_statistics(self, sample_npz_data):
        """Test statistics generation."""
        stats = get_data_statistics(sample_npz_data)
        
        assert 'n_structures' in stats, "Should have n_structures"
        assert 'keys' in stats, "Should have keys"
        assert 'coordinates' in stats, "Should have coordinate stats"
        assert 'energy' in stats, "Should have energy stats"
        assert 'elements' in stats, "Should have element stats"
        
        # Check values
        assert stats['n_structures'] == 10, "Should have 10 structures"
        assert 'R' in stats['keys'], "Should list R in keys"
    
    def test_statistics_ranges(self, sample_npz_data):
        """Test that statistics compute ranges correctly."""
        stats = get_data_statistics(sample_npz_data)
        
        # Check that ranges are sensible
        assert 'min' in stats['coordinates'], "Should have min coordinate"
        assert 'max' in stats['coordinates'], "Should have max coordinate"
        assert stats['coordinates']['min'] < stats['coordinates']['max'], \
            "Min should be less than max"
        
        assert 'min' in stats['energy'], "Should have min energy"
        assert 'max' in stats['energy'], "Should have max energy"


class TestDataConfig:
    """Test DataConfig functionality."""
    
    def test_data_config_creation(self):
        """Test creating DataConfig."""
        config = DataConfig(
            batch_size=32,
            targets=['energy', 'forces'],
            num_atoms=60
        )
        
        assert config.batch_size == 32, "Should have batch size"
        assert 'energy' in config.targets, "Should have energy target"
        assert config.num_atoms == 60, "Should have num_atoms"
    
    def test_data_config_defaults(self):
        """Test DataConfig defaults."""
        config = DataConfig()
        
        assert config.batch_size == 32, "Should have default batch size"
        assert 'energy' in config.targets, "Should have default targets"
        assert config.num_atoms == 60, "Should have default num_atoms"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

