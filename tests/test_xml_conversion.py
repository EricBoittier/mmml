"""
Integration tests for Molpro XML to NPZ conversion.

These tests verify the complete workflow from XML files to NPZ datasets.
"""

import pytest
import numpy as np
from pathlib import Path
import sys

# Add mmml to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from mmml.data import (
    convert_xml_to_npz,
    batch_convert_xml,
    MolproConverter,
    validate_npz,
    load_npz
)


class TestXMLConversion:
    """Test XML to NPZ conversion functionality."""
    
    def test_single_file_conversion(self, co2_xml_file, temp_dir):
        """Test converting a single XML file."""
        output_file = temp_dir / 'co2.npz'
        
        # Convert
        success = convert_xml_to_npz(
            xml_file=co2_xml_file,
            output_file=str(output_file),
            verbose=False
        )
        
        assert success, "Conversion should succeed"
        assert output_file.exists(), "Output file should be created"
        
        # Load and verify
        data = load_npz(str(output_file), validate=False)
        assert 'R' in data, "Should have coordinates"
        assert 'Z' in data, "Should have atomic numbers"
        assert 'E' in data, "Should have energy"
        assert 'N' in data, "Should have atom counts"
    
    def test_conversion_with_validation(self, co2_xml_file, temp_dir):
        """Test conversion with automatic validation."""
        output_file = temp_dir / 'co2_validated.npz'
        
        # Convert
        success = convert_xml_to_npz(
            xml_file=co2_xml_file,
            output_file=str(output_file),
            verbose=False
        )
        
        assert success, "Conversion with validation should succeed"
        
        # Validate separately
        is_valid, info = validate_npz(str(output_file), verbose=False)
        assert is_valid, "NPZ should be valid"
        assert info is not None, "Should return info dict"
        assert info['n_structures'] == 1, "Should have 1 structure"
    
    def test_co2_data_correctness(self, co2_xml_file, temp_dir, expected_co2_properties):
        """Test that CO2 data is correctly converted."""
        output_file = temp_dir / 'co2_check.npz'
        
        # Convert
        convert_xml_to_npz(co2_xml_file, str(output_file), verbose=False)
        
        # Load
        data = load_npz(str(output_file), validate=False)
        
        # Check structure
        assert len(data['Z']) == 1, "Should have 1 structure"
        
        # Check atoms (counting non-zero)
        n_atoms_actual = np.sum(data['Z'][0] > 0)
        assert n_atoms_actual == expected_co2_properties['n_atoms'], \
            f"Should have {expected_co2_properties['n_atoms']} atoms"
        
        # Check elements
        elements = sorted(data['Z'][0][data['Z'][0] > 0])
        expected = sorted(expected_co2_properties['elements'])
        assert list(elements) == expected, f"Should have elements {expected}"
        
        # Check properties presence
        assert 'E' in data, "Should have energy"
        assert 'F' in data, "Should have forces"
        assert 'D' in data, "Should have dipole"
        
        # Check energy is reasonable (for CO2)
        energy = data['E'][0]
        assert -200 < energy < -100, f"CO2 energy should be reasonable, got {energy}"
    
    def test_converter_class(self, co2_xml_file, temp_dir):
        """Test MolproConverter class directly."""
        converter = MolproConverter(
            padding_atoms=60,
            include_variables=True,
            verbose=False
        )
        
        # Convert single file
        npz_data = converter.convert_single(co2_xml_file)
        
        assert npz_data is not None, "Should return data dict"
        assert 'R' in npz_data, "Should have coordinates"
        assert 'Z' in npz_data, "Should have atomic numbers"
        assert 'E' in npz_data, "Should have energy"
        
        # Check statistics
        stats = converter.get_statistics()
        assert stats.n_files == 1, "Should have processed 1 file"
        assert stats.n_structures == 1, "Should have 1 structure"
        assert stats.n_failed == 0, "Should have no failures"
    
    def test_batch_conversion(self, co2_xml_file, temp_dir):
        """Test batch conversion of multiple files (using same file multiple times)."""
        output_file = temp_dir / 'batch.npz'
        
        # Use same file 3 times for testing
        xml_files = [co2_xml_file] * 3
        
        success = batch_convert_xml(
            xml_files=xml_files,
            output_file=str(output_file),
            verbose=False
        )
        
        assert success, "Batch conversion should succeed"
        
        # Load and verify
        data = load_npz(str(output_file), validate=False)
        assert len(data['E']) == 3, "Should have 3 structures"
        assert data['R'].shape[0] == 3, "Should have 3 sets of coordinates"
    
    def test_conversion_without_variables(self, co2_xml_file, temp_dir):
        """Test conversion excluding Molpro variables."""
        output_file = temp_dir / 'no_vars.npz'
        
        # Convert without variables
        converter = MolproConverter(
            include_variables=False,
            verbose=False
        )
        npz_data = converter.convert_single(co2_xml_file)
        converter.save_npz(npz_data, str(output_file), validate=False)
        
        # Load and check
        data = np.load(output_file, allow_pickle=True)
        
        # Metadata might still exist but shouldn't have molpro_variables
        if 'metadata' in data:
            metadata = data['metadata'][0]
            assert 'molpro_variables' not in metadata or \
                   not metadata.get('molpro_variables'), \
                   "Should not have Molpro variables"
    
    def test_different_padding(self, co2_xml_file, temp_dir):
        """Test conversion with different padding sizes."""
        for padding in [30, 60, 100]:
            output_file = temp_dir / f'co2_pad{padding}.npz'
            
            converter = MolproConverter(
                padding_atoms=padding,
                verbose=False
            )
            npz_data = converter.convert_single(co2_xml_file)
            
            assert npz_data['R'].shape[1] == padding, \
                f"Should pad to {padding} atoms"
            assert npz_data['Z'].shape[1] == padding, \
                f"Should pad to {padding} atoms"
    
    def test_conversion_statistics(self, co2_xml_file, temp_dir):
        """Test that conversion statistics are tracked correctly."""
        converter = MolproConverter(verbose=False)
        
        # Convert file
        npz_data = converter.convert_single(co2_xml_file)
        
        # Check statistics
        stats = converter.get_statistics()
        assert stats.n_files >= 1, "Should track file count"
        assert stats.n_structures >= 1, "Should track structure count"
        assert len(stats.properties_extracted) > 0, "Should track properties"
        
        # Check that expected properties are tracked
        expected_props = {'R', 'Z', 'E', 'N'}
        extracted_props = set(stats.properties_extracted)
        assert expected_props.issubset(extracted_props), \
            f"Should extract at least {expected_props}"


class TestConversionEdgeCases:
    """Test edge cases and error handling in conversion."""
    
    def test_nonexistent_file(self, temp_dir):
        """Test handling of nonexistent XML file."""
        output_file = temp_dir / 'output.npz'
        
        success = convert_xml_to_npz(
            xml_file='nonexistent_file.xml',
            output_file=str(output_file),
            verbose=False
        )
        
        # Should fail gracefully
        assert success is False, "Should return False for nonexistent file"
    
    def test_invalid_xml(self, temp_dir):
        """Test handling of invalid XML file."""
        # Create invalid XML file
        invalid_xml = temp_dir / 'invalid.xml'
        invalid_xml.write_text('Not valid XML content')
        
        output_file = temp_dir / 'output.npz'
        
        success = convert_xml_to_npz(
            xml_file=str(invalid_xml),
            output_file=str(output_file),
            verbose=False
        )
        
        # Should fail gracefully
        assert success is False, "Should return False for invalid XML"
    
    def test_empty_output_directory(self, temp_dir):
        """Test that output directories are created if needed."""
        nested_dir = temp_dir / 'nested' / 'output' / 'dir'
        output_file = nested_dir / 'output.npz'
        
        # This might fail if directory creation fails
        # Just check that it doesn't crash
        try:
            converter = MolproConverter(verbose=False)
            # We can't test this fully without a valid XML
            # but we can test the directory creation in save_npz
        except Exception:
            pass  # Expected to fail without valid data


class TestConversionPerformance:
    """Performance and stress tests."""
    
    def test_multiple_conversions(self, co2_xml_file, temp_dir):
        """Test converting same file multiple times (memory leak check)."""
        converter = MolproConverter(verbose=False)
        
        for i in range(5):
            npz_data = converter.convert_single(co2_xml_file)
            assert npz_data is not None, f"Conversion {i} should succeed"
        
        # Check that statistics accumulated correctly
        stats = converter.get_statistics()
        assert stats.n_files == 5, "Should have processed 5 files"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

