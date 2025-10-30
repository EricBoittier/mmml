"""
Tests for CLI commands.
"""

import pytest
import subprocess
import sys
from pathlib import Path
import json

sys.path.insert(0, str(Path(__file__).parent.parent))


class TestCLIXml2npz:
    """Test xml2npz CLI command."""
    
    def test_xml2npz_help(self):
        """Test that help command works."""
        result = subprocess.run(
            [sys.executable, '-m', 'mmml.cli', 'xml2npz', '--help'],
            capture_output=True,
            text=True
        )
        
        assert result.returncode == 0, "Help should exit with 0"
        assert 'Convert Molpro XML' in result.stdout, "Should show description"
        assert '--output' in result.stdout, "Should show options"
        assert 'Examples:' in result.stdout, "Should show examples"
    
    def test_xml2npz_single_file(self, co2_xml_file, temp_dir):
        """Test converting single file via CLI."""
        output_file = temp_dir / 'cli_output.npz'
        
        result = subprocess.run(
            [
                sys.executable, '-m', 'mmml.cli', 'xml2npz',
                co2_xml_file,
                '-o', str(output_file),
                '--quiet'
            ],
            capture_output=True,
            text=True
        )
        
        assert result.returncode == 0, f"Should succeed. Stderr: {result.stderr}"
        assert output_file.exists(), "Output file should be created"
    
    def test_xml2npz_with_validation(self, co2_xml_file, temp_dir):
        """Test converting with validation."""
        output_file = temp_dir / 'cli_validated.npz'
        
        result = subprocess.run(
            [
                sys.executable, '-m', 'mmml.cli', 'xml2npz',
                co2_xml_file,
                '-o', str(output_file),
                '--validate'
            ],
            capture_output=True,
            text=True
        )
        
        assert result.returncode == 0, "Should succeed with validation"
        assert 'Validating output' in result.stdout or \
               '✓' in result.stdout, "Should show validation"
    
    def test_xml2npz_with_summary(self, co2_xml_file, temp_dir):
        """Test generating summary JSON."""
        output_file = temp_dir / 'cli_with_summary.npz'
        summary_file = temp_dir / 'summary.json'
        
        result = subprocess.run(
            [
                sys.executable, '-m', 'mmml.cli', 'xml2npz',
                co2_xml_file,
                '-o', str(output_file),
                '--summary', str(summary_file),
                '--quiet'
            ],
            capture_output=True,
            text=True
        )
        
        assert result.returncode == 0, "Should succeed"
        assert summary_file.exists(), "Summary file should be created"
        
        # Check summary content
        with open(summary_file) as f:
            summary = json.load(f)
        
        assert 'input_files' in summary, "Should have input_files"
        assert 'dataset_info' in summary, "Should have dataset_info"
        assert summary['input_files'] >= 1, "Should have at least 1 input"
    
    def test_xml2npz_missing_output(self, co2_xml_file):
        """Test that missing output arg gives error."""
        result = subprocess.run(
            [
                sys.executable, '-m', 'mmml.cli', 'xml2npz',
                co2_xml_file
            ],
            capture_output=True,
            text=True
        )
        
        assert result.returncode != 0, "Should fail without output"
        assert 'required' in result.stderr.lower() or \
               'output' in result.stderr.lower(), "Should mention output requirement"
    
    def test_xml2npz_nonexistent_file(self, temp_dir):
        """Test handling of nonexistent input file."""
        output_file = temp_dir / 'output.npz'
        
        result = subprocess.run(
            [
                sys.executable, '-m', 'mmml.cli', 'xml2npz',
                'nonexistent.xml',
                '-o', str(output_file),
                '--quiet'
            ],
            capture_output=True,
            text=True
        )
        
        # Should fail or report no files
        assert result.returncode != 0 or 'No XML files found' in result.stdout


class TestCLIValidate:
    """Test validate CLI command."""
    
    def test_validate_help(self):
        """Test that validate help works."""
        # Note: validate doesn't have its own --help, it's part of main CLI
        result = subprocess.run(
            [sys.executable, '-m', 'mmml.cli', '--help'],
            capture_output=True,
            text=True
        )
        
        # The main CLI should show validate in help
        assert 'validate' in result.stdout.lower() and result.returncode == 0
    
    def test_validate_existing_file(self, sample_npz_file):
        """Test validating existing NPZ file."""
        result = subprocess.run(
            [
                sys.executable, '-m', 'mmml.cli', 'validate',
                str(sample_npz_file)
            ],
            capture_output=True,
            text=True
        )
        
        assert result.returncode == 0, "Valid file should pass"
        assert 'valid' in result.stdout.lower(), "Should indicate success"
    
    def test_validate_nonexistent_file(self, temp_dir):
        """Test validating nonexistent file."""
        result = subprocess.run(
            [
                sys.executable, '-m', 'mmml.cli', 'validate',
                str(temp_dir / 'nonexistent.npz')
            ],
            capture_output=True,
            text=True
        )
        
        assert result.returncode != 0, "Should fail for nonexistent file"
    
    def test_validate_no_args(self):
        """Test validate with no arguments."""
        result = subprocess.run(
            [sys.executable, '-m', 'mmml.cli', 'validate'],
            capture_output=True,
            text=True
        )
        
        # Should show error or help
        assert result.returncode != 0 or 'Usage' in result.stdout


class TestCLIMainDispatcher:
    """Test main CLI dispatcher."""
    
    def test_cli_help(self):
        """Test main CLI help."""
        result = subprocess.run(
            [sys.executable, '-m', 'mmml.cli', '--help'],
            capture_output=True,
            text=True
        )
        
        assert result.returncode == 0, "Help should work"
        assert 'MMML' in result.stdout, "Should show MMML"
        assert 'xml2npz' in result.stdout, "Should list xml2npz"
        assert 'validate' in result.stdout, "Should list validate"
    
    def test_cli_no_command(self):
        """Test CLI with no command."""
        result = subprocess.run(
            [sys.executable, '-m', 'mmml.cli'],
            capture_output=True,
            text=True
        )
        
        # Should show error or help
        assert result.returncode != 0 or 'usage' in result.stdout.lower()
    
    def test_cli_invalid_command(self):
        """Test CLI with invalid command."""
        result = subprocess.run(
            [sys.executable, '-m', 'mmml.cli', 'invalid_command'],
            capture_output=True,
            text=True
        )
        
        assert result.returncode != 0, "Invalid command should fail"
    
    def test_cli_train_implemented(self):
        """Test that train command is now implemented."""
        result = subprocess.run(
            [sys.executable, '-m', 'mmml.cli', 'train'],
            capture_output=True,
            text=True
        )
        
        # Should show error about missing --train (not "coming soon")
        assert '--train required' in result.stderr or \
               'required' in result.stderr.lower(), "Should show train is implemented"
    
    def test_cli_evaluate_implemented(self):
        """Test that evaluate command is now implemented."""
        result = subprocess.run(
            [sys.executable, '-m', 'mmml.cli', 'evaluate'],
            capture_output=True,
            text=True
        )
        
        # Should show error about missing --model/--data (not "coming soon")
        assert 'required' in result.stderr.lower(), "Should show evaluate is implemented"
        assert 'coming soon' not in result.stderr.lower(), "Should not be placeholder"


class TestCLIIntegration:
    """Integration tests for full CLI workflows."""
    
    def test_full_workflow_xml_to_validation(self, co2_xml_file, temp_dir):
        """Test complete workflow: convert -> validate."""
        output_file = temp_dir / 'workflow.npz'
        
        # Step 1: Convert
        result1 = subprocess.run(
            [
                sys.executable, '-m', 'mmml.cli', 'xml2npz',
                co2_xml_file,
                '-o', str(output_file),
                '--quiet'
            ],
            capture_output=True,
            text=True
        )
        
        assert result1.returncode == 0, "Conversion should succeed"
        
        # Step 2: Validate
        result2 = subprocess.run(
            [
                sys.executable, '-m', 'mmml.cli', 'validate',
                str(output_file)
            ],
            capture_output=True,
            text=True
        )
        
        assert result2.returncode == 0, "Validation should succeed"
    
    def test_workflow_with_summary(self, co2_xml_file, temp_dir):
        """Test workflow with summary generation."""
        output_file = temp_dir / 'workflow_summary.npz'
        summary_file = temp_dir / 'workflow_summary.json'
        
        # Convert with summary
        result = subprocess.run(
            [
                sys.executable, '-m', 'mmml.cli', 'xml2npz',
                co2_xml_file,
                '-o', str(output_file),
                '--summary', str(summary_file),
                '--validate',
                '--quiet'
            ],
            capture_output=True,
            text=True
        )
        
        assert result.returncode == 0, "Should succeed"
        assert output_file.exists(), "NPZ should exist"
        assert summary_file.exists(), "Summary should exist"
        
        # Check summary is valid JSON
        with open(summary_file) as f:
            summary = json.load(f)
            assert isinstance(summary, dict), "Summary should be dict"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

