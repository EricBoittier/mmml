"""
Tests for train CLI command.
"""

import pytest
import subprocess
import sys
from pathlib import Path
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))

from mmml.cli.train import TrainConfig, save_config, load_config_file


class TestTrainConfig:
    """Test training configuration."""
    
    def test_train_config_creation(self):
        """Test creating training configuration."""
        config = TrainConfig(
            model='dcmnet',
            train_file='train.npz',
            batch_size=32
        )
        
        assert config.model == 'dcmnet'
        assert config.train_file == 'train.npz'
        assert config.batch_size == 32
    
    def test_train_config_defaults(self):
        """Test default configuration values."""
        config = TrainConfig()
        
        assert config.model == 'dcmnet'
        assert config.batch_size == 32
        assert config.max_epochs == 1000
        assert 'energy' in config.targets
    
    def test_save_and_load_config(self, temp_dir):
        """Test saving and loading configuration."""
        config = TrainConfig(
            model='physnetjax',
            train_file='test_train.npz',
            batch_size=64
        )
        
        config_file = temp_dir / 'config.yaml'
        save_config(config, str(config_file))
        
        assert config_file.exists()
        
        # Load it back
        loaded_config = load_config_file(str(config_file))
        
        assert loaded_config.model == 'physnetjax'
        assert loaded_config.batch_size == 64


class TestTrainCLI:
    """Test train CLI command."""
    
    def test_train_help(self):
        """Test that help command works."""
        result = subprocess.run(
            [sys.executable, '-m', 'mmml.cli', 'train', '--help'],
            capture_output=True,
            text=True
        )
        
        assert result.returncode == 0
        assert 'Train MMML models' in result.stdout
        assert '--model' in result.stdout
        assert '--train' in result.stdout
    
    def test_train_missing_train_file(self):
        """Test error when train file not provided."""
        result = subprocess.run(
            [sys.executable, '-m', 'mmml.cli', 'train', '--model', 'dcmnet'],
            capture_output=True,
            text=True
        )
        
        assert result.returncode != 0
        assert '--train required' in result.stdout or 'required' in result.stderr.lower()
    
    def test_train_dry_run_dcmnet(self, sample_npz_file):
        """Test dry run with DCMNet."""
        result = subprocess.run(
            [
                sys.executable, '-m', 'mmml.cli', 'train',
                '--model', 'dcmnet',
                '--train', str(sample_npz_file),
                '--dry-run'
            ],
            capture_output=True,
            text=True
        )
        
        # Dry run should succeed
        assert result.returncode == 0
        assert 'Dry run' in result.stdout or 'dry run' in result.stdout.lower()
    
    def test_train_dry_run_physnetjax(self, sample_npz_file):
        """Test dry run with PhysNetJAX."""
        result = subprocess.run(
            [
                sys.executable, '-m', 'mmml.cli', 'train',
                '--model', 'physnetjax',
                '--train', str(sample_npz_file),
                '--dry-run',
                '--quiet'
            ],
            capture_output=True,
            text=True
        )
        
        # Dry run should succeed
        assert result.returncode == 0
    
    def test_train_with_config_file(self, temp_dir, sample_npz_file):
        """Test training with configuration file."""
        # Create config
        config = TrainConfig(
            model='dcmnet',
            train_file=str(sample_npz_file),
            batch_size=8,
            max_epochs=10
        )
        
        config_file = temp_dir / 'train_config.yaml'
        save_config(config, str(config_file))
        
        # Run with config
        result = subprocess.run(
            [
                sys.executable, '-m', 'mmml.cli', 'train',
                '--config', str(config_file),
                '--dry-run',
                '--quiet'
            ],
            capture_output=True,
            text=True
        )
        
        assert result.returncode == 0
    
    def test_train_save_config(self, temp_dir):
        """Test saving configuration."""
        output_config = temp_dir / 'saved_config.yaml'
        
        result = subprocess.run(
            [
                sys.executable, '-m', 'mmml.cli', 'train',
                '--model', 'dcmnet',
                '--train', 'dummy.npz',
                '--batch-size', '64',
                '--save-config', str(output_config)
            ],
            capture_output=True,
            text=True
        )
        
        assert result.returncode == 0
        assert output_config.exists()
        
        # Check config content
        with open(output_config) as f:
            config_data = yaml.safe_load(f)
        
        assert config_data['model'] == 'dcmnet'
        assert config_data['batch_size'] == 64
    
    def test_train_with_train_valid_split(self, sample_npz_file):
        """Test training with automatic train/valid split."""
        result = subprocess.run(
            [
                sys.executable, '-m', 'mmml.cli', 'train',
                '--model', 'dcmnet',
                '--train', str(sample_npz_file),
                '--train-fraction', '0.8',
                '--dry-run',
                '--quiet'
            ],
            capture_output=True,
            text=True
        )
        
        assert result.returncode == 0
    
    def test_train_with_separate_valid_file(self, temp_dir, sample_npz_data):
        """Test training with separate validation file."""
        import numpy as np
        
        # Create two files
        train_file = temp_dir / 'train.npz'
        valid_file = temp_dir / 'valid.npz'
        
        np.savez_compressed(train_file, **sample_npz_data)
        np.savez_compressed(valid_file, **sample_npz_data)
        
        result = subprocess.run(
            [
                sys.executable, '-m', 'mmml.cli', 'train',
                '--model', 'dcmnet',
                '--train', str(train_file),
                '--valid', str(valid_file),
                '--dry-run',
                '--quiet'
            ],
            capture_output=True,
            text=True
        )
        
        assert result.returncode == 0
    
    def test_train_verbose_output(self, sample_npz_file):
        """Test verbose training output."""
        result = subprocess.run(
            [
                sys.executable, '-m', 'mmml.cli', 'train',
                '--model', 'dcmnet',
                '--train', str(sample_npz_file),
                '--dry-run',
                '--verbose'
            ],
            capture_output=True,
            text=True
        )
        
        assert result.returncode == 0
        # Check for expected verbose output
        assert 'Loading' in result.stdout or 'loading' in result.stdout.lower()
        assert 'Train' in result.stdout or 'train' in result.stdout.lower()


class TestTrainIntegration:
    """Integration tests for training workflow."""
    
    def test_end_to_end_dcmnet_prep(self, sample_npz_file):
        """Test end-to-end DCMNet preparation."""
        result = subprocess.run(
            [
                sys.executable, '-m', 'mmml.cli', 'train',
                '--model', 'dcmnet',
                '--train', str(sample_npz_file),
                '--batch-size', '4',
                '--targets', 'energy', 'forces',
                '--dry-run',
                '--verbose'
            ],
            capture_output=True,
            text=True
        )
        
        assert result.returncode == 0
        # Should mention batch preparation
        assert 'batch' in result.stdout.lower() or 'Batch' in result.stdout
    
    def test_end_to_end_physnet_prep(self, sample_npz_file):
        """Test end-to-end PhysNetJAX preparation."""
        result = subprocess.run(
            [
                sys.executable, '-m', 'mmml.cli', 'train',
                '--model', 'physnetjax',
                '--train', str(sample_npz_file),
                '--batch-size', '4',
                '--dry-run',
                '--verbose'
            ],
            capture_output=True,
            text=True
        )
        
        assert result.returncode == 0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

