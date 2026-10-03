"""Tests for pycharmm safeguards module.

This module tests the optional safeguards system including:
- Enable/disable functionality
- Logging and checkpoints
- Pre-validation of commands
- Context manager usage
"""

import os
import tempfile

import pycharmm.lingo as lingo
import pycharmm.safeguards as safeguards


class TestSafeguardsEnableDisable:
    """Tests for enable/disable functionality."""

    def test_disabled_by_default(self):
        """Safeguards should be disabled by default."""
        # Reset to default state
        safeguards.disable()
        assert safeguards.is_enabled() is False

    def test_enable_creates_log_file(self):
        """enable() should create a log file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            log_file = os.path.join(tmpdir, "test.log")
            safeguards.enable(log_file=log_file)
            try:
                assert safeguards.is_enabled() is True
                assert safeguards.get_log_file() == log_file
                # Log file should exist after first write
                safeguards.log_message("Test message")
                assert os.path.exists(log_file)
            finally:
                safeguards.disable()

    def test_disable_stops_logging(self):
        """disable() should stop logging."""
        with tempfile.TemporaryDirectory() as tmpdir:
            log_file = os.path.join(tmpdir, "test.log")
            safeguards.enable(log_file=log_file)
            safeguards.disable()
            assert safeguards.is_enabled() is False

    def test_get_config(self):
        """get_config() should return current configuration."""
        with tempfile.TemporaryDirectory() as tmpdir:
            log_file = os.path.join(tmpdir, "test.log")
            safeguards.enable(log_file=log_file, validate_files=True, validate_gpu=False)
            try:
                config = safeguards.get_config()
                assert config["enabled"] is True
                assert config["log_file"] == log_file
                assert config["validate_files"] is True
                assert config["validate_gpu"] is False
            finally:
                safeguards.disable()


class TestSafeguardsContextManager:
    """Tests for context manager usage."""

    def test_enabled_context_manager(self):
        """Context manager should enable and disable safeguards."""
        safeguards.disable()
        assert safeguards.is_enabled() is False

        with tempfile.TemporaryDirectory() as tmpdir:
            log_file = os.path.join(tmpdir, "test.log")
            with safeguards.enabled(log_file=log_file):
                assert safeguards.is_enabled() is True

            assert safeguards.is_enabled() is False

    def test_context_manager_restores_state(self):
        """Context manager should restore previous state."""
        safeguards.disable()

        with tempfile.TemporaryDirectory() as tmpdir:
            log_file = os.path.join(tmpdir, "test.log")
            with safeguards.enabled(log_file=log_file):
                pass

            # Should be disabled again
            assert safeguards.is_enabled() is False


class TestLogging:
    """Tests for logging functionality."""

    def test_log_message(self):
        """log_message() should write to log file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            log_file = os.path.join(tmpdir, "test.log")
            safeguards.enable(log_file=log_file)
            try:
                safeguards.log_message("Test message 123")

                with open(log_file) as f:
                    content = f.read()
                    assert "Test message 123" in content
            finally:
                safeguards.disable()

    def test_log_checkpoint(self):
        """log_checkpoint() should write checkpoint to log file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            log_file = os.path.join(tmpdir, "test.log")
            safeguards.enable(log_file=log_file)
            try:
                safeguards.log_checkpoint("After minimization")

                with open(log_file) as f:
                    content = f.read()
                    assert "CHECKPOINT" in content
                    assert "After minimization" in content
            finally:
                safeguards.disable()

    def test_commands_are_logged(self):
        """CHARMM commands should be logged when enabled."""
        with tempfile.TemporaryDirectory() as tmpdir:
            log_file = os.path.join(tmpdir, "test.log")
            safeguards.enable(log_file=log_file)
            try:
                # Run a simple CHARMM command
                lingo.charmm_script("prnlev 5")

                with open(log_file) as f:
                    content = f.read()
                    assert "prnlev" in content.lower() or "Command:" in content
            finally:
                safeguards.disable()


class TestFileValidation:
    """Tests for file validation."""

    def test_missing_file_detected(self):
        """Missing file should be detected by validator."""
        # Test the validator directly without running CHARMM
        warnings = safeguards._validate_file_exists("read rtf card name /nonexistent/file.rtf")
        assert len(warnings) > 0
        assert "not found" in warnings[0][0].lower()

    def test_existing_file_no_warning(self):
        """Existing file should not generate warning."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a test file
            test_file = os.path.join(tmpdir, "test.rtf")
            with open(test_file, "w") as f:
                f.write("* Test\n*\n")

            # Test the validator directly
            warnings = safeguards._validate_file_exists(f"read rtf card name {test_file}")
            assert len(warnings) == 0

    def test_stream_file_validation(self):
        """Stream file validation should detect missing files."""
        warnings = safeguards._validate_stream_file("stream /nonexistent/script.str")
        assert len(warnings) > 0
        assert "not found" in warnings[0][0].lower()


class TestGPUValidation:
    """Tests for GPU validation."""

    def test_gpu_check_function(self):
        """_check_cuda_available should return tuple."""
        available, reason = safeguards._check_cuda_available()
        assert isinstance(available, bool)
        assert isinstance(reason, str)

    def test_gpu_validator_returns_warnings(self):
        """GPU validation should return warnings about GPU availability."""
        # Test the validator directly without running CHARMM
        warnings = safeguards._validate_gpu_operation("blade on")
        # Should return a warning tuple (either about GPU or success)
        assert isinstance(warnings, list)
        # If GPU not available, should have a warning
        available, _ = safeguards._check_cuda_available()
        if not available:
            assert len(warnings) > 0


class TestCustomValidators:
    """Tests for custom validator registration."""

    def test_register_custom_validator(self):
        """Custom validators should be registered and callable."""
        with tempfile.TemporaryDirectory() as tmpdir:
            log_file = os.path.join(tmpdir, "test.log")
            safeguards.enable(log_file=log_file)

            custom_called = []

            def custom_validator(script):
                custom_called.append(script)
                if "CUSTOM_TEST" in script:
                    return [("Custom warning triggered", "Custom tip")]
                return []

            safeguards.register_validator(r"CUSTOM_TEST", custom_validator)

            try:
                # Test by calling the hook directly
                safeguards._pre_execute_hook("echo CUSTOM_TEST")

                # Validator should have been called
                assert len(custom_called) > 0
                assert "CUSTOM_TEST" in custom_called[0]

                with open(log_file) as f:
                    content = f.read()
                    assert "Custom warning" in content
            finally:
                safeguards.disable()


class TestNoBlockingBehavior:
    """Tests that safeguards never block operations."""

    def test_warnings_do_not_block(self):
        """Warnings should not prevent command execution."""
        with tempfile.TemporaryDirectory() as tmpdir:
            log_file = os.path.join(tmpdir, "test.log")
            safeguards.enable(log_file=log_file, validate_files=True)
            try:
                # This will generate a warning but should still execute
                # (CHARMM handles the actual error)
                status = lingo.charmm_script("prnlev 5")
                # Command should have executed
                assert status is not None
            finally:
                safeguards.disable()

    def test_disabled_safeguards_no_overhead(self):
        """Disabled safeguards should have minimal overhead."""
        safeguards.disable()

        # This should work fine with safeguards disabled
        status = lingo.charmm_script("prnlev 5")
        assert status is not None
