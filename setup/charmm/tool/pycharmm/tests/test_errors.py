"""Tests for CHARMM error handling functionality.

This module tests the error handling features including:
- CharmmScriptError exception raising
- Global and per-call error handling configuration
- Context managers for strict/permissive modes
- Context managers for bomb/warn/verbosity levels
"""

from types import SimpleNamespace

import pytest

import pycharmm.lingo as lingo
import pycharmm.settings as settings
from pycharmm.safeguards import CharmmError, CharmmScriptError


@pytest.fixture
def mock_charmm_status(monkeypatch):
    """Force a deterministic CHARMM return code for lingo error-handling tests."""

    def _mock(status):
        fake_lib = SimpleNamespace(eval_charmm_script=lambda *_args: status)
        monkeypatch.setattr(lingo, "lib", fake_lib)
        monkeypatch.setattr(lingo, "_invalidate_cache", lambda: None)

    return _mock


class TestCharmmScriptErrorHandling:
    """Tests for charmm_script() error handling."""

    def test_default_no_raise(self, mock_charmm_status):
        """Default behavior should not raise on error, just return status."""
        mock_charmm_status(0)
        # Ensure default is False
        old = lingo.set_default_error_handling(False)
        try:
            # This invalid command should return non-1 status but not raise
            status = lingo.charmm_script("THIS_IS_NOT_A_VALID_CHARMM_COMMAND_XYZ123")
            assert status == 0
        finally:
            lingo.set_default_error_handling(old)

    def test_raise_on_error_true(self, mock_charmm_status):
        """raise_on_error=True should raise CharmmScriptError on failure."""
        mock_charmm_status(0)
        with pytest.raises(CharmmScriptError) as exc_info:
            lingo.charmm_script("INVALID_COMMAND_THAT_WILL_FAIL_ABC", raise_on_error=True)

        # Check exception attributes
        assert "INVALID_COMMAND_THAT_WILL_FAIL_ABC" in exc_info.value.script
        assert exc_info.value.status != 1

    def test_raise_on_error_false_overrides_global(self, mock_charmm_status):
        """raise_on_error=False should override global True setting."""
        mock_charmm_status(0)
        old = lingo.set_default_error_handling(True)
        try:
            # This should NOT raise because we explicitly set raise_on_error=False
            status = lingo.charmm_script("INVALID_XYZ", raise_on_error=False)
            assert status == 0
        finally:
            lingo.set_default_error_handling(old)

    def test_global_default_setting(self, mock_charmm_status):
        """set_default_error_handling(True) should make all calls raise."""
        mock_charmm_status(0)
        old = lingo.set_default_error_handling(True)
        try:
            with pytest.raises(CharmmScriptError):
                lingo.charmm_script("BAD_COMMAND_123")
        finally:
            lingo.set_default_error_handling(old)

    def test_get_default_error_handling(self):
        """get_default_error_handling() should return current setting."""
        old = lingo.get_default_error_handling()

        lingo.set_default_error_handling(True)
        assert lingo.get_default_error_handling() is True

        lingo.set_default_error_handling(False)
        assert lingo.get_default_error_handling() is False

        # Restore
        lingo.set_default_error_handling(old)

    def test_set_default_returns_old_value(self):
        """set_default_error_handling() should return previous value."""
        original = lingo.get_default_error_handling()

        returned = lingo.set_default_error_handling(True)
        assert returned == original

        returned = lingo.set_default_error_handling(False)
        assert returned is True

        # Restore
        lingo.set_default_error_handling(original)


class TestErrorHandlingContextManagers:
    """Tests for strict_mode() and permissive_mode() context managers."""

    def test_strict_mode_enables_raising(self, mock_charmm_status):
        """strict_mode() context should enable error raising."""
        mock_charmm_status(0)
        # Ensure global default is False
        old = lingo.set_default_error_handling(False)
        try:
            with lingo.strict_mode():
                with pytest.raises(CharmmScriptError):
                    lingo.charmm_script("INVALID_IN_STRICT")

            # Outside context, should not raise
            status = lingo.charmm_script("INVALID_OUTSIDE_STRICT")
            assert status == 0
        finally:
            lingo.set_default_error_handling(old)

    def test_permissive_mode_disables_raising(self, mock_charmm_status):
        """permissive_mode() context should disable error raising."""
        mock_charmm_status(0)
        old = lingo.set_default_error_handling(True)
        try:
            with lingo.permissive_mode():
                # This should NOT raise even though global is True
                status = lingo.charmm_script("INVALID_IN_PERMISSIVE")
                assert status == 0

            # Outside context, should raise again
            with pytest.raises(CharmmScriptError):
                lingo.charmm_script("INVALID_OUTSIDE_PERMISSIVE")
        finally:
            lingo.set_default_error_handling(old)

    def test_strict_mode_restores_on_exception(self):
        """strict_mode() should restore setting even if exception occurs."""
        old = lingo.set_default_error_handling(False)
        try:
            try:
                with lingo.strict_mode():
                    raise RuntimeError("Simulated error")
            except RuntimeError:
                pass

            # Setting should be restored to False
            assert lingo.get_default_error_handling() is False
        finally:
            lingo.set_default_error_handling(old)


class TestSettingsContextManagers:
    """Tests for settings module context managers."""

    def test_bomb_level_context_manager(self):
        """bomb_level() should temporarily change and restore bomb level."""
        # Get current level by setting and getting back
        original = settings.set_bomb_level(0)
        settings.set_bomb_level(original)  # Restore

        with settings.bomb_level(-5) as old_level:
            assert old_level == original
            # Verify it was changed
            current = settings.set_bomb_level(-5)
            assert current == -5
            settings.set_bomb_level(-5)  # Put it back

        # Should be restored
        restored = settings.set_bomb_level(original)
        assert restored == original

    def test_warn_level_context_manager(self):
        """warn_level() should temporarily change and restore warn level."""
        original = settings.set_warn_level(0)
        settings.set_warn_level(original)

        with settings.warn_level(-5) as old_level:
            assert old_level == original

        # Should be restored
        restored = settings.set_warn_level(original)
        assert restored == original

    def test_verbosity_context_manager(self):
        """verbosity() should temporarily change and restore verbosity."""
        original = settings.set_verbosity(5)
        settings.set_verbosity(original)

        with settings.verbosity(0) as old_level:
            assert old_level == original

        # Should be restored
        restored = settings.set_verbosity(original)
        assert restored == original

    def test_error_levels_combined(self):
        """error_levels() should set multiple levels at once."""
        # Save originals
        orig_bomb = settings.set_bomb_level(0)
        settings.set_bomb_level(orig_bomb)
        orig_warn = settings.set_warn_level(0)
        settings.set_warn_level(orig_warn)
        orig_verb = settings.set_verbosity(5)
        settings.set_verbosity(orig_verb)

        with settings.error_levels(bomb=-1, warn=-5, print_level=0):
            # Check levels were set
            current_bomb = settings.set_bomb_level(-1)
            assert current_bomb == -1
            settings.set_bomb_level(-1)

        # Should all be restored
        assert settings.set_bomb_level(orig_bomb) == orig_bomb
        assert settings.set_warn_level(orig_warn) == orig_warn
        assert settings.set_verbosity(orig_verb) == orig_verb

    def test_error_levels_partial(self):
        """error_levels() with only some parameters should only affect those."""
        orig_bomb = settings.set_bomb_level(0)
        settings.set_bomb_level(orig_bomb)

        # Only set bomb, not warn or verbosity
        with settings.error_levels(bomb=-2):
            current = settings.set_bomb_level(-2)
            assert current == -2
            settings.set_bomb_level(-2)

        # Bomb should be restored
        assert settings.set_bomb_level(orig_bomb) == orig_bomb


class TestExceptionHierarchy:
    """Tests for exception class hierarchy."""

    def test_charmm_script_error_is_charmm_error(self):
        """CharmmScriptError should be a subclass of CharmmError."""
        assert issubclass(CharmmScriptError, CharmmError)

    def test_can_catch_with_base_class(self, mock_charmm_status):
        """Should be able to catch CharmmScriptError with CharmmError."""
        mock_charmm_status(0)
        with pytest.raises(CharmmError):
            lingo.charmm_script("INVALID_CATCH_TEST", raise_on_error=True)

    def test_exception_attributes(self, mock_charmm_status):
        """CharmmScriptError should have script and status attributes."""
        mock_charmm_status(0)
        with pytest.raises(CharmmScriptError) as exc_info:
            lingo.charmm_script("TEST_SCRIPT_ATTRS", raise_on_error=True)
        e = exc_info.value
        assert hasattr(e, "script")
        assert hasattr(e, "status")
        assert e.script == "TEST_SCRIPT_ATTRS"
        assert isinstance(e.status, int)

    def test_exception_message_truncation(self, mock_charmm_status):
        """Long scripts should be truncated in exception message."""
        mock_charmm_status(0)
        long_script = "X" * 200
        with pytest.raises(CharmmScriptError) as exc_info:
            lingo.charmm_script(long_script, raise_on_error=True)
        e = exc_info.value
        # Full script should be in attribute
        assert len(e.script) == 200
        # Message should be truncated
        assert "..." in str(e)


class TestCommandScriptErrorHandling:
    """Tests for CommandScript.run() error handling."""

    def test_command_script_raise_on_error(self, mock_charmm_status):
        """CommandScript.run() should accept raise_on_error parameter."""
        mock_charmm_status(0)
        from pycharmm.script import CommandScript

        script = CommandScript("INVALID_COMMAND_SCRIPT_TEST")

        with pytest.raises(CharmmScriptError):
            script.run(raise_on_error=True)

    def test_command_script_default_behavior(self, mock_charmm_status):
        """CommandScript.run() should respect global default."""
        mock_charmm_status(0)
        from pycharmm.script import CommandScript

        old = lingo.set_default_error_handling(False)
        try:
            script = CommandScript("INVALID_NO_RAISE")
            # Should not raise with default False
            result = script.run()
            assert result is script  # Returns self
        finally:
            lingo.set_default_error_handling(old)
