"""Tests for pycharmm.script_factory — parameterized CHARMM script wrappers."""

from pycharmm import script_factory


class TestScriptFactory:
    """Smoke tests that script_factory produces runnable CHARMM scripts.

    These don't assert on CHARMM's stdout (which is not easily captured
    through the lingo bridge); they verify that constructed scripts
    invoke without raising.
    """

    def test_no_arg_factory(self):
        """A factory with a literal command produces a callable Script."""
        Echo = script_factory("echo hello")
        Echo().run()  # would raise on parse / dispatch failure

    def test_factory_with_named_argument(self):
        """A factory with one named arg interpolates correctly at run time."""
        Echo = script_factory("echo", ["msg"])
        Echo(msg="help me").run()

    def test_factory_with_quoted_argument(self):
        """Quoted arguments survive the factory's string substitution."""
        Echo = script_factory("echo", ["msg"])
        Echo(msg='"help me"').run()
