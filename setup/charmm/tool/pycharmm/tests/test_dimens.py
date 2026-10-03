"""
Tests for lazy initialization of the CHARMM C library and dimens integration.
"""

import sys
import unittest

# Assuming pycharmm is installable/importable in the test environment.
# If running this script directly from its directory, PYTHONPATH might need
# tool/pycharmm to be included. Test runners usually handle this.
import pycharmm
import pycharmm.lingo  # For triggering C calls
import pycharmm.loader  # For inspecting library initialization state
import pycharmm.scalar  # For verifying C calls

# pycharmm.dimens is directly available as pycharmm.dimens after 'import pycharmm'

# Helper to print consistently to stderr
INFO_PREFIX = "INFO:"


def print_info(test_name, message):
    print(f"\n{INFO_PREFIX} {test_name}: {message}", file=sys.stderr)


class TestCharmmCoreInitialization(unittest.TestCase):
    charmm_was_initialized_at_class_setup = False
    charmm_initialized_by_this_test_class = False

    @classmethod
    def setUpClass(cls):
        """
        Called once before tests in this class. Notes if CHARMM seems pre-initialized.
        """
        cls.charmm_was_initialized_at_class_setup = pycharmm.loader.is_initialized()
        if cls.charmm_was_initialized_at_class_setup:
            print(
                "\nWARNING: TestCharmmCoreInitialization.setUpClass: CHARMM C library "
                "appears to be already initialized before these tests started. "
                "Some assertions about 'initial state' might not reflect a truly fresh import.",
                file=sys.stderr,
            )
        else:
            print(
                f"\n{INFO_PREFIX} setUpClass: CHARMM C library is NOT initialized at class setup.",
                file=sys.stderr,
            )

    @classmethod
    def tearDownClass(cls):
        if not cls.charmm_was_initialized_at_class_setup and pycharmm.loader.is_initialized():
            cls.charmm_initialized_by_this_test_class = (
                True  # Mark that tests in this class initialized it
            )

        if cls.charmm_initialized_by_this_test_class:
            print(
                f"\n{INFO_PREFIX} tearDownClass: CHARMM C library was initialized by tests in this class.",
                file=sys.stderr,
            )
        elif not cls.charmm_was_initialized_at_class_setup:
            print(
                f"\n{INFO_PREFIX} tearDownClass: CHARMM C library remained uninitialized by tests in this class.",
                file=sys.stderr,
            )
        else:  # Was initialized before, still initialized
            print(
                f"\n{INFO_PREFIX} tearDownClass: CHARMM C library was already initialized at setup and remains so.",
                file=sys.stderr,
            )

    def test_01_initial_import_does_not_load_c_library(self):
        """
        Tests that the CHARMM C library's internal pointer (_lib) is None
        after import, assuming no prior C-requiring calls in the session.
        """
        print_info(
            "test_01_initial_import", "Starting test: Verifying C library is not loaded on import."
        )
        if self.charmm_was_initialized_at_class_setup:
            print_info("test_01_initial_import", "Skipping due to pre-initialized C library.")
            self.skipTest(
                "CHARMM C library was already initialized; cannot test initial import state."
            )

        lib_state = "Initialized" if pycharmm.loader.is_initialized() else "NOT Initialized"
        print_info(
            "test_01_initial_import",
            f"State of pycharmm.loader.is_initialized() before assertion: {lib_state}",
        )
        self.assertFalse(
            pycharmm.loader.is_initialized(),
            "CHARMM C library should not be initialized "
            "immediately after import, before any C-requiring calls.",
        )
        print_info("test_01_initial_import", "Assertion successful: C library is not initialized.")

    def test_02_dimens_accessible_and_modifiable_before_c_init(self):
        """
        Tests that pycharmm.dimens can be accessed and its attributes (e.g., maxa)
        can be modified. If the C library is not yet initialized, these
        modified dimens should be used upon its subsequent initialization.
        """
        print_info(
            "test_02_dimens", "Starting test: Verifying pycharmm.dimens access and modification."
        )
        original_maxa = pycharmm.dimens.maxa
        print_info("test_02_dimens", f"Original pycharmm.dimens.maxa: {original_maxa}")

        test_maxa_val = 98765
        if original_maxa == test_maxa_val:
            test_maxa_val += 1
        print_info("test_02_dimens", f"Target test_maxa_val: {test_maxa_val}")

        lib_state_before_dimen_change = (
            "Initialized" if pycharmm.loader.is_initialized() else "NOT Initialized"
        )
        print_info(
            "test_02_dimens",
            f"C library state before pycharmm.dimens.set_maxa(): {lib_state_before_dimen_change}",
        )

        pycharmm.dimens.set_maxa(test_maxa_val)
        print_info(
            "test_02_dimens",
            f"Called pycharmm.dimens.set_maxa({test_maxa_val}). New pycharmm.dimens.maxa: {pycharmm.dimens.maxa}",
        )
        if lib_state_before_dimen_change == "NOT Initialized":
            self.assertEqual(
                pycharmm.dimens.maxa,
                test_maxa_val,
                "pycharmm.dimens.maxa should reflect the new value set on the Python side "
                "before CHARMM initializes.",
            )
            print_info(
                "test_02_dimens",
                f"C library was not initialized. Modified pycharmm.dimens.maxa ({pycharmm.dimens.maxa}) will be used on first C call.",
            )
        else:
            self.assertEqual(
                pycharmm.dimens.maxa,
                original_maxa,
                "pycharmm.dimens.maxa should remain unchanged once CHARMM is initialized.",
            )
            print_info(
                "test_02_dimens",
                f"C library was already initialized. pycharmm.dimens.maxa changed to {pycharmm.dimens.maxa}, but original dimens were used at C init time.",
            )
        print_info("test_02_dimens", "Test assertions complete.")

    def test_03_c_library_initializes_on_first_use_and_works(self):
        """
        Tests that the CHARMM C library initializes upon the first function call
        that requires it (e.g., a lingo script). Also verifies the call works.
        If CHARMM initializes here, it should use dimens potentially set by test_02.
        """
        print_info(
            "test_03_c_init_on_use",
            "Starting test: Verifying C library initializes on first use and works.",
        )
        lib_was_none_before_call = not pycharmm.loader.is_initialized()
        lib_state_before_call = (
            "NOT Initialized" if lib_was_none_before_call else "Already Initialized"
        )
        print_info(
            "test_03_c_init_on_use",
            f"C library state before C-requiring call: {lib_state_before_call}",
        )

        test_lingo_var_name = "PYTEST_LINGO_VAR"
        test_lingo_var_val = 42.01
        print_info(
            "test_03_c_init_on_use",
            f"Attempting to set lingo variable '{test_lingo_var_name}' to {test_lingo_var_val}.",
        )

        current_maxa_before_call = pycharmm.dimens.maxa
        print_info(
            "test_03_c_init_on_use",
            f"pycharmm.dimens.maxa value before C-requiring call: {current_maxa_before_call}",
        )

        try:
            pycharmm.lingo.set_charmm_variable(test_lingo_var_name, test_lingo_var_val)
            print_info("test_03_c_init_on_use", "lingo.set_charmm_variable call successful.")
        except Exception as e:
            self.fail(f"lingo.set_charmm_variable call failed unexpectedly: {e}")

        lib_state_after_call = (
            "Initialized" if pycharmm.loader.is_initialized() else "NOT Initialized"
        )
        print_info(
            "test_03_c_init_on_use",
            f"C library state after C-requiring call: {lib_state_after_call}",
        )
        self.assertTrue(
            pycharmm.loader.is_initialized(),
            "CHARMM C library should be initialized after a C-requiring function call.",
        )

        print_info(
            "test_03_c_init_on_use",
            f"Attempting to retrieve lingo variable '{test_lingo_var_name}'.",
        )
        retrieved_val_str = pycharmm.lingo.get_charmm_variable(test_lingo_var_name)
        print_info(
            "test_03_c_init_on_use", f"Retrieved lingo variable as string: '{retrieved_val_str}'."
        )
        self.assertIsNotNone(
            retrieved_val_str,
            f"CHARMM lingo variable '{test_lingo_var_name}' was not found after setting.",
        )

        retrieved_val_float = -1.0  # Default if conversion fails before assertion
        try:
            retrieved_val_float = float(retrieved_val_str)
            print_info(
                "test_03_c_init_on_use",
                f"Converted retrieved value to float: {retrieved_val_float}.",
            )
        except (ValueError, TypeError) as e:
            self.fail(
                f"Could not convert retrieved lingo variable '{retrieved_val_str}' to float. Error: {e}"
            )

        self.assertAlmostEqual(
            retrieved_val_float,
            test_lingo_var_val,
            places=5,
            msg=(
                "CHARMM lingo variable set/get mismatch. "
                f"Expected {test_lingo_var_val}, got {retrieved_val_float} (from str '{retrieved_val_str}')."
            ),
        )
        print_info("test_03_c_init_on_use", "Lingo variable set/get assertions successful.")

        if lib_was_none_before_call:
            print_info(
                "test_03_c_init_on_use",
                f"CHARMM C library was initialized by this test. The pycharmm.dimens.maxa value at initialization was {current_maxa_before_call}.",
            )
            # Set the flag for tearDownClass
            TestCharmmCoreInitialization.charmm_initialized_by_this_test_class = True
        else:
            print_info(
                "test_03_c_init_on_use",
                f"CHARMM C library was already initialized. Current pycharmm.dimens.maxa is {current_maxa_before_call}.",
            )
        print_info("test_03_c_init_on_use", "Test assertions complete.")

    def test_04_dimens_not_modifiable_in_c_after_init(self):
        """
        Tests that attempting to change pycharmm.dimens attributes after CHARMM C library
        is initialized prints a warning and does not affect the running C library's dimensions.
        """
        print_info(
            "test_04_dimens_after_init", "Starting test: Verifying dimens behavior after C init."
        )

        # 1. Ensure CHARMM is initialized
        if not pycharmm.loader.is_initialized():
            print_info(
                "test_04_dimens_after_init", "CHARMM not initialized, initializing it first..."
            )
            try:
                pycharmm.lingo.charmm_script(
                    "scalar DUMMY_INIT_VAR set 1.0"
                )  # Any C-requiring call
                TestCharmmCoreInitialization.charmm_initialized_by_this_test_class = True
                print_info("test_04_dimens_after_init", "CHARMM initialized.")
            except Exception as e:
                self.fail(f"Failed to initialize CHARMM for test_04: {e}")
        else:
            print_info("test_04_dimens_after_init", "CHARMM is already initialized.")

        self.assertTrue(
            pycharmm.loader.is_initialized(),
            "CHARMM C library should be initialized for this test.",
        )

        # 2. Store current dimens value and attempt to change it
        original_maxa_in_python = pycharmm.dimens.maxa
        new_maxa_val = original_maxa_in_python + 12345  # A clearly different value
        print_info(
            "test_04_dimens_after_init",
            f"Before change: Python pycharmm.dimens.maxa = {original_maxa_in_python}. Attempting to set to {new_maxa_val}.",
        )

        # 3. Capture stderr to check for the warning
        import sys
        from io import StringIO

        old_stderr = sys.stderr
        sys.stderr = captured_stderr = StringIO()

        pycharmm.dimens.set_maxa(new_maxa_val)

        sys.stderr = old_stderr  # Restore stderr
        warning_output = captured_stderr.getvalue()

        print_info(
            "test_04_dimens_after_init", f"stderr output from set_maxa: {warning_output.strip()}"
        )

        # 4. Verify Python-side object did NOT change due to the safeguard
        self.assertEqual(
            pycharmm.dimens.maxa,
            original_maxa_in_python,
            "Python-side pycharmm.dimens.maxa should NOT have changed if CHARMM was already initialized due to safeguard.",
        )
        print_info(
            "test_04_dimens_after_init",
            f"After attempted change: Python pycharmm.dimens.maxa = {pycharmm.dimens.maxa} (should be original value).",
        )

        # 5. Verify warning was printed
        self.assertIn(
            "Warning: CHARMM is already initialized. Dimension 'maxa' cannot be changed.",
            warning_output,
            "Warning message about CHARMM being already initialized was not found in stderr.",
        )
        print_info("test_04_dimens_after_init", "Correct warning was printed to stderr.")
        print_info("test_04_dimens_after_init", "Test assertions complete.")


if __name__ == "__main__":
    # This allows the test script to be run directly, e.g., `python test_charmm_initialization.py`.
    # For this to work correctly if pycharmm is not installed site-wide,
    # the `tool/pycharmm` directory (one level up from `pycharmm/pycharmm` and `pycharmm/tests`)
    # might need to be in PYTHONPATH.
    # Example: export PYTHONPATH=/path/to/charmm/tool/pycharmm:$PYTHONPATH
    unittest.main()
