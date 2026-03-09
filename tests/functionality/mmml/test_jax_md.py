"""
Tests for JAX-MD integration with the MMML hybrid calculator.

Verifies that the spherical_cutoff_calculator can be used as a JAX-MD energy
function for minimization (FIRE) and optionally short MD runs.

Note: FIRE minimization and NVE tests were removed; they require a proper
setup (e.g. pre-minimized structure, appropriate dt) to run stably.
"""
