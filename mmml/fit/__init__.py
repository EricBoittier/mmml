"""Fit ML/MM hybrid parameters to experimental liquid observables.

Trajectory reweighting (DiffTRe-style): sample at theta_0, reweight frames to
theta, differentiate the reweighted averages; re-sample when the effective
sample size drops.
"""
