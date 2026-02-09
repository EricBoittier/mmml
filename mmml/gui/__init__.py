"""
MMML GUI - Molecular viewer for NPZ, ASE trajectory, and PDB files.

Usage:
    mmml gui --data-dir ./trajectories
    mmml gui --file simulation.npz
"""

from .api import app, create_app

__all__ = ['app', 'create_app']
