"""
KARML GUI - Molecular viewer for NPZ, ASE trajectory, and PDB files.

Usage:
    karml gui --data-dir ./trajectories
    karml gui --file simulation.npz
"""

from .api import app, create_app

__all__ = ['app', 'create_app']
