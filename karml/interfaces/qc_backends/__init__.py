"""Supplementary QC backends for cross-checking ML and PySCF results."""

from mmml.interfaces.qc_backends.protocol import BackendSpec, QCEvaluator

__all__ = [
    "BackendSpec",
    "QCEvaluator",
]
