"""Padding sanitization vs nonfinite *physical* ML/MM contributions.

Padded dimer/monomer slots are dummy gathers. Zeroing them is intentional.
A nonfinite value on a live atom or live dimer is a real force-field failure
and must abort with diagnostics — not ``where(isfinite, x, 0)``.
"""

from __future__ import annotations

from typing import Any

import numpy as np


class NonfinitePhysicalContribution(RuntimeError):
    """A live (unmasked) energy or force contribution was NaN/Inf."""

    def __init__(self, name: str, n_bad: int, *, detail: str | None = None):
        self.name = name
        self.n_bad = int(n_bad)
        extra = f" {detail}" if detail else ""
        super().__init__(
            f"nonfinite physical {name} ({self.n_bad} values).{extra} "
            "Padding slots are zeroed by atom masks; this is a real contribution. "
            "Aborting."
        )


def require_host_finite(
    energy: Any,
    forces: Any,
    *,
    name: str = "ML USER",
) -> None:
    """Host-side check after ``device_get``. Raises on any nonfinite value."""
    e = np.asarray(energy)
    f = np.asarray(forces)
    n_bad_e = int(e.size - np.isfinite(e).sum()) if e.size else 0
    n_bad_f = int(f.size - np.isfinite(f).sum()) if f.size else 0
    n_bad = n_bad_e + n_bad_f
    if n_bad:
        detail = f"energy_bad={n_bad_e} force_bad={n_bad_f}."
        raise NonfinitePhysicalContribution(name, n_bad, detail=detail)
