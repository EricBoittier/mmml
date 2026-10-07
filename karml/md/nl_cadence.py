"""Neighbor-list refresh cadence shared by the JAX-MD drivers.

``jaxmd_runner`` grew ensemble-aware NL cadence first (see
``ENSEMBLE_JAXMD_UPDATE_INTERVAL`` there); λ-dynamics and the ``mmml.md`` driver
hardcoded their own intervals. This module holds the canonical policy so all
three agree, without the lighter drivers having to import ``jaxmd_runner``
(which pulls in rich, HDF5 reporters and the pycharmm interface).

``tests/unit/test_nl_cadence.py`` asserts parity with ``jaxmd_runner`` so the
two copies cannot drift silently.
"""

from __future__ import annotations

__all__ = [
    "ENSEMBLE_UPDATE_INTERVAL",
    "FREE_SPACE_UPDATE_INTERVAL",
    "resolve_update_interval",
    "resolve_block_steps",
    "verlet_reuse_displacement_limit_A",
]

# NVT can batch more aggressively (the thermostat absorbs force noise). NpT
# needs fresher pairs because the cell moves every step. NVE is in between:
# stale pairs show up as E_tot drift.
ENSEMBLE_UPDATE_INTERVAL: dict[str, int] = {
    "nvt": 10,
    "npt": 5,
    "nve": 5,
}

# No dynamic MM pairs in free space, so the interval only bounds block size.
FREE_SPACE_UPDATE_INTERVAL = 100


def resolve_update_interval(
    ensemble: str | None,
    requested: int | None,
    *,
    use_pbc: bool = True,
) -> int:
    """Resolve the MM neighbor refresh cadence in MD steps.

    An explicit positive ``requested`` always wins. ``None``/``<= 0`` falls back
    to the ensemble default under PBC, or ``FREE_SPACE_UPDATE_INTERVAL``
    otherwise.
    """
    if requested is not None and int(requested) > 0:
        return int(requested)
    if not use_pbc:
        return FREE_SPACE_UPDATE_INTERVAL
    key = str(ensemble or "nve").strip().lower()
    return int(ENSEMBLE_UPDATE_INTERVAL.get(key, 1))


def resolve_block_steps(
    *,
    steps_per_recording: int,
    use_pbc: bool,
    has_update_fn: bool,
    update_interval: int | None,
    ensemble: str | None = None,
) -> int:
    """Return the compiled-block size, which also sets the MM pair refresh cadence.

    The pair list enters the compiled step as data, so one number means two
    things: how often Python rebuilds the list, and how many MD steps a single
    compiled block advances. The result always divides ``steps_per_recording``
    so a recording boundary lands exactly on a neighbor-list refresh.
    """
    if use_pbc and has_update_fn:
        requested = resolve_update_interval(ensemble, update_interval, use_pbc=True)
    else:
        requested = (
            int(update_interval)
            if update_interval is not None and int(update_interval) > 0
            else FREE_SPACE_UPDATE_INTERVAL
        )

    max_block = min(requested, int(steps_per_recording))
    for candidate in range(max_block, 0, -1):
        if int(steps_per_recording) % candidate == 0:
            return candidate
    return max_block


def verlet_reuse_displacement_limit_A(skin_A: float) -> float:
    """Max per-atom displacement that keeps a list built at ``Rcut + skin`` valid.

    Two atoms can approach each other by up to twice the per-atom displacement,
    so the safe bound is ``skin / 2``. Mirrors the MM-side helper of the same
    name in ``mm_energy_forces`` (kept in sync by ``test_nl_cadence.py``).
    """
    return 0.5 * float(max(0.0, skin_A))
