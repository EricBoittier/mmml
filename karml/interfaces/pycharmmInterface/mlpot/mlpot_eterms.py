"""Which CHARMM energy terms hold the MLpot callback energy.

Up to c49 (and the ``charmm_npr256_virial`` build) ``mlpot_call`` added the
callback return to ``ETERM(USER)``. CHARMM c52a1 gives MLpot its own terms:
``MLPO`` (125, the callback return) and ``MLEL`` (126, the ML-charge/MM-charge
electrostatics callback), and USER stays 0. Their CETERM names are assigned on
the first MLpot energy call, so ``get_term_by_name("MLPO")`` raises
``ValueError`` before that and on c49 libraries.

The MLpot energy is therefore USER + MLPO + MLEL on either library: on c49
MLPO/MLEL are absent (0), on c52a1 USER is 0 unless a genuine ``func_set``
user term is also registered (karml never does that alongside MLpot).
"""

from __future__ import annotations

import math
from typing import Mapping

MLPOT_ETERM_KEYS: tuple[str, ...] = ("USER", "MLPO", "MLEL")


def mlpot_eterm_kcal_from_terms(terms: Mapping[str, float] | None) -> float:
    """Sum USER + MLPO + MLEL (kcal/mol) from an energy-term row; missing keys count 0."""
    if not terms:
        return 0.0
    total = 0.0
    for key in MLPOT_ETERM_KEYS:
        try:
            total += float(terms.get(key, 0.0))
        except (TypeError, ValueError):
            continue
    return total


def read_mlpot_eterm_kcal(energy_mod=None) -> float | None:
    """Read USER + MLPO + MLEL from the current CHARMM ETERM array (no ``ENER``).

    Returns None when none of the three terms can be read or the sum is not finite.
    """
    if energy_mod is None:
        import pycharmm.energy as energy_mod

    get_term_by_name = getattr(energy_mod, "get_term_by_name", None)
    if get_term_by_name is None:
        return None
    total = 0.0
    found = False
    for key in MLPOT_ETERM_KEYS:
        try:
            value = float(get_term_by_name(key))
        except (ValueError, IndexError, TypeError, AttributeError):
            continue
        found = True
        total += value
    if not found or not math.isfinite(total):
        return None
    return total
