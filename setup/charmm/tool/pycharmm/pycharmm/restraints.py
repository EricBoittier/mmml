"""Unified restraints module for pycharmm.

This module provides a Pythonic interface to CHARMM's restraint facilities:

- **NOE**: Nuclear Overhauser Effect distance restraints
- **PNOE**: Point NOE restraints (atom to fixed coordinate)
- **RESD**: Restrained distances (reaction coordinates)
- **Harmonic**: Positional restraints (absolute, best-fit, relative, PCA)
- **SCAT**: Constrained atom scaling (via block.py)

The API follows the patterns established in block.py for consistency,
including state tracking, command buffering, and direct memory access.

Examples
--------
>>> import pycharmm.restraints as restraints
>>> from pycharmm.select_atoms import SelectAtoms

>>> # NOE distance restraints
>>> with restraints.NOE(reset=True) as noe:
...     noe.assign(sel1, sel2, kmax=5.0, rmax=3.0)

>>> # Point NOE (atom to coordinate)
>>> with restraints.NOE() as noe:
...     noe.assign_pnoe(sel, cnox=10.0, cnoy=20.0, cnoz=15.0, kmax=10.0, rmax=2.0)

>>> # Restrained distances (reaction coordinate)
>>> restraints.resd_add([
...     (1.0, ('MAIN', 11, 'OG'), ('MAIN', 11, 'HG')),
...     (-1.0, ('MAIN', 11, 'HG'), ('MAIN', 23, 'OD1'))
... ], kval=2000.0, rval=-1.0)

>>> # Harmonic positional restraints
>>> restraints.harmonic_absolute(selection=bb_sel, force_const=10.0)

Note on SCAT (Constrained Atom Scaling)
---------------------------------------
SCAT operates within the BLOCK facility and is already implemented in block.py:

    >>> import pycharmm.block as block
    >>> with block.Block(3) as b:
    ...     block.constrained_atom_scaling(k=300)  # SCAT K 300
    ...     block.define_constrained_atoms(sel)   # CATS selection

See block.constrained_atom_scaling() and block.define_constrained_atoms().
Wrapper functions are also available in this module: scat_enable(), scat_disable().

Backend Compatibility
---------------------
- NOE: Standard (full), DOMDEC (full), BLaDE (full), OpenMM (NOT supported)
- RESD: Standard (full), DOMDEC (full), BLaDE (NOT supported), OpenMM (NOT supported)
- Harmonic: Standard (full), DOMDEC (full), BLaDE (full), OpenMM (full)
- SCAT: All backends supported (via block.py)

Author: Stanislav Cherepanov <stanislc@umich.edu>
"""

from contextlib import contextmanager
import ctypes
import logging
import warnings

import pycharmm.lingo as lingo

# Module logger
logger = logging.getLogger(__name__)
from pycharmm.loader import lib


# =============================================================================
# Exception Classes
# =============================================================================

class RestraintError(Exception):
    """Base exception for restraint module errors."""
    pass


class IncompatibleBackendError(RestraintError):
    """Raised when a restraint type is not supported by the current backend.

    Attributes
    ----------
    restraint_type : str
        The type of restraint that was attempted
    backend : str
        The backend that doesn't support this restraint
    reason : str, optional
        Additional explanation
    """

    def __init__(self, restraint_type: str, backend: str, reason: str = None):
        self.restraint_type = restraint_type
        self.backend = backend
        self.reason = reason
        msg = f"{restraint_type} restraints not supported on {backend} backend"
        if reason:
            msg += f": {reason}"
        super().__init__(msg)


class IncompatibleRestraintError(RestraintError):
    """Raised when two restraint types cannot be combined.

    Attributes
    ----------
    restraint1 : str
        First restraint type
    restraint2 : str
        Second restraint type that conflicts
    selection_overlap : list, optional
        Atom indices or descriptions where restraints overlap
    """

    def __init__(self, restraint1: str, restraint2: str, reason: str = None,
                 selection_overlap: list = None):
        self.restraint1 = restraint1
        self.restraint2 = restraint2
        self.selection_overlap = selection_overlap
        msg = f"Cannot combine {restraint1} and {restraint2} restraints"
        if reason:
            msg += f": {reason}"
        if selection_overlap:
            overlap_str = ', '.join(str(x) for x in selection_overlap[:5])
            if len(selection_overlap) > 5:
                overlap_str += '...'
            msg += f" (overlapping atoms: {overlap_str})"
        super().__init__(msg)


class RestraintStateError(RestraintError):
    """Raised when an operation is invalid in the current restraint state."""
    pass


# =============================================================================
# RESD API Error Codes (must match api_resd.F90)
# =============================================================================

RESD_ERR_NPAIRS = -10    # Invalid npairs (< 1)
RESD_ERR_REDMAX = -11    # Maximum restraint count exceeded
RESD_ERR_REDMX2 = -12    # Maximum atom pair count exceeded
RESD_ERR_DISABLED = -20  # RESD module not compiled

# Error messages for RESD API error codes
_RESD_ERROR_MESSAGES = {
    RESD_ERR_NPAIRS: "Invalid number of atom pairs (npairs < 1)",
    RESD_ERR_REDMAX: "Maximum restraint count exceeded (limit: {max_restraints})",
    RESD_ERR_REDMX2: "Maximum atom pair count exceeded (limit: {max_pairs})",
    RESD_ERR_DISABLED: "RESD module not compiled in this CHARMM build",
}


def _interpret_resd_error(code: int) -> str:
    """Convert RESD API error code to human-readable message.

    Parameters
    ----------
    code : int
        Error code from resddata_add()

    Returns
    -------
    str
        Human-readable error message
    """
    if code not in _RESD_ERROR_MESSAGES:
        return f"Unknown RESD error code: {code}"

    msg = _RESD_ERROR_MESSAGES[code]

    # Try to get actual limits if available
    if '{max_restraints}' in msg or '{max_pairs}' in msg:
        try:
            if _init_resddata_bindings():
                max_restraints = lib.resddata_get_max_restraints()
                max_pairs = lib.resddata_get_max_pairs()
                msg = msg.format(max_restraints=max_restraints, max_pairs=max_pairs)
            else:
                msg = msg.format(max_restraints="unknown", max_pairs="unknown")
        except AttributeError:
            msg = msg.format(max_restraints="unknown", max_pairs="unknown")

    return msg


# =============================================================================
# Backend Incompatibility Rules (for strict validation)
# =============================================================================

BACKEND_INCOMPATIBILITY = {
    'openmm': {
        'disallowed': ['NOE', 'PNOE', 'RESD', 'IC', 'DROPLET'],
        'reason': 'OpenMM interface does not implement these restraint types'
    },
    'blade': {
        'disallowed': ['RESD', 'IC', 'DROPLET'],
        'notes': {
            'NOE': 'Single atom NOE and PNOE supported. '
                   'Complex multi-atom averaging may have limitations.'
        }
    },
    'domdec': {
        'disallowed': [],
        'notes': {}
    },
    'standard': {
        'disallowed': [],
        'notes': {}
    }
}

# Mutual exclusion rules - restraints that cannot be combined on same atoms
MUTUAL_EXCLUSION_RULES = {
    ('FIX', 'HARMONIC_ABSOLUTE'): (
        "Cannot both fix atoms and apply absolute harmonic restraints to same atoms. "
        "FIX immobilizes atoms completely, while harmonic restraints allow movement."
    ),
    ('FIX', 'HARMONIC_BEST_FIT'): (
        "Cannot both fix atoms and apply best-fit harmonic restraints to same atoms."
    ),
    ('FIX', 'HARMONIC_RELATIVE'): (
        "Cannot both fix atoms and apply relative harmonic restraints to same atoms."
    ),
    ('FIX', 'HARMONIC_PCA'): (
        "Cannot both fix atoms and apply PCA harmonic restraints to same atoms."
    ),
}


# =============================================================================
# Backend Support Matrix
# =============================================================================

BACKEND_SUPPORT = {
    'NOE': {
        'standard': True,
        'domdec': True,
        'blade': True,
        'openmm': False,
        'notes': {
            'blade': 'Single atom/atoms NOE and PNOE supported. '
                    'Complex multi-atom averaging may have limitations.',
            'domdec': 'Full NOE support in DOMDEC parallel.',
            'openmm': 'NOE restraints are NOT implemented in OpenMM interface.'
        }
    },
    'SCAT': {
        'location': 'block.py',
        'description': 'Constrained atom scaling is part of BLOCK facility. '
                      'Access via block.constrained_atom_scaling() and '
                      'block.define_constrained_atoms()',
        'functions': ['constrained_atom_scaling', 'define_constrained_atoms'],
        'standard': True,
        'domdec': True,
        'blade': True,
        'openmm': True
    },
    'RESD': {
        'standard': True,
        'domdec': True,
        'blade': False,
        'openmm': False,
        'notes': {
            'blade': 'RESD restraints not mentioned in BLaDE documentation.'
        }
    },
    'HARMONIC': {
        'location': 'cons_harm.py',
        'description': 'Harmonic positional restraints via CONS HARMonic. '
                      'Wrapper functions in restraints.py provide unified API.',
        'functions': ['harmonic_absolute', 'harmonic_best_fit',
                     'harmonic_relative', 'harmonic_pca', 'harmonic_turn_off'],
        'standard': True,
        'domdec': True,
        'blade': True,
        'openmm': True,
        'notes': {
            'blade': 'Full support for all harmonic restraint types.',
            'openmm': 'Full support via CustomExternalForce.'
        }
    },
    'FIX': {
        'location': 'cons_fix.py',
        'description': 'Fixed atom constraints via CONS FIX. '
                      'Completely immobilizes selected atoms.',
        'functions': ['fix_atoms', 'fix_turn_off'],
        'standard': True,
        'domdec': True,
        'blade': True,
        'openmm': True,
        'notes': {
            'blade': 'Full support for fixed atom constraints.',
            'openmm': 'Full support via CustomExternalForce with large k.'
        }
    },
    'DIHE': {
        'location': 'cons_methods.py',
        'description': 'Dihedral angle restraints via CONS DIHE.',
        'functions': ['dihe_restraint'],
        'standard': True,
        'domdec': True,
        'blade': True,  # CONS DIHE supported on BLaDE GPU
        'openmm': True,
        'notes': {
            'blade': 'Full GPU acceleration via BLaDE dihedral restraint kernel.'
        }
    },
    'MMFP_DIHE': {
        'location': 'restraints.py',
        'description': 'MMFP GEO dihedral restraints (BLaDE GPU compatible).',
        'functions': ['mmfp_dihedral'],
        'standard': True,
        'domdec': True,
        'blade': True,  # Full GPU support via MMFP GEO
        'openmm': True,
        'notes': {
            'blade': 'Full GPU acceleration via MMFP GEO sphere dihedral.',
            'energy': 'E = 0.5 * k * (phi - phi0)^2 (factor of 2 vs CONS DIHE)'
        }
    },
    'IC': {
        'location': 'cons_methods.py',
        'description': 'Internal coordinate restraints via CONS IC.',
        'functions': ['ic_restraint'],
        'standard': True,
        'domdec': True,
        'blade': False,
        'openmm': False,
        'notes': {
            'blade': 'IC restraints not explicitly documented for BLaDE.'
        }
    },
    'DROPLET': {
        'location': 'cons_methods.py',
        'description': 'Quartic droplet potential via CONS DROPlet.',
        'functions': ['droplet_restraint'],
        'standard': True,
        'domdec': True,
        'blade': False,
        'openmm': False
    }
}


# =============================================================================
# Internal State Tracking
# =============================================================================

class _RestraintState:
    """Internal state tracker for restraint facilities.

    Mirrors the pattern established in block.py. Maintains Python-side
    copies of restraint configurations for efficient queries without
    parsing CHARMM output.

    Provides strict validation for backend compatibility and mutual
    exclusion rules.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        """Reset all restraint state to initial values."""
        # Backend tracking
        self._current_backend = 'standard'

        # Active restraints registry - tracks all enabled restraints
        self.active = {
            # Atom-based restraints
            'fix': {'enabled': False, 'selections': [], 'options': {}},
            'harmonic_absolute': {'enabled': False, 'selections': [], 'options': {}},
            'harmonic_best_fit': {'enabled': False, 'selections': [], 'options': {}},
            'harmonic_relative': {'enabled': False, 'pairs': [], 'options': {}},
            'harmonic_pca': {'enabled': False, 'selections': [], 'options': {}},

            # Distance restraints
            'noe': {'enabled': False, 'count': 0, 'scale': 1.0, 'restraints': []},
            'pnoe': {'enabled': False, 'count': 0, 'restraints': []},
            'resd': {'enabled': False, 'count': 0, 'scale': 1.0, 'restraints': []},

            # Angle restraints
            'dihedral': {'enabled': False, 'restraints': []},
            'mmfp_dihedral': {'enabled': False, 'restraints': []},

            # Position restraints
            'droplet': {'enabled': False, 'settings': {}},

            # Internal coordinate restraints
            'ic_bond': {'enabled': False, 'force': None},
            'ic_angle': {'enabled': False, 'force': None},
            'ic_dihedral': {'enabled': False, 'force': None},
            'ic_improper': {'enabled': False, 'force': None},
        }

        # NOE state (legacy compatibility)
        self.noe = {
            'active': False,
            'count': 0,
            'scale': 1.0,
            'restraints': [],
            'charmm_synced': False
        }

        # Command buffer for NOE (block.py pattern)
        self._noe_command_buffer = []
        self._in_noe_context = False

        # RESD state (legacy compatibility)
        self.resd = {
            'active': False,
            'count': 0,
            'scale': 1.0,
            'restraints': []
        }

    def set_backend(self, backend: str):
        """Set current backend and validate active restraints.

        Parameters
        ----------
        backend : str
            One of 'standard', 'domdec', 'blade', 'openmm'

        Raises
        ------
        ValueError
            If backend is not recognized
        IncompatibleBackendError
            If active restraints are incompatible with the new backend
        """
        valid_backends = {'standard', 'domdec', 'blade', 'openmm'}
        if backend not in valid_backends:
            raise ValueError(f"Unknown backend: {backend}. Must be one of: {valid_backends}")

        # Check for incompatible active restraints
        conflicts = self._check_backend_conflicts(backend)
        if conflicts:
            raise IncompatibleBackendError(
                ', '.join(conflicts), backend,
                "These restraints are currently active and incompatible with the new backend"
            )
        self._current_backend = backend

    def get_current_backend(self) -> str:
        """Get current backend name."""
        return self._current_backend

    def _check_backend_conflicts(self, backend: str) -> list:
        """Check if any active restraints conflict with backend."""
        conflicts = []
        rules = BACKEND_INCOMPATIBILITY.get(backend, {})
        disallowed = rules.get('disallowed', [])

        # Map restraint types to active keys
        type_to_keys = {
            'NOE': ['noe'],
            'PNOE': ['pnoe'],
            'RESD': ['resd'],
            'IC': ['ic_bond', 'ic_angle', 'ic_dihedral', 'ic_improper'],
            'DROPLET': ['droplet'],
            'HARMONIC': ['harmonic_absolute', 'harmonic_best_fit',
                        'harmonic_relative', 'harmonic_pca'],
            'FIX': ['fix'],
            'DIHE': ['dihedral'],
        }

        for restraint_type in disallowed:
            keys = type_to_keys.get(restraint_type, [restraint_type.lower()])
            for key in keys:
                if key in self.active and self.active[key].get('enabled'):
                    conflicts.append(restraint_type)
                    break  # Only add once per type
        return conflicts

    def get_active_restraints(self) -> dict:
        """Get dictionary of all currently active restraints.

        Returns
        -------
        dict
            Dictionary of restraint type -> state for enabled restraints
        """
        return {k: v.copy() for k, v in self.active.items() if v.get('enabled')}

    def validate_can_add(self, restraint_type: str, selection=None) -> None:
        """Validate that a restraint can be added. Raises on failure.

        Parameters
        ----------
        restraint_type : str
            Type of restraint being added (e.g., 'FIX', 'HARMONIC_ABSOLUTE')
        selection : optional
            Atom selection (for checking mutual exclusion on same atoms)

        Raises
        ------
        IncompatibleBackendError
            If restraint not supported on current backend
        IncompatibleRestraintError
            If restraint conflicts with currently active restraints
        """
        backend = self._current_backend
        restraint_upper = restraint_type.upper()

        # Map specific types to general category for backend check
        category_map = {
            'HARMONIC_ABSOLUTE': 'HARMONIC',
            'HARMONIC_BEST_FIT': 'HARMONIC',
            'HARMONIC_RELATIVE': 'HARMONIC',
            'HARMONIC_PCA': 'HARMONIC',
            'IC_BOND': 'IC',
            'IC_ANGLE': 'IC',
            'IC_DIHEDRAL': 'IC',
            'IC_IMPROPER': 'IC',
            'PNOE': 'NOE',  # PNOE uses NOE backend support
        }
        category = category_map.get(restraint_upper, restraint_upper)

        # Check backend compatibility
        rules = BACKEND_INCOMPATIBILITY.get(backend, {})
        disallowed = rules.get('disallowed', [])
        if category in disallowed:
            reason = rules.get('reason', '')
            raise IncompatibleBackendError(restraint_type, backend, reason)

        # Check mutual exclusion rules
        self._check_mutual_exclusions(restraint_upper, selection)

    def _check_mutual_exclusions(self, restraint_type: str, selection) -> None:
        """Check for incompatible restraint combinations."""
        for (type1, type2), reason in MUTUAL_EXCLUSION_RULES.items():
            if restraint_type in (type1, type2):
                other = type2 if restraint_type == type1 else type1
                other_key = other.lower()

                # Check if the other restraint is active
                if self.active.get(other_key, {}).get('enabled'):
                    # If we have selection info, check for overlap
                    overlap = None
                    if selection is not None:
                        overlap = self._check_selection_overlap(other_key, selection)
                        if overlap:
                            raise IncompatibleRestraintError(
                                restraint_type, other, reason, overlap
                            )
                    else:
                        # No selection info - raise if potentially conflicting
                        raise IncompatibleRestraintError(restraint_type, other, reason)

    def _check_selection_overlap(self, restraint_key: str, selection) -> list:
        """Check if selection overlaps with existing restraint selections.

        Returns list of overlapping atom indices, or empty list if no overlap.
        """
        # Get atom indices from new selection
        new_atoms = set()
        if hasattr(selection, 'get_indices'):
            new_atoms = set(selection.get_indices())
        elif hasattr(selection, 'get_n_selected'):
            # SelectAtoms object - get the indices
            try:
                from pycharmm.select_atoms import SelectAtoms
                if isinstance(selection, SelectAtoms):
                    # Selection might not expose indices directly
                    # For now, return empty (conservative - no overlap assumed)
                    return []
            except ImportError:
                return []

        if not new_atoms:
            return []

        # Get existing selection atoms
        existing_atoms = set()
        restraint_state = self.active.get(restraint_key, {})
        for sel_str in restraint_state.get('selections', []):
            # Would need to parse selection string - complex
            # For now, return overlap as the selection if any selections exist
            if sel_str:
                return list(new_atoms)[:10]  # Return some atoms as overlap indicator

        return []

    def to_dict(self):
        """Export complete state as dictionary."""
        return {
            'noe': self.noe.copy(),
            'resd': self.resd.copy(),
            'active': {k: v.copy() for k, v in self.active.items()},
            'backend': self._current_backend
        }

    def noe_add_restraint(self, params):
        """Track a new NOE restraint in Python state."""
        self.noe['restraints'].append(params.copy())
        self.noe['count'] += 1
        self.noe['active'] = True
        self.active['noe']['enabled'] = True
        self.active['noe']['count'] = self.noe['count']
        self.active['noe']['restraints'] = self.noe['restraints']
        return self.noe['count']

    def noe_clear(self):
        """Clear all NOE restraints from Python state."""
        self.noe['restraints'] = []
        self.noe['count'] = 0
        self.noe['scale'] = 1.0
        self.noe['active'] = False
        self.noe['charmm_synced'] = False
        self.active['noe']['enabled'] = False
        self.active['noe']['count'] = 0
        self.active['noe']['restraints'] = []

    def resd_add_restraint(self, params):
        """Track a new RESD restraint in Python state."""
        self.resd['restraints'].append(params.copy())
        self.resd['count'] += 1
        self.resd['active'] = True
        self.active['resd']['enabled'] = True
        self.active['resd']['count'] = self.resd['count']
        self.active['resd']['restraints'] = self.resd['restraints']
        return self.resd['count']

    def resd_clear(self):
        """Clear all RESD restraints from Python state."""
        self.resd['restraints'] = []
        self.resd['count'] = 0
        self.resd['scale'] = 1.0
        self.resd['active'] = False
        self.active['resd']['enabled'] = False
        self.active['resd']['count'] = 0
        self.active['resd']['restraints'] = []

    # Methods to track other restraint types
    def set_fix_active(self, enabled: bool, selection=None, options=None):
        """Update fix restraint state."""
        self.active['fix']['enabled'] = enabled
        if selection:
            self.active['fix']['selections'].append(str(selection))
        if options:
            self.active['fix']['options'] = options
        if not enabled:
            self.active['fix']['selections'] = []
            self.active['fix']['options'] = {}

    def set_harmonic_active(self, variant: str, enabled: bool, selection=None, options=None):
        """Update harmonic restraint state.

        Parameters
        ----------
        variant : str
            One of 'absolute', 'best_fit', 'relative', 'pca'
        enabled : bool
            Whether the restraint is active
        selection : optional
            Atom selection
        options : dict, optional
            Restraint options
        """
        key = f'harmonic_{variant}'
        if key in self.active:
            self.active[key]['enabled'] = enabled
            if selection:
                if 'selections' in self.active[key]:
                    self.active[key]['selections'].append(str(selection))
                elif 'pairs' in self.active[key]:
                    self.active[key]['pairs'].append(str(selection))
            if options:
                self.active[key]['options'] = options
            if not enabled:
                if 'selections' in self.active[key]:
                    self.active[key]['selections'] = []
                if 'pairs' in self.active[key]:
                    self.active[key]['pairs'] = []
                self.active[key]['options'] = {}

    def set_dihedral_active(self, enabled: bool, restraint_info=None):
        """Update dihedral restraint state."""
        self.active['dihedral']['enabled'] = enabled
        if restraint_info:
            self.active['dihedral']['restraints'].append(restraint_info)
        if not enabled:
            self.active['dihedral']['restraints'] = []

    def set_droplet_active(self, enabled: bool, settings=None):
        """Update droplet restraint state."""
        self.active['droplet']['enabled'] = enabled
        if settings:
            self.active['droplet']['settings'] = settings
        if not enabled:
            self.active['droplet']['settings'] = {}

    def set_ic_active(self, component: str, enabled: bool, force=None):
        """Update IC restraint state.

        Parameters
        ----------
        component : str
            One of 'bond', 'angle', 'dihedral', 'improper'
        enabled : bool
            Whether the restraint is active
        force : float, optional
            Force constant
        """
        key = f'ic_{component}'
        if key in self.active:
            self.active[key]['enabled'] = enabled
            self.active[key]['force'] = force if enabled else None


# Module-level singleton
_state = _RestraintState()


# =============================================================================
# ctypes Structures for Harmonic Restraints (from cons_harm.py)
# =============================================================================

_HARMONIC_OPTIONS_FIELDS = [
    ('expo', ctypes.c_int),
    ('x_scale', ctypes.c_double),
    ('y_scale', ctypes.c_double),
    ('z_scale', ctypes.c_double),
    ('q_no_rot', ctypes.c_int),
    ('q_no_trans', ctypes.c_int),
    ('q_mass', ctypes.c_int),
    ('q_weight', ctypes.c_int),
    ('force_const', ctypes.c_double)
]


class _HarmonicOptions(ctypes.Structure):
    """A ctypes struct to hold harmonic constraint settings.

    This structure is passed to CHARMM's cons_harm_* routines.

    Attributes
    ----------
    expo : int
        Exponent on diff between atom and ref atom (default: 2)
    x_scale : float
        Global scale factor for the x component (default: 1.0)
    y_scale : float
        Global scale factor for the y component (default: 1.0)
    z_scale : float
        Global scale factor for the z component (default: 1.0)
    q_no_rot : int
        Do not do rotational restraint (default: 0)
    q_no_trans : int
        Do not do translational restraint (default: 0)
    q_mass : int
        Multiply k by atom mass (default: 0)
    q_weight : int
        Use weight array for k(i) (default: 0)
    force_const : float
        Restraint force constant k (default: 0.0)
    """
    _fields_ = _HARMONIC_OPTIONS_FIELDS


# Default values: expo=2, scales=1.0, all flags=0, force_const=0.0
_HARMONIC_OPTIONS_DEFAULTS = (2, 1.0, 1.0, 1.0, 0, 0, 0, 0, 0.0)


def _make_harmonic_opts(valid_fields: list, settings: dict) -> _HarmonicOptions:
    """Create a new instance of _HarmonicOptions from settings.

    Parameters
    ----------
    valid_fields : list[str]
        List of valid field names used to filter settings.
        Different harmonic restraint types use different subsets:
        - absolute: expo, x_scale, y_scale, z_scale, q_mass, q_weight, force_const
        - best_fit: q_no_rot, q_no_trans, q_mass, q_weight, force_const
        - relative: q_no_rot, q_no_trans, q_mass, q_weight, force_const
        - pca: expo, x_scale, y_scale, z_scale, q_mass, q_weight, force_const
    settings : dict
        Name and value for harmonic restraint settings

    Returns
    -------
    _HarmonicOptions
        An instance filled with values from settings

    Raises
    ------
    TypeError
        If a setting value cannot be converted to the expected type
    ValueError
        If a setting value is invalid for the expected type
    """
    new_opts = _HarmonicOptions(*_HARMONIC_OPTIONS_DEFAULTS)
    fields_types = dict(_HARMONIC_OPTIONS_FIELDS)
    valid_settings = [(k, v) for k, v in settings.items() if k in valid_fields]

    for k, v in valid_settings:
        try:
            setattr(new_opts, k, fields_types[k](v))
        except TypeError as e:
            raise TypeError(
                f"Cannot convert '{k}' value {v!r} to {fields_types[k].__name__}: {e}"
            ) from e
        except ValueError as e:
            raise ValueError(
                f"Invalid value for '{k}': {v!r} - {e}"
            ) from e

    return new_opts


# Field sets for different harmonic restraint types
_HARMONIC_ABSOLUTE_FIELDS = ['expo', 'x_scale', 'y_scale', 'z_scale',
                              'q_mass', 'q_weight', 'force_const']
_HARMONIC_BEST_FIT_FIELDS = ['q_no_rot', 'q_no_trans',
                              'q_mass', 'q_weight', 'force_const']
_HARMONIC_RELATIVE_FIELDS = ['q_no_rot', 'q_no_trans',
                              'q_mass', 'q_weight', 'force_const']
_HARMONIC_PCA_FIELDS = ['expo', 'x_scale', 'y_scale', 'z_scale',
                         'q_mass', 'q_weight', 'force_const']


# =============================================================================
# Low-level Implementation Functions (Direct lib calls)
# =============================================================================
# These functions call lib directly and are used by both namespace
# classes and backward-compatible wrapper functions.

def _harmonic_setup_absolute(selection, comparison: bool = False, **kwargs) -> bool:
    """Low-level absolute harmonic restraint setup.

    Calls lib.cons_harm_setup_absolute directly.

    Parameters
    ----------
    selection : SelectAtoms
        Atom selection object
    comparison : bool
        If True, apply to comparison set
    **kwargs : dict
        Harmonic restraint options (expo, x_scale, y_scale, z_scale,
        q_mass, q_weight, force_const)

    Returns
    -------
    bool
        True if successful
    """
    import pycharmm

    if selection is None:
        selection = pycharmm.SelectAtoms().all_atoms()

    opts = _make_harmonic_opts(_HARMONIC_ABSOLUTE_FIELDS, kwargs)
    c_sel = selection.as_ctypes()
    c_comp = ctypes.c_int(comparison)

    status = lib.cons_harm_setup_absolute(
        c_sel,
        ctypes.byref(c_comp),
        ctypes.byref(opts)
    )
    return bool(status)


def _harmonic_setup_best_fit(selection, comparison: bool = False, **kwargs) -> bool:
    """Low-level best-fit harmonic restraint setup.

    Calls lib.cons_harm_setup_best_fit directly.

    Parameters
    ----------
    selection : SelectAtoms
        Atom selection object
    comparison : bool
        If True, apply to comparison set
    **kwargs : dict
        Harmonic restraint options (q_no_rot, q_no_trans, q_mass,
        q_weight, force_const)

    Returns
    -------
    bool
        True if successful
    """
    import pycharmm

    if selection is None:
        selection = pycharmm.SelectAtoms().all_atoms()

    opts = _make_harmonic_opts(_HARMONIC_BEST_FIT_FIELDS, kwargs)
    c_sel = selection.as_ctypes()
    c_comp = ctypes.c_int(comparison)

    status = lib.cons_harm_setup_best_fit(
        c_sel,
        ctypes.byref(c_comp),
        ctypes.byref(opts)
    )
    return bool(status)


def _harmonic_setup_relative(iselection, jselection, comparison: bool = False,
                             **kwargs) -> bool:
    """Low-level relative harmonic restraint setup.

    Calls lib.cons_harm_setup_relative directly.

    Parameters
    ----------
    iselection : SelectAtoms
        First atom selection
    jselection : SelectAtoms
        Second atom selection (must match iselection count)
    comparison : bool
        If True, apply to comparison set
    **kwargs : dict
        Harmonic restraint options (q_no_rot, q_no_trans, q_mass,
        q_weight, force_const)

    Returns
    -------
    bool
        True if successful

    Raises
    ------
    ValueError
        If selections have different numbers of atoms
    """
    # Validate that selections have the same number of atoms
    i_count = iselection.get_n_selected()
    j_count = jselection.get_n_selected()
    if i_count != j_count:
        raise ValueError(
            f"Selection size mismatch: iselection has {i_count} atoms, "
            f"jselection has {j_count} atoms. Both must have the same count."
        )

    opts = _make_harmonic_opts(_HARMONIC_RELATIVE_FIELDS, kwargs)
    c_isel = iselection.as_ctypes()
    c_jsel = jselection.as_ctypes()
    c_comp = ctypes.c_int(comparison)

    status = lib.cons_harm_setup_relative(
        c_isel, c_jsel,
        ctypes.byref(c_comp),
        ctypes.byref(opts)
    )
    return bool(status)


def _harmonic_setup_pca(selection, comparison: bool = False, **kwargs) -> bool:
    """Low-level PCA harmonic restraint setup.

    Calls lib.cons_harm_setup_pca directly.

    Parameters
    ----------
    selection : SelectAtoms
        Atom selection object
    comparison : bool
        If True, apply to comparison set
    **kwargs : dict
        Harmonic restraint options (expo, x_scale, y_scale, z_scale,
        q_mass, q_weight, force_const)

    Returns
    -------
    bool
        True if successful
    """
    import pycharmm

    if selection is None:
        selection = pycharmm.SelectAtoms().all_atoms()

    opts = _make_harmonic_opts(_HARMONIC_PCA_FIELDS, kwargs)
    c_sel = selection.as_ctypes()
    c_comp = ctypes.c_int(comparison)

    status = lib.cons_harm_setup_pca(
        c_sel,
        ctypes.byref(c_comp),
        ctypes.byref(opts)
    )
    return bool(status)


def _harmonic_turn_off() -> bool:
    """Low-level harmonic restraint turn off.

    Calls lib.cons_harm_turn_off directly.

    Returns
    -------
    bool
        True if successful
    """
    status = lib.cons_harm_turn_off()
    return bool(status)


def _fix_setup(selection, comparison: bool = False, purge: bool = False,
               bond: bool = False, angle: bool = False, phi: bool = False,
               imp: bool = False, cmap: bool = False) -> bool:
    """Low-level fix constraint setup.

    Calls lib.cons_fix_setup directly.

    Parameters
    ----------
    selection : SelectAtoms
        Atom selection object
    comparison : bool
        If True, apply to comparison set
    purge : bool
        If True, use PURGE option (modifies PSF irrevocably)
    bond : bool
        If True, also fix bonds involving selected atoms
    angle : bool
        If True, also fix angles involving selected atoms
    phi : bool
        If True, also fix dihedrals involving selected atoms
    imp : bool
        If True, also fix impropers involving selected atoms
    cmap : bool
        If True, also fix CMAP terms involving selected atoms

    Returns
    -------
    bool
        True if successful
    """
    c_sel = selection.as_ctypes()
    c_comp = ctypes.c_int(comparison)
    c_purge = ctypes.c_int(purge)
    c_bond = ctypes.c_int(bond)
    c_angle = ctypes.c_int(angle)
    c_phi = ctypes.c_int(phi)
    c_imp = ctypes.c_int(imp)
    c_cmap = ctypes.c_int(cmap)

    status = lib.cons_fix_setup(
        c_sel,
        ctypes.byref(c_comp),
        ctypes.byref(c_purge),
        ctypes.byref(c_bond),
        ctypes.byref(c_angle),
        ctypes.byref(c_phi),
        ctypes.byref(c_imp),
        ctypes.byref(c_cmap)
    )
    return bool(status)


def _fix_turn_off(comparison: bool = False) -> bool:
    """Low-level fix constraint turn off.

    Turns off fix constraints by setting empty selection.

    Parameters
    ----------
    comparison : bool
        If True, turn off constraints on comparison set

    Returns
    -------
    bool
        True if successful
    """
    import pycharmm
    none_selected = pycharmm.SelectAtoms()
    return _fix_setup(none_selected, comparison=comparison)


def _parse_dihe_selection(selection: str):
    """Parse dihedral selection string to get 4 atom indices.

    Handles 'bynum i j k l' format.

    Parameters
    ----------
    selection : str
        Selection string for 4 atoms

    Returns
    -------
    list or None
        List of 4 atom indices (1-based), or None if parsing fails
    """
    sel = selection.strip().lower()
    if sel.startswith('bynum'):
        # Format: bynum 7 9 15 17
        parts = sel.split()
        if len(parts) == 5:
            try:
                return [int(parts[i]) for i in range(1, 5)]
            except ValueError:
                return None
    return None


def _dihe_restraint(selection: str = '', force: float = 0, clear: bool = False,
                    **kwargs) -> None:
    """Low-level dihedral restraint setup.

    Tries direct API access first, falls back to CHARMM script command.

    Parameters
    ----------
    selection : str
        Atom selection for dihedral (4 atoms)
    force : float
        Force constant
    clear : bool
        If True, clear all dihedral restraints
    **kwargs : dict
        Additional options: minimum, period, width, comp, main
    """
    import math

    # Try direct API first
    try:
        if _init_consdata_bindings():
            if clear or len(selection) == 0:
                # Clear all dihedral restraints
                lib.consdata_clear_dihe()
                return
            else:
                # Parse selection to get 4 atom indices
                atom_indices = _parse_dihe_selection(selection)
                if atom_indices is not None and len(atom_indices) == 4:
                    i_atom, j_atom, k_atom, l_atom = atom_indices

                    # Get parameters
                    minimum = kwargs.get('minimum', 0.0)
                    period = kwargs.get('period', 0)
                    width = kwargs.get('width', 0.0)

                    # Convert minimum from degrees to radians
                    min_rad = math.radians(minimum)

                    # Add the dihedral restraint
                    result = lib.consdata_add_dihe(
                        i_atom, j_atom, k_atom, l_atom,
                        float(force), min_rad, int(period), float(width)
                    )
                    if result > 0:
                        return
    except (AttributeError, OSError, RuntimeError):
        pass

    # Fall back to script-based approach
    import pycharmm.script

    if clear or len(selection) == 0:
        cons_dihe = pycharmm.script.CommandScript('cons cldh')
    else:
        cons_command = 'cons dihe ' + str(selection)
        cons_dihe = pycharmm.script.CommandScript(cons_command,
                                                  force=force,
                                                  **kwargs)
    cons_dihe.run()


def _ic_restraint(**kwargs) -> None:
    """Low-level IC restraint setup.

    Tries direct API access first, falls back to CHARMM script command.

    Parameters
    ----------
    **kwargs : dict
        Options: bond, angle, dihedral, improper, exponent, upper
    """
    # Try direct API first
    try:
        if _init_consdata_bindings():
            # Extract parameters
            bond_force = float(kwargs.get('bond', 0.0))
            angle_force = float(kwargs.get('angle', 0.0))
            dihe_force = float(kwargs.get('dihedral', 0.0))
            impr_force = float(kwargs.get('improper', 0.0))
            exponent = int(kwargs.get('exponent', 2))
            upper = 1 if kwargs.get('upper', False) else 0

            # Set IC restraint parameters
            lib.consdata_set_ic_params(
                bond_force, angle_force, dihe_force, impr_force,
                exponent, upper
            )
            return
    except (AttributeError, OSError, RuntimeError):
        pass

    # Fall back to script-based approach
    import pycharmm.script

    cons_ic = pycharmm.script.CommandScript('cons ic', **kwargs)
    cons_ic.run()


def _droplet_restraint(**kwargs) -> None:
    """Low-level droplet restraint setup.

    Tries direct API access first, falls back to CHARMM script command.

    Parameters
    ----------
    **kwargs : dict
        Options: force, exponent, nomass
    """
    # Try direct API first
    try:
        if _init_consdata_bindings():
            # Extract parameters
            force = float(kwargs.get('force', 0.0))
            exponent = int(kwargs.get('exponent', 4))
            # nomass means NOT mass weighted, so mass_weight is opposite
            mass_weight = 0 if kwargs.get('nomass', False) else 1

            # Set droplet restraint parameters
            lib.consdata_set_droplet_params(
                force, exponent, mass_weight
            )
            return
    except (AttributeError, OSError, RuntimeError):
        pass

    # Fall back to script-based approach
    import pycharmm.script

    cons_droplet = pycharmm.script.CommandScript('cons droplet', **kwargs)
    cons_droplet.run()


# =============================================================================
# Helper Functions
# =============================================================================

def _build_selection_string(selection):
    """Convert selection to CHARMM selection string.

    Handles SelectAtoms objects, strings, and objects with get_atom_indexes().

    Parameters
    ----------
    selection : SelectAtoms, str, or selection-like
        The selection to convert

    Returns
    -------
    str
        CHARMM selection string ending with END
    """
    from pycharmm.select_atoms import SelectAtoms

    if isinstance(selection, SelectAtoms):
        # SelectAtoms uses 0-based indices internally
        # CHARMM BYNUM uses 1-based atom numbers
        indices = selection.get_atom_indexes()
        if len(indices) == 0:
            raise ValueError("Selection contains no atoms")
        # Convert 0-based to 1-based for CHARMM
        atom_nums = [str(i + 1) for i in indices]
        return f"SELE BYNUM {' '.join(atom_nums)} END"
    elif isinstance(selection, str):
        # Ensure it ends with END
        sel = selection.strip()
        if not sel.upper().endswith('END'):
            sel = f'SELE {sel} END'
        elif not sel.upper().startswith('SELE'):
            sel = f'SELE {sel}'
        return sel
    elif hasattr(selection, 'get_atom_indexes'):
        # Handle other selection-like objects with get_atom_indexes()
        indices = selection.get_atom_indexes()
        if len(indices) == 0:
            raise ValueError("Selection contains no atoms")
        atom_nums = [str(i + 1) for i in indices]
        return f"SELE BYNUM {' '.join(atom_nums)} END"
    else:
        raise TypeError(f"Invalid selection type: {type(selection)}")


def _execute_noe_command(cmd):
    """Accumulate command for NOE context or execute immediately.

    If inside an NOE context (with statement), commands are buffered
    for batch execution. Otherwise, wraps in NOE/END and executes.
    """
    if _state._in_noe_context:
        _state._noe_command_buffer.append(cmd)
    else:
        # Execute immediately in standalone mode
        script = f'NOE\n{cmd}\nEND'
        lingo.charmm_script(script)


def _flush_noe_command_buffer():
    """Execute all accumulated NOE commands as single batch."""
    if not _state._noe_command_buffer:
        return

    script = '\n'.join(_state._noe_command_buffer)
    _state._noe_command_buffer = []
    lingo.charmm_script(script)


# =============================================================================
# Namespace Classes for Organized API
# =============================================================================
# These classes provide a namespace-based API: restraints.atoms.fix(),
# restraints.distances.noe(), etc.

class _AtomRestraints:
    """Namespace for atom-based restraints (FIX, HARMONIC).

    Provides methods for fixing atoms or applying harmonic positional
    restraints with backend validation and state tracking.

    Examples
    --------
    >>> import pycharmm.restraints as restraints
    >>> from pycharmm.select_atoms import SelectAtoms

    >>> # Fix atoms
    >>> bb = SelectAtoms(atom_type=['CA', 'C', 'N'])
    >>> restraints.atoms.fix(bb)

    >>> # Harmonic restraints
    >>> restraints.atoms.harmonic_absolute(selection=bb, force_const=10.0)

    >>> # Turn off
    >>> restraints.atoms.harmonic_turn_off()
    >>> restraints.atoms.fix_turn_off()
    """

    def __init__(self, state: _RestraintState):
        self._state = state

    def fix(self, selection, comparison: bool = False, purge: bool = False,
            bond: bool = False, angle: bool = False, phi: bool = False,
            imp: bool = False, cmap: bool = False) -> bool:
        """Fix selected atoms in place (immobilize).

        Fixed atoms have zero velocity and forces are not computed.
        Maps to CHARMM command: CONS FIX ...

        Parameters
        ----------
        selection : SelectAtoms
            Atoms to fix (immobilize)
        comparison : bool, optional
            If True, apply to comparison coordinate set. Default False.
        purge : bool, optional
            If True, use PURGE option (modifies PSF irrevocably). Default False.
        bond : bool, optional
            If True, also fix bonds involving selected atoms. Default False.
        angle : bool, optional
            If True, also fix angles involving selected atoms. Default False.
        phi : bool, optional
            If True, also fix dihedrals involving selected atoms. Default False.
        imp : bool, optional
            If True, also fix impropers involving selected atoms. Default False.
        cmap : bool, optional
            If True, also fix CMAP terms involving selected atoms. Default False.

        Returns
        -------
        bool
            True if successful

        Raises
        ------
        IncompatibleBackendError
            If FIX not supported on current backend
        IncompatibleRestraintError
            If FIX conflicts with active harmonic restraints on same atoms
        """
        self._state.validate_can_add('FIX', selection)
        result = _fix_setup(selection, comparison=comparison, purge=purge,
                           bond=bond, angle=angle, phi=phi, imp=imp, cmap=cmap)
        if result:
            options = {'comparison': comparison, 'purge': purge,
                      'bond': bond, 'angle': angle, 'phi': phi,
                      'imp': imp, 'cmap': cmap}
            self._state.set_fix_active(True, selection, options)
        return result

    def fix_turn_off(self, comparison: bool = False) -> bool:
        """Turn off all fixed atom constraints.

        Maps to CHARMM command: CONS FIX SELE NONE END

        Parameters
        ----------
        comparison : bool, optional
            If True, turn off constraints on comparison set. Default False.

        Returns
        -------
        bool
            True if successful
        """
        result = _fix_turn_off(comparison=comparison)
        if result:
            self._state.set_fix_active(False)
        return result

    def harmonic_absolute(self, selection=None, force_const: float = 0.0,
                          expo: int = 2, x_scale: float = 1.0,
                          y_scale: float = 1.0, z_scale: float = 1.0,
                          comparison: bool = False, mass_weighted: bool = False,
                          use_weights: bool = False, q_mass=None, q_weight=None) -> bool:
        """Apply absolute harmonic positional restraints.

        Restrains atoms to fixed reference positions (current coordinates
        or comparison set).
        Maps to CHARMM command: CONS HARMonic ABSOlute ...

        Parameters
        ----------
        selection : SelectAtoms, optional
            Atoms to restrain. If None, all atoms are restrained.
        force_const : float, optional
            Force constant k (kcal/mol/A^expo). Default 0.0.
        expo : int, optional
            Exponent on displacement (2 for harmonic). Default 2.
        x_scale, y_scale, z_scale : float, optional
            Scale factors for each component. Default 1.0.
        comparison : bool, optional
            If True, use comparison coordinate set. Default False.
        mass_weighted : bool, optional
            If True, multiply k by atomic mass. Default False.
        use_weights : bool, optional
            If True, use weight array for k(i). Default False.

        Returns
        -------
        bool
            True if successful

        Raises
        ------
        IncompatibleBackendError
            If HARMONIC not supported on current backend
        IncompatibleRestraintError
            If conflicts with FIX on same atoms
        """
        if q_mass is not None:
            warnings.warn(
                "q_mass is deprecated; use mass_weighted instead",
                DeprecationWarning,
                stacklevel=2
            )
            mass_weighted = bool(q_mass)
        if q_weight is not None:
            warnings.warn(
                "q_weight is deprecated; use use_weights instead",
                DeprecationWarning,
                stacklevel=2
            )
            use_weights = bool(q_weight)

        self._state.validate_can_add('HARMONIC_ABSOLUTE', selection)
        result = _harmonic_setup_absolute(
            selection, comparison=comparison,
            force_const=force_const, expo=expo,
            x_scale=x_scale, y_scale=y_scale, z_scale=z_scale,
            q_mass=int(mass_weighted), q_weight=int(use_weights)
        )
        if result:
            options = {'force_const': force_const, 'expo': expo,
                      'x_scale': x_scale, 'y_scale': y_scale, 'z_scale': z_scale,
                      'comparison': comparison, 'mass_weighted': mass_weighted,
                      'use_weights': use_weights}
            self._state.set_harmonic_active('absolute', True, selection, options)
        return result

    def harmonic_best_fit(self, selection=None, force_const: float = 0.0,
                          comparison: bool = False, mass_weighted: bool = False,
                          use_weights: bool = False, no_rotation: bool = False,
                          no_translation: bool = False,
                          q_mass=None, q_weight=None,
                          q_no_rot=None, q_no_trans=None) -> bool:
        """Apply best-fit harmonic positional restraints.

        Restrains atoms after optimal superposition to reference.
        Maps to CHARMM command: CONS HARMonic BESTfit ...

        Parameters
        ----------
        selection : SelectAtoms, optional
            Atoms to restrain. If None, all atoms are restrained.
        force_const : float, optional
            Force constant k (kcal/mol/A^2). Default 0.0.
        comparison : bool, optional
            If True, use comparison coordinate set. Default False.
        mass_weighted : bool, optional
            If True, multiply k by atomic mass. Default False.
        use_weights : bool, optional
            If True, use weight array for k(i). Default False.
        no_rotation : bool, optional
            If True, disable rotational component. Default False.
        no_translation : bool, optional
            If True, disable translational component. Default False.

        Returns
        -------
        bool
            True if successful
        """
        if q_mass is not None:
            warnings.warn(
                "q_mass is deprecated; use mass_weighted instead",
                DeprecationWarning,
                stacklevel=2
            )
            mass_weighted = bool(q_mass)
        if q_weight is not None:
            warnings.warn(
                "q_weight is deprecated; use use_weights instead",
                DeprecationWarning,
                stacklevel=2
            )
            use_weights = bool(q_weight)
        if q_no_rot is not None:
            warnings.warn(
                "q_no_rot is deprecated; use no_rotation instead",
                DeprecationWarning,
                stacklevel=2
            )
            no_rotation = bool(q_no_rot)
        if q_no_trans is not None:
            warnings.warn(
                "q_no_trans is deprecated; use no_translation instead",
                DeprecationWarning,
                stacklevel=2
            )
            no_translation = bool(q_no_trans)

        self._state.validate_can_add('HARMONIC_BEST_FIT', selection)
        result = _harmonic_setup_best_fit(
            selection, comparison=comparison,
            force_const=force_const,
            q_mass=int(mass_weighted), q_weight=int(use_weights),
            q_no_rot=int(no_rotation), q_no_trans=int(no_translation)
        )
        if result:
            options = {'force_const': force_const, 'comparison': comparison,
                      'no_rotation': no_rotation, 'no_translation': no_translation,
                      'mass_weighted': mass_weighted, 'use_weights': use_weights}
            self._state.set_harmonic_active('best_fit', True, selection, options)
        return result

    def harmonic_relative(self, selection1, selection2, force_const: float = 0.0,
                          comparison: bool = False, mass_weighted: bool = False,
                          use_weights: bool = False, no_rotation: bool = False,
                          no_translation: bool = False,
                          q_mass=None, q_weight=None,
                          q_no_rot=None, q_no_trans=None) -> bool:
        """Apply relative harmonic restraints between two selections.

        Restrains atoms in selection1 relative to atoms in selection2.
        Both selections must have the same number of atoms.
        Maps to CHARMM command: CONS HARMonic RELAtive ...

        Parameters
        ----------
        selection1 : SelectAtoms
            First selection of atoms
        selection2 : SelectAtoms
            Second selection of atoms (must match selection1 count)
        force_const : float, optional
            Force constant k (kcal/mol/A^2). Default 0.0.
        comparison : bool, optional
            If True, use comparison coordinate set. Default False.
        mass_weighted : bool, optional
            If True, multiply k by atomic mass. Default False.
        use_weights : bool, optional
            If True, use weight array for k(i). Default False.
        no_rotation : bool, optional
            If True, disable rotational component. Default False.
        no_translation : bool, optional
            If True, disable translational component. Default False.

        Returns
        -------
        bool
            True if successful

        Raises
        ------
        ValueError
            If selections have different numbers of atoms
        """
        if q_mass is not None:
            warnings.warn(
                "q_mass is deprecated; use mass_weighted instead",
                DeprecationWarning,
                stacklevel=2
            )
            mass_weighted = bool(q_mass)
        if q_weight is not None:
            warnings.warn(
                "q_weight is deprecated; use use_weights instead",
                DeprecationWarning,
                stacklevel=2
            )
            use_weights = bool(q_weight)
        if q_no_rot is not None:
            warnings.warn(
                "q_no_rot is deprecated; use no_rotation instead",
                DeprecationWarning,
                stacklevel=2
            )
            no_rotation = bool(q_no_rot)
        if q_no_trans is not None:
            warnings.warn(
                "q_no_trans is deprecated; use no_translation instead",
                DeprecationWarning,
                stacklevel=2
            )
            no_translation = bool(q_no_trans)

        self._state.validate_can_add('HARMONIC_RELATIVE', selection1)
        result = _harmonic_setup_relative(
            selection1, selection2, comparison=comparison,
            force_const=force_const,
            q_mass=int(mass_weighted), q_weight=int(use_weights),
            q_no_rot=int(no_rotation), q_no_trans=int(no_translation)
        )
        if result:
            options = {'force_const': force_const, 'comparison': comparison,
                      'no_rotation': no_rotation, 'no_translation': no_translation,
                      'mass_weighted': mass_weighted, 'use_weights': use_weights}
            self._state.set_harmonic_active('relative', True,
                                           f"{selection1}:{selection2}", options)
        return result

    def harmonic_turn_off(self) -> bool:
        """Turn off all harmonic restraints.

        Maps to CHARMM command: CONS HARMonic CLEAR

        Returns
        -------
        bool
            True if successful
        """
        result = _harmonic_turn_off()
        if result:
            for variant in ['absolute', 'best_fit', 'relative', 'pca']:
                self._state.set_harmonic_active(variant, False)
        return result


class _DistanceRestraints:
    """Namespace for distance restraints (NOE, PNOE, RESD).

    Provides methods for NOE distance restraints, Point NOE restraints,
    and restrained distances (reaction coordinates).

    Examples
    --------
    >>> import pycharmm.restraints as restraints
    >>> from pycharmm.select_atoms import SelectAtoms

    >>> # NOE via context manager
    >>> with restraints.distances.NOE(reset=True) as noe:
    ...     noe.assign(sel1, sel2, kmax=5.0, rmax=3.0)

    >>> # Direct NOE functions
    >>> restraints.distances.noe_reset()
    >>> restraints.distances.noe_scale(0.5)

    >>> # RESD for reaction coordinates
    >>> restraints.distances.resd([...], kval=2000.0, rval=-1.0)
    """

    def __init__(self, state: _RestraintState):
        self._state = state

    def NOE(self, reset: bool = False):
        """Get NOE context manager.

        Parameters
        ----------
        reset : bool, optional
            If True, reset all NOE restraints on entry. Default False.

        Returns
        -------
        NOE
            Context manager for NOE restraints

        Examples
        --------
        >>> with restraints.distances.NOE(reset=True) as noe:
        ...     noe.assign(sel1, sel2, kmax=5.0, rmax=3.0)
        """
        # Import NOE class (defined later in this module)
        return NOE(reset=reset)

    def noe_reset(self) -> None:
        """Reset all NOE restraints."""
        noe_reset()

    def noe_scale(self, factor: float) -> None:
        """Set NOE scale factor.

        Parameters
        ----------
        factor : float
            Scale factor for NOE energies/forces
        """
        noe_scale(factor)

    def resd(self, distances: list, kval: float = 0.0, rval: float = 0.0,
             power: int = 2, rpower: int = 2, **kwargs) -> None:
        """Add restrained distance (reaction coordinate).

        Parameters
        ----------
        distances : list
            List of distance definitions. Each element can be:
            - String format: "coef segid1 resid1 atom1 segid2 resid2 atom2"
            - Tuple format: (coef, (seg1, res1, atom1), (seg2, res2, atom2))
        kval : float
            Force constant (kcal/mol/A^power)
        rval : float
            Target/reference distance (A)
        power : int
            Exponent for energy function. Default 2.
        rpower : int
            Exponent for distance combination. Default 2.
        **kwargs
            Additional RESD options
        """
        self._state.validate_can_add('RESD')
        resd_add(distances, kval=kval, rval=rval, power=power,
                rpower=rpower, **kwargs)

    def resd_reset(self) -> None:
        """Reset all RESD restraints."""
        resd_reset()

    def resd_scale(self, factor: float) -> None:
        """Set RESD scale factor.

        Parameters
        ----------
        factor : float
            Scale factor for RESD energies/forces
        """
        resd_scale(factor)


class _AngleRestraints:
    """Namespace for angle/dihedral restraints.

    Provides methods for dihedral angle restraints.

    Examples
    --------
    >>> import pycharmm.restraints as restraints

    >>> # Dihedral restraint
    >>> restraints.angles.dihedral(selection='bynum 7 9 15 17',
    ...                            force=10.0, minimum=-60.0)

    >>> # Clear dihedral restraints
    >>> restraints.angles.dihedral_clear()
    """

    def __init__(self, state: _RestraintState):
        self._state = state

    def dihedral(self, selection: str = '', force: float = 0,
                 minimum: float = None, period: int = None,
                 width: float = None, comp: bool = False,
                 main: bool = True) -> None:
        """Apply dihedral angle restraints.

        Maps to CHARMM command: CONS DIHE ...

        Parameters
        ----------
        selection : str
            Atom selection for dihedral (4 atoms). Can be:
            - 'bynum int int int int' format
            - '4x(segid resid iupac)' format
        force : float
            Force constant (kcal/mol/rad^2)
        minimum : float, optional
            Target dihedral angle (degrees)
        period : int, optional
            Periodicity of the potential
        width : float, optional
            Width parameter for flat-bottom potential
        comp : bool, optional
            If True, use comparison coordinates. Default False.
        main : bool, optional
            If True, use main coordinates. Default True.
        """
        self._state.validate_can_add('DIHE')
        kwargs = {}
        if minimum is not None:
            kwargs['minimum'] = minimum
        if period is not None:
            kwargs['period'] = period
        if width is not None:
            kwargs['width'] = width
        if comp:
            kwargs['comp'] = comp
        if not main:
            kwargs['main'] = main

        _dihe_restraint(selection=selection, force=force, **kwargs)

        if selection:
            self._state.set_dihedral_active(True,
                                           {'selection': selection, 'force': force})

    def dihedral_clear(self) -> None:
        """Clear all dihedral restraints.

        Maps to CHARMM command: CONS CLDH
        """
        _dihe_restraint(clear=True)
        self._state.set_dihedral_active(False)


class _PositionRestraints:
    """Namespace for position-based restraints (DROPLET, PCA harmonic).

    Provides methods for droplet boundary potential and PCA-style
    harmonic restraints.

    Examples
    --------
    >>> import pycharmm.restraints as restraints

    >>> # Droplet boundary
    >>> restraints.positions.droplet(force=10.0, exponent=2)

    >>> # PCA harmonic
    >>> restraints.positions.harmonic_pca(selection=sel, force_const=5.0)
    """

    def __init__(self, state: _RestraintState):
        self._state = state

    def droplet(self, force: float = 0, exponent: int = 2,
                nomass: bool = False) -> None:
        """Apply quartic droplet boundary potential.

        Maps to CHARMM command: CONS DROPlet ...

        Parameters
        ----------
        force : float
            Force constant
        exponent : int
            Exponent for potential. Default 2.
        nomass : bool
            If True, disable mass weighting. Default False.
        """
        self._state.validate_can_add('DROPLET')
        kwargs = {'force': force, 'exponent': exponent}
        if nomass:
            kwargs['nomass'] = nomass
        _droplet_restraint(**kwargs)
        self._state.set_droplet_active(True, kwargs)

    def harmonic_pca(self, selection=None, force_const: float = 0.0,
                     expo: int = 2, x_scale: float = 1.0,
                     y_scale: float = 1.0, z_scale: float = 1.0,
                     comparison: bool = False, mass_weighted: bool = False,
                     use_weights: bool = False) -> bool:
        """Apply PCA-style harmonic restraints.

        Similar to absolute restraints but designed for PCA analysis.
        Maps to CHARMM command: CONS HARMonic PCA ...

        Parameters
        ----------
        selection : SelectAtoms, optional
            Atoms to restrain. If None, all atoms are restrained.
        force_const : float, optional
            Force constant k (kcal/mol/A^expo). Default 0.0.
        expo : int, optional
            Exponent on displacement. Default 2.
        x_scale, y_scale, z_scale : float, optional
            Scale factors for each component. Default 1.0.
        comparison : bool, optional
            If True, use comparison coordinate set. Default False.
        mass_weighted : bool, optional
            If True, multiply k by atomic mass. Default False.
        use_weights : bool, optional
            If True, use weight array for k(i). Default False.

        Returns
        -------
        bool
            True if successful
        """
        self._state.validate_can_add('HARMONIC_PCA', selection)
        result = _harmonic_setup_pca(
            selection, comparison=comparison,
            force_const=force_const, expo=expo,
            x_scale=x_scale, y_scale=y_scale, z_scale=z_scale,
            q_mass=int(mass_weighted), q_weight=int(use_weights)
        )
        if result:
            options = {'force_const': force_const, 'expo': expo}
            self._state.set_harmonic_active('pca', True, selection, options)
        return result


class _ICRestraints:
    """Namespace for internal coordinate restraints.

    Provides methods for IC-based bond, angle, dihedral, and improper
    restraints.

    Examples
    --------
    >>> import pycharmm.restraints as restraints

    >>> # Apply all IC restraints at once
    >>> restraints.internal_coords.all(bond=100.0, angle=50.0)

    >>> # Or individually
    >>> restraints.internal_coords.bond(force=100.0)
    >>> restraints.internal_coords.angle(force=50.0)
    """

    def __init__(self, state: _RestraintState):
        self._state = state

    def all(self, bond: float = None, angle: float = None,
            dihedral: float = None, improper: float = None,
            exponent: int = None, upper: bool = False) -> None:
        """Apply IC restraints for multiple coordinate types.

        Maps to CHARMM command: CONS IC ...

        Parameters
        ----------
        bond : float, optional
            Force constant for bond restraints
        angle : float, optional
            Force constant for angle restraints
        dihedral : float, optional
            Force constant for dihedral restraints
        improper : float, optional
            Force constant for improper restraints
        exponent : int, optional
            Exponent for energy function
        upper : bool, optional
            If True, use upper bound restraint. Default False.
        """
        self._state.validate_can_add('IC')
        kwargs = {}
        if bond is not None:
            kwargs['bond'] = bond
            self._state.set_ic_active('bond', True, bond)
        if angle is not None:
            kwargs['angle'] = angle
            self._state.set_ic_active('angle', True, angle)
        if dihedral is not None:
            kwargs['dihedral'] = dihedral
            self._state.set_ic_active('dihedral', True, dihedral)
        if improper is not None:
            kwargs['improper'] = improper
            self._state.set_ic_active('improper', True, improper)
        if exponent is not None:
            kwargs['exponent'] = exponent
        if upper:
            kwargs['upper'] = upper

        _ic_restraint(**kwargs)

    def bond(self, force: float, exponent: int = None,
             upper: bool = False) -> None:
        """Apply IC bond restraints.

        Parameters
        ----------
        force : float
            Force constant for bond restraints
        exponent : int, optional
            Exponent for energy function
        upper : bool, optional
            If True, use upper bound restraint. Default False.
        """
        self._state.validate_can_add('IC_BOND')
        kwargs = {'bond': force}
        if exponent is not None:
            kwargs['exponent'] = exponent
        if upper:
            kwargs['upper'] = upper
        _ic_restraint(**kwargs)
        self._state.set_ic_active('bond', True, force)

    def angle(self, force: float, exponent: int = None,
              upper: bool = False) -> None:
        """Apply IC angle restraints.

        Parameters
        ----------
        force : float
            Force constant for angle restraints
        exponent : int, optional
            Exponent for energy function
        upper : bool, optional
            If True, use upper bound restraint. Default False.
        """
        self._state.validate_can_add('IC_ANGLE')
        kwargs = {'angle': force}
        if exponent is not None:
            kwargs['exponent'] = exponent
        if upper:
            kwargs['upper'] = upper
        _ic_restraint(**kwargs)
        self._state.set_ic_active('angle', True, force)

    def dihedral(self, force: float, exponent: int = None,
                 upper: bool = False) -> None:
        """Apply IC dihedral restraints.

        Parameters
        ----------
        force : float
            Force constant for dihedral restraints
        exponent : int, optional
            Exponent for energy function
        upper : bool, optional
            If True, use upper bound restraint. Default False.
        """
        self._state.validate_can_add('IC_DIHEDRAL')
        kwargs = {'dihedral': force}
        if exponent is not None:
            kwargs['exponent'] = exponent
        if upper:
            kwargs['upper'] = upper
        _ic_restraint(**kwargs)
        self._state.set_ic_active('dihedral', True, force)

    def improper(self, force: float, exponent: int = None,
                 upper: bool = False) -> None:
        """Apply IC improper restraints.

        Parameters
        ----------
        force : float
            Force constant for improper restraints
        exponent : int, optional
            Exponent for energy function
        upper : bool, optional
            If True, use upper bound restraint. Default False.
        """
        self._state.validate_can_add('IC_IMPROPER')
        kwargs = {'improper': force}
        if exponent is not None:
            kwargs['exponent'] = exponent
        if upper:
            kwargs['upper'] = upper
        _ic_restraint(**kwargs)
        self._state.set_ic_active('improper', True, force)


# =============================================================================
# NOE Context Manager
# =============================================================================

class NOE:
    """Context manager for NOE restraint facility.

    Provides a clean interface for setting up NOE restraints with
    automatic entry/exit of the NOE command parser. Commands are
    buffered and executed as a single batch on exit.

    Backend Compatibility
    ---------------------
    - Standard CHARMM: Full support
    - DOMDEC: Full support
    - BLaDE: Single atom/atoms NOE and PNOE supported
    - OpenMM: NOT SUPPORTED

    Examples
    --------
    >>> import pycharmm.restraints as restraints
    >>> from pycharmm.select_atoms import SelectAtoms

    >>> # Using context manager
    >>> with restraints.NOE(reset=True) as noe:
    ...     noe.assign(sel1, sel2, kmax=5.0, rmax=3.0)
    ...     noe.assign_pnoe(sel, cnox=0, cnoy=0, cnoz=0, kmax=10.0, rmax=2.0)
    ...     noe.scale(2.0)

    Equivalent to CHARMM:
    ::

        NOE
          RESET
          ASSIGN KMAX 5.0 RMAX 3.0 SELE ... END SELE ... END
          ASSIGN CNOX 0 CNOY 0 CNOZ 0 KMAX 10.0 RMAX 2.0 SELE ... END
          SCALE 2.0
        END

    See Also
    --------
    noe_assign : Add distance restraint between two selections
    noe_assign_pnoe : Add point NOE restraint
    """

    def __init__(self, reset=False):
        """Initialize NOE context.

        Parameters
        ----------
        reset : bool, optional
            If True, clear all existing NOE restraints on entry.
            Default False.
        """
        self._reset_on_enter = reset

    def __enter__(self):
        """Enter NOE context - set up state tracking.

        Since direct memory API is used, commands are executed immediately
        without buffering. The context manager just tracks state.

        Raises
        ------
        RuntimeError
            If already inside an NOE context (nesting not allowed)
        """
        if _state._in_noe_context:
            raise RuntimeError("NOE context cannot be nested - already in NOE block")

        _state._in_noe_context = True
        _state.noe['active'] = True

        if self._reset_on_enter:
            noe_reset()

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit NOE context - finalize state.

        Since direct memory API is used, all commands were executed
        immediately. Just update state tracking.
        """
        _state._in_noe_context = False

        if exc_type is not None:
            # Exception occurred
            logger.warning(f"NOE context exited with exception: {exc_val}")
            return False  # Re-raise the exception

        # Normal exit - state is already synced via direct memory
        _state.noe['charmm_synced'] = True
        return False

    # Instance methods delegate to module functions
    def assign(self, selection1, selection2, **kwargs):
        """Add NOE distance restraint. See noe_assign() for details."""
        return noe_assign(selection1, selection2, **kwargs)

    def assign_pnoe(self, selection, cnox, cnoy, cnoz, **kwargs):
        """Add point NOE restraint. See noe_assign_pnoe() for details."""
        return noe_assign_pnoe(selection, cnox, cnoy, cnoz, **kwargs)

    def scale(self, factor):
        """Set scale factor. See noe_scale() for details."""
        return noe_scale(factor)

    def reset(self):
        """Clear all restraints. See noe_reset() for details."""
        return noe_reset()

    def print_analysis(self, cut=None):
        """Print restraints with analysis. See noe_print() for details."""
        return noe_print(analysis=True, cut=cut)


# =============================================================================
# Core NOE Functions
# =============================================================================

def _get_selection_indices(selection):
    """Get atom indices from a selection object, indices, or BYNUM string.

    Parameters
    ----------
    selection : SelectAtoms, int, list, or str
        Selection to convert to indices:
        - SelectAtoms: extracts indices from selection
        - int: single atom index (1-based)
        - list of int: multiple atom indices (1-based)
        - str: "bynum 100" or "bynum 100 101 102"

    Returns
    -------
    list or None
        List of 1-based atom indices, or None if conversion fails
    """
    import numpy as np
    from pycharmm.select_atoms import SelectAtoms

    # Direct index support - fastest path
    if isinstance(selection, (int, np.integer)):
        return [int(selection)]
    
    if isinstance(selection, (list, tuple)):
        # Check if it's a list of integers
        if all(isinstance(x, (int, np.integer)) for x in selection):
            return [int(x) for x in selection]
    
    if isinstance(selection, np.ndarray):
        if selection.dtype in (np.int32, np.int64, np.intp, int):
            return [int(x) for x in selection]

    if isinstance(selection, SelectAtoms):
        try:
            # SelectAtoms.get_atom_indexes() returns 0-based indices
            # Convert to 1-based for CHARMM
            if hasattr(selection, 'get_atom_indexes'):
                indices_0based = selection.get_atom_indexes()
                if indices_0based is not None and len(indices_0based) > 0:
                    return [i + 1 for i in indices_0based]
            # Fallback: use internal _selection array
            elif hasattr(selection, '_selection'):
                arr = selection._selection
                if arr is not None:
                    return [i + 1 for i, v in enumerate(arr) if v]
        except Exception:
            pass
    elif isinstance(selection, str):
        # Parse BYNUM strings: "bynum 100" or "bynum 100 101 102"
        sel_lower = selection.strip().lower()
        if sel_lower.startswith('bynum'):
            try:
                # Extract numbers after "bynum"
                parts = selection.split()
                if len(parts) >= 2:
                    indices = [int(x) for x in parts[1:]]
                    if indices:
                        return indices
            except (ValueError, IndexError):
                pass
    return None


def noe_assign(selection1, selection2, kmin=0.0, rmin=0.0, kmax=0.0,
               rmax=9999.0, fmax=9999.0, tcon=0.0, rexp=1.0,
               rswi=None, sexp=1.0, sumr=False, mindist=False):
    """Add NOE distance restraint between two atom selections.

    Maps to CHARMM command: ASSIGN ... atom-selection atom-selection

    The restraint energy follows a flat-bottom well potential::

        E(R) = 0.5*KMIN*(R-RMIN)^2     if R < RMIN
             = 0                        if RMIN <= R <= RMAX
             = 0.5*KMAX*(R-RMAX)^2     if R > RMAX (up to FMAX limit)

    Where R is the (optionally averaged) distance between selections.
    Uses direct API when available, falls back to script-based approach.

    Backend Compatibility
    ---------------------
    - Standard CHARMM: Full support
    - DOMDEC: Full support
    - BLaDE: Supported for single atom selections
    - OpenMM: NOT SUPPORTED

    Parameters
    ----------
    selection1 : SelectAtoms, str
        First atom selection (or center of mass for multi-atom)
    selection2 : SelectAtoms, str
        Second atom selection
    kmin : float, optional
        Force constant below rmin (kcal/mol/A^2). Default 0.0.
    rmin : float, optional
        Minimum distance (A). Default 0.0.
    kmax : float, optional
        Force constant above rmax (kcal/mol/A^2). Default 0.0.
    rmax : float, optional
        Maximum distance (A). Default 9999.0.
    fmax : float, optional
        Maximum force limit (kcal/mol/A). Default 9999.0.
    tcon : float, optional
        Time constant for R^-3 averaging (ps). 0=instantaneous. Default 0.0.
    rexp : float, optional
        Distance averaging exponent. Default 1.0 (simple average).
        Use 3.0 for NOE-style R^-3 averaging.
    rswi : float, optional
        Switch distance for soft asymptote. If specified, enables
        soft-square NOE potential beyond rmax+rswi.
    sexp : float, optional
        Soft asymptote exponent. Default 1.0.
    sumr : bool, optional
        Use sum-averaging mode (Sum_ij instead of average). Default False.
    mindist : bool, optional
        Use minimum distance between groups at each step. Default False.

    Returns
    -------
    int
        Index of the added restraint (1-based)

    Raises
    ------
    RuntimeError
        If not in NOE context and standalone execution fails

    Examples
    --------
    >>> import pycharmm.restraints as restraints
    >>> from pycharmm.select_atoms import SelectAtoms

    >>> sel1 = SelectAtoms(atom_type='CA', resid=10)
    >>> sel2 = SelectAtoms(atom_type='CA', resid=20)

    >>> # Simple distance restraint
    >>> with restraints.NOE(reset=True) as noe:
    ...     idx = noe.assign(sel1, sel2, kmax=10.0, rmax=5.0)

    >>> # NOE-style with R^-3 averaging
    >>> with restraints.NOE() as noe:
    ...     noe.assign(sel1, sel2, kmax=1.0, rmax=5.5, rexp=3.0)

    See Also
    --------
    noe_assign_pnoe : Point NOE restraints to fixed coordinates
    NOE : Context manager for batched restraint setup
    """
    # Always try direct API first (silent, no CHARMM output)
    try:
        if _init_noe_assign_bindings():
            idx1 = _get_selection_indices(selection1)
            idx2 = _get_selection_indices(selection2)
            if idx1 is not None and idx2 is not None:
                # Prepare ctypes arrays (ensure Python ints for ctypes)
                ni = len(idx1)
                nj = len(idx2)
                ilist = (ctypes.c_int * ni)(*[int(x) for x in idx1])
                jlist = (ctypes.c_int * nj)(*[int(x) for x in idx2])

                # Handle rswi - use -1.0 to indicate not set
                rswi_val = rswi if rswi is not None else -1.0

                result = lib.noedata_assign(
                    ctypes.c_int(ni), ilist, ctypes.c_int(nj), jlist,
                    ctypes.c_double(kmin), ctypes.c_double(rmin),
                    ctypes.c_double(kmax), ctypes.c_double(rmax),
                    ctypes.c_double(fmax), ctypes.c_double(tcon),
                    ctypes.c_double(rexp), ctypes.c_double(rswi_val),
                    ctypes.c_double(sexp), ctypes.c_int(1 if mindist else 0)
                )

                if result > 0:
                    # Track in Python state
                    params = {
                        'index': result,
                        'kmin': kmin, 'rmin': rmin,
                        'kmax': kmax, 'rmax': rmax, 'fmax': fmax,
                        'tcon': tcon, 'rexp': rexp,
                        'rswi': rswi, 'sexp': sexp,
                        'sumr': sumr, 'mindist': mindist,
                        'is_pnoe': False,
                        'selection1': str(selection1),
                        'selection2': str(selection2)
                    }
                    _state.noe_add_restraint(params)
                    return result
    except (AttributeError, OSError, RuntimeError):
        pass

    # Fall back to script-based approach (only if direct API unavailable)
    # Build selection strings
    sel1_str = _build_selection_string(selection1)
    sel2_str = _build_selection_string(selection2)

    # Build ASSIGN command
    cmd_parts = ['ASSIGN']

    # Parameters
    if kmin != 0.0:
        cmd_parts.append(f'KMIN {kmin}')
    if rmin != 0.0:
        cmd_parts.append(f'RMIN {rmin}')
    if kmax != 0.0:
        cmd_parts.append(f'KMAX {kmax}')
    if rmax != 9999.0:
        cmd_parts.append(f'RMAX {rmax}')
    if fmax != 9999.0:
        cmd_parts.append(f'FMAX {fmax}')
    if tcon != 0.0:
        cmd_parts.append(f'TCON {tcon}')
    if rexp != 1.0:
        cmd_parts.append(f'REXP {rexp}')
    if rswi is not None:
        cmd_parts.append(f'RSWI {rswi}')
        if sexp != 1.0:
            cmd_parts.append(f'SEXP {sexp}')
    if sumr:
        cmd_parts.append('SUMR')
    if mindist:
        cmd_parts.append('MINDIST')

    # Selections (must be at end)
    cmd_parts.append(sel1_str)
    cmd_parts.append(sel2_str)

    cmd = ' '.join(cmd_parts)
    _execute_noe_command(cmd)

    # Track in Python state
    params = {
        'index': _state.noe['count'] + 1,
        'kmin': kmin, 'rmin': rmin,
        'kmax': kmax, 'rmax': rmax, 'fmax': fmax,
        'tcon': tcon, 'rexp': rexp,
        'rswi': rswi, 'sexp': sexp,
        'sumr': sumr, 'mindist': mindist,
        'is_pnoe': False,
        'selection1': str(selection1),
        'selection2': str(selection2)
    }

    return _state.noe_add_restraint(params)


def noe_assign_pnoe(selection, cnox, cnoy, cnoz, kmin=0.0, rmin=0.0,
                    kmax=0.0, rmax=9999.0, fmax=9999.0, **kwargs):
    """Add point NOE restraint from atoms to fixed coordinates.

    Maps to CHARMM command: ASSIGN CNOX x CNOY y CNOZ z atom-selection

    Point NOE restraints (PNOE) apply distance restraints between selected
    atoms and a fixed point in space. Useful for:

    - Docking restraints
    - Loop refinement
    - Ligand positioning

    Uses direct API when available, falls back to script-based approach.

    Backend Compatibility
    ---------------------
    - Standard CHARMM: Full support
    - DOMDEC: Full support
    - BLaDE: SUPPORTED (tested in test_pnoe_blade.py)
    - OpenMM: NOT SUPPORTED

    Parameters
    ----------
    selection : SelectAtoms, str
        Atom selection to restrain
    cnox, cnoy, cnoz : float
        Point coordinates (x, y, z) in Angstroms
    kmin : float, optional
        Force constant below rmin (kcal/mol/A^2). Default 0.0.
    rmin : float, optional
        Minimum distance (A). Default 0.0.
    kmax : float, optional
        Force constant above rmax (kcal/mol/A^2). Default 0.0.
    rmax : float, optional
        Maximum distance (A). Default 9999.0.
    fmax : float, optional
        Maximum force limit (kcal/mol/A). Default 9999.0.
    **kwargs
        Additional parameters passed to noe_assign (tcon, rexp, etc.)

    Returns
    -------
    int
        Index of the added restraint (1-based)

    Examples
    --------
    >>> import pycharmm.restraints as restraints
    >>> from pycharmm.select_atoms import SelectAtoms

    >>> # Restrain ligand atom to binding site coordinate
    >>> sel = SelectAtoms(seg_id='LGND', atom_type='N1')
    >>> with restraints.NOE(reset=True) as noe:
    ...     noe.assign_pnoe(sel, cnox=10.5, cnoy=22.3, cnoz=15.0,
    ...                     kmax=10.0, rmax=1.5)

    See Also
    --------
    noe_assign : Standard two-selection distance restraints
    noe_mpnoe : Define moving point NOE
    """
    # Always try direct API first (silent, no CHARMM output)
    try:
        if _init_noe_assign_bindings():
            idx1 = _get_selection_indices(selection)
            if idx1 is not None:
                # Prepare ctypes array (ensure Python ints for ctypes)
                ni = len(idx1)
                ilist = (ctypes.c_int * ni)(*[int(x) for x in idx1])

                tcon = kwargs.get('tcon', 0.0)
                rexp = kwargs.get('rexp', 1.0)
                result = lib.noedata_assign_pnoe(
                    ctypes.c_int(ni), ilist,
                    ctypes.c_double(cnox), ctypes.c_double(cnoy),
                    ctypes.c_double(cnoz),
                    ctypes.c_double(kmin), ctypes.c_double(rmin),
                    ctypes.c_double(kmax), ctypes.c_double(rmax),
                    ctypes.c_double(fmax), ctypes.c_double(tcon),
                    ctypes.c_double(rexp)
                )

                if result > 0:
                    # Track in Python state
                    params = {
                        'index': result,
                        'kmin': kmin, 'rmin': rmin,
                        'kmax': kmax, 'rmax': rmax, 'fmax': fmax,
                        'tcon': kwargs.get('tcon', 0.0),
                        'rexp': kwargs.get('rexp', 1.0),
                        'rswi': kwargs.get('rswi'),
                        'sexp': kwargs.get('sexp', 1.0),
                        'is_pnoe': True,
                        'cnox': cnox, 'cnoy': cnoy, 'cnoz': cnoz,
                        'is_moving': False,
                        'selection1': str(selection)
                    }
                    _state.noe_add_restraint(params)
                    return result
    except (AttributeError, OSError, RuntimeError):
        pass

    # Fall back to script-based approach (only if direct API unavailable)
    # Build selection string
    sel_str = _build_selection_string(selection)

    # Build ASSIGN command with point coordinates
    cmd_parts = ['ASSIGN']

    # Point coordinates first
    cmd_parts.append(f'CNOX {cnox}')
    cmd_parts.append(f'CNOY {cnoy}')
    cmd_parts.append(f'CNOZ {cnoz}')

    # Force constant parameters
    if kmin != 0.0:
        cmd_parts.append(f'KMIN {kmin}')
    if rmin != 0.0:
        cmd_parts.append(f'RMIN {rmin}')
    if kmax != 0.0:
        cmd_parts.append(f'KMAX {kmax}')
    if rmax != 9999.0:
        cmd_parts.append(f'RMAX {rmax}')
    if fmax != 9999.0:
        cmd_parts.append(f'FMAX {fmax}')

    # Optional parameters from kwargs
    if kwargs.get('tcon', 0.0) != 0.0:
        cmd_parts.append(f'TCON {kwargs["tcon"]}')
    if kwargs.get('rexp', 1.0) != 1.0:
        cmd_parts.append(f'REXP {kwargs["rexp"]}')
    if kwargs.get('rswi') is not None:
        cmd_parts.append(f'RSWI {kwargs["rswi"]}')
        if kwargs.get('sexp', 1.0) != 1.0:
            cmd_parts.append(f'SEXP {kwargs["sexp"]}')

    # Selection (at end for PNOE)
    cmd_parts.append(sel_str)

    cmd = ' '.join(cmd_parts)
    _execute_noe_command(cmd)

    # Track in Python state
    params = {
        'index': _state.noe['count'] + 1,
        'kmin': kmin, 'rmin': rmin,
        'kmax': kmax, 'rmax': rmax, 'fmax': fmax,
        'tcon': kwargs.get('tcon', 0.0),
        'rexp': kwargs.get('rexp', 1.0),
        'rswi': kwargs.get('rswi'),
        'sexp': kwargs.get('sexp', 1.0),
        'is_pnoe': True,
        'cnox': cnox, 'cnoy': cnoy, 'cnoz': cnoz,
        'is_moving': False,
        'selection1': str(selection)
    }

    return _state.noe_add_restraint(params)


def noe_reset():
    """Reset all NOE restraints.

    Maps to CHARMM command: RESET

    Clears all existing NOE restraints and resets scale factor to 1.0.
    Uses direct API when available, falls back to script-based approach.

    Examples
    --------
    >>> with restraints.NOE() as noe:
    ...     noe.reset()  # Clear all existing
    ...     noe.assign(...)  # Add new
    """
    # Try direct API first
    try:
        if _init_noe_assign_bindings():
            result = lib.noedata_reset()
            if result >= 0:
                _state.noe_clear()
                return
    except (AttributeError, OSError, RuntimeError):
        pass

    # Fall back to script-based approach
    _execute_noe_command('RESET')
    _state.noe_clear()


def noe_scale(factor):
    """Set scale factor for NOE energy and forces.

    Maps to CHARMM command: SCALE real

    Parameters
    ----------
    factor : float
        Scale factor (default 1.0). All NOE energies and forces
        are multiplied by this value.

    Examples
    --------
    >>> with restraints.NOE() as noe:
    ...     noe.assign(sel1, sel2, kmax=5.0, rmax=3.0)
    ...     noe.scale(2.0)  # Double all NOE energies
    """
    _execute_noe_command(f'SCALE {factor}')
    _state.noe['scale'] = factor


def noe_print(analysis=False, cut=None):
    """Print NOE restraints to output.

    Maps to CHARMM command: PRINT [ANAL [CUT real]]

    Parameters
    ----------
    analysis : bool, optional
        If True, include distance and energy analysis. Default False.
    cut : float, optional
        Only list restraints exceeding RMAX by more than this value.
        Only used when analysis=True.

    Examples
    --------
    >>> with restraints.NOE() as noe:
    ...     noe.assign(sel1, sel2, kmax=5.0, rmax=3.0)
    ...     noe.print_analysis(cut=0.5)  # List violations > 0.5 A
    """
    if analysis:
        if cut is not None:
            _execute_noe_command(f'PRINT ANAL CUT {cut}')
        else:
            _execute_noe_command('PRINT ANAL')
    else:
        _execute_noe_command('PRINT')


def noe_read(unit):
    """Read NOE restraints from file.

    Maps to CHARMM command: READ UNIT int

    Parameters
    ----------
    unit : int
        Fortran unit number for input file
    """
    _execute_noe_command(f'READ UNIT {unit}')
    _state.noe['charmm_synced'] = False


def noe_write(unit, analysis=False):
    """Write NOE restraints to file.

    Maps to CHARMM command: WRITE UNIT int [ANAL]

    Parameters
    ----------
    unit : int
        Fortran unit number for output file
    analysis : bool, optional
        If True, include distance/energy analysis. Default False.
    """
    if analysis:
        _execute_noe_command(f'WRITE UNIT {unit} ANAL')
    else:
        _execute_noe_command(f'WRITE UNIT {unit}')


# =============================================================================
# Batch NOE Functions (DEPRECATED - will be removed in future release)
# =============================================================================

def noe_assign_batch(restraints_list, reset=False, use_script=False):
    """Add multiple NOE restraints efficiently in a single batch.

    .. deprecated::
        This function is deprecated and will be removed in a future release.
        Use noe_assign() within a NOE context manager instead.

    This function is optimized for adding many restraints at once,
    avoiding the per-call overhead of individual noe_assign() calls.

    Parameters
    ----------
    restraints_list : list of dict
        Each dict should contain:
        - 'selection1': SelectAtoms or str - first atom selection
        - 'selection2': SelectAtoms or str - second atom selection
        - 'kmin', 'rmin', 'kmax', 'rmax', 'fmax', 'tcon', 'rexp', etc. (optional)
    reset : bool, optional
        If True, clear existing restraints before adding. Default False.
    use_script : bool, optional
        If True, force script-based approach (slower but more compatible).
        If False (default), use direct API when available.

    Returns
    -------
    list of int
        Indices of added restraints (1-based)

    Examples
    --------
    >>> import pycharmm.restraints as restraints
    >>> from pycharmm.select_atoms import SelectAtoms

    >>> # Define many restraints
    >>> restraints_data = [
    ...     {'selection1': SelectAtoms(seg='A', resid=10, atom_type='CA'),
    ...      'selection2': SelectAtoms(seg='A', resid=20, atom_type='CA'),
    ...      'kmax': 5.0, 'rmax': 8.0},
    ...     {'selection1': SelectAtoms(seg='A', resid=15, atom_type='CA'),
    ...      'selection2': SelectAtoms(seg='A', resid=25, atom_type='CA'),
    ...      'kmax': 5.0, 'rmax': 10.0},
    ...     # ... hundreds more
    ... ]

    >>> # Add all at once - much faster than individual calls
    >>> indices = restraints.noe_assign_batch(restraints_data, reset=True)

    Notes
    -----
    For best performance:
    - Pre-create SelectAtoms objects before calling
    - Use atom indices directly if possible (faster than string selections)
    - Batch sizes of 100-1000 restraints work well
    """
    warnings.warn(
        "noe_assign_batch() is deprecated and will be removed in a future release. "
        "Use noe_assign() within a NOE context manager instead.",
        DeprecationWarning,
        stacklevel=2
    )
    if not restraints_list:
        return []

    indices = []

    if reset:
        noe_reset()

    # Try batch API if available and not forcing script
    if not use_script:
        try:
            if _init_noe_assign_bindings():
                # Pre-extract all indices
                for i, r in enumerate(restraints_list):
                    idx1 = _get_selection_indices(r['selection1'])
                    idx2 = _get_selection_indices(r['selection2'])

                    if idx1 is not None and idx2 is not None:
                        ni = len(idx1)
                        nj = len(idx2)
                        ilist = (ctypes.c_int * ni)(*idx1)
                        jlist = (ctypes.c_int * nj)(*idx2)

                        kmin = r.get('kmin', 0.0)
                        rmin = r.get('rmin', 0.0)
                        kmax = r.get('kmax', 0.0)
                        rmax = r.get('rmax', 9999.0)
                        fmax = r.get('fmax', 9999.0)
                        tcon = r.get('tcon', 0.0)
                        rexp = r.get('rexp', 1.0)
                        rswi = r.get('rswi')
                        rswi_val = rswi if rswi is not None else -1.0
                        sexp = r.get('sexp', 1.0)
                        mindist = r.get('mindist', False)

                        result = lib.noedata_assign(
                            ni, ilist, nj, jlist,
                            ctypes.c_double(kmin), ctypes.c_double(rmin),
                            ctypes.c_double(kmax), ctypes.c_double(rmax),
                            ctypes.c_double(fmax), ctypes.c_double(tcon),
                            ctypes.c_double(rexp), ctypes.c_double(rswi_val),
                            ctypes.c_double(sexp), ctypes.c_int(1 if mindist else 0)
                        )

                        if result > 0:
                            indices.append(result)
                            # Minimal state tracking for batch
                            _state.noe['count'] += 1
                        else:
                            indices.append(-1)
                    else:
                        indices.append(-1)

                if indices and all(i > 0 for i in indices):
                    return indices
        except (AttributeError, OSError, RuntimeError):
            pass

    # Fall back to optimized script-based approach
    # Build all commands at once, execute in single batch
    cmd_lines = ['NOE']

    for r in restraints_list:
        sel1_str = _build_selection_string(r['selection1'])
        sel2_str = _build_selection_string(r['selection2'])

        cmd_parts = ['ASSIGN']

        kmin = r.get('kmin', 0.0)
        if kmin != 0.0:
            cmd_parts.append(f'KMIN {kmin}')

        rmin = r.get('rmin', 0.0)
        if rmin != 0.0:
            cmd_parts.append(f'RMIN {rmin}')

        kmax = r.get('kmax', 0.0)
        if kmax != 0.0:
            cmd_parts.append(f'KMAX {kmax}')

        rmax = r.get('rmax', 9999.0)
        if rmax != 9999.0:
            cmd_parts.append(f'RMAX {rmax}')

        fmax = r.get('fmax', 9999.0)
        if fmax != 9999.0:
            cmd_parts.append(f'FMAX {fmax}')

        tcon = r.get('tcon', 0.0)
        if tcon != 0.0:
            cmd_parts.append(f'TCON {tcon}')

        rexp = r.get('rexp', 1.0)
        if rexp != 1.0:
            cmd_parts.append(f'REXP {rexp}')

        rswi = r.get('rswi')
        if rswi is not None:
            cmd_parts.append(f'RSWI {rswi}')
            sexp = r.get('sexp', 1.0)
            if sexp != 1.0:
                cmd_parts.append(f'SEXP {sexp}')

        if r.get('sumr', False):
            cmd_parts.append('SUMR')

        if r.get('mindist', False):
            cmd_parts.append('MINDIST')

        cmd_parts.append(sel1_str)
        cmd_parts.append(sel2_str)

        cmd_lines.append(' '.join(cmd_parts))

    cmd_lines.append('END')

    # Execute all at once
    full_cmd = '\n'.join(cmd_lines)
    lingo.charmm_script(full_cmd)

    # Update state
    n_added = len(restraints_list)
    base_idx = _state.noe['count'] + 1
    indices = list(range(base_idx, base_idx + n_added))
    _state.noe['count'] += n_added

    return indices


def noe_assign_pnoe_batch(restraints_list, reset=False, use_script=False):
    """Add multiple point NOE (PNOE) restraints efficiently in a single batch.

    .. deprecated::
        This function is deprecated and will be removed in a future release.
        Use noe_assign_pnoe() within a NOE context manager instead.

    This function is optimized for adding many PNOE restraints at once.

    Parameters
    ----------
    restraints_list : list of dict
        Each dict should contain:
        - 'selection': SelectAtoms or str - atom selection
        - 'cnox', 'cnoy', 'cnoz': float - target point coordinates
        - 'kmin', 'rmin', 'kmax', 'rmax', 'fmax', etc. (optional)
    reset : bool, optional
        If True, clear existing restraints before adding. Default False.
    use_script : bool, optional
        If True, force script-based approach. Default False.

    Returns
    -------
    list of int
        Indices of added restraints (1-based)

    Examples
    --------
    >>> import pycharmm.restraints as restraints

    >>> # Define PNOE restraints for water molecules to grid points
    >>> pnoe_data = [
    ...     {'selection': f'BYNUM {atom_idx}',
    ...      'cnox': x, 'cnoy': y, 'cnoz': z,
    ...      'kmin': 300, 'rmin': 5, 'rmax': 5}
    ...     for atom_idx, (x, y, z) in enumerate(grid_points, start=1)
    ... ]

    >>> # Add all at once
    >>> indices = restraints.noe_assign_pnoe_batch(pnoe_data, reset=True)
    """
    warnings.warn(
        "noe_assign_pnoe_batch() is deprecated and will be removed in a future release. "
        "Use noe_assign_pnoe() within a NOE context manager instead.",
        DeprecationWarning,
        stacklevel=2
    )
    if not restraints_list:
        return []

    indices = []

    if reset:
        noe_reset()

    # Try batch API if available
    if not use_script:
        try:
            if _init_noe_assign_bindings():
                for r in restraints_list:
                    idx = _get_selection_indices(r['selection'])

                    if idx is not None:
                        ni = len(idx)
                        ilist = (ctypes.c_int * ni)(*idx)

                        cnox = r['cnox']
                        cnoy = r['cnoy']
                        cnoz = r['cnoz']
                        kmin = r.get('kmin', 0.0)
                        rmin = r.get('rmin', 0.0)
                        kmax = r.get('kmax', 0.0)
                        rmax = r.get('rmax', 9999.0)
                        fmax = r.get('fmax', 9999.0)
                        tcon = r.get('tcon', 0.0)
                        rexp = r.get('rexp', 1.0)

                        result = lib.noedata_assign_pnoe(
                            ni, ilist,
                            ctypes.c_double(cnox), ctypes.c_double(cnoy),
                            ctypes.c_double(cnoz),
                            ctypes.c_double(kmin), ctypes.c_double(rmin),
                            ctypes.c_double(kmax), ctypes.c_double(rmax),
                            ctypes.c_double(fmax), ctypes.c_double(tcon),
                            ctypes.c_double(rexp)
                        )

                        if result > 0:
                            indices.append(result)
                            _state.noe['count'] += 1
                        else:
                            indices.append(-1)
                    else:
                        indices.append(-1)

                if indices and all(i > 0 for i in indices):
                    return indices
        except (AttributeError, OSError, RuntimeError):
            pass

    # Fall back to optimized script-based approach
    cmd_lines = ['NOE']

    for r in restraints_list:
        sel_str = _build_selection_string(r['selection'])

        cmd_parts = ['ASSIGN']
        cmd_parts.append(f"CNOX {r['cnox']}")
        cmd_parts.append(f"CNOY {r['cnoy']}")
        cmd_parts.append(f"CNOZ {r['cnoz']}")

        kmin = r.get('kmin', 0.0)
        if kmin != 0.0:
            cmd_parts.append(f'KMIN {kmin}')

        rmin = r.get('rmin', 0.0)
        if rmin != 0.0:
            cmd_parts.append(f'RMIN {rmin}')

        kmax = r.get('kmax', 0.0)
        if kmax != 0.0:
            cmd_parts.append(f'KMAX {kmax}')

        rmax = r.get('rmax', 9999.0)
        if rmax != 9999.0:
            cmd_parts.append(f'RMAX {rmax}')

        fmax = r.get('fmax', 9999.0)
        if fmax != 9999.0:
            cmd_parts.append(f'FMAX {fmax}')

        cmd_parts.append(sel_str)
        cmd_lines.append(' '.join(cmd_parts))

    cmd_lines.append('END')

    # Execute all at once
    full_cmd = '\n'.join(cmd_lines)
    lingo.charmm_script(full_cmd)

    # Update state
    n_added = len(restraints_list)
    base_idx = _state.noe['count'] + 1
    indices = list(range(base_idx, base_idx + n_added))
    _state.noe['count'] += n_added

    return indices


def noe_assign_from_indices_batch(index_pairs, params_list=None,
                                  default_kmax=0.0, default_rmax=9999.0,
                                  reset=False):
    """Add NOE restraints using pre-computed atom indices for maximum speed.

    .. deprecated::
        This function is deprecated and will be removed in a future release.
        Use noe_assign() within a NOE context manager instead.

    This is the fastest method when you already have atom indices.
    Skips selection parsing entirely.

    Parameters
    ----------
    index_pairs : list of tuple
        List of (idx1, idx2) where each is either:
        - int: single atom index (1-based)
        - list of int: multiple atom indices (1-based)
    params_list : list of dict, optional
        Parameters for each restraint. If None, uses defaults.
    default_kmax : float, optional
        Default force constant. Default 0.0.
    default_rmax : float, optional
        Default max distance. Default 9999.0.
    reset : bool, optional
        If True, clear existing restraints first. Default False.

    Returns
    -------
    list of int
        Indices of added restraints

    Examples
    --------
    >>> # Pre-computed atom pairs (1-based indices)
    >>> pairs = [(100, 200), (150, 250), (175, 275)]
    >>> params = [{'kmax': 5.0, 'rmax': 8.0}] * len(pairs)
    >>> indices = restraints.noe_assign_from_indices_batch(pairs, params, reset=True)
    """
    warnings.warn(
        "noe_assign_from_indices_batch() is deprecated and will be removed in a future release. "
        "Use noe_assign() within a NOE context manager instead.",
        DeprecationWarning,
        stacklevel=2
    )
    if not index_pairs:
        return []

    if reset:
        noe_reset()

    indices = []

    # Try direct API
    import numpy as np
    try:
        if _init_noe_assign_bindings():
            for i, (idx1, idx2) in enumerate(index_pairs):
                # Convert single indices to lists (handle numpy integer types)
                if isinstance(idx1, (int, np.integer)):
                    idx1 = [idx1]
                elif idx1 is None:
                    raise ValueError(f"NOE pair {i}: idx1 is None")
                if isinstance(idx2, (int, np.integer)):
                    idx2 = [idx2]
                elif idx2 is None:
                    raise ValueError(f"NOE pair {i}: idx2 is None")

                ni = len(idx1)
                nj = len(idx2)

                # Validate both selections have atoms
                if ni == 0:
                    raise ValueError(f"NOE pair {i}: first selection has 0 atoms")
                if nj == 0:
                    raise ValueError(f"NOE pair {i}: second selection has 0 atoms")

                # Ensure Python ints for ctypes (numpy integers can cause issues)
                ilist = (ctypes.c_int * ni)(*[int(x) for x in idx1])
                jlist = (ctypes.c_int * nj)(*[int(x) for x in idx2])

                # Get parameters
                if params_list and i < len(params_list):
                    p = params_list[i]
                    kmin = p.get('kmin', 0.0)
                    rmin = p.get('rmin', 0.0)
                    kmax = p.get('kmax', default_kmax)
                    rmax = p.get('rmax', default_rmax)
                    fmax = p.get('fmax', 9999.0)
                    tcon = p.get('tcon', 0.0)
                    rexp = p.get('rexp', 1.0)
                    rswi = p.get('rswi')
                    rswi_val = rswi if rswi is not None else -1.0
                    sexp = p.get('sexp', 1.0)
                    mindist = p.get('mindist', False)
                else:
                    kmin, rmin = 0.0, 0.0
                    kmax, rmax = default_kmax, default_rmax
                    fmax, tcon, rexp = 9999.0, 0.0, 1.0
                    rswi_val, sexp = -1.0, 1.0
                    mindist = False

                result = lib.noedata_assign(
                    ctypes.c_int(ni), ilist, ctypes.c_int(nj), jlist,
                    ctypes.c_double(kmin), ctypes.c_double(rmin),
                    ctypes.c_double(kmax), ctypes.c_double(rmax),
                    ctypes.c_double(fmax), ctypes.c_double(tcon),
                    ctypes.c_double(rexp), ctypes.c_double(rswi_val),
                    ctypes.c_double(sexp), ctypes.c_int(1 if mindist else 0)
                )

                if result > 0:
                    indices.append(result)
                    _state.noe['count'] += 1
                else:
                    indices.append(-1)

            return indices
    except (AttributeError, OSError, RuntimeError):
        pass

    # Fall back to script (slower)
    cmd_lines = ['NOE']

    import numpy as np
    for i, (idx1, idx2) in enumerate(index_pairs):
        # Handle single integers (including numpy integer types)
        if isinstance(idx1, (int, np.integer)):
            idx1 = [idx1]
        elif idx1 is None:
            raise ValueError(f"NOE pair {i}: idx1 is None")
        if isinstance(idx2, (int, np.integer)):
            idx2 = [idx2]
        elif idx2 is None:
            raise ValueError(f"NOE pair {i}: idx2 is None")

        # Validate both selections have atoms
        if len(idx1) == 0:
            raise ValueError(f"NOE pair {i}: first selection has 0 atoms")
        if len(idx2) == 0:
            raise ValueError(f"NOE pair {i}: second selection has 0 atoms")

        sel1_str = f"SELE BYNUM {' '.join(str(x) for x in idx1)} END"
        sel2_str = f"SELE BYNUM {' '.join(str(x) for x in idx2)} END"

        cmd_parts = ['ASSIGN']

        if params_list and i < len(params_list):
            p = params_list[i]
            if p.get('kmin', 0.0) != 0.0:
                cmd_parts.append(f"KMIN {p['kmin']}")
            if p.get('rmin', 0.0) != 0.0:
                cmd_parts.append(f"RMIN {p['rmin']}")
            kmax = p.get('kmax', default_kmax)
            rmax = p.get('rmax', default_rmax)
        else:
            kmax, rmax = default_kmax, default_rmax

        if kmax != 0.0:
            cmd_parts.append(f'KMAX {kmax}')
        if rmax != 9999.0:
            cmd_parts.append(f'RMAX {rmax}')

        cmd_parts.append(sel1_str)
        cmd_parts.append(sel2_str)
        cmd_lines.append(' '.join(cmd_parts))

    cmd_lines.append('END')
    lingo.charmm_script('\n'.join(cmd_lines))

    n_added = len(index_pairs)
    base_idx = _state.noe['count'] + 1
    indices = list(range(base_idx, base_idx + n_added))
    _state.noe['count'] += n_added

    return indices


def noe_assign_pnoe_from_indices_batch(atom_indices, coordinates, params_list=None,
                                        default_kmin=0.0, default_rmin=0.0,
                                        default_kmax=0.0, default_rmax=9999.0,
                                        reset=False):
    """Add PNOE restraints using pre-computed atom indices for maximum speed.

    .. deprecated::
        This function is deprecated and will be removed in a future release.
        Use noe_assign_pnoe() within a NOE context manager instead.

    This is the fastest method for PNOE when you have atom indices.

    Parameters
    ----------
    atom_indices : list of int
        1-based atom indices for each restraint
    coordinates : list of tuple
        (x, y, z) target coordinates for each restraint
    params_list : list of dict, optional
        Parameters for each restraint. If None, uses defaults.
    default_kmin, default_rmin, default_kmax, default_rmax : float
        Default force parameters
    reset : bool, optional
        If True, clear existing restraints first. Default False.

    Returns
    -------
    list of int
        Indices of added restraints

    Examples
    --------
    >>> # Pre-computed: water oxygens → grid points
    >>> atom_indices = [100, 200, 300, 400]  # 1-based
    >>> coordinates = [(0.0, 0.0, 0.0), (5.0, 0.0, 0.0), ...]
    >>> indices = restraints.noe_assign_pnoe_from_indices_batch(
    ...     atom_indices, coordinates,
    ...     default_kmin=300, default_rmin=5, default_rmax=5,
    ...     reset=True
    ... )
    """
    warnings.warn(
        "noe_assign_pnoe_from_indices_batch() is deprecated and will be removed in a future release. "
        "Use noe_assign_pnoe() within a NOE context manager instead.",
        DeprecationWarning,
        stacklevel=2
    )
    if not atom_indices or not coordinates:
        return []

    if len(atom_indices) != len(coordinates):
        raise ValueError("atom_indices and coordinates must have same length")

    if reset:
        noe_reset()

    indices = []

    # Try direct API
    try:
        if _init_noe_assign_bindings():
            for i, (atom_idx, (cnox, cnoy, cnoz)) in enumerate(zip(atom_indices, coordinates)):
                # Ensure Python int for ctypes (numpy integers can cause issues)
                ilist = (ctypes.c_int * 1)(int(atom_idx))

                if params_list and i < len(params_list):
                    p = params_list[i]
                    kmin = p.get('kmin', default_kmin)
                    rmin = p.get('rmin', default_rmin)
                    kmax = p.get('kmax', default_kmax)
                    rmax = p.get('rmax', default_rmax)
                    fmax = p.get('fmax', 9999.0)
                    tcon = p.get('tcon', 0.0)
                    rexp = p.get('rexp', 1.0)
                else:
                    kmin, rmin = default_kmin, default_rmin
                    kmax, rmax = default_kmax, default_rmax
                    fmax, tcon, rexp = 9999.0, 0.0, 1.0

                result = lib.noedata_assign_pnoe(
                    ctypes.c_int(1), ilist,
                    ctypes.c_double(cnox), ctypes.c_double(cnoy),
                    ctypes.c_double(cnoz),
                    ctypes.c_double(kmin), ctypes.c_double(rmin),
                    ctypes.c_double(kmax), ctypes.c_double(rmax),
                    ctypes.c_double(fmax), ctypes.c_double(tcon),
                    ctypes.c_double(rexp)
                )

                if result > 0:
                    indices.append(result)
                    _state.noe['count'] += 1
                else:
                    indices.append(-1)

            return indices
    except (AttributeError, OSError, RuntimeError):
        pass

    # Fall back to optimized script
    cmd_lines = ['NOE']

    for i, (atom_idx, (cnox, cnoy, cnoz)) in enumerate(zip(atom_indices, coordinates)):
        cmd_parts = ['ASSIGN']
        cmd_parts.append(f'CNOX {cnox}')
        cmd_parts.append(f'CNOY {cnoy}')
        cmd_parts.append(f'CNOZ {cnoz}')

        if params_list and i < len(params_list):
            p = params_list[i]
            kmin = p.get('kmin', default_kmin)
            rmin = p.get('rmin', default_rmin)
            kmax = p.get('kmax', default_kmax)
            rmax = p.get('rmax', default_rmax)
        else:
            kmin, rmin = default_kmin, default_rmin
            kmax, rmax = default_kmax, default_rmax

        if kmin != 0.0:
            cmd_parts.append(f'KMIN {kmin}')
        if rmin != 0.0:
            cmd_parts.append(f'RMIN {rmin}')
        if kmax != 0.0:
            cmd_parts.append(f'KMAX {kmax}')
        if rmax != 9999.0:
            cmd_parts.append(f'RMAX {rmax}')

        cmd_parts.append(f'SELE BYNUM {atom_idx} END')
        cmd_lines.append(' '.join(cmd_parts))

    cmd_lines.append('END')
    lingo.charmm_script('\n'.join(cmd_lines))

    n_added = len(atom_indices)
    base_idx = _state.noe['count'] + 1
    indices = list(range(base_idx, base_idx + n_added))
    _state.noe['count'] += n_added

    return indices


# =============================================================================
# Moving Point NOE Functions
# =============================================================================

def noe_mpnoe(inoe, tnox, tnoy, tnoz):
    """Define moving point NOE target position.

    Maps to CHARMM command: MPNOE INOE int TNOX x TNOY y TNOZ z

    Converts a point NOE to a moving point NOE that gradually moves
    from its initial position (CNOX, CNOY, CNOZ) to the target position
    (TNOX, TNOY, TNOZ) over NMPNOE steps.

    Uses direct API when available, falls back to script-based approach.

    Parameters
    ----------
    inoe : int
        Index of the NOE restraint to convert (1-based)
    tnox, tnoy, tnoz : float
        Target position coordinates (Angstroms)

    See Also
    --------
    noe_nmpnoe : Set number of steps for moving PNOE
    noe_assign_pnoe : Create initial point NOE

    Examples
    --------
    >>> with restraints.NOE(reset=True) as noe:
    ...     # Create PNOE at initial position
    ...     idx = noe.assign_pnoe(sel, cnox=0, cnoy=0, cnoz=0, kmax=10.0, rmax=2.0)
    ...     # Set target position
    ...     restraints.noe_mpnoe(idx, tnox=10.0, tnoy=10.0, tnoz=10.0)
    ...     # Set number of steps
    ...     restraints.noe_nmpnoe(10000)
    """
    # Try direct API first (only when not in NOE context)
    if not _state._in_noe_context:
        try:
            if _init_noe_assign_bindings():
                result = lib.noedata_set_mpnoe_target(
                    inoe,
                    ctypes.c_double(tnox), ctypes.c_double(tnoy),
                    ctypes.c_double(tnoz)
                )
                if result == 1:
                    # Update state if tracking this restraint
                    if inoe <= len(_state.noe['restraints']):
                        _state.noe['restraints'][inoe - 1]['is_moving'] = True
                        _state.noe['restraints'][inoe - 1]['tnox'] = tnox
                        _state.noe['restraints'][inoe - 1]['tnoy'] = tnoy
                        _state.noe['restraints'][inoe - 1]['tnoz'] = tnoz
                    return
        except (AttributeError, OSError, RuntimeError):
            pass

    # Fall back to script-based approach
    cmd = f'MPNOE INOE {inoe} TNOX {tnox} TNOY {tnoy} TNOZ {tnoz}'
    _execute_noe_command(cmd)

    # Update state if tracking this restraint
    if inoe <= len(_state.noe['restraints']):
        _state.noe['restraints'][inoe - 1]['is_moving'] = True
        _state.noe['restraints'][inoe - 1]['tnox'] = tnox
        _state.noe['restraints'][inoe - 1]['tnoy'] = tnoy
        _state.noe['restraints'][inoe - 1]['tnoz'] = tnoz


def noe_nmpnoe(nsteps):
    """Set number of steps for moving point NOE.

    Maps to CHARMM command: NMPNOE int

    Uses direct API when available, falls back to script-based approach.

    Parameters
    ----------
    nsteps : int
        Number of dynamics steps over which point NOEs move from
        initial (CNOX, CNOY, CNOZ) to target (TNOX, TNOY, TNOZ) positions.

    See Also
    --------
    noe_mpnoe : Define target position for moving PNOE
    """
    # Try direct API first (only when not in NOE context)
    if not _state._in_noe_context:
        try:
            if _init_noe_assign_bindings():
                lib.noedata_set_nmpnoe(nsteps)
                return
        except (AttributeError, OSError, RuntimeError):
            pass

    # Fall back to script-based approach
    _execute_noe_command(f'NMPNOE {nsteps}')


def noe_temperature(temp):
    """Set temperature for old-format NOE variance conversion.

    Maps to CHARMM command: TEMPERATURE real

    Parameters
    ----------
    temp : float
        Temperature in Kelvin
    """
    _execute_noe_command(f'TEMPERATURE {temp}')


# =============================================================================
# State Query Functions
# =============================================================================

def noe_get_count(direct=False):
    """Get number of NOE restraints.

    Parameters
    ----------
    direct : bool, optional
        If True, query CHARMM directly (requires api_noe).
        If False (default), return Python state count.

    Returns
    -------
    int
        Number of NOE restraints
    """
    if direct:
        count = _noe_get_count_direct()
        if count is not None:
            return count
    return _state.noe['count']


def noe_get_state():
    """Get current NOE state dictionary.

    Returns
    -------
    dict
        Copy of NOE state including count, scale, and restraint list
    """
    return _state.noe.copy()


def noe_is_active():
    """Check if currently inside NOE context.

    Returns
    -------
    bool
        True if in NOE context manager
    """
    return _state._in_noe_context


def noe_get_restraints():
    """Get list of tracked NOE restraints.

    Returns
    -------
    list
        List of restraint parameter dictionaries
    """
    return [r.copy() for r in _state.noe['restraints']]


def noe_get_scale():
    """Get current NOE scale factor.

    Returns
    -------
    float
        Scale factor (default 1.0)
    """
    return _state.noe['scale']


# =============================================================================
# Backend Compatibility Functions
# =============================================================================

def check_backend_support(restraint_type, backend=None):
    """Check if restraint type is supported on specified backend.

    Parameters
    ----------
    restraint_type : str
        'NOE', 'RESD', or 'SCAT'
    backend : str, optional
        'standard', 'domdec', 'blade', or 'openmm'.
        If None, returns full support dict for restraint type.

    Returns
    -------
    dict or bool
        If backend specified: {'supported': bool, 'notes': str}
        If backend None: Full support dictionary

    Examples
    --------
    >>> restraints.check_backend_support('NOE', 'blade')
    {'supported': True, 'notes': 'Single atom/atoms NOE and PNOE supported...'}

    >>> restraints.check_backend_support('NOE', 'openmm')
    {'supported': False, 'notes': 'NOE restraints are NOT implemented...'}
    """
    restraint_type = restraint_type.upper()

    if restraint_type not in BACKEND_SUPPORT:
        raise ValueError(f"Unknown restraint type: {restraint_type}. "
                        f"Known types: {list(BACKEND_SUPPORT.keys())}")

    support = BACKEND_SUPPORT[restraint_type]

    if backend is None:
        return support.copy()

    backend = backend.lower()
    if backend not in ['standard', 'domdec', 'blade', 'openmm']:
        raise ValueError(f"Unknown backend: {backend}")

    supported = support.get(backend, False)
    notes = support.get('notes', {}).get(backend, '')

    return {'supported': supported, 'notes': notes}


# =============================================================================
# Direct Memory Access (ctypes bindings for api_noe.F90)
# =============================================================================

_noedata_initialized = False


def _init_noedata_bindings():
    """Initialize ctypes bindings for NOE data access."""
    global _noedata_initialized
    if _noedata_initialized:
        return True

    try:


        # Basic query functions
        lib.noedata_is_active.restype = ctypes.c_int
        lib.noedata_is_active.argtypes = []

        lib.noedata_get_count.restype = ctypes.c_int
        lib.noedata_get_count.argtypes = []

        lib.noedata_get_scale.restype = ctypes.c_double
        lib.noedata_get_scale.argtypes = []

        lib.noedata_set_scale.restype = None
        lib.noedata_set_scale.argtypes = [ctypes.c_double]

        # Parameter retrieval
        lib.noedata_get_params.restype = ctypes.c_int
        lib.noedata_get_params.argtypes = [
            ctypes.c_int,  # idx
            ctypes.POINTER(ctypes.c_double),  # kmin
            ctypes.POINTER(ctypes.c_double),  # rmin
            ctypes.POINTER(ctypes.c_double),  # kmax
            ctypes.POINTER(ctypes.c_double),  # rmax
            ctypes.POINTER(ctypes.c_double),  # fmax
            ctypes.POINTER(ctypes.c_double),  # tcon
            ctypes.POINTER(ctypes.c_double),  # rexp
        ]

        lib.noedata_get_soft_params.restype = ctypes.c_int
        lib.noedata_get_soft_params.argtypes = [
            ctypes.c_int,  # idx
            ctypes.POINTER(ctypes.c_double),  # rswi
            ctypes.POINTER(ctypes.c_double),  # sexp
        ]

        lib.noedata_is_mindist.restype = ctypes.c_int
        lib.noedata_is_mindist.argtypes = [ctypes.c_int]

        # Array retrieval
        lib.noedata_get_params_array.restype = ctypes.c_int
        lib.noedata_get_params_array.argtypes = [
            ctypes.c_int,  # n
            ctypes.POINTER(ctypes.c_double),  # kmin_arr
            ctypes.POINTER(ctypes.c_double),  # rmin_arr
            ctypes.POINTER(ctypes.c_double),  # kmax_arr
            ctypes.POINTER(ctypes.c_double),  # rmax_arr
        ]

        # Atom indices retrieval
        lib.noedata_get_atoms.restype = ctypes.c_int
        lib.noedata_get_atoms.argtypes = [
            ctypes.c_int,  # idx
            ctypes.POINTER(ctypes.c_int),  # ni
            ctypes.POINTER(ctypes.c_int),  # nj
            ctypes.POINTER(ctypes.c_int),  # ilist
            ctypes.POINTER(ctypes.c_int),  # jlist
            ctypes.c_int,  # max_atoms
        ]

        _noedata_initialized = True
        return True
    except (AttributeError, OSError, RuntimeError):
        return False


def _init_pnoe_bindings():
    """Initialize ctypes bindings for PNOE functions (if available)."""
    try:


        lib.noedata_is_pnoe.restype = ctypes.c_int
        lib.noedata_is_pnoe.argtypes = [ctypes.c_int]

        lib.noedata_get_pnoe_coords.restype = ctypes.c_int
        lib.noedata_get_pnoe_coords.argtypes = [
            ctypes.c_int,  # idx
            ctypes.POINTER(ctypes.c_double),  # cx
            ctypes.POINTER(ctypes.c_double),  # cy
            ctypes.POINTER(ctypes.c_double),  # cz
        ]

        lib.noedata_set_pnoe_coords.restype = ctypes.c_int
        lib.noedata_set_pnoe_coords.argtypes = [
            ctypes.c_int,  # idx
            ctypes.c_double,  # cx
            ctypes.c_double,  # cy
            ctypes.c_double,  # cz
        ]

        lib.noedata_is_moving_pnoe.restype = ctypes.c_int
        lib.noedata_is_moving_pnoe.argtypes = [ctypes.c_int]

        lib.noedata_get_mpnoe_steps.restype = None
        lib.noedata_get_mpnoe_steps.argtypes = [
            ctypes.POINTER(ctypes.c_int),  # current_step
            ctypes.POINTER(ctypes.c_int),  # total_steps
        ]

        return True
    except (AttributeError, OSError, RuntimeError):
        return False


def _init_noe_assign_bindings():
    """Initialize ctypes bindings for NOE ASSIGN functions (if available)."""
    try:


        lib.noedata_reset.restype = ctypes.c_int
        lib.noedata_reset.argtypes = []

        lib.noedata_assign.restype = ctypes.c_int
        lib.noedata_assign.argtypes = [
            ctypes.c_int,  # ni
            ctypes.POINTER(ctypes.c_int),  # ilist
            ctypes.c_int,  # nj
            ctypes.POINTER(ctypes.c_int),  # jlist
            ctypes.c_double,  # kmin
            ctypes.c_double,  # rmin
            ctypes.c_double,  # kmax
            ctypes.c_double,  # rmax
            ctypes.c_double,  # fmax
            ctypes.c_double,  # tcon
            ctypes.c_double,  # rexp
            ctypes.c_double,  # rswi
            ctypes.c_double,  # sexp
            ctypes.c_int,  # mindist
        ]

        lib.noedata_assign_pnoe.restype = ctypes.c_int
        lib.noedata_assign_pnoe.argtypes = [
            ctypes.c_int,  # ni
            ctypes.POINTER(ctypes.c_int),  # ilist
            ctypes.c_double,  # cnox
            ctypes.c_double,  # cnoy
            ctypes.c_double,  # cnoz
            ctypes.c_double,  # kmin
            ctypes.c_double,  # rmin
            ctypes.c_double,  # kmax
            ctypes.c_double,  # rmax
            ctypes.c_double,  # fmax
            ctypes.c_double,  # tcon
            ctypes.c_double,  # rexp
        ]

        lib.noedata_set_mpnoe_target.restype = ctypes.c_int
        lib.noedata_set_mpnoe_target.argtypes = [
            ctypes.c_int,  # idx
            ctypes.c_double,  # tnox
            ctypes.c_double,  # tnoy
            ctypes.c_double,  # tnoz
        ]

        lib.noedata_set_nmpnoe.restype = None
        lib.noedata_set_nmpnoe.argtypes = [ctypes.c_int]  # nsteps

        return True
    except (AttributeError, OSError, RuntimeError):
        return False


_resddata_initialized = False


def _init_resddata_bindings():
    """Initialize ctypes bindings for RESD data access."""
    global _resddata_initialized
    if _resddata_initialized:
        return True

    try:


        lib.resddata_is_active.restype = ctypes.c_int
        lib.resddata_is_active.argtypes = []

        lib.resddata_get_count.restype = ctypes.c_int
        lib.resddata_get_count.argtypes = []

        lib.resddata_get_scale.restype = ctypes.c_double
        lib.resddata_get_scale.argtypes = []

        lib.resddata_set_scale.restype = None
        lib.resddata_set_scale.argtypes = [ctypes.c_double]

        lib.resddata_reset.restype = ctypes.c_int
        lib.resddata_reset.argtypes = []

        lib.resddata_add.restype = ctypes.c_int
        lib.resddata_add.argtypes = [
            ctypes.c_int,  # npairs
            ctypes.POINTER(ctypes.c_int),  # atom_i
            ctypes.POINTER(ctypes.c_int),  # atom_j
            ctypes.POINTER(ctypes.c_double),  # factors
            ctypes.c_double,  # kval
            ctypes.c_double,  # rval
            ctypes.c_int,  # eval_exp
            ctypes.c_int,  # ival
            ctypes.c_int,  # mval
        ]

        # NEW: Dimension query functions (may not be available in older CHARMM)
        try:
            lib.resddata_get_max_restraints.restype = ctypes.c_int
            lib.resddata_get_max_restraints.argtypes = []

            lib.resddata_get_max_pairs.restype = ctypes.c_int
            lib.resddata_get_max_pairs.argtypes = []

            lib.resddata_get_current_pair_count.restype = ctypes.c_int
            lib.resddata_get_current_pair_count.argtypes = []
        except AttributeError:
            # These functions may not be available in older CHARMM builds
            logger.debug("RESD dimension query functions not available")

        _resddata_initialized = True
        return True
    except (AttributeError, OSError, RuntimeError):
        return False


_consdata_initialized = False


def _init_consdata_bindings():
    """Initialize ctypes bindings for CONS (dihedral, IC, droplet) data access."""
    global _consdata_initialized
    if _consdata_initialized:
        return True

    try:


        # Dihedral restraints
        lib.consdata_get_ndihe.restype = ctypes.c_int
        lib.consdata_get_ndihe.argtypes = []

        lib.consdata_clear_dihe.restype = None
        lib.consdata_clear_dihe.argtypes = []

        lib.consdata_add_dihe.restype = ctypes.c_int
        lib.consdata_add_dihe.argtypes = [
            ctypes.c_int,  # i_atom
            ctypes.c_int,  # j_atom
            ctypes.c_int,  # k_atom
            ctypes.c_int,  # l_atom
            ctypes.c_double,  # force
            ctypes.c_double,  # min_angle
            ctypes.c_int,  # period
            ctypes.c_double,  # width
        ]

        lib.consdata_get_dihe_params.restype = ctypes.c_int
        lib.consdata_get_dihe_params.argtypes = [
            ctypes.c_int,  # idx
            ctypes.POINTER(ctypes.c_int),  # i_atom
            ctypes.POINTER(ctypes.c_int),  # j_atom
            ctypes.POINTER(ctypes.c_int),  # k_atom
            ctypes.POINTER(ctypes.c_int),  # l_atom
            ctypes.POINTER(ctypes.c_double),  # force
            ctypes.POINTER(ctypes.c_double),  # min_angle
            ctypes.POINTER(ctypes.c_int),  # period
            ctypes.POINTER(ctypes.c_double),  # width
        ]

        # IC restraints
        lib.consdata_ic_is_active.restype = ctypes.c_int
        lib.consdata_ic_is_active.argtypes = []

        lib.consdata_get_ic_params.restype = None
        lib.consdata_get_ic_params.argtypes = [
            ctypes.POINTER(ctypes.c_double),  # bond_force
            ctypes.POINTER(ctypes.c_double),  # angle_force
            ctypes.POINTER(ctypes.c_double),  # dihe_force
            ctypes.POINTER(ctypes.c_double),  # impr_force
            ctypes.POINTER(ctypes.c_int),  # exponent
            ctypes.POINTER(ctypes.c_int),  # upper
        ]

        lib.consdata_set_ic_params.restype = None
        lib.consdata_set_ic_params.argtypes = [
            ctypes.c_double,  # bond_force
            ctypes.c_double,  # angle_force
            ctypes.c_double,  # dihe_force
            ctypes.c_double,  # impr_force
            ctypes.c_int,  # exponent
            ctypes.c_int,  # upper
        ]

        lib.consdata_clear_ic.restype = None
        lib.consdata_clear_ic.argtypes = []

        # Droplet restraints
        lib.consdata_droplet_is_active.restype = ctypes.c_int
        lib.consdata_droplet_is_active.argtypes = []

        lib.consdata_get_droplet_params.restype = None
        lib.consdata_get_droplet_params.argtypes = [
            ctypes.POINTER(ctypes.c_double),  # force
            ctypes.POINTER(ctypes.c_int),  # exponent
            ctypes.POINTER(ctypes.c_int),  # mass_weight
        ]

        lib.consdata_set_droplet_params.restype = None
        lib.consdata_set_droplet_params.argtypes = [
            ctypes.c_double,  # force
            ctypes.c_int,  # exponent
            ctypes.c_int,  # mass_weight
        ]

        lib.consdata_clear_droplet.restype = None
        lib.consdata_clear_droplet.argtypes = []

        _consdata_initialized = True
        return True
    except (AttributeError, OSError, RuntimeError):
        return False


def _check_noedata_available():
    """Check if NOE direct access API is available."""
    return _init_noedata_bindings()


def _noe_get_count_direct():
    """Get NOE count directly from CHARMM memory.

    Returns
    -------
    int or None
        Number of restraints, or None if API not available
    """
    if not _init_noedata_bindings():
        return None
    try:
        return lib.noedata_get_count()
    except Exception:
        return None


def _noe_get_scale_direct():
    """Get NOE scale factor directly from CHARMM memory.

    Returns
    -------
    float or None
        Scale factor, or None if API not available
    """
    if not _init_noedata_bindings():
        return None
    try:
        return lib.noedata_get_scale()
    except Exception:
        return None


def _noe_set_scale_direct(scale):
    """Set NOE scale factor directly in CHARMM memory.

    Parameters
    ----------
    scale : float
        New scale factor

    Returns
    -------
    bool
        True if successful
    """
    if not _init_noedata_bindings():
        return False
    try:
        lib.noedata_set_scale(ctypes.c_double(scale))
        return True
    except Exception:
        return False


def noe_get_parameters_direct(idx):
    """Get parameters for a specific NOE restraint from CHARMM memory.

    Parameters
    ----------
    idx : int
        1-based restraint index

    Returns
    -------
    dict or None
        Parameter dictionary with keys: kmin, rmin, kmax, rmax, fmax, tcon, rexp
        Returns None if index invalid or API unavailable
    """
    if not _init_noedata_bindings():
        return None

    try:
        kmin = ctypes.c_double()
        rmin = ctypes.c_double()
        kmax = ctypes.c_double()
        rmax = ctypes.c_double()
        fmax = ctypes.c_double()
        tcon = ctypes.c_double()
        rexp = ctypes.c_double()

        result = lib.noedata_get_params(
            idx,
            ctypes.byref(kmin), ctypes.byref(rmin),
            ctypes.byref(kmax), ctypes.byref(rmax),
            ctypes.byref(fmax), ctypes.byref(tcon),
            ctypes.byref(rexp)
        )

        if result != 1:
            return None

        return {
            'kmin': kmin.value,
            'rmin': rmin.value,
            'kmax': kmax.value,
            'rmax': rmax.value,
            'fmax': fmax.value,
            'tcon': tcon.value,
            'rexp': rexp.value
        }
    except Exception:
        return None


def noe_get_soft_params_direct(idx):
    """Get soft-core parameters for a specific NOE restraint.

    Parameters
    ----------
    idx : int
        1-based restraint index

    Returns
    -------
    dict or None
        Parameter dictionary with keys: rswi, sexp
        Returns None if index invalid or API unavailable
    """
    if not _init_noedata_bindings():
        return None

    try:
        rswi = ctypes.c_double()
        sexp = ctypes.c_double()

        result = lib.noedata_get_soft_params(
            idx, ctypes.byref(rswi), ctypes.byref(sexp)
        )

        if result != 1:
            return None

        return {'rswi': rswi.value, 'sexp': sexp.value}
    except Exception:
        return None


def noe_is_pnoe_direct(idx):
    """Check if restraint is a Point NOE from CHARMM memory.

    Parameters
    ----------
    idx : int
        1-based restraint index

    Returns
    -------
    bool or None
        True if PNOE, False if not, None if unavailable
    """
    if not _init_pnoe_bindings():
        return None

    try:
        result = lib.noedata_is_pnoe(idx)
        if result < 0:
            return None
        return result == 1
    except Exception:
        return None


def noe_get_pnoe_coords_direct(idx):
    """Get PNOE reference coordinates from CHARMM memory.

    Parameters
    ----------
    idx : int
        1-based restraint index

    Returns
    -------
    tuple or None
        (cx, cy, cz) coordinates, or None if not PNOE or unavailable
    """
    if not _init_pnoe_bindings():
        return None

    try:
        cx = ctypes.c_double()
        cy = ctypes.c_double()
        cz = ctypes.c_double()

        result = lib.noedata_get_pnoe_coords(
            idx, ctypes.byref(cx), ctypes.byref(cy), ctypes.byref(cz)
        )

        if result != 1:
            return None

        return (cx.value, cy.value, cz.value)
    except Exception:
        return None


def noe_set_pnoe_coords_direct(idx, cx, cy, cz):
    """Set PNOE reference coordinates in CHARMM memory.

    Useful for moving PNOE without re-issuing commands.

    Parameters
    ----------
    idx : int
        1-based restraint index
    cx, cy, cz : float
        New coordinates

    Returns
    -------
    bool
        True if successful
    """
    if not _init_pnoe_bindings():
        return False

    try:
        result = lib.noedata_set_pnoe_coords(
            idx, ctypes.c_double(cx), ctypes.c_double(cy), ctypes.c_double(cz)
        )
        return result == 1
    except Exception:
        return False


def noe_get_atoms_direct(idx, max_atoms=1000):
    """Get atom indices for an NOE restraint from CHARMM memory.

    Parameters
    ----------
    idx : int
        1-based restraint index
    max_atoms : int
        Maximum number of atoms per selection (default 1000)

    Returns
    -------
    dict or None
        {'i_atoms': list, 'j_atoms': list} with 1-based atom indices,
        or None if unavailable. For PNOE, j_atoms will be a dummy entry.
    """
    if not _init_noedata_bindings():
        return None

    try:
        ni = ctypes.c_int()
        nj = ctypes.c_int()
        ilist = (ctypes.c_int * max_atoms)()
        jlist = (ctypes.c_int * max_atoms)()

        result = lib.noedata_get_atoms(
            idx,
            ctypes.byref(ni),
            ctypes.byref(nj),
            ilist,
            jlist,
            max_atoms
        )

        if result != 1:
            return None

        return {
            'i_atoms': list(ilist[:ni.value]),
            'j_atoms': list(jlist[:nj.value])
        }
    except Exception:
        return None


def is_direct_access_available():
    """Check which direct access APIs are available.

    Returns
    -------
    dict
        Dictionary of API names to availability status
    """
    return {
        'noedata': _init_noedata_bindings(),
        'pnoe': _init_pnoe_bindings(),
        'noe_assign': _init_noe_assign_bindings(),
        'resddata': _init_resddata_bindings(),
        'consdata': _init_consdata_bindings()
    }


def noe_verify_with_charmm():
    """Verify Python NOE state matches CHARMM memory.

    Compares Python-side tracked values against CHARMM's internal
    data structures via direct memory access.

    Returns
    -------
    dict
        {'match': bool, 'details': dict, 'errors': list}
    """
    result = {'match': True, 'details': {}, 'errors': []}

    if not _init_noedata_bindings():
        result['errors'].append('Direct access API not available')
        result['match'] = False
        return result

    # Check count
    py_count = _state.noe['count']
    charmm_count = _noe_get_count_direct()

    result['details']['count'] = {
        'python': py_count,
        'charmm': charmm_count,
        'match': py_count == charmm_count
    }

    if py_count != charmm_count:
        result['match'] = False
        result['errors'].append(
            f'Count mismatch: Python={py_count}, CHARMM={charmm_count}'
        )

    # Check scale
    py_scale = _state.noe['scale']
    charmm_scale = _noe_get_scale_direct()

    if charmm_scale is not None:
        result['details']['scale'] = {
            'python': py_scale,
            'charmm': charmm_scale,
            'match': abs(py_scale - charmm_scale) < 1e-6
        }
        if not result['details']['scale']['match']:
            result['match'] = False
            result['errors'].append(
                f'Scale mismatch: Python={py_scale}, CHARMM={charmm_scale}'
            )

    return result


# =============================================================================
# SCAT Functions (Wrapper to block.py for API consistency)
# =============================================================================

def _scat_auto_context(func):
    """Decorator to auto-wrap SCAT calls in block.modify() if needed.

    SCAT commands must be inside a BLOCK n ... END context. If called
    outside a BLOCK context (after block.end()), this decorator
    automatically wraps the call in block.modify().

    Re-entering BLOCK doesn't reset state until block.clear() is called,
    so this is safe for incremental SCAT setup.
    """
    import functools

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        import pycharmm.block as block

        # Check if we're already in a BLOCK context
        if block._state.active and not block._state._in_block_context:
            # Outside BLOCK context - auto-wrap with modify()
            with block.modify():
                return func(*args, **kwargs)
        else:
            # Inside BLOCK context or during initial setup - call directly
            return func(*args, **kwargs)

    return wrapper


@_scat_auto_context
def scat_enable(mode='on', k=None):
    """Enable scaling of constrained atoms (SCAT).

    Wrapper around block.constrained_atom_scaling() for API consistency.
    Automatically enters BLOCK context if needed (no manual block.modify() required).

    Maps to CHARMM command: SCAT [ON | K real | OFF]

    Backend Compatibility
    ---------------------
    - Standard CHARMM: Full support
    - DOMDEC: Full support
    - BLaDE: Full support
    - OpenMM: Full support (via BLOCK)

    Parameters
    ----------
    mode : str, optional
        'on' to enable, 'off' to disable. Default 'on'.
    k : float, optional
        If specified, sets force constant for constraint scaling.
        Overrides mode parameter.

    Examples
    --------
    >>> import pycharmm.restraints as restraints

    >>> # Can be called directly after block.end() - no modify() needed
    >>> restraints.scat_enable('on')
    >>> restraints.scat_enable(k=300)
    >>> restraints.scat_define_atoms(sel)

    See Also
    --------
    scat_define_atoms : Define which atoms are constrained
    scat_get_state : Query current SCAT settings
    block.constrained_atom_scaling : Underlying implementation
    """
    import pycharmm.block as block
    block.constrained_atom_scaling(mode=mode, k=k)


@_scat_auto_context
def scat_disable():
    """Disable scaling of constrained atoms.

    Wrapper around block.constrained_atom_scaling('off').
    Automatically enters BLOCK context if needed.

    Examples
    --------
    >>> restraints.scat_disable()
    """
    import pycharmm.block as block
    block.constrained_atom_scaling(mode='off')


@_scat_auto_context
def scat_define_atoms(selection):
    """Define constrained atom group for SCAT.

    Wrapper around block.define_constrained_atoms() for API consistency.
    Automatically enters BLOCK context if needed (no manual block.modify() required).

    Maps to CHARMM command: CATS atom-selection

    Parameters
    ----------
    selection : SelectAtoms, str
        Atom selection for constrained atoms

    Examples
    --------
    >>> from pycharmm.select_atoms import SelectAtoms
    >>> import pycharmm.restraints as restraints

    >>> # Can be called directly after block.end()
    >>> restraints.scat_enable(k=300)
    >>> restraints.scat_define_atoms(sel_oe2)
    >>> restraints.scat_define_atoms(sel_cd)

    See Also
    --------
    scat_enable : Enable SCAT
    block.define_constrained_atoms : Underlying implementation
    """
    import pycharmm.block as block
    block.define_constrained_atoms(selection)


def scat_get_state():
    """Query current SCAT settings.

    Wrapper around block.get_constrained_atom_scaling().

    Returns
    -------
    dict
        {'enabled': bool, 'mode': str or None, 'k': float or None}

    Examples
    --------
    >>> restraints.scat_enable(k=300)
    >>> state = restraints.scat_get_state()
    >>> print(state['k'])  # 300.0
    """
    import pycharmm.block as block
    return block.get_constrained_atom_scaling()


def scat_setup(k, *selections):
    """Set up SCAT with force constant and atom groups in a single BLOCK context.

    Convenience function that batches all SCAT setup commands into a single
    BLOCK/END pair for efficiency. Equivalent to calling scat_enable() and
    scat_define_atoms() for each selection, but more efficient.

    Parameters
    ----------
    k : float
        Force constant for constraint scaling (typically kT = 298.15)
    *selections : SelectAtoms or str
        Variable number of atom selections for constrained atom groups

    Examples
    --------
    >>> import pycharmm.restraints as restraints
    >>> from pycharmm import SelectAtoms

    >>> # Set up SCAT with multiple atom groups in one call
    >>> restraints.scat_setup(298.15, sel_oe2, sel_cd, sel_oe1, sel_cg)

    >>> # Equivalent to (but more efficient than):
    >>> # restraints.scat_enable(k=298.15)
    >>> # restraints.scat_define_atoms(sel_oe2)
    >>> # restraints.scat_define_atoms(sel_cd)
    >>> # ...

    See Also
    --------
    scat_enable : Enable SCAT (individual call)
    scat_define_atoms : Define atom groups (individual call)
    """
    import pycharmm.block as block

    if not block._state.active:
        raise ValueError("BLOCK not initialized. Call block.initialize() first.")

    with block.modify():
        block.constrained_atom_scaling(mode='on')
        block.constrained_atom_scaling(k=k)
        for sel in selections:
            block.define_constrained_atoms(sel)


# =============================================================================
# RESD Functions (Restrained Distances)
# =============================================================================

def _build_atom_spec(atom):
    """Convert atom specification to CHARMM format (segid resid atomname).

    RESD uses atom specification format 'SEGID RESID ATOMNAME', not
    SELE...END syntax.

    Parameters
    ----------
    atom : tuple or str
        If tuple: (segid, resid, atomname)
        If str: 'SEGID RESID ATOMNAME' format

    Returns
    -------
    str
        CHARMM atom specification string
    """
    if isinstance(atom, tuple):
        if len(atom) == 3:
            segid, resid, atomname = atom
            return f'{segid} {resid} {atomname}'
        else:
            raise ValueError(f"Atom tuple must have 3 elements: {atom}")
    elif isinstance(atom, str):
        # Assume already in correct format
        return atom
    else:
        raise ValueError(f"Invalid atom specification: {atom}")


def _get_atom_index_from_spec(atom_spec):
    """Get 1-based atom index from atom specification.

    Parameters
    ----------
    atom_spec : tuple or str
        Atom specification as (segid, resid, atomname) tuple or string

    Returns
    -------
    int or None
        1-based atom index, or None if lookup fails
    """
    try:
        from pycharmm.select_atoms import SelectAtoms

        if isinstance(atom_spec, tuple):
            segid, resid, atomname = atom_spec
            sel = SelectAtoms(seg_id=segid, res_id=resid, atom_type=atomname)
        elif isinstance(atom_spec, str):
            parts = atom_spec.split()
            if len(parts) == 3:
                segid, resid, atomname = parts
                sel = SelectAtoms(seg_id=segid, res_id=int(resid), atom_type=atomname)
            else:
                return None
        else:
            return None

        indices = _get_selection_indices(sel)
        if indices and len(indices) == 1:
            return indices[0]
    except Exception:
        pass
    return None


def resd_add(distances, kval, rval, eval_exp=2, ival=1, positive=False,
             negative=False):
    """Add a restrained distance constraint.

    Maps to CHARMM command: RESDistance KVAL real RVAL real ...

    Restrained distances allow specifying reaction coordinates as linear
    combinations of multiple distances. The energy is:

        E = (1/EVAL) * KVAL * Dref**EVAL

    Where Dref = sum_i(weight_i * R_i**IVAL) - RVAL

    Uses direct API when available, falls back to script-based approach.

    Backend Compatibility
    ---------------------
    - Standard CHARMM: Full support
    - DOMDEC: Full support
    - BLaDE: NOT SUPPORTED
    - OpenMM: NOT SUPPORTED

    Parameters
    ----------
    distances : list of tuples
        Each tuple is (weight, atom1, atom2) where:
        - weight: float - coefficient for this distance
        - atom1, atom2: atom specifications as tuples (segid, resid, atomname)
          or strings in 'SEGID RESID ATOMNAME' format
    kval : float
        Force constant (kcal/mol/A^EVAL)
    rval : float
        Target value for Dref
    eval_exp : int, optional
        Exponent for energy term. Default 2 (harmonic).
    ival : int, optional
        Exponent for individual distances. Default 1.
    positive : bool, optional
        Only apply when Dref > 0. Default False.
    negative : bool, optional
        Only apply when Dref < 0. Default False.

    Returns
    -------
    int
        Index of the added restraint (1-based)

    Raises
    ------
    ValueError
        If distances list is empty or has invalid format

    Examples
    --------
    >>> import pycharmm.restraints as restraints

    >>> # Simple reaction coordinate: R(O-H) - R(H-O')
    >>> # Using tuple format (segid, resid, atomname)
    >>> atom1 = ('MAIN', 11, 'OG')
    >>> atom2 = ('MAIN', 11, 'HG')
    >>> atom3 = ('MAIN', 23, 'OD1')
    >>>
    >>> distances = [
    ...     (1.0, atom1, atom2),   # +1.0 * R(O-H)
    ...     (-1.0, atom2, atom3),  # -1.0 * R(H-O')
    ... ]
    >>> restraints.resd_add(distances, kval=2000.0, rval=-1.0)

    >>> # Using string format
    >>> restraints.resd_add([
    ...     (1.0, 'ADP 1 CA', 'ADP 1 C'),
    ... ], kval=100.0, rval=1.5)

    >>> # Equilateral triangle constraint
    >>> restraints.resd_add([
    ...     (1.0, atom1, atom2),
    ...     (1.0, atom1, atom3),
    ...     (-2.0, atom2, atom3)
    ... ], kval=1000.0, rval=0.0)

    See Also
    --------
    resd_reset : Clear all RESD restraints
    resd_scale : Set scale factor
    resd_print : Print current restraints
    """
    if not distances:
        raise ValueError("distances list cannot be empty")

    # Validate parameters (matches CHARMM script mode behavior)
    if kval == 0.0:
        raise ValueError("Force constant (kval) cannot be zero")
    if eval_exp <= 0:
        raise ValueError(f"Exponent (eval_exp) must be positive, got {eval_exp}")
    if positive and negative:
        raise ValueError("Cannot specify both positive=True and negative=True")
    if rval < 0 and not (positive or negative):
        # Just a warning-worthy case, but not an error
        pass

    # Try direct API first
    try:
        if _init_resddata_bindings():
            # Try to convert all atom specifications to indices
            atom_i_list = []
            atom_j_list = []
            factor_list = []

            for weight, atom1, atom2 in distances:
                idx1 = _get_atom_index_from_spec(atom1)
                idx2 = _get_atom_index_from_spec(atom2)
                if idx1 is None or idx2 is None:
                    break
                atom_i_list.append(idx1)
                atom_j_list.append(idx2)
                factor_list.append(weight)

            if len(atom_i_list) == len(distances):
                # All indices resolved, use direct API
                npairs = len(distances)
                atom_i = (ctypes.c_int * npairs)(*atom_i_list)
                atom_j = (ctypes.c_int * npairs)(*atom_j_list)
                factors = (ctypes.c_double * npairs)(*factor_list)

                # Convert mode flags to mval: 0=both, 1=positive, -1=negative
                mval = 0
                if positive:
                    mval = 1
                elif negative:
                    mval = -1

                result = lib.resddata_add(
                    npairs, atom_i, atom_j, factors,
                    ctypes.c_double(kval), ctypes.c_double(rval),
                    eval_exp, ival, mval
                )

                if result > 0:
                    # Track in Python state
                    params = {
                        'index': result,
                        'kval': kval,
                        'rval': rval,
                        'eval': eval_exp,
                        'ival': ival,
                        'positive': positive,
                        'negative': negative,
                        'distances': [(w, str(a1), str(a2)) for w, a1, a2 in distances]
                    }
                    _state.resd_add_restraint(params)
                    return result
                else:
                    # Direct API returned an error code
                    error_msg = _interpret_resd_error(result)
                    logger.warning(f"Direct RESD API returned error: {error_msg}")
                    # Fall through to script-based approach
    except (AttributeError, OSError) as e:
        logger.debug(f"Direct RESD API unavailable ({type(e).__name__}), using script fallback")
    except RuntimeError as e:
        logger.warning(f"Direct RESD API failed ({e}), falling back to script mode")

    # Fall back to script-based approach
    # Build RESD command
    cmd_parts = ['RESDistance']
    cmd_parts.append(f'KVAL {kval}')
    cmd_parts.append(f'RVAL {rval}')

    if eval_exp != 2:
        cmd_parts.append(f'EVAL {eval_exp}')
    if ival != 1:
        cmd_parts.append(f'IVAL {ival}')
    if positive:
        cmd_parts.append('POSITIVE')
    if negative:
        cmd_parts.append('NEGATIVE')

    # Add distance specifications using RESD format (segid resid atomname)
    # Each distance pair on its own line to handle negative weights correctly
    distance_lines = []
    for i, (weight, atom1, atom2) in enumerate(distances):
        atom1_str = _build_atom_spec(atom1)
        atom2_str = _build_atom_spec(atom2)

        # Format: weight atom1 atom2 with continuation on separate lines
        if i < len(distances) - 1:
            distance_lines.append(f'  {weight} {atom1_str} {atom2_str} -')
        else:
            distance_lines.append(f'  {weight} {atom1_str} {atom2_str}')

    # Build command with proper line breaks
    cmd_header = ' '.join(cmd_parts)
    cmd = cmd_header + ' -\n' + '\n'.join(distance_lines)
    lingo.charmm_script(cmd)

    # Track in Python state
    params = {
        'index': _state.resd['count'] + 1,
        'kval': kval,
        'rval': rval,
        'eval': eval_exp,
        'ival': ival,
        'positive': positive,
        'negative': negative,
        'distances': [(w, str(a1), str(a2)) for w, a1, a2 in distances]
    }

    return _state.resd_add_restraint(params)


def resd_reset():
    """Reset all restrained distance constraints.

    Maps to CHARMM command: RESDistance RESET

    Clears all RESD restraints and resets scale to 1.0.
    Uses direct API when available, falls back to script-based approach.

    Examples
    --------
    >>> restraints.resd_reset()
    """
    # Try direct API first
    try:
        if _init_resddata_bindings():
            result = lib.resddata_reset()
            if result >= 0:
                _state.resd_clear()
                return
    except (AttributeError, OSError, RuntimeError):
        pass

    # Fall back to script-based approach
    lingo.charmm_script('RESDistance RESET')
    _state.resd_clear()


def resd_scale(factor):
    """Set scale factor for all RESD restraints.

    Maps to CHARMM command: RESDistance SCALE real

    Uses direct API when available, falls back to script-based approach.
    Implements transactional state updates to avoid inconsistencies.

    Parameters
    ----------
    factor : float
        Scale factor for RESD energy contribution

    Raises
    ------
    ValueError
        If factor is negative
    RuntimeError
        If the operation fails

    Examples
    --------
    >>> restraints.resd_scale(0.5)  # Reduce restraint strength by half
    """
    if factor < 0:
        raise ValueError(f"Scale factor must be non-negative, got {factor}")

    old_scale = _state.resd.get('scale', 1.0)

    # Try direct API first
    try:
        if _init_resddata_bindings():
            lib.resddata_set_scale(ctypes.c_double(factor))
            _state.resd['scale'] = factor
            return
    except (AttributeError, OSError) as e:
        logger.debug(f"Direct RESD API unavailable ({type(e).__name__}), using script fallback")
    except RuntimeError as e:
        logger.warning(f"Direct RESD API failed ({e}), falling back to script mode")

    # Fall back to script-based approach
    try:
        lingo.charmm_script(f'RESDistance SCALE {factor}')
        _state.resd['scale'] = factor
    except Exception as e:
        # Attempt to recover actual state from Fortran
        try:
            if _init_resddata_bindings():
                actual = lib.resddata_get_scale()
                _state.resd['scale'] = actual
            else:
                _state.resd['scale'] = old_scale
        except (AttributeError, OSError, RuntimeError):
            _state.resd['scale'] = old_scale
        raise RuntimeError(f"Failed to set RESD scale: {e}") from e


def resd_print():
    """Print current restrained distance settings.

    Maps to CHARMM command: PRINT RESDistances

    Examples
    --------
    >>> restraints.resd_print()
    """
    lingo.charmm_script('PRINT RESDistances')


def resd_get_count():
    """Get number of RESD restraints.

    Returns
    -------
    int
        Number of restrained distance constraints
    """
    return _state.resd['count']


def resd_get_state():
    """Get current RESD state dictionary.

    Returns
    -------
    dict
        Copy of RESD state including count, scale, and restraint list
    """
    return _state.resd.copy()


def resd_get_restraints():
    """Get list of tracked RESD restraints.

    Returns
    -------
    list
        List of restraint parameter dictionaries
    """
    return [r.copy() for r in _state.resd['restraints']]


def resd_is_active():
    """Check if any RESD restraints are defined.

    Returns
    -------
    bool
        True if RESD restraints exist
    """
    return _state.resd['active']


# =============================================================================
# Module-Level Convenience Functions
# =============================================================================

def reset():
    """Reset all restraint state (NOE, RESD, etc.)."""
    _state.reset()


def get_state():
    """Get complete restraint state dictionary."""
    return _state.to_dict()


# =============================================================================
# Harmonic Restraints (Wrapper to cons_harm.py for API consistency)
# =============================================================================

def harmonic_absolute(selection=None, force_const=0.0, expo=2,
                      x_scale=1.0, y_scale=1.0, z_scale=1.0,
                      comparison=False, mass_weighted=False,
                      use_weights=False, q_mass=None, q_weight=None):
    """Apply absolute harmonic positional restraints.

    Wrapper around cons_harm.setup_absolute() for API consistency.
    Restrains atoms to fixed reference positions (current coordinates
    or comparison set).

    Maps to CHARMM command: CONS HARMonic ABSOlute ...

    The restraint energy is::

        E = sum_i k(i) * [mass(i)] * |r(i) - r_ref(i)|^expo

    Backend Compatibility
    ---------------------
    - Standard CHARMM: Full support
    - DOMDEC: Full support
    - BLaDE: Full support
    - OpenMM: Full support

    Parameters
    ----------
    selection : SelectAtoms, optional
        Atoms to restrain. If None, all atoms are restrained.
    force_const : float, optional
        Force constant k (kcal/mol/A^expo). Default 0.0.
    expo : int, optional
        Exponent on displacement (2 for harmonic). Default 2.
    x_scale : float, optional
        Scale factor for x component. Default 1.0.
    y_scale : float, optional
        Scale factor for y component. Default 1.0.
    z_scale : float, optional
        Scale factor for z component. Default 1.0.
    comparison : bool, optional
        If True, use comparison coordinate set. Default False.
    mass_weighted : bool, optional
        If True, multiply k by atomic mass. Default False.
    use_weights : bool, optional
        If True, use weight array for k(i). Default False.

    Returns
    -------
    bool
        True if successful

    Examples
    --------
    >>> import pycharmm.restraints as restraints
    >>> from pycharmm.select_atoms import SelectAtoms

    >>> # Restrain all atoms with k=10 kcal/mol/A^2
    >>> restraints.harmonic_absolute(force_const=10.0)

    >>> # Restrain backbone atoms only
    >>> bb = SelectAtoms(atom_type=['CA', 'C', 'N', 'O'])
    >>> restraints.harmonic_absolute(selection=bb, force_const=5.0)

    >>> # Restrain only in z direction
    >>> restraints.harmonic_absolute(force_const=10.0,
    ...                              x_scale=0.0, y_scale=0.0, z_scale=1.0)

    See Also
    --------
    harmonic_best_fit : Best-fit (superposition) restraints
    harmonic_relative : Relative restraints between selections
    harmonic_turn_off : Turn off all harmonic restraints
    """
    if q_mass is not None:
        warnings.warn(
            "q_mass is deprecated; use mass_weighted instead",
            DeprecationWarning,
            stacklevel=2
        )
        mass_weighted = bool(q_mass)
    if q_weight is not None:
        warnings.warn(
            "q_weight is deprecated; use use_weights instead",
            DeprecationWarning,
            stacklevel=2
        )
        use_weights = bool(q_weight)

    return _harmonic_setup_absolute(
        selection, comparison=comparison,
        force_const=force_const, expo=expo,
        x_scale=x_scale, y_scale=y_scale, z_scale=z_scale,
        q_mass=int(mass_weighted), q_weight=int(use_weights)
    )


def harmonic_best_fit(selection=None, force_const=0.0, comparison=False,
                      mass_weighted=False, use_weights=False,
                      no_rotation=False, no_translation=False,
                      q_mass=None, q_weight=None,
                      q_no_rot=None, q_no_trans=None):
    """Apply best-fit harmonic positional restraints.

    Wrapper around cons_harm.setup_best_fit() for API consistency.
    Restrains atoms after optimal superposition to reference.

    Maps to CHARMM command: CONS HARMonic BESTfit ...

    Backend Compatibility
    ---------------------
    - Standard CHARMM: Full support
    - DOMDEC: Full support
    - BLaDE: Full support
    - OpenMM: Full support

    Parameters
    ----------
    selection : SelectAtoms, optional
        Atoms to restrain. If None, all atoms are restrained.
    force_const : float, optional
        Force constant k (kcal/mol/A^2). Default 0.0.
    comparison : bool, optional
        If True, use comparison coordinate set. Default False.
    mass_weighted : bool, optional
        If True, multiply k by atomic mass. Default False.
    use_weights : bool, optional
        If True, use weight array for k(i). Default False.
    no_rotation : bool, optional
        If True, disable rotational component. Default False.
    no_translation : bool, optional
        If True, disable translational component. Default False.

    Returns
    -------
    bool
        True if successful

    Examples
    --------
    >>> import pycharmm.restraints as restraints
    >>> from pycharmm.select_atoms import SelectAtoms

    >>> # Best-fit restraints on CA atoms
    >>> ca = SelectAtoms(atom_type='CA')
    >>> restraints.harmonic_best_fit(selection=ca, force_const=1.0)

    See Also
    --------
    harmonic_absolute : Absolute positional restraints
    harmonic_relative : Relative restraints between selections
    """
    if q_mass is not None:
        warnings.warn(
            "q_mass is deprecated; use mass_weighted instead",
            DeprecationWarning,
            stacklevel=2
        )
        mass_weighted = bool(q_mass)
    if q_weight is not None:
        warnings.warn(
            "q_weight is deprecated; use use_weights instead",
            DeprecationWarning,
            stacklevel=2
        )
        use_weights = bool(q_weight)
    if q_no_rot is not None:
        warnings.warn(
            "q_no_rot is deprecated; use no_rotation instead",
            DeprecationWarning,
            stacklevel=2
        )
        no_rotation = bool(q_no_rot)
    if q_no_trans is not None:
        warnings.warn(
            "q_no_trans is deprecated; use no_translation instead",
            DeprecationWarning,
            stacklevel=2
        )
        no_translation = bool(q_no_trans)

    return _harmonic_setup_best_fit(
        selection, comparison=comparison,
        force_const=force_const,
        q_mass=int(mass_weighted), q_weight=int(use_weights),
        q_no_rot=int(no_rotation), q_no_trans=int(no_translation)
    )


def harmonic_relative(selection1, selection2, force_const=0.0,
                      comparison=False, mass_weighted=False,
                      use_weights=False, no_rotation=False,
                      no_translation=False,
                      q_mass=None, q_weight=None,
                      q_no_rot=None, q_no_trans=None):
    """Apply relative harmonic restraints between two selections.

    Wrapper around cons_harm.setup_relative() for API consistency.
    Restrains atoms in selection1 relative to atoms in selection2.
    Both selections must have the same number of atoms.

    Maps to CHARMM command: CONS HARMonic RELAtive ...

    Backend Compatibility
    ---------------------
    - Standard CHARMM: Full support
    - DOMDEC: Full support
    - BLaDE: Full support
    - OpenMM: Full support

    Parameters
    ----------
    selection1 : SelectAtoms
        First selection of atoms
    selection2 : SelectAtoms
        Second selection of atoms (must match selection1 count)
    force_const : float, optional
        Force constant k (kcal/mol/A^2). Default 0.0.
    comparison : bool, optional
        If True, use comparison coordinate set. Default False.
    mass_weighted : bool, optional
        If True, multiply k by atomic mass. Default False.
    use_weights : bool, optional
        If True, use weight array for k(i). Default False.
    no_rotation : bool, optional
        If True, disable rotational component. Default False.
    no_translation : bool, optional
        If True, disable translational component. Default False.

    Returns
    -------
    bool
        True if successful

    Raises
    ------
    ValueError
        If selections have different numbers of atoms

    Examples
    --------
    >>> import pycharmm.restraints as restraints
    >>> from pycharmm.select_atoms import SelectAtoms

    >>> # Relative restraints between two protein chains
    >>> chainA = SelectAtoms(seg_id='PROA', atom_type='CA')
    >>> chainB = SelectAtoms(seg_id='PROB', atom_type='CA')
    >>> restraints.harmonic_relative(chainA, chainB, force_const=5.0)

    See Also
    --------
    harmonic_absolute : Absolute positional restraints
    harmonic_best_fit : Best-fit restraints
    """
    if q_mass is not None:
        warnings.warn(
            "q_mass is deprecated; use mass_weighted instead",
            DeprecationWarning,
            stacklevel=2
        )
        mass_weighted = bool(q_mass)
    if q_weight is not None:
        warnings.warn(
            "q_weight is deprecated; use use_weights instead",
            DeprecationWarning,
            stacklevel=2
        )
        use_weights = bool(q_weight)
    if q_no_rot is not None:
        warnings.warn(
            "q_no_rot is deprecated; use no_rotation instead",
            DeprecationWarning,
            stacklevel=2
        )
        no_rotation = bool(q_no_rot)
    if q_no_trans is not None:
        warnings.warn(
            "q_no_trans is deprecated; use no_translation instead",
            DeprecationWarning,
            stacklevel=2
        )
        no_translation = bool(q_no_trans)

    return _harmonic_setup_relative(
        selection1, selection2, comparison=comparison,
        force_const=force_const,
        q_mass=int(mass_weighted), q_weight=int(use_weights),
        q_no_rot=int(no_rotation), q_no_trans=int(no_translation)
    )


def harmonic_pca(selection=None, force_const=0.0, expo=2,
                 x_scale=1.0, y_scale=1.0, z_scale=1.0,
                 comparison=False, mass_weighted=False,
                 use_weights=False):
    """Apply PCA-style harmonic restraints.

    Wrapper around cons_harm.setup_pca() for API consistency.
    Similar to absolute restraints but designed for PCA analysis.

    Maps to CHARMM command: CONS HARMonic PCA ...

    Parameters
    ----------
    selection : SelectAtoms, optional
        Atoms to restrain. If None, all atoms are restrained.
    force_const : float, optional
        Force constant k (kcal/mol/A^expo). Default 0.0.
    expo : int, optional
        Exponent on displacement. Default 2.
    x_scale : float, optional
        Scale factor for x component. Default 1.0.
    y_scale : float, optional
        Scale factor for y component. Default 1.0.
    z_scale : float, optional
        Scale factor for z component. Default 1.0.
    comparison : bool, optional
        If True, use comparison coordinate set. Default False.
    mass_weighted : bool, optional
        If True, multiply k by atomic mass. Default False.
    use_weights : bool, optional
        If True, use weight array for k(i). Default False.

    Returns
    -------
    bool
        True if successful
    """
    return _harmonic_setup_pca(
        selection, comparison=comparison,
        force_const=force_const, expo=expo,
        x_scale=x_scale, y_scale=y_scale, z_scale=z_scale,
        q_mass=int(mass_weighted), q_weight=int(use_weights)
    )


def harmonic_turn_off():
    """Turn off all harmonic restraints.

    Wrapper around cons_harm.turn_off() for API consistency.

    Maps to CHARMM command: CONS HARMonic CLEAR

    Returns
    -------
    bool
        True if successful

    Examples
    --------
    >>> import pycharmm.restraints as restraints
    >>> restraints.harmonic_turn_off()
    """
    return _harmonic_turn_off()


# =============================================================================
# Fixed Atom Constraints (Wrapper to cons_fix.py)
# =============================================================================

def fix_atoms(selection, comparison=False, purge=False,
              bond=False, angle=False, phi=False, imp=False, cmap=False):
    """Fix selected atoms in place (immobilize).

    Wrapper around cons_fix.setup() for API consistency.
    Fixed atoms have zero velocity and forces are not computed.

    Maps to CHARMM command: CONS FIX ...

    Backend Compatibility
    ---------------------
    - Standard CHARMM: Full support
    - DOMDEC: Full support
    - BLaDE: Full support
    - OpenMM: Full support

    Parameters
    ----------
    selection : SelectAtoms
        Atoms to fix (immobilize)
    comparison : bool, optional
        If True, apply to comparison coordinate set. Default False.
    purge : bool, optional
        If True, use PURGE option (modifies PSF irrevocably). Default False.
    bond : bool, optional
        If True, also fix bonds involving selected atoms. Default False.
    angle : bool, optional
        If True, also fix angles involving selected atoms. Default False.
    phi : bool, optional
        If True, also fix dihedrals involving selected atoms. Default False.
    imp : bool, optional
        If True, also fix impropers involving selected atoms. Default False.
    cmap : bool, optional
        If True, also fix CMAP terms involving selected atoms. Default False.

    Returns
    -------
    bool
        True if successful

    Examples
    --------
    >>> import pycharmm.restraints as restraints
    >>> from pycharmm.select_atoms import SelectAtoms

    >>> # Fix backbone atoms
    >>> bb = SelectAtoms(atom_type=['CA', 'C', 'N'])
    >>> restraints.fix_atoms(bb)

    >>> # Fix entire segment
    >>> restraints.fix_atoms(SelectAtoms(seg_id='PROT'))

    See Also
    --------
    fix_turn_off : Turn off fixed atom constraints
    harmonic_absolute : Apply soft positional restraints instead
    """
    return _fix_setup(
        selection, comparison=comparison, purge=purge,
        bond=bond, angle=angle, phi=phi, imp=imp, cmap=cmap
    )


def fix_turn_off(comparison=False):
    """Turn off all fixed atom constraints.

    Wrapper around cons_fix.turn_off() for API consistency.

    Maps to CHARMM command: CONS FIX SELE NONE END

    Parameters
    ----------
    comparison : bool, optional
        If True, turn off constraints on comparison set. Default False.

    Returns
    -------
    bool
        True if successful

    Examples
    --------
    >>> import pycharmm.restraints as restraints
    >>> restraints.fix_turn_off()
    """
    return _fix_turn_off(comparison=comparison)


# =============================================================================
# Dihedral and IC Constraints (Wrapper to cons_methods.py)
# =============================================================================

def dihe_restraint(selection='', force=0, minimum=None, period=None,
                   width=None, comp=False, main=True, clear=False):
    """Apply dihedral angle restraints.

    Wrapper around cons_methods.dihe() for API consistency.

    Maps to CHARMM command: CONS DIHE ...

    Backend Compatibility
    ---------------------
    - Standard CHARMM: Full support
    - DOMDEC: Full support
    - BLaDE: Full support
    - OpenMM: Full support

    Parameters
    ----------
    selection : str
        Atom selection for dihedral (4 atoms). Can be:
        - 'bynum int int int int' format
        - '4x(segid resid iupac)' format
        - '4x(resnumber iupac)' format
    force : float
        Force constant (kcal/mol/rad^2)
    minimum : float, optional
        Target dihedral angle (degrees)
    period : int, optional
        Periodicity of the potential
    width : float, optional
        Width parameter for flat-bottom potential
    comp : bool, optional
        If True, use comparison coordinates. Default False.
    main : bool, optional
        If True, use main coordinates. Default True.
    clear : bool, optional
        If True, clear all dihedral restraints. Default False.

    Examples
    --------
    >>> import pycharmm.restraints as restraints

    >>> # Apply dihedral restraint
    >>> restraints.dihe_restraint(
    ...     selection='bynum 7 9 15 17',
    ...     force=10.0, minimum=-60, width=0
    ... )

    >>> # Clear dihedral restraints
    >>> restraints.dihe_restraint(clear=True)

    See Also
    --------
    ic_restraint : Internal coordinate restraints
    """
    kwargs = {}
    if minimum is not None:
        kwargs['minimum'] = minimum
    if period is not None:
        kwargs['period'] = period
    if width is not None:
        kwargs['width'] = width
    if comp:
        kwargs['comp'] = comp
    if not main:
        kwargs['main'] = main

    _dihe_restraint(selection=selection, force=force, clear=clear, **kwargs)


def mmfp_dihedral(atom1: str, atom2: str, atom3: str, atom4: str,
                  force: float, target_angle: float,
                  clear: bool = False) -> None:
    """Apply MMFP GEO dihedral restraint (BLaDE GPU compatible).

    This function uses MMFP GEO sphere dihedral harmonic symmetric restraints,
    which are fully supported on BLaDE GPU. Use this instead of dihe_restraint()
    when running with BLaDE backend.

    Parameters
    ----------
    atom1, atom2, atom3, atom4 : str
        Selection strings for the four atoms defining the dihedral.
        Example: 'type C1', 'segid MAIN .and. resid 1 .and. type CA'
    force : float
        Force constant in kcal/mol/rad^2.
        Note: MMFP uses E = 0.5 * k * (dphi)^2, so k_mmfp = 2 * k_cons
    target_angle : float
        Target dihedral angle in degrees.
    clear : bool, optional
        If True, clear all MMFP GEO restraints. Default False.

    Examples
    --------
    >>> import pycharmm.restraints as restraints

    >>> # Apply MMFP dihedral restraint (BLaDE compatible)
    >>> restraints.mmfp_dihedral(
    ...     atom1='type C1', atom2='type C2',
    ...     atom3='type C3', atom4='type C4',
    ...     force=100.0, target_angle=-60.0
    ... )

    >>> # Clear MMFP restraints
    >>> restraints.mmfp_dihedral(clear=True)

    Notes
    -----
    - MMFP energy: E = 0.5 * k * (phi - phi0)^2
    - CONS DIHE energy: E = k * (phi - phi0)^2
    - To match CONS DIHE energies, use k_mmfp = 2 * k_cons

    See Also
    --------
    dihe_restraint : CONS DIHE restraints (CPU only)
    smart_dihedral : Auto-selects between CONS DIHE and MMFP based on backend
    """
    if clear:
        lingo.charmm_script("MMFP\ngeo reset\nEND")
        _state.active['mmfp_dihedral'] = {'enabled': False, 'restraints': []}
        return

    # Validate backend supports MMFP_DIHE
    backend = get_backend()
    if backend == 'openmm':
        raise IncompatibleBackendError(
            'MMFP_DIHE',
            backend,
            "Use dihe_restraint() for OpenMM."
        )

    # Validate inputs
    if force <= 0:
        raise ValueError(f"Force constant must be positive, got {force}")

    # Build MMFP command
    cmd = f"""MMFP
geo sphere dihedral harmonic symmetric force {force} tref {target_angle} -
select {atom1} end -
select {atom2} end -
select {atom3} end -
select {atom4} end
END"""

    lingo.charmm_script(cmd)

    # Update state
    _state.active['mmfp_dihedral']['enabled'] = True
    _state.active['mmfp_dihedral']['restraints'].append({
        'atoms': [atom1, atom2, atom3, atom4],
        'force': force,
        'target': target_angle
    })


def mmfp_dihedral_clear() -> None:
    """Clear all MMFP GEO restraints.

    Examples
    --------
    >>> restraints.mmfp_dihedral_clear()
    """
    mmfp_dihedral(atom1='', atom2='', atom3='', atom4='',
                  force=0, target_angle=0, clear=True)


def smart_dihedral(selection: str = None,
                   atom1: str = None, atom2: str = None,
                   atom3: str = None, atom4: str = None,
                   force: float = 0, minimum: float = None,
                   auto_adjust_force: bool = True, **kwargs) -> None:
    """Apply dihedral restraint with automatic backend selection.

    Automatically chooses between CONS DIHE and MMFP GEO based on
    the current backend:
    - BLaDE: Uses mmfp_dihedral() (MMFP GEO)
    - Others: Uses dihe_restraint() (CONS DIHE)

    Parameters
    ----------
    selection : str, optional
        CONS DIHE style selection (e.g., 'bynum 7 9 15 17').
        Used when backend is not BLaDE.
    atom1, atom2, atom3, atom4 : str, optional
        Individual atom selections. Required for BLaDE backend.
    force : float
        Force constant in kcal/mol/rad^2.
    minimum : float, optional
        Target dihedral angle in degrees.
    auto_adjust_force : bool, optional
        If True (default), automatically adjust force constant
        for MMFP (multiply by 2) to match CONS DIHE energies.

    Examples
    --------
    >>> import pycharmm.restraints as restraints

    >>> # For BLaDE backend
    >>> restraints.set_backend('blade')
    >>> restraints.smart_dihedral(
    ...     atom1='type C1', atom2='type C2',
    ...     atom3='type C3', atom4='type C4',
    ...     force=100.0, minimum=-60.0
    ... )

    >>> # For CPU backend
    >>> restraints.set_backend('standard')
    >>> restraints.smart_dihedral(
    ...     selection='bynum 7 9 15 17',
    ...     force=100.0, minimum=-60.0
    ... )

    Notes
    -----
    When auto_adjust_force=True (default), the force constant is automatically
    doubled for MMFP to produce equivalent energies:
    - CONS DIHE: E = k * (phi - phi0)^2
    - MMFP: E = 0.5 * k * (phi - phi0)^2

    See Also
    --------
    dihe_restraint : CONS DIHE restraints (CPU only)
    mmfp_dihedral : MMFP GEO restraints (BLaDE GPU compatible)
    """
    backend = get_backend()

    # Validate required parameters
    if minimum is None:
        raise ValueError("minimum (target angle in degrees) is required")

    if backend == 'blade':
        if not all([atom1, atom2, atom3, atom4]):
            raise ValueError(
                "BLaDE backend requires atom1-4 selections. "
                "Use atom1='type X' style selections."
            )
        # MMFP uses E = 0.5*k*(dphi)^2, so double the force constant
        mmfp_force = force * 2 if auto_adjust_force else force
        mmfp_dihedral(atom1, atom2, atom3, atom4, mmfp_force, minimum)
    else:
        if selection is None:
            raise ValueError(
                "Non-BLaDE backends require 'selection' parameter."
            )
        dihe_restraint(selection=selection, force=force, minimum=minimum, **kwargs)


def ic_restraint(bond=None, angle=None, dihedral=None, improper=None,
                 exponent=2, upper=False):
    """Apply internal coordinate restraints.

    Wrapper around cons_methods.ic() for API consistency.

    Maps to CHARMM command: CONS IC ...

    Backend Compatibility
    ---------------------
    - Standard CHARMM: Full support
    - DOMDEC: Full support
    - BLaDE: NOT SUPPORTED
    - OpenMM: NOT SUPPORTED

    Parameters
    ----------
    bond : float, optional
        Force constant for bond distances
    angle : float, optional
        Force constant for bond angles
    dihedral : float, optional
        Force constant for dihedral angles
    improper : float, optional
        Force constant for improper dihedrals
    exponent : int, optional
        Exponent in restraint energy. Default 2.
    upper : bool, optional
        If True, use upper bound constraint. Default False.

    Examples
    --------
    >>> import pycharmm.restraints as restraints

    >>> # Restrain all internal coordinates
    >>> restraints.ic_restraint(bond=100, angle=50, dihedral=10)
    """
    _state.validate_can_add('IC')

    kwargs = {}
    if bond is not None:
        kwargs['bond'] = bond
    if angle is not None:
        kwargs['angle'] = angle
    if dihedral is not None:
        kwargs['dihedral'] = dihedral
    if improper is not None:
        kwargs['improper'] = improper
    kwargs['exponent'] = exponent
    if upper:
        kwargs['upper'] = upper

    _ic_restraint(**kwargs)

    for component, force in (
        ('bond', bond),
        ('angle', angle),
        ('dihedral', dihedral),
        ('improper', improper),
    ):
        if force is not None:
            _state.set_ic_active(component, True, force)


def droplet_restraint(force=None, exponent=None, nomass=False):
    """Apply quartic droplet restraint.

    Wrapper around cons_methods.droplet() for API consistency.
    Applies a spherical boundary potential to keep molecules within a droplet.

    Maps to CHARMM command: CONS DROPlet ...

    Backend Compatibility
    ---------------------
    - Standard CHARMM: Full support
    - DOMDEC: Full support
    - BLaDE: NOT SUPPORTED
    - OpenMM: NOT SUPPORTED

    Parameters
    ----------
    force : float, optional
        Force constant for droplet boundary
    exponent : int, optional
        Exponent in boundary potential
    nomass : bool, optional
        If True, don't use mass-weighting. Default False.

    Examples
    --------
    >>> import pycharmm.restraints as restraints

    >>> # Apply droplet boundary
    >>> restraints.droplet_restraint(force=1.0, exponent=4)
    """
    _state.validate_can_add('DROPLET')

    kwargs = {}
    if force is not None:
        kwargs['force'] = force
    if exponent is not None:
        kwargs['exponent'] = exponent
    if nomass:
        kwargs['nomass'] = nomass

    _droplet_restraint(**kwargs)
    _state.set_droplet_active(True, kwargs)


# =============================================================================
# Module-level Namespace Instances
# =============================================================================
# These provide the namespace-based API: restraints.atoms.fix(),
# restraints.distances.noe(), restraints.angles.dihedral(), etc.

atoms = _AtomRestraints(_state)
"""Namespace for atom-based restraints (FIX, HARMONIC).

Examples
--------
>>> import pycharmm.restraints as restraints

>>> # Fix atoms
>>> restraints.atoms.fix(selection)

>>> # Harmonic restraints
>>> restraints.atoms.harmonic_absolute(selection=sel, force_const=10.0)
>>> restraints.atoms.harmonic_best_fit(selection=sel, force_const=5.0)
>>> restraints.atoms.harmonic_relative(sel1, sel2, force_const=10.0)

>>> # Turn off
>>> restraints.atoms.harmonic_turn_off()
>>> restraints.atoms.fix_turn_off()
"""

distances = _DistanceRestraints(_state)
"""Namespace for distance restraints (NOE, PNOE, RESD).

Examples
--------
>>> import pycharmm.restraints as restraints

>>> # NOE via context manager
>>> with restraints.distances.NOE(reset=True) as noe:
...     noe.assign(sel1, sel2, kmax=5.0, rmax=3.0)

>>> # RESD for reaction coordinates
>>> restraints.distances.resd([...], kval=2000.0, rval=-1.0)

>>> # Scale and reset
>>> restraints.distances.noe_scale(0.5)
>>> restraints.distances.noe_reset()
"""

angles = _AngleRestraints(_state)
"""Namespace for angle/dihedral restraints.

Examples
--------
>>> import pycharmm.restraints as restraints

>>> # Dihedral restraint
>>> restraints.angles.dihedral(selection='bynum 7 9 15 17', force=10.0)

>>> # Clear
>>> restraints.angles.dihedral_clear()
"""

positions = _PositionRestraints(_state)
"""Namespace for position-based restraints (DROPLET, PCA harmonic).

Examples
--------
>>> import pycharmm.restraints as restraints

>>> # Droplet boundary
>>> restraints.positions.droplet(force=10.0, exponent=2)

>>> # PCA harmonic
>>> restraints.positions.harmonic_pca(selection=sel, force_const=5.0)
"""

internal_coords = _ICRestraints(_state)
"""Namespace for internal coordinate restraints.

Examples
--------
>>> import pycharmm.restraints as restraints

>>> # Apply all IC restraints at once
>>> restraints.internal_coords.all(bond=100.0, angle=50.0)

>>> # Or individually
>>> restraints.internal_coords.bond(force=100.0)
>>> restraints.internal_coords.angle(force=50.0)
>>> restraints.internal_coords.dihedral(force=25.0)
>>> restraints.internal_coords.improper(force=10.0)
"""


# =============================================================================
# Module-level Functions for State Management
# =============================================================================

def get_active_restraints() -> dict:
    """Get all currently active restraints.

    Returns a dictionary of all enabled restraint types and their
    current settings.

    Returns
    -------
    dict
        Dictionary mapping restraint type names to their state dictionaries.
        Only includes restraints that are currently enabled.

    Examples
    --------
    >>> import pycharmm.restraints as restraints

    >>> restraints.atoms.harmonic_absolute(force_const=10.0)
    >>> active = restraints.get_active_restraints()
    >>> print(active.keys())
    dict_keys(['harmonic_absolute'])
    """
    return _state.get_active_restraints()


def set_backend(backend: str) -> None:
    """Set the compute backend and validate active restraints.

    Sets the backend used for computation and validates that all
    currently active restraints are compatible with the new backend.

    Parameters
    ----------
    backend : str
        One of 'standard', 'domdec', 'blade', 'openmm'

    Raises
    ------
    ValueError
        If backend is not recognized
    IncompatibleBackendError
        If any active restraints are not supported on the new backend

    Examples
    --------
    >>> import pycharmm.restraints as restraints

    >>> restraints.set_backend('blade')
    >>> restraints.set_backend('openmm')  # Raises if NOE is active
    """
    _state.set_backend(backend)


def get_backend() -> str:
    """Get the current compute backend.

    Returns
    -------
    str
        Current backend name ('standard', 'domdec', 'blade', or 'openmm')

    Examples
    --------
    >>> import pycharmm.restraints as restraints

    >>> backend = restraints.get_backend()
    >>> print(backend)
    'standard'
    """
    return _state.get_current_backend()


def reset_state() -> None:
    """Reset all restraint state tracking.

    Clears all Python-side state tracking. Does NOT clear restraints
    in CHARMM - use the specific turn_off/reset functions for that.

    Use this function when you need to resync the Python state with
    CHARMM state, e.g., after running CHARMM scripts that modify
    restraints directly.

    Examples
    --------
    >>> import pycharmm.restraints as restraints

    >>> restraints.reset_state()  # Clear Python state tracking
    """
    _state.reset()


# =============================================================================
# Convenience Aliases
# =============================================================================

# Dihedral restraint aliases (bidirectional consistency)
dihedral = dihe_restraint
dihe = dihe_restraint

def dihedral_clear():
    """Clear all CONS DIHE restraints. Alias for angles.dihedral_clear()."""
    angles.dihedral_clear()

# MMFP dihedral aliases
mmfp_dihe = mmfp_dihedral
mmfp_dihe_clear = mmfp_dihedral_clear


# =============================================================================
# Bond/Angle Helper Functions
# =============================================================================

def bond(force: float, exponent: int = 2, upper: bool = False):
    """Apply bond distance restraint via IC restraint facility.

    Convenience wrapper for ic_restraint(bond=force, ...).

    Parameters
    ----------
    force : float
        Force constant in kcal/mol/A^2.
    exponent : int, optional
        Exponent in restraint energy. Default 2.
    upper : bool, optional
        If True, use upper bound constraint. Default False.

    Examples
    --------
    >>> import pycharmm.restraints as restraints
    >>> restraints.bond(force=100.0)
    """
    return ic_restraint(bond=force, exponent=exponent, upper=upper)


def angle(force: float, exponent: int = 2, upper: bool = False):
    """Apply bond angle restraint via IC restraint facility.

    Convenience wrapper for ic_restraint(angle=force, ...).

    Parameters
    ----------
    force : float
        Force constant in kcal/mol/rad^2.
    exponent : int, optional
        Exponent in restraint energy. Default 2.
    upper : bool, optional
        If True, use upper bound constraint. Default False.

    Examples
    --------
    >>> import pycharmm.restraints as restraints
    >>> restraints.angle(force=50.0)
    """
    return ic_restraint(angle=force, exponent=exponent, upper=upper)
