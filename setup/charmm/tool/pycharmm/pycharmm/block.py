# pycharmm: molecular dynamics in python with CHARMM
# block module - CHARMM BLOCK facility interface
# Copyright (C) 2025 Stanislav Cherepanov

# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

"""Functions for CHARMM BLOCK facility for energy partitioning and free energy.

Corresponds to CHARMM command `BLOCK`
See https://academiccharmm.org/documentation/latest/block

The BLOCK facility partitions a molecular system into blocks and scales
interaction energies (and forces) between them. Primary applications:

- Free energy simulations with component analysis
- Lambda-dynamics for efficient alchemical free energy calculations
- Multi-site lambda dynamics (MSLD)
- Energy partitioning as alternative to INTERACTION command

Classes
=======
- `Block` -- Context manager for BLOCK facility setup

Core Functions
==============
- `initialize` -- Initialize BLOCK facility with specified number of blocks
- `get_nrep` -- Get number of replicas for replica exchange
- `call` -- Assign atoms to a block
- `unassign_block` -- Remove atoms from a block (auto-updates tracking)
- `reassign_block` -- Replace block assignment (auto-handles previous)
- `coef` -- Set interaction coefficient between blocks
- `set_lambda` -- Set lambda value (3-block system convenience)
- `clear` -- Clear all BLOCK facility data
- `end` -- Exit BLOCK facility
- `set_force` -- Enable or disable force calculations
- `get_force_enabled` -- Query force calculation setting
- `add_exclusion` -- Create exclusions between blocks
- `add_exclusion_extend` / `adexcl` -- Add to existing exclusions
- `get_exclusions` -- Query current exclusions

State Query Functions
=====================
- `get_nblocks` -- Get current number of blocks
- `is_active` -- Check if BLOCK facility is active
- `get_coefficients` -- Get coefficient matrix
- `get_coefficient` -- Get specific coefficient value
- `get_block_assignments` -- Get all block assignments
- `get_assigned_atoms` -- Get all assigned atom indices
- `get_state` -- Get complete current state
- `get_lambda` -- Get current lambda value

Free Energy Functions
=====================
- `free_energy` -- Exponential formula free energy evaluation
- `energy_average` -- Thermodynamic integration (dV/dlambda)
- `component_analysis` -- Component-wise free energy analysis

Lambda-Dynamics Functions
=========================
- `enable_lambda_dynamics` -- Enable lambda-dynamics
- `disable_lambda_dynamics` -- Disable lambda-dynamics
- `get_lambda_dynamics_state` -- Query lambda-dynamics settings
- `ldin` -- Initialize lambda parameters for a block (supports pH-MD)
- `get_ldin_params` -- Query LDIN parameters
- `phmd_ph` -- Set pH for pH-dependent lambda dynamics
- `get_phmd_ph` -- Query current pH value
- `ldmatrix` -- Auto-populate coefficient matrix from lambda values
- `set_langevin` -- Couple lambda to Langevin thermostat
- `disable_langevin` -- Disable Langevin coupling
- `get_langevin_state` -- Query Langevin settings
- `set_bias_count` -- Set number of biasing potentials (usually auto-managed)
- `add_bias` -- Add biasing potential (auto-indexes, auto-manages count)
- `remove_bias` -- Remove bias(es) with auto re-indexing
- `clear_biases` -- Remove all biases
- `get_biases` -- Query current biases
- `rmla` -- Remove lambda scaling for specific energy terms
- `restart_ld` -- Restart lambda-dynamics
- `write_ld` -- Write lambda histogram output
- `restraining_potential` -- Set restraining potential for unbound states

MSLD Functions
==============
- `msld` -- Initialize multi-site lambda dynamics
- `msmatrix` -- Auto-populate MSLD coefficient matrix
- `assign_block_to_site` -- Assign a block to a site
- `theta_bias` -- Set theta-biasing for MSLD
- `get_msld_state` -- Query MSLD settings

Advanced Functions
==================
- `soft_core` -- MSLD soft core potentials
- `get_soft_core_state` -- Query soft core settings
- `soft_omm` -- CHARMM/OpenMM soft core for vdW
- `get_soft_omm_state` -- Query OpenMM soft core setting
- `pssp` -- Dual-topology soft core potentials
- `no_pssp` -- Turn off dual-topology soft core
- `get_pssp_state` -- Query PSSP settings
- `pmel` -- MSLD PME electrostatics handling
- `get_pmel_mode` -- Query PME mode
- `constrained_atom_scaling` -- Scaling of constrained atoms
- `get_constrained_atom_scaling` -- Query constrained atom scaling
- `define_constrained_atoms` -- Define constrained atom group
- `soft_bond_setup` -- Setup soft bonds
- `define_soft_bond` -- Define individual soft bond
- `hybrid_hamiltonian` -- Enable hybrid Hamiltonian
- `get_hybrid_hamiltonian_state` -- Query hybrid Hamiltonian settings
- `set_hybh_output` -- Set output unit for dE/dl
- `print_hybh` -- Print dE/dl terms
- `write_hybh` -- Write dE/dl to output unit
- `clear_hybh` -- Clear hybrid Hamiltonian data
- `enable_mc_md` -- Enable hybrid MC/MD
- `disable_mc_md` -- Disable hybrid MC/MD
- `get_mc_md_state` -- Query MC/MD settings
- `mc_intermediate` -- Define intermediate lambda states
- `mc_step_size` -- Define lambda step increment
- `mc_clear` -- Clear MC/MD data

Direct Memory Access Functions
==============================
These functions use ctypes to read directly from CHARMM memory,
providing faster access than command-based queries. They require
CHARMM to be compiled with the appropriate API support.

Dynamics Data Collection (api_lambdata, api_msldata):
- `is_direct_access_available` -- Check if direct access APIs are available
- `get_nblock_direct` -- Get number of blocks directly
- `get_nbiasv_direct` -- Get number of biases directly
- `get_lambda_squared_direct` -- Get lambda^2 values from dynamics
- `get_bias_data_direct` -- Get lambda-dynamics bias data
- `get_msld_blocks_direct` -- Get MSLD block data from dynamics
- `get_msld_bias_direct` -- Get MSLD bias data from dynamics
- `get_msld_step_direct` -- Get MSLD step data (nblocks, nsites, etc.)
- `get_msld_theta_direct` -- Get MSLD theta values from dynamics
- `get_msld_nsubs_direct` -- Get MSLD nsubs per site from dynamics
- `enable_lambda_data_collection` -- Enable lambda data collection
- `disable_lambda_data_collection` -- Disable lambda data collection
- `enable_msld_data_collection` -- Enable MSLD data collection
- `disable_msld_data_collection` -- Disable MSLD data collection

Configuration Data (api_blockdata) - Read/Write Access:
- `set_coefficient_direct` -- Set single coefficient value
- `get_lambda_values_direct` -- Get current lambda values
- `set_ldin_params_direct` -- Set LDIN parameters for a block
- `get_bias_params_direct` -- Get bias parameters
- `get_temperature_direct` -- Get lambda dynamics temperature
- `set_temperature_direct` -- Set lambda dynamics temperature
- `is_lambda_dynamics_enabled_direct` -- Check if QLDM enabled
- `is_theta_enabled_direct` -- Check if theta mode enabled
- `is_langevin_enabled_direct` -- Check if Langevin enabled
- `get_nsites_direct` -- Get number of MSLD sites
- `get_site_assignments_direct` -- Get site assignments per block
- `get_softcore_mode_direct` -- Get soft-core mode
- `get_pme_mode_direct` -- Get PME handling mode
- `get_fnex_direct` -- Get FNEX factor
- `get_ph_direct` -- Get pH value (constant-pH MD)
- `set_ph_direct` -- Set pH value (constant-pH MD)
- `sync_state_from_charmm` -- Synchronize Python state from CHARMM

Convenience Functions
=====================
- `setup_dual_topology` -- Quick setup for standard 3-block dual-topology FEP
- `get_lambda_schedule` -- Generate lambda values for FEP windows

Examples
========
>>> import pycharmm
>>> import pycharmm.block as block

Basic 3-block setup with context manager:
>>> with block.Block(3) as b:
...     b.call(2, reactant_selection)
...     b.call(3, product_selection)
...     b.set_lambda(0.5)

Equivalent without context manager:
>>> block.initialize(3)
>>> block.call(2, reactant_selection)
>>> block.call(3, product_selection)
>>> block.set_lambda(0.5)
>>> block.end()

Set individual coefficients:
>>> block.initialize(3)
>>> block.coef(1, 2, 0.8, elec=0.7, vdw=0.9)
>>> block.end()

Lambda-dynamics setup:
>>> with block.Block(4) as b:
...     b.call(2, 'site1sub1')
...     b.call(3, 'site1sub2')
...     b.call(4, 'site1sub3')
...     block.enable_lambda_dynamics(theta=True)
...     block.set_langevin(temp=310.0)
...     block.ldin(1, 1.0, 0.0, 12.0, 0.0, 5.0)
...     block.ldin(2, 0.5, 0.0, 12.0, 0.0, 5.0)
...     block.ldin(3, 0.3, 0.0, 12.0, 3.2, 5.0)
...     block.ldin(4, 0.2, 0.0, 12.0, -1.0, 5.0)
...     block.rmla('bond', 'theta')
...     block.ldmatrix()

Query current state:
>>> coeffs = block.get_coefficients()
>>> assignments = block.get_block_assignments()
>>> print(f"Number of blocks: {block.get_nblocks()}")
"""

import ctypes
from contextlib import contextmanager

import numpy as np
import pandas as pd

import pycharmm.lingo as lingo
import pycharmm.select_atoms as select_atoms
from pycharmm.loader import lib


# =============================================================================
# Internal State Tracking
# =============================================================================

class _BlockState:
    """Internal state tracker - mirrors CHARMM BLOCK state.

    This class maintains a Python-side copy of the BLOCK facility state
    for efficient queries without parsing CHARMM output.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        """Reset to initial state."""
        self.active = False
        self.nblocks = 0
        self.assignments = {}  # {block_id: {'selection_str': str, 'atom_indices': set}}
        self.assigned_atoms = set()  # All assigned atom indices (for overlap checking)
        self.coefficients = {}  # {(i,j): {'default': v, 'bond': v, ...}}
        self.lambda_value = None
        self.force_enabled = True
        self.exclusions = []

        # Command buffer for batching BLOCK commands
        # CHARMM's BLOCK facility is a sub-language that requires all commands
        # to be sent together: BLOCK n ... commands ... END
        self._command_buffer = []
        self._in_block_context = False  # Track if we've started a BLOCK session

        # Track if BLOCK commands have been flushed to CHARMM
        # True after first end() call - indicates CHARMM has received the setup
        self.charmm_initialized = False

        # Lambda-dynamics state
        self.lambda_dynamics = {
            'enabled': False,
            'theta': False,
            'langevin_temp': None,
            'ldin_params': {},  # {block_id: {'lambda_sq': v, 'vel': v, ...}}
            'biases': [],
            'bias_count': 0,
            'rmla_terms': set()
        }

        # MSLD state
        self.msld = {
            'enabled': False,
            'site_assignments': {},  # {block_id: site_id}
            'fnex': None,
            'fnxs': None,
            'functional_form': None
        }

        # Advanced features state
        self.soft_core = {'enabled': False, 'mode': None, 'w14': False}
        self.pssp_state = {'enabled': False, 'alam': None, 'dlam': None}
        self.hybh = {'enabled': False, 'lambda': None, 'output_unit': None}
        self.mc_md = {'enabled': False, 'params': {}}

        # Additional state variables (previously added dynamically)
        self.nrep = None
        self.phmd_ph = None
        self.langevin_enabled = False
        self.soft_omm_enabled = False
        self.pmel_mode = None
        self.scat_state = {'enabled': False, 'mode': None, 'k': None}

    def to_dict(self):
        """Export state as dictionary."""
        return {
            'active': self.active,
            'nblocks': self.nblocks,
            'nrep': self.nrep,
            'assignments': {k: {'selection_str': v['selection_str'],
                               'atom_indices': list(v['atom_indices'])}
                           for k, v in self.assignments.items()},
            'assigned_atoms': list(self.assigned_atoms),
            'coefficients': self.coefficients.copy(),
            'lambda_value': self.lambda_value,
            'force_enabled': self.force_enabled,
            'exclusions': self.exclusions.copy(),
            'lambda_dynamics': self.lambda_dynamics.copy(),
            'msld': self.msld.copy(),
            'soft_core': self.soft_core.copy(),
            'pssp': self.pssp_state.copy(),
            'hybh': self.hybh.copy(),
            'mc_md': self.mc_md.copy(),
            'phmd_ph': self.phmd_ph,
            'langevin_enabled': self.langevin_enabled,
            'soft_omm_enabled': self.soft_omm_enabled,
            'pmel_mode': self.pmel_mode,
            'scat_state': self.scat_state.copy()
        }

    def initialize_coefficients(self, nblocks):
        """Initialize coefficient matrix to default values (1.0)."""
        self.coefficients = {}
        for i in range(1, nblocks + 1):
            for j in range(i, nblocks + 1):
                self.coefficients[(i, j)] = {
                    'default': 1.0,
                    'bond': None,
                    'angl': None,
                    'dihe': None,
                    'elec': None,
                    'vdw': None,
                    'vdwa': None,
                    'vdwr': None
                }


# Module-level singleton
_state = _BlockState()


# =============================================================================
# Helper Functions
# =============================================================================

def _get_selection_object(selection):
    """Convert selection to SelectAtoms object if possible.

    Parameters
    ----------
    selection : SelectAtoms, str, or tuple
        Atom selection specification

    Returns
    -------
    SelectAtoms or None
        SelectAtoms object if available, None for string selections
        (CHARMM will validate string selections directly)
    """
    if isinstance(selection, select_atoms.SelectAtoms):
        return selection
    elif isinstance(selection, str):
        # For string selections, we cannot easily get atom indices
        # without running through CHARMM. Return None and let CHARMM
        # handle validation when the CALL command is executed.
        return None
    elif hasattr(selection, 'get_selection'):
        sel = selection.get_selection()
        return select_atoms.SelectAtoms(selection=sel)
    else:
        raise TypeError(f"Invalid selection type: {type(selection)}")


def _generate_selection_name():
    """Generate a unique name for a stored selection."""
    import random
    import string
    # Generate a random 8-character uppercase name with BLCK prefix
    rand_part = ''.join(random.choice(string.ascii_uppercase)
                        for _ in range(4))
    return f'BLCK{rand_part}'


def _build_selection_string(selection):
    """Convert selection to CHARMM selection string for BLOCK commands.

    Parameters
    ----------
    selection : SelectAtoms, str, or tuple
        Atom selection specification

    Returns
    -------
    str
        CHARMM-compatible selection string using stored selection name

    Note
    ----
    For BLOCK facility commands, selections must be stored first and
    referenced by name. Inline selections with the 'end' keyword cause
    conflicts with BLOCK's END command parsing. This function stores
    the selection immediately (outside BLOCK context) and returns a
    reference to the stored name.
    """
    if isinstance(selection, str):
        # For string selections, use inline syntax with uppercase SELECT/END
        # to match common BLOCK command patterns
        sel_str = selection.strip()

        # Strip existing sele/select prefix and end suffix if present
        sel_lower = sel_str.lower()
        if sel_lower.startswith('sele'):
            sel_str = sel_str[4:].strip()
        elif sel_lower.startswith('select'):
            sel_str = sel_str[6:].strip()
        if sel_str.lower().endswith('end'):
            sel_str = sel_str[:-3].strip()

        # Use uppercase SELECT/END for consistency with BLOCK command patterns
        return f'SELECT {sel_str} END'
    elif isinstance(selection, select_atoms.SelectAtoms):
        # SelectAtoms object - get stored name or store it
        if not selection.is_stored():
            selection.store()
        # Use uppercase for consistency with BLOCK command patterns
        return f'SELECT {selection.get_stored_name()} END'
    elif hasattr(selection, 'get_selection'):
        # Some other selection-like object
        sel = selection.get_selection()
        sel_obj = select_atoms.SelectAtoms(selection=sel)
        sel_obj.store()
        # Use uppercase for consistency with BLOCK command patterns
        return f'SELECT {sel_obj.get_stored_name()} END'
    else:
        raise TypeError(f"Invalid selection type: {type(selection)}")


def _execute_block_command(cmd):
    """Accumulate a command for the BLOCK context.

    Commands are buffered and sent together when end() is called.
    This is required because CHARMM's BLOCK facility is a sub-language
    that enters an interactive mode after 'BLOCK n' until 'END' is sent.
    """
    _state._command_buffer.append(cmd)


def _flush_command_buffer():
    """Execute all accumulated BLOCK commands as a single batch.

    This sends all buffered commands (including the initial BLOCK n
    and final END) to CHARMM as a single script.
    """
    if not _state._command_buffer:
        return

    # Join all commands with newlines and execute as single script
    script = '\n'.join(_state._command_buffer)
    _state._command_buffer = []
    _state._in_block_context = False
    lingo.charmm_script(script)


# =============================================================================
# Context Manager Class
# =============================================================================

class Block:
    """Context manager for BLOCK facility setup.

    This class provides a convenient way to set up the BLOCK facility
    with automatic cleanup (END command) when exiting the context.

    Parameters
    ----------
    nblocks : int, optional
        Number of blocks to allocate (default: 3)
    nrep : int, optional
        Number of replicas for replica exchange (pH-REX, etc.)

    Examples
    --------
    >>> with block.Block(3) as b:
    ...     b.call(2, reactant_sel)
    ...     b.call(3, product_sel)
    ...     b.set_lambda(0.5)
    ... # END called automatically

    >>> with block.Block(4) as b:
    ...     b.call(2, 'segid LIG1')
    ...     b.call(3, 'segid LIG2')
    ...     b.call(4, 'segid LIG3')
    ...     b.coef(1, 2, 0.8)
    ...     b.coef(1, 3, 0.2)

    >>> # For pH-REX with 8 replicas
    >>> with block.Block(31, nrep=8) as b:
    ...     # Setup titratable residues...
    """

    def __init__(self, nblocks=3, nrep=None):
        self.nblocks = nblocks
        self.nrep = nrep
        self._entered = False

    def __enter__(self):
        initialize(self.nblocks, nrep=self.nrep)
        self._entered = True
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._entered:
            end()
            self._entered = False
        return False

    # Delegate methods to module functions
    def call(self, block_id, selection):
        """Assign atoms to a block."""
        return call(block_id, selection)

    def coef(self, i, j, value, **kwargs):
        """Set interaction coefficient between blocks."""
        return coef(i, j, value, **kwargs)

    def set_lambda(self, value):
        """Set lambda value (3-block system convenience)."""
        return set_lambda(value)

    def set_force(self, enabled=True):
        """Enable or disable force calculations."""
        return set_force(enabled)

    def clear(self):
        """Clear all BLOCK facility data."""
        return clear()


# =============================================================================
# Core Functions
# =============================================================================

def initialize(nblocks=3, nrep=None):
    """Initialize BLOCK facility with specified number of blocks.

    Maps to CHARMM command: BLOCK [int] [NREP int]

    Parameters
    ----------
    nblocks : int, optional
        Number of blocks to allocate (default: 3)
    nrep : int, optional
        Number of replicas for replica exchange simulations.
        Used in pH-REX (replica exchange with pH) and related methods.
        If None, replica exchange is not enabled.

    Notes
    -----
    The number of blocks is only read on the initial BLOCK call.
    Subsequent calls will enter the BLOCK facility but not change
    the number of blocks.

    Examples
    --------
    >>> block.initialize(3)              # Standard 3-block setup
    >>> block.initialize(31, nrep=8)     # pH-REX with 8 replicas

    See Also
    --------
    get_nblocks : Query number of blocks
    get_nrep : Query number of replicas
    """
    # Clear any previous command buffer and start fresh
    _state._command_buffer = []
    _state._in_block_context = True

    _state.nblocks = nblocks
    _state.active = True
    _state.initialize_coefficients(nblocks)
    _state.nrep = nrep

    cmd = f'BLOCK {nblocks}'
    if nrep is not None:
        cmd += f' NREP {nrep}'
    _execute_block_command(cmd)


def get_nrep():
    """Get number of replicas for replica exchange.

    Returns
    -------
    int or None
        Number of replicas, or None if not using replica exchange
    """
    return getattr(_state, 'nrep', None)


def call(block_id, selection):
    """Assign atoms to a block.

    Maps to CHARMM command: CALL int atom-selection

    Parameters
    ----------
    block_id : int
        Block number (1 to nblocks). Block 1 is typically the environment.
    selection : SelectAtoms, str, or selection-like
        Atom selection specifying which atoms to assign

    Raises
    ------
    ValueError
        If block_id is out of range, BLOCK not initialized, selection is empty,
        or atoms are already assigned to another block
    """
    if not _state.active:
        raise ValueError("BLOCK facility not initialized. Call initialize() first.")

    if block_id < 1 or block_id > _state.nblocks:
        raise ValueError(f"block_id must be between 1 and {_state.nblocks}")

    # Try to get selection as SelectAtoms object for validation
    sel_obj = _get_selection_object(selection)

    if sel_obj is not None:
        # SelectAtoms object - can validate fully
        atom_indices = set(sel_obj.get_atom_indexes())
        n_selected = sel_obj.get_n_selected()

        # Validate selection is not empty
        if n_selected == 0:
            raise ValueError("Selection is empty. At least one atom must be selected.")

        # Check for overlap with previously assigned atoms
        overlap = atom_indices & _state.assigned_atoms
        if overlap:
            # Find which block(s) have the overlapping atoms
            overlap_info = []
            for other_block_id, assignment in _state.assignments.items():
                other_overlap = atom_indices & assignment['atom_indices']
                if other_overlap:
                    overlap_info.append(f"block {other_block_id} ({len(other_overlap)} atoms)")
            raise ValueError(
                f"Selection contains {len(overlap)} atom(s) already assigned to: "
                f"{', '.join(overlap_info)}. Each atom can only be assigned to one block."
            )

        # Store assignment in state with atom indices
        _state.assignments[block_id] = {
            'selection_str': str(selection),
            'atom_indices': atom_indices
        }
        _state.assigned_atoms.update(atom_indices)
    else:
        # String selection - CHARMM will validate
        # Store assignment without atom indices (limited tracking)
        _state.assignments[block_id] = {
            'selection_str': str(selection),
            'atom_indices': set()  # Unknown for string selections
        }

    # Build selection string for CHARMM command
    sel_str = _build_selection_string(selection)

    cmd = f'CALL {block_id} {sel_str}'
    _execute_block_command(cmd)


def unassign_block(block_id):
    """Remove atoms from a block assignment.

    This function clears the atom assignment for a specific block,
    updating the Python state tracking. The atoms become available
    for assignment to other blocks.

    Note: In CHARMM, calling CALL with 'none' or an empty selection
    effectively clears a block. This function uses 'sele none end'.

    Parameters
    ----------
    block_id : int
        Block number to unassign

    Returns
    -------
    dict or None
        The previous assignment info if block was assigned, None otherwise.
        Contains 'selection_str' and 'atom_indices'.

    Examples
    --------
    >>> block.call(2, 'segid LIG1')
    >>> block.call(3, 'segid LIG2')
    >>> old = block.unassign_block(2)  # Remove LIG1 from block 2
    >>> print(old['selection_str'])     # 'segid LIG1'
    >>> block.call(2, 'segid LIG3')     # Now assign LIG3 to block 2

    See Also
    --------
    call : Assign atoms to a block
    get_block_assignments : Query all assignments
    """
    if not _state.active:
        raise ValueError("BLOCK facility not initialized. Call initialize() first.")

    if block_id < 1 or block_id > _state.nblocks:
        raise ValueError(f"block_id must be between 1 and {_state.nblocks}")

    # Get previous assignment if any
    previous = None
    if block_id in _state.assignments:
        previous = _state.assignments[block_id].copy()
        # Remove atoms from global tracking
        _state.assigned_atoms -= previous['atom_indices']
        # Remove assignment
        del _state.assignments[block_id]

    # Issue CALL with empty selection to CHARMM
    _execute_block_command(f'CALL {block_id} sele none end')

    return previous


def reassign_block(block_id, selection):
    """Reassign atoms to a block, automatically handling previous assignment.

    This is a convenience function that combines unassign_block() and call().
    It properly removes the previous assignment's atoms from tracking before
    assigning new atoms.

    Parameters
    ----------
    block_id : int
        Block number to reassign
    selection : SelectAtoms, str, or selection-like
        New atom selection

    Returns
    -------
    dict or None
        The previous assignment info if block was previously assigned

    Examples
    --------
    >>> block.call(2, 'segid LIG1')
    >>> old = block.reassign_block(2, 'segid LIG2')  # Replace LIG1 with LIG2
    >>> print(old['selection_str'])  # 'segid LIG1'

    See Also
    --------
    call : Assign atoms to a block
    unassign_block : Remove block assignment
    """
    previous = unassign_block(block_id)
    call(block_id, selection)
    return previous


def coef(i, j, value, *, bond=None, angl=None, dihe=None,
         elec=None, vdw=None, vdwa=None, vdwr=None):
    """Set interaction coefficient between blocks.

    Maps to CHARMM command: COEF int int real [BOND real] [ANGL real] ...

    Parameters
    ----------
    i, j : int
        Block indices (order doesn't matter, matrix is symmetric)
    value : float
        Default coefficient for all energy terms
    bond : float, optional
        Bond energy scaling coefficient
    angl : float, optional
        Angle energy scaling coefficient
    dihe : float, optional
        Dihedral energy scaling coefficient
    elec : float, optional
        Electrostatic energy scaling coefficient
    vdw : float, optional
        Van der Waals energy scaling coefficient
    vdwa : float, optional
        VdW attractive (C6/r^6) scaling coefficient
    vdwr : float, optional
        VdW repulsive (C12/r^12) scaling coefficient

    Notes
    -----
    The coefficient matrix is symmetric: COEF i j == COEF j i
    """
    if not _state.active:
        raise ValueError("BLOCK facility not initialized. Call initialize() first.")

    # Ensure i <= j for consistent storage
    if i > j:
        i, j = j, i

    # Update state
    key = (i, j)
    if key not in _state.coefficients:
        _state.coefficients[key] = {}

    _state.coefficients[key]['default'] = value
    if bond is not None:
        _state.coefficients[key]['bond'] = bond
    if angl is not None:
        _state.coefficients[key]['angl'] = angl
    if dihe is not None:
        _state.coefficients[key]['dihe'] = dihe
    if elec is not None:
        _state.coefficients[key]['elec'] = elec
    if vdw is not None:
        _state.coefficients[key]['vdw'] = vdw
    if vdwa is not None:
        _state.coefficients[key]['vdwa'] = vdwa
    if vdwr is not None:
        _state.coefficients[key]['vdwr'] = vdwr

    # Build command
    cmd = f'COEF {i} {j} {value}'
    if bond is not None:
        cmd += f' BOND {bond}'
    if angl is not None:
        cmd += f' ANGL {angl}'
    if dihe is not None:
        cmd += f' DIHE {dihe}'
    if elec is not None:
        cmd += f' ELEC {elec}'
    if vdw is not None:
        cmd += f' VDW {vdw}'
    if vdwa is not None:
        cmd += f' VDWA {vdwa}'
    if vdwr is not None:
        cmd += f' VDWR {vdwr}'

    _execute_block_command(cmd)


def set_lambda(value):
    """Set lambda value (3-block system convenience).

    Maps to CHARMM command: LAMBda real

    This is a convenience function for the standard 3-block dual-topology
    setup where:
    - Block 1 = Environment
    - Block 2 = Reactant
    - Block 3 = Product

    The coefficient matrix is set as:
    - COEF 1 1 = 1.0 (env-env)
    - COEF 1 2 = 1-lambda (env-reactant)
    - COEF 1 3 = lambda (env-product)
    - COEF 2 2 = 1-lambda (reactant-reactant)
    - COEF 2 3 = 0.0 (reactant-product)
    - COEF 3 3 = lambda (product-product)

    Parameters
    ----------
    value : float
        Lambda value between 0.0 (reactant) and 1.0 (product)

    Raises
    ------
    ValueError
        If value is not between 0 and 1, or if not using 3 blocks
    """
    if not _state.active:
        raise ValueError("BLOCK facility not initialized. Call initialize() first.")

    if not 0.0 <= value <= 1.0:
        raise ValueError("Lambda value must be between 0.0 and 1.0")

    _state.lambda_value = value

    # Update coefficient state for 3-block system
    if _state.nblocks >= 3:
        _state.coefficients[(1, 1)] = {'default': 1.0}
        _state.coefficients[(1, 2)] = {'default': 1.0 - value}
        _state.coefficients[(1, 3)] = {'default': value}
        _state.coefficients[(2, 2)] = {'default': 1.0 - value}
        _state.coefficients[(2, 3)] = {'default': 0.0}
        _state.coefficients[(3, 3)] = {'default': value}

    cmd = f'LAMBDA {value}'
    _execute_block_command(cmd)


def clear():
    """Clear all BLOCK facility data.

    Maps to CHARMM command: CLEAr

    This resets the BLOCK facility to its initial state,
    removing all block assignments and coefficients.

    Note: This sends a complete BLOCK...CLEAR...END sequence
    to CHARMM since CLEAR is a BLOCK sub-command.
    """
    nblocks = _state.nblocks if _state.nblocks > 0 else 3
    _state.reset()

    # Send a complete BLOCK...CLEAR...END sequence
    script = f'BLOCK {nblocks}\nCLEAR\nEND'
    lingo.charmm_script(script)


def end():
    """Exit BLOCK facility and print coefficient matrix.

    Maps to CHARMM command: END

    This must be called to finalize the BLOCK setup and exit
    the BLOCK command context. The coefficient matrix will be
    printed to the output.

    Note: This function flushes all accumulated BLOCK commands
    to CHARMM as a single batch, including the initial BLOCK n
    command and all subsequent commands.
    """
    _execute_block_command('END')
    _flush_command_buffer()
    # Keep state.active as True since BLOCK is now configured
    # _state.active = False  # Don't reset - BLOCK is now active in CHARMM
    _state.charmm_initialized = True


@contextmanager
def modify():
    """Re-enter BLOCK to modify parameters after end().

    Uses CHARMM's native re-entry capability - arrays remain allocated
    after END, so BLOCK n re-enters the existing BLOCK state rather than
    reinitializing. This allows modifying coefficients, lambda parameters,
    and other settings without full re-initialization.

    Yields
    ------
    None
        Yields control back to allow BLOCK commands within the context

    Raises
    ------
    ValueError
        If BLOCK was not initialized (call initialize() first)

    Examples
    --------
    >>> # Initial BLOCK setup
    >>> block.initialize(3)
    >>> block.coef(1, 2, 0.5)
    >>> block.end()
    >>>
    >>> # Later, modify a coefficient
    >>> with block.modify():
    ...     block.coef(1, 2, 0.8)  # Change coefficient
    ...     block.coef(2, 3, 0.3)  # Add another

    Notes
    -----
    After END (not CLEAR), CHARMM keeps BLOCK arrays allocated and QBLOCK=TRUE.
    Re-entering with "BLOCK n" allows modifying existing state. This context
    manager handles the BLOCK/END wrapping automatically.

    The context manager buffers commands and flushes them on exit, just like
    the normal initialize/end workflow.

    See Also
    --------
    initialize : Initial BLOCK setup
    end : Exit BLOCK context
    clear : Full BLOCK reset (deallocates arrays)
    """
    if not _state.active:
        raise ValueError("BLOCK not initialized. Call initialize() first.")

    if _state.nblocks <= 0:
        raise ValueError("No blocks defined. Initialize BLOCK first.")

    # Save current context state
    old_in_context = _state._in_block_context

    # Start fresh buffer for modifications - re-enter BLOCK
    _state._command_buffer = [f'BLOCK {_state.nblocks}']
    _state._in_block_context = True

    try:
        yield
        # Flush modifications with END
        _execute_block_command('END')
        _flush_command_buffer()
    finally:
        # Restore context state (buffer is now empty after flush)
        _state._in_block_context = old_in_context
        # Don't restore old buffer - it should have been flushed in initial end()


def set_force(enabled=True):
    """Enable or disable force calculations for BLOCK.

    Maps to CHARMM commands: FORCe / NOFOrce

    Controls whether forces are calculated when using the BLOCK facility.
    Disabling forces can speed up energy calculations when forces are
    not needed (e.g., post-processing trajectories).

    Parameters
    ----------
    enabled : bool, optional
        - True: Enable force calculations (default)
                Required for dynamics/minimization
        - False: Disable force calculations
                Faster for energy-only post-processing

    Examples
    --------
    >>> block.set_force(True)   # Enable forces (for dynamics)
    >>> block.set_force(False)  # Disable forces (for post-processing)

    See Also
    --------
    get_force_enabled : Query current force setting
    """
    if not _state.active:
        raise ValueError("BLOCK facility not initialized. Call initialize() first.")

    _state.force_enabled = enabled

    if enabled:
        _execute_block_command('FORCE')
    else:
        _execute_block_command('NOFORCE')


def get_force_enabled():
    """Query whether force calculations are enabled.

    Returns
    -------
    bool
        True if forces are calculated

    Examples
    --------
    >>> block.set_force(False)
    >>> print(block.get_force_enabled())  # False
    """
    return _state.force_enabled


# =============================================================================
# State Query Functions
# =============================================================================

def get_nblocks(direct=True):
    """Get current number of blocks.

    Parameters
    ----------
    direct : bool, optional
        If True (default), read from CHARMM internal state when available.
        If False, read from Python cache only.

    Returns
    -------
    int
        Number of blocks in the BLOCK facility

    Notes
    -----
    When Python hasn't initialized BLOCK yet (_state.nblocks == 0),
    returns 0 regardless of CHARMM's state. This ensures consistent
    behavior from the user's perspective.
    """
    # If Python says no blocks initialized, trust Python
    # (avoids returning stale CHARMM state from previous operations)
    if _state.nblocks == 0:
        return 0

    if direct:
        charmm_val = _get_nblock_from_charmm_internal()
        if charmm_val is not None:
            return charmm_val
        # Fall back to cache if CHARMM API unavailable
    return _state.nblocks


def is_active(direct=True):
    """Check if BLOCK facility is active.

    Parameters
    ----------
    direct : bool, optional
        If True (default), check CHARMM's QBLOCK flag when available.
        If False, check Python state cache only.

    Returns
    -------
    bool
        True if BLOCK facility is currently active

    Notes
    -----
    When commands are being buffered (between initialize() and end()),
    Python state returns True even though CHARMM doesn't know about
    the BLOCK yet. In this case, Python state takes precedence.
    """
    # If Python says we're in buffering mode, trust Python
    # (CHARMM won't know until end() is called)
    if _state.active and _state._in_block_context:
        return True

    if direct:
        charmm_val = _is_block_active_in_charmm_internal()
        if charmm_val is not None:
            return charmm_val
        # Fall back to cache if CHARMM API unavailable
    return _state.active


def get_coefficients():
    """Get coefficient matrix as pandas DataFrame.

    Returns
    -------
    pd.DataFrame
        Symmetric matrix with block pairs as index/columns.
        Each cell contains a dict with 'default' and optional
        per-term coefficients.
    """
    if _state.nblocks == 0:
        return pd.DataFrame()

    # Create symmetric matrix
    n = _state.nblocks
    data = {}
    for i in range(1, n + 1):
        row = {}
        for j in range(1, n + 1):
            key = (min(i, j), max(i, j))
            if key in _state.coefficients:
                row[j] = _state.coefficients[key].get('default', 1.0)
            else:
                row[j] = 1.0
        data[i] = row

    return pd.DataFrame(data).T


def get_coefficient_matrix(direct=True):
    """Get full coefficient matrix as numpy array.

    Parameters
    ----------
    direct : bool, optional
        If True (default), read from CHARMM's BLCOEP array.
        If False, read from Python cache.

    Returns
    -------
    numpy.ndarray or None
        2D symmetric coefficient matrix (nblock x nblock), or None if not available
    """
    if direct:
        charmm_matrix = _get_coefficient_matrix_from_charmm_internal()
        if charmm_matrix is not None:
            return charmm_matrix
        # Fall back to cache if CHARMM API unavailable

    return _get_coefficient_matrix_cached()


def get_coefficient(i, j, term=None, direct=True):
    """Get specific coefficient value.

    Parameters
    ----------
    i, j : int
        Block indices (1-based)
    term : str, optional
        Specific term ('bond', 'elec', 'vdw', etc.) or None for default.
        Note: term-specific coefficients are only available from Python cache.
    direct : bool, optional
        If True (default), read from CHARMM's BLCOEP array (default coefficient only).
        If False, read from Python cache.

    Returns
    -------
    float or None
        Coefficient value, or None if not set
    """
    # Direct CHARMM access only supports default coefficient (no per-term)
    if direct and term is None:
        charmm_val = _get_coefficient_from_charmm_internal(i, j)
        if charmm_val is not None:
            return charmm_val
        # Fall back to cache if CHARMM API unavailable

    # Python cache lookup
    key = (min(i, j), max(i, j))
    if key not in _state.coefficients:
        return 1.0 if term is None else None

    coef_dict = _state.coefficients[key]
    if term is None:
        return coef_dict.get('default', 1.0)
    return coef_dict.get(term)


def get_block_assignments():
    """Get all block assignments.

    Returns
    -------
    dict
        {block_id: {'selection_str': str, 'atom_indices': set, 'n_atoms': int}}
        mapping of block IDs to their assignment information
    """
    result = {}
    for block_id, assignment in _state.assignments.items():
        result[block_id] = {
            'selection_str': assignment['selection_str'],
            'atom_indices': assignment['atom_indices'].copy(),
            'n_atoms': len(assignment['atom_indices'])
        }
    return result


def get_assigned_atoms():
    """Get all assigned atom indices.

    Returns
    -------
    set
        Set of all atom indices that have been assigned to blocks
    """
    return _state.assigned_atoms.copy()


def get_state():
    """Get complete current state as dictionary.

    Returns
    -------
    dict
        Complete state including assignments, coefficients,
        lambda-dynamics settings, MSLD settings, etc.
    """
    return _state.to_dict()


def get_lambda():
    """Get current lambda value.

    Returns
    -------
    float or None
        Current lambda value, or None if not set
    """
    return _state.lambda_value


# =============================================================================
# Exclusion Functions
# =============================================================================

def add_exclusion(*pairs):
    """Create exclusions between blocks (replaces existing exclusions).

    Maps to CHARMM command: EXCLusion int int [int int] ...

    **WARNING**: This command OVERWRITES any previously defined exclusions.
    If you want to ADD to existing exclusions, use `add_exclusion_extend()`
    or `adexcl()` instead.

    Excluded block pairs have no nonbonded interactions between them.
    This is essential for MSLD/CpHMD where different protonation states
    of the same residue should not interact with each other.

    Parameters
    ----------
    *pairs : tuple of (int, int)
        Block pairs to exclude from nonbonded interactions

    Examples
    --------
    >>> # Set exclusions (replaces any existing)
    >>> block.add_exclusion((2, 3), (2, 4), (3, 4))

    >>> # For incremental addition, use adexcl instead:
    >>> block.adexcl(2, 3)
    >>> block.adexcl(2, 4)
    >>> block.adexcl(3, 4)

    See Also
    --------
    add_exclusion_extend : Add to existing exclusions (does not overwrite)
    adexcl : Shorthand alias for add_exclusion_extend
    get_exclusions : Query current exclusions
    """
    if not _state.active:
        raise ValueError("BLOCK facility not initialized. Call initialize() first.")

    # EXCL overwrites previous exclusions, so clear the state
    _state.exclusions = []

    cmd_parts = ['EXCLUSION']
    for pair in pairs:
        if len(pair) != 2:
            raise ValueError("Each exclusion must be a pair of block indices")
        cmd_parts.append(f'{pair[0]} {pair[1]}')
        _state.exclusions.append(pair)

    _execute_block_command(' '.join(cmd_parts))


def add_exclusion_extend(i, j):
    """Add to existing exclusions (extend exclusion list).

    Maps to CHARMM command: ADEXclusion int int

    Adds a block pair to the exclusion list. Excluded block pairs
    have no nonbonded interactions between them. This is essential
    for MSLD/CpHMD where different protonation states of the same
    residue should not interact with each other.

    Parameters
    ----------
    i, j : int
        Block pair to add to exclusions

    Examples
    --------
    >>> # Exclude interactions between 3 protonation states of same residue
    >>> block.add_exclusion_extend(2, 3)
    >>> block.add_exclusion_extend(2, 4)
    >>> block.add_exclusion_extend(3, 4)

    See Also
    --------
    add_exclusion : Add multiple exclusions at once
    adexcl : Shorthand alias for add_exclusion_extend
    get_exclusions : Query current exclusions
    """
    if not _state.active:
        raise ValueError("BLOCK facility not initialized. Call initialize() first.")

    _state.exclusions.append((i, j))
    _execute_block_command(f'ADEXCLUSION {i} {j}')


# Alias for add_exclusion_extend (matches CHARMM shorthand)
adexcl = add_exclusion_extend


def get_exclusions():
    """Query current block exclusions.

    Returns
    -------
    list
        List of (i, j) tuples representing excluded block pairs

    Examples
    --------
    >>> block.adexcl(2, 3)
    >>> block.adexcl(2, 4)
    >>> print(block.get_exclusions())  # [(2, 3), (2, 4)]
    """
    return _state.exclusions.copy()


# =============================================================================
# Free Energy Analysis Functions
# =============================================================================

def free_energy(oldl, newl, first, *, nunit=1, begin=1, stop=-1, skip=1,
                temp=298.15, cont=0, ihbf=-1, inbf=-1, imgf=-1):
    """Exponential formula free energy evaluation.

    Maps to CHARMM command: FREE_energy_evaluation ...

    Calculates: dA = -kT * ln<exp(-(U_new - U_old)/kT)>

    Parameters
    ----------
    oldl : float
        Lambda value at which trajectory was generated
    newl : float
        Lambda value for which to calculate free energy
    first : int
        First Fortran unit number for trajectory file(s)
    nunit : int, optional
        Number of trajectory files (default: 1)
    begin : int, optional
        First frame to process (default: 1)
    stop : int, optional
        Last frame to process (default: -1 for all)
    skip : int, optional
        Process every nth frame (default: 1)
    temp : float, optional
        Temperature in Kelvin (default: 298.15)
    cont : int, optional
        Continuous output mode: +n for cumulative, -n for binned
    ihbf, inbf, imgf : int, optional
        Update frequencies for hbond, nonbond, and image lists
    """
    if not _state.active:
        raise ValueError("BLOCK facility not initialized. Call initialize() first.")

    cmd = f'FREE OLDL {oldl} NEWL {newl} FIRST {first}'
    if nunit != 1:
        cmd += f' NUNIT {nunit}'
    if begin != 1:
        cmd += f' BEGIN {begin}'
    if stop != -1:
        cmd += f' STOP {stop}'
    if skip != 1:
        cmd += f' SKIP {skip}'
    cmd += f' TEMP {temp}'
    if cont != 0:
        cmd += f' CONT {cont}'
    if ihbf != -1:
        cmd += f' IHBF {ihbf}'
    if inbf != -1:
        cmd += f' INBF {inbf}'
    if imgf != -1:
        cmd += f' IMGF {imgf}'

    _execute_block_command(cmd)


def energy_average(oldl, newl, first, *, nunit=1, begin=1, stop=-1, skip=1,
                   cont=0, ihbf=-1, inbf=-1, imgf=-1):
    """Thermodynamic integration (dV/dlambda) calculation.

    Maps to CHARMM command: Energy_AVeraGe ...

    Calculates ensemble averages of <U_new - U_old> for thermodynamic
    integration.

    Parameters
    ----------
    oldl : float
        Lambda value for reference state
    newl : float
        Lambda value for target state
    first : int
        First Fortran unit number for trajectory file(s)
    nunit : int, optional
        Number of trajectory files (default: 1)
    begin : int, optional
        First frame to process (default: 1)
    stop : int, optional
        Last frame to process (default: -1 for all)
    skip : int, optional
        Process every nth frame (default: 1)
    cont : int, optional
        Continuous output mode
    ihbf, inbf, imgf : int, optional
        Update frequencies
    """
    if not _state.active:
        raise ValueError("BLOCK facility not initialized. Call initialize() first.")

    cmd = f'EAVG OLDL {oldl} NEWL {newl} FIRST {first}'
    if nunit != 1:
        cmd += f' NUNIT {nunit}'
    if begin != 1:
        cmd += f' BEGIN {begin}'
    if stop != -1:
        cmd += f' STOP {stop}'
    if skip != 1:
        cmd += f' SKIP {skip}'
    if cont != 0:
        cmd += f' CONT {cont}'
    if ihbf != -1:
        cmd += f' IHBF {ihbf}'
    if inbf != -1:
        cmd += f' INBF {inbf}'
    if imgf != -1:
        cmd += f' IMGF {imgf}'

    _execute_block_command(cmd)


def component_analysis(dell, ndel, first, *, nunit=1, begin=1, stop=-1, skip=1,
                       temp=298.15, ihbf=-1, inbf=-1, imgf=-1):
    """Component-wise free energy analysis.

    Maps to CHARMM command: COMPonent_analysis ...

    Evaluates dV/dlambda at lambda +/- k*dell for k=0,1,...,ndel
    using perturbation theory. Requires exactly 4 blocks.

    Parameters
    ----------
    dell : float
        Lambda increment for perturbation
    ndel : int
        Number of perturbation steps
    first : int
        First Fortran unit number for trajectory file(s)
    nunit : int, optional
        Number of trajectory files (default: 1)
    begin : int, optional
        First frame to process (default: 1)
    stop : int, optional
        Last frame to process (default: -1 for all)
    skip : int, optional
        Process every nth frame (default: 1)
    temp : float, optional
        Temperature in Kelvin (default: 298.15)
    ihbf, inbf, imgf : int, optional
        Update frequencies
    """
    if not _state.active:
        raise ValueError("BLOCK facility not initialized. Call initialize() first.")

    cmd = f'COMP DELL {dell} NDEL {ndel} FIRST {first}'
    if nunit != 1:
        cmd += f' NUNIT {nunit}'
    if begin != 1:
        cmd += f' BEGIN {begin}'
    if stop != -1:
        cmd += f' STOP {stop}'
    if skip != 1:
        cmd += f' SKIP {skip}'
    cmd += f' TEMP {temp}'
    if ihbf != -1:
        cmd += f' IHBF {ihbf}'
    if inbf != -1:
        cmd += f' INBF {inbf}'
    if imgf != -1:
        cmd += f' IMGF {imgf}'

    _execute_block_command(cmd)


# =============================================================================
# Lambda-Dynamics Functions
# =============================================================================

def enable_lambda_dynamics(theta=False):
    """Enable lambda-dynamics for enhanced sampling.

    Maps to CHARMM command: QLDM [THETa]

    Lambda-dynamics treats lambda as a dynamic variable that evolves
    during the simulation, allowing efficient sampling of alchemical
    space. This is more efficient than running separate simulations
    at fixed lambda values.

    Parameters
    ----------
    theta : bool, optional
        If True, use theta-dynamics variant (default: False).
        - False: Standard lambda-dynamics with lambda^2 functional form
        - True: Theta-dynamics with sin^2(theta) functional form,
                which provides smoother transitions and is often
                preferred for MSLD simulations.

    Examples
    --------
    >>> block.enable_lambda_dynamics()           # Standard lambda-dynamics
    >>> block.enable_lambda_dynamics(theta=True) # Theta-dynamics

    See Also
    --------
    disable_lambda_dynamics : Disable lambda-dynamics
    get_lambda_dynamics_state : Query current settings
    ldin : Initialize lambda parameters for a block
    ldmatrix : Auto-populate coefficient matrix
    set_langevin : Couple lambda to thermostat
    """
    if not _state.active:
        raise ValueError("BLOCK facility not initialized. Call initialize() first.")

    _state.lambda_dynamics['enabled'] = True
    _state.lambda_dynamics['theta'] = theta

    cmd = 'QLDM'
    if theta:
        cmd += ' THETA'

    _execute_block_command(cmd)


def disable_lambda_dynamics():
    """Disable lambda-dynamics.

    Maps to CHARMM command: NQLDM

    Disables lambda-dynamics, reverting to fixed lambda behavior.
    Use this to switch between dynamic and fixed lambda modes.

    Examples
    --------
    >>> block.enable_lambda_dynamics(theta=True)  # Enable
    >>> # ... run simulation ...
    >>> block.disable_lambda_dynamics()            # Disable

    See Also
    --------
    enable_lambda_dynamics : Enable lambda-dynamics
    get_lambda_dynamics_state : Query current settings
    """
    if not _state.active:
        raise ValueError("BLOCK facility not initialized. Call initialize() first.")

    _state.lambda_dynamics['enabled'] = False
    _state.lambda_dynamics['theta'] = False
    _execute_block_command('NQLDM')


def get_lambda_dynamics_state():
    """Query current lambda-dynamics settings.

    Returns
    -------
    dict
        Complete lambda-dynamics state including:
        - 'enabled': bool - whether lambda-dynamics is active
        - 'theta': bool - whether theta-dynamics is used
        - 'langevin_temp': float or None - Langevin thermostat temperature
        - 'ldin_params': dict - per-block lambda parameters
        - 'biases': list - biasing potentials
        - 'bias_count': int - number of biases
        - 'rmla_terms': set - terms excluded from lambda scaling

    Examples
    --------
    >>> block.enable_lambda_dynamics(theta=True)
    >>> state = block.get_lambda_dynamics_state()
    >>> print(state['enabled'])  # True
    >>> print(state['theta'])    # True
    """
    return _state.lambda_dynamics.copy()


def ldin(block_id, lambda_sq, velocity, mass, bias, force_const=None,
         friction=None, ph_mode=None, ph_value=None):
    """Initialize lambda parameters for a block.

    Maps to CHARMM command: LDINitialize int real real real real [real] [NONE|UNEG real|UPOS real]

    Parameters
    ----------
    block_id : int
        Block number
    lambda_sq : float
        Lambda squared value (lambda = sqrt(lambda_sq)).
        Initial lambda = sqrt(lambda_sq), so lambda_sq=1.0 gives lambda=1.0
    velocity : float
        Initial lambda velocity. Typically 0.0 for equilibrated start.
    mass : float
        Lambda mass (fictitious mass for lambda dynamics).
        - Typical values: 5.0-20.0 amu
        - Larger mass = slower lambda dynamics
        - Smaller mass = faster sampling but may affect stability
    bias : float
        Biasing potential value for enhancing sampling.
        - 0.0 = no bias
        - Positive values push lambda toward endpoints
    force_const : float, optional
        Deprecated alias for ``friction``. Kept for backward compatibility.
    friction : float, optional
        Per-block friction coefficient for Langevin lambda dynamics.
        Only used when Langevin dynamics is enabled via ``set_langevin()``.
        - CHARMM reads this as the 6th positional field of LDIN when
          Langevin lambda dynamics (ILALDM) is active.
        - If not provided or <= 0, CHARMM defaults to 50.0 kcal/mol/A^2/ps.
        - Use different values per block to control lambda fluctuation rates.
    ph_mode : str, optional
        pH-dependent lambda dynamics mode (for PHMD simulations).
        Used together with PHMD PH setting.
        - None: Standard lambda dynamics (default)
        - 'none': Explicitly disable pH coupling for this block
        - 'uneg': Unprotonated state is negatively charged
        - 'upos': Unprotonated state is positively charged
    ph_value : float, optional
        Reference pKa or pH-related value for pH-dependent modes.
        Required when ph_mode is 'uneg' or 'upos'.

    Examples
    --------
    >>> # Standard lambda dynamics initialization
    >>> block.ldin(2, 1.0, 0.0, 12.0, 0.0)

    >>> # With per-block friction for Langevin lambda dynamics
    >>> block.ldin(2, 1.0, 0.0, 12.0, 0.0, friction=50.0)

    >>> # With pH-MD for titratable residue (carboxylic acid)
    >>> block.ldin(2, 1.0, 0.0, 12.0, 0.0, friction=50.0,
    ...            ph_mode='uneg', ph_value=4.0)

    >>> # Backward-compatible: force_const is an alias for friction
    >>> block.ldin(2, 1.0, 0.0, 12.0, 0.0, force_const=50.0)

    See Also
    --------
    set_langevin : Enable Langevin lambda dynamics (required for friction)
    ldmatrix : Auto-populate coefficient matrix from LDIN values
    get_ldin_params : Query LDIN parameters for a block
    """
    if not _state.active:
        raise ValueError("BLOCK facility not initialized. Call initialize() first.")

    # Handle force_const/friction aliasing
    if force_const is not None and friction is not None:
        raise ValueError(
            "Cannot specify both 'force_const' and 'friction'. "
            "'force_const' is a deprecated alias for 'friction'."
        )
    if force_const is not None:
        friction = force_const

    # Validate pH mode parameters
    if ph_mode is not None:
        ph_mode_lower = ph_mode.lower()
        if ph_mode_lower not in {'none', 'uneg', 'upos'}:
            raise ValueError(f"Invalid ph_mode '{ph_mode}'. Valid: 'none', 'uneg', 'upos'")
        if ph_mode_lower in {'uneg', 'upos'} and ph_value is None:
            raise ValueError(f"ph_value is required when ph_mode is '{ph_mode}'")

    _state.lambda_dynamics['ldin_params'][block_id] = {
        'lambda_sq': lambda_sq,
        'velocity': velocity,
        'mass': mass,
        'bias': bias,
        'friction': friction,
        'ph_mode': ph_mode,
        'ph_value': ph_value
    }

    cmd = f'LDIN {block_id} {lambda_sq} {velocity} {mass} {bias}'
    if friction is not None:
        cmd += f' {friction}'

    # Add pH-MD flags
    if ph_mode is not None:
        ph_mode_upper = ph_mode.upper()
        if ph_mode_upper == 'NONE':
            cmd += ' NONE'
        elif ph_mode_upper == 'UNEG':
            cmd += f' UNEG {ph_value}'
        elif ph_mode_upper == 'UPOS':
            cmd += f' UPOS {ph_value}'

    _execute_block_command(cmd)


def get_ldin_params(block_id=None, direct=True):
    """Query LDIN parameters for one or all blocks.

    Parameters
    ----------
    block_id : int, optional
        Block number to query. If None, returns all blocks.
    direct : bool, optional
        If True (default), read from CHARMM memory (single block only).
        If False, read from Python cache.

    Returns
    -------
    dict
        If block_id specified: parameters for that block
        If block_id is None: {block_id: params} for all blocks

    Examples
    --------
    >>> block.ldin(2, 1.0, 0.0, 12.0, 0.0, 5.0)
    >>> params = block.get_ldin_params(2)
    >>> print(params['mass'])  # 12.0

    >>> all_params = block.get_ldin_params()
    >>> print(all_params.keys())  # dict_keys([2])
    """
    if block_id is not None:
        # Single block query
        if direct:
            charmm_val = _get_ldin_params_from_charmm_internal(block_id)
            if charmm_val is not None:
                return charmm_val
            # Fall back to cache if CHARMM API unavailable
        return _state.lambda_dynamics['ldin_params'].get(block_id, {}).copy()

    # All blocks query (direct reads each block from CHARMM)
    if direct:
        result = {}
        for bid in _state.lambda_dynamics['ldin_params'].keys():
            charmm_val = _get_ldin_params_from_charmm_internal(bid)
            if charmm_val is not None:
                result[bid] = charmm_val
            else:
                result[bid] = _state.lambda_dynamics['ldin_params'].get(bid, {}).copy()
        return result

    return {k: v.copy() for k, v in _state.lambda_dynamics['ldin_params'].items()}


def phmd_ph(ph_value):
    """Set pH value for pH-dependent lambda dynamics (PHMD).

    Maps to CHARMM command: PHMD PH real

    Sets the simulation pH for constant-pH molecular dynamics (CpHMD).
    This pH value is used together with the pKa values specified in
    ldin() via the ph_mode parameter.

    The driving force for protonation/deprotonation is:
    dG = 2.303 * RT * (pH - pKa)

    Parameters
    ----------
    ph_value : float
        Simulation pH value.
        - Typical range: 0.0-14.0
        - pH < pKa: favors protonated state
        - pH > pKa: favors deprotonated state

    Examples
    --------
    >>> # Set up pH-dependent lambda dynamics
    >>> block.enable_lambda_dynamics(theta=True)
    >>> block.phmd_ph(7.0)  # Simulate at pH 7.0
    >>> block.ldin(2, 1.0, 0.0, 12.0, 0.0, 5.0, ph_mode='uneg', ph_value=4.0)

    See Also
    --------
    ldin : Initialize lambda parameters with pH coupling
    get_phmd_ph : Query current pH value
    """
    if not _state.active:
        raise ValueError("BLOCK facility not initialized. Call initialize() first.")

    _state.phmd_ph = ph_value
    _execute_block_command(f'PHMD PH {ph_value}')


def get_phmd_ph():
    """Query current PHMD pH value.

    Returns
    -------
    float or None
        Current simulation pH, or None if not set

    Examples
    --------
    >>> block.phmd_ph(7.0)
    >>> print(block.get_phmd_ph())  # 7.0
    """
    return _state.phmd_ph


def ldmatrix():
    """Auto-populate coefficient matrix from LDIN values.

    Maps to CHARMM command: LDMAtrix

    This command automatically sets the coefficient matrix based on
    the lambda^2 values specified with LDIN.
    """
    if not _state.active:
        raise ValueError("BLOCK facility not initialized. Call initialize() first.")

    _execute_block_command('LDMATRIX')


def set_langevin(temp=None):
    """Couple lambda variable to Langevin thermostat.

    Maps to CHARMM command: LANG [TEMP real]

    In lambda-dynamics, the lambda variable evolves dynamically.
    This function couples lambda to a Langevin thermostat to maintain
    proper thermal distribution in lambda space.

    Parameters
    ----------
    temp : float, optional
        Temperature for Langevin dynamics in Kelvin.
        - If None: uses the simulation temperature
        - Typical values: 298.15 - 310.0 K
        - Higher temperatures increase lambda fluctuations

    Examples
    --------
    >>> block.set_langevin()           # Use system temperature
    >>> block.set_langevin(temp=310.0) # Set specific temperature
    >>> block.disable_langevin()       # Disable Langevin coupling

    See Also
    --------
    disable_langevin : Disable Langevin coupling
    get_langevin_state : Query current settings
    enable_lambda_dynamics : Enable lambda-dynamics (required first)
    """
    if not _state.active:
        raise ValueError("BLOCK facility not initialized. Call initialize() first.")

    _state.lambda_dynamics['langevin_temp'] = temp
    _state.langevin_enabled = True

    cmd = 'LANG'
    if temp is not None:
        cmd += f' TEMP {temp}'

    _execute_block_command(cmd)


def disable_langevin():
    """Disable Langevin coupling for lambda variable.

    Note: CHARMM does not have a NOLANG command. Langevin coupling is
    disabled by calling CLEAR or re-initializing BLOCK. This function
    only updates Python state tracking.

    To fully disable Langevin coupling during a session, use:
    - block.clear() to reset all BLOCK state, or
    - Re-initialize with block.initialize() without calling set_langevin()

    Examples
    --------
    >>> block.set_langevin(temp=310.0)  # Enable
    >>> # ... run simulation ...
    >>> block.disable_langevin()         # Update Python state only

    See Also
    --------
    set_langevin : Enable Langevin coupling
    get_langevin_state : Query current settings
    clear : Clear all BLOCK state including Langevin
    """
    if not _state.active:
        raise ValueError("BLOCK facility not initialized. Call initialize() first.")

    _state.lambda_dynamics['langevin_temp'] = None
    _state.langevin_enabled = False
    # Note: CHARMM has no NOLANG command - Langevin is disabled via CLEAR
    # Only updating Python state here


def get_langevin_state():
    """Query current Langevin thermostat settings for lambda.

    Returns
    -------
    dict
        {'enabled': bool, 'temp': float or None}

    Examples
    --------
    >>> block.set_langevin(temp=310.0)
    >>> state = block.get_langevin_state()
    >>> print(state['enabled'])  # True
    >>> print(state['temp'])     # 310.0
    """
    return {
        'enabled': _state.langevin_enabled,
        'temp': _state.lambda_dynamics.get('langevin_temp')
    }


def set_bias_count(n):
    """Set number of biasing potentials.

    Maps to CHARMM command: LDBI int

    Note: This function is typically not needed when using `add_bias()`,
    which automatically manages the bias count. Use this only if you need
    to pre-allocate a specific number of bias slots or for Fortran-style
    compatibility.

    Parameters
    ----------
    n : int
        Number of biasing potentials to allocate

    See Also
    --------
    add_bias : Add biasing potential (auto-manages count)
    """
    if not _state.active:
        raise ValueError("BLOCK facility not initialized. Call initialize() first.")

    _state.lambda_dynamics['bias_count'] = n
    _execute_block_command(f'LDBI {n}')


def add_bias(block_i, block_j, cls, ref, cforce, npower, index=None):
    """Add biasing potential.

    Maps to CHARMM command: LDBV int int int int real real int

    This function automatically handles bias indexing and count management:
    - If `index` is not provided, it auto-increments based on existing biases
    - Automatically issues LDBI command when bias count increases

    Parameters
    ----------
    block_i : int
        First block index
    block_j : int
        Second block index
    cls : int
        Bias class (1-12, defines functional form)
    ref : float
        Reference value
    cforce : float
        Force constant
    npower : int
        Power/exponent
    index : int, optional
        Bias potential index. If not provided, auto-increments from 1.

    Notes
    -----
    Classes define different functional forms:
    1: V = cforce*(lambda-ref)^npower if lambda < ref
    2: V = cforce*(lambda-ref)^npower if lambda > ref
    3: V = cforce*[lambda(i) - lambda(j)]^npower
    etc.

    Examples
    --------
    >>> # Auto-indexed (recommended Pythonic way)
    >>> block.add_bias(2, 3, 5, 0.0, 0.0, 0)  # Auto-assigns index 1
    >>> block.add_bias(2, 4, 5, 0.0, 0.0, 0)  # Auto-assigns index 2

    >>> # Explicit index (Fortran-style)
    >>> block.add_bias(2, 3, 5, 0.0, 0.0, 0, index=1)
    """
    if not _state.active:
        raise ValueError("BLOCK facility not initialized. Call initialize() first.")

    # Auto-generate index if not provided
    if index is None:
        index = len(_state.lambda_dynamics['biases']) + 1

    # Auto-update bias count if needed (must be >= number of biases)
    new_count = max(_state.lambda_dynamics['bias_count'], index)
    if new_count > _state.lambda_dynamics['bias_count']:
        _state.lambda_dynamics['bias_count'] = new_count
        _execute_block_command(f'LDBI {new_count}')

    _state.lambda_dynamics['biases'].append({
        'index': index,
        'block_i': block_i,
        'block_j': block_j,
        'cls': cls,
        'ref': ref,
        'cforce': cforce,
        'npower': npower
    })

    cmd = f'LDBV {index} {block_i} {block_j} {cls} {ref} {cforce} {npower}'
    _execute_block_command(cmd)


def remove_bias(index=None, block_i=None, block_j=None):
    """Remove biasing potential(s) and re-issue remaining biases.

    Since CHARMM doesn't have a direct "remove bias" command, this function:
    1. Removes matching bias(es) from Python state
    2. Clears all biases in CHARMM (LDBI 0)
    3. Re-issues remaining biases with updated indices

    Parameters
    ----------
    index : int, optional
        Remove bias with this specific index
    block_i : int, optional
        Remove all biases involving this block (as first block)
    block_j : int, optional
        Remove all biases involving this block (as second block)

    Returns
    -------
    int
        Number of biases removed

    Examples
    --------
    >>> block.add_bias(2, 3, 5, 0.0, 0.0, 0)  # index 1
    >>> block.add_bias(2, 4, 5, 0.0, 0.0, 0)  # index 2
    >>> block.add_bias(3, 4, 5, 0.0, 0.0, 0)  # index 3
    >>> block.remove_bias(index=2)             # Remove bias 2, reindex remaining
    >>> # Now: bias 1 (2,3) and bias 2 (3,4) - auto re-indexed

    >>> block.remove_bias(block_i=2)           # Remove all biases with block_i=2

    See Also
    --------
    add_bias : Add biasing potential
    clear_biases : Remove all biases
    get_biases : Query current biases
    """
    if not _state.active:
        raise ValueError("BLOCK facility not initialized. Call initialize() first.")

    if index is None and block_i is None and block_j is None:
        raise ValueError("Must specify at least one of: index, block_i, block_j")

    # Find biases to remove
    original_count = len(_state.lambda_dynamics['biases'])
    remaining_biases = []

    for bias in _state.lambda_dynamics['biases']:
        remove = False
        if index is not None and bias['index'] == index:
            remove = True
        if block_i is not None and bias['block_i'] == block_i:
            remove = True
        if block_j is not None and bias['block_j'] == block_j:
            remove = True

        if not remove:
            remaining_biases.append(bias)

    removed_count = original_count - len(remaining_biases)

    if removed_count == 0:
        return 0

    # Clear all biases in CHARMM
    _execute_block_command('LDBI 0')

    # Update state
    _state.lambda_dynamics['biases'] = []
    _state.lambda_dynamics['bias_count'] = 0

    # Re-issue remaining biases with new indices
    if remaining_biases:
        new_count = len(remaining_biases)
        _state.lambda_dynamics['bias_count'] = new_count
        _execute_block_command(f'LDBI {new_count}')

        for new_idx, bias in enumerate(remaining_biases, start=1):
            bias['index'] = new_idx  # Update index
            _state.lambda_dynamics['biases'].append(bias)
            cmd = f"LDBV {new_idx} {bias['block_i']} {bias['block_j']} {bias['cls']} {bias['ref']} {bias['cforce']} {bias['npower']}"
            _execute_block_command(cmd)

    return removed_count


def clear_biases():
    """Remove all biasing potentials.

    Maps to CHARMM command: LDBI 0

    Clears all biases and resets the bias count to zero.

    Examples
    --------
    >>> block.add_bias(2, 3, 5, 0.0, 0.0, 0)
    >>> block.add_bias(2, 4, 5, 0.0, 0.0, 0)
    >>> block.clear_biases()  # All biases removed
    >>> print(len(block.get_biases()))  # 0

    See Also
    --------
    add_bias : Add biasing potential
    remove_bias : Remove specific bias(es)
    """
    if not _state.active:
        raise ValueError("BLOCK facility not initialized. Call initialize() first.")

    _state.lambda_dynamics['biases'] = []
    _state.lambda_dynamics['bias_count'] = 0
    _execute_block_command('LDBI 0')


def get_biases():
    """Query current biasing potentials.

    Returns
    -------
    list
        List of bias dictionaries, each containing:
        - 'index': int
        - 'block_i': int
        - 'block_j': int
        - 'cls': int
        - 'ref': float
        - 'cforce': float
        - 'npower': int

    Examples
    --------
    >>> block.add_bias(2, 3, 5, 0.0, 0.0, 0)
    >>> biases = block.get_biases()
    >>> print(biases[0]['block_i'])  # 2
    """
    return [b.copy() for b in _state.lambda_dynamics['biases']]


def rmla(*terms):
    """Remove lambda scaling for specific energy terms.

    Maps to CHARMM command: RMLA {BOND | THETa | DIHEd | ...}

    Parameters
    ----------
    *terms : str
        Terms to exclude from lambda scaling. Valid values:
        'bond', '12bond', '13bond', 'theta', 'angle', 'phi',
        'dihed', 'imphi', 'impr', 'cmap'

    Examples
    --------
    >>> block.rmla('bond', 'theta')
    >>> block.rmla('bond')
    """
    if not _state.active:
        raise ValueError("BLOCK facility not initialized. Call initialize() first.")

    valid_terms = {'bond', '12bond', '13bond', 'theta', 'angle',
                   'phi', 'dihed', 'imphi', 'impr', 'cmap'}

    for term in terms:
        term_lower = term.lower()
        if term_lower not in valid_terms:
            raise ValueError(f"Invalid term '{term}'. Valid: {valid_terms}")
        _state.lambda_dynamics['rmla_terms'].add(term_lower)

    cmd = 'RMLA ' + ' '.join(t.upper() for t in terms)
    _execute_block_command(cmd)


def restart_ld():
    """Restart lambda-dynamics.

    Maps to CHARMM command: LDRStart
    """
    if not _state.active:
        raise ValueError("BLOCK facility not initialized. Call initialize() first.")

    _execute_block_command('LDRS')


def write_ld(unit, nsavl):
    """Write lambda histogram output.

    Maps to CHARMM command: LDWRite IUNL int NSAVL int

    Parameters
    ----------
    unit : int
        Fortran unit number for output
    nsavl : int
        Save frequency for lambda values
    """
    if not _state.active:
        raise ValueError("BLOCK facility not initialized. Call initialize() first.")

    _execute_block_command(f'LDWR IUNL {unit} NSAVL {nsavl}')


def restraining_potential(block_id, value):
    """Set restraining potential for unbound states.

    Maps to CHARMM command: RSTP int real

    Parameters
    ----------
    block_id : int
        Block number
    value : float
        Restraining potential value
    """
    if not _state.active:
        raise ValueError("BLOCK facility not initialized. Call initialize() first.")

    _execute_block_command(f'RSTP {block_id} {value}')


# =============================================================================
# Multi-Site Lambda Dynamics (MSLD) Functions
# =============================================================================

def msld(site_assignments=None, *, nsite=None, fnex=5.5, fnxs=None,
         fnsin=False, f2ex=False, f2sin=False, ffix=False):
    """Initialize multi-site lambda dynamics (MSLD).

    Maps to CHARMM command: MSLD [int_1 int_2 ... | NSITe int] {FNEXponential | ...}

    MSLD enables simultaneous sampling of multiple alchemical sites,
    which is essential for constant-pH MD (CpHMD) simulations with
    multiple titratable residues.

    Parameters
    ----------
    site_assignments : list, optional
        List of site IDs for each block [site_0, site_1, site_2, ...].
        - Block 1 (index 0 in list) is typically site 0 (environment/core)
        - Each subsequent block is assigned to a site number
        - Blocks with the same site number belong to the same titratable group

        Example for 3 ASP residues with 3 protonation states each:
        [0, 1, 1, 1, 2, 2, 2, 3, 3, 3] means:
        - Block 1 → site 0 (environment)
        - Blocks 2,3,4 → site 1 (first ASP: ASPO, ASH1, ASH2)
        - Blocks 5,6,7 → site 2 (second ASP)
        - Blocks 8,9,10 → site 3 (third ASP)

    nsite : int, optional
        Number of sites (alternative to site_assignments).
        Use assign_block_to_site() to set individual assignments.
    fnex : float, optional
        FNEX parameter controlling lambda functional form (default: 5.5).
        Higher values = sharper transitions between states.
    fnxs : list, optional
        Per-site FNEX values for heterogeneous site behavior.
    fnsin : bool, optional
        Use sin^2 functional form instead of exponential.
    f2ex : bool, optional
        Use 2-block exponential form.
    f2sin : bool, optional
        Use 2-block sin^2 form.
    ffix : bool, int, or list of int, optional
        Fixed lambda mode. Three forms:
        - ``False`` (default): No fixed blocks.
        - ``True``: Global FFIX — all blocks use fixed lambda (FEP/MBAR).
          Mutually exclusive with other functional forms.
        - ``int`` or ``list of int``: Partial FFIX — fix specific blocks
          by block index while remaining blocks use the chosen functional
          form (FNEX, FNXS, etc.). For solute tempering or hybrid schemes.

    Examples
    --------
    >>> # CpHMD setup with 10 titratable sites (from reference file)
    >>> # Each ASP/GLU has 3 states: charged, protonated-O1, protonated-O2
    >>> # HSP has 3 states: charged, delta-protonated, epsilon-protonated
    >>> site_list = [0,  # Block 1: environment
    ...              1, 1, 1,    # Blocks 2-4: ASP site 1
    ...              2, 2, 2,    # Blocks 5-7: ASP site 2
    ...              3, 3, 3,    # Blocks 8-10: ASP site 3
    ...              # ... more sites ...
    ...              10, 10, 10] # Blocks 29-31: HSP site 10
    >>> block.msld(site_list, fnex=5.5)

    >>> # Using nsite for manual assignment
    >>> block.msld(nsite=3)
    >>> block.assign_block_to_site(2, 1)  # Block 2 → site 1
    >>> block.assign_block_to_site(3, 1)  # Block 3 → site 1
    >>> block.assign_block_to_site(4, 2)  # Block 4 → site 2

    >>> # Partial FFIX: fix block 2 at its ldin value (solute tempering)
    >>> block.msld(site_list, fnex=5.5, ffix=[2])

    >>> # Partial FFIX: fix multiple blocks
    >>> block.msld(site_list, fnex=5.5, ffix=[2, 5])

    >>> # Single int also works
    >>> block.msld(site_list, fnex=5.5, ffix=2)

    See Also
    --------
    msmatrix : Auto-populate coefficient matrix
    assign_block_to_site : Manual block-to-site assignment
    get_msld_state : Query MSLD settings
    """
    if not _state.active:
        raise ValueError("BLOCK facility not initialized. Call initialize() first.")

    # CRITICAL: MSLD requires theta mode (QLDM THETA) to be enabled first
    # See lambdadyn.F90:3084-3085 - wrndie if qthetadm is false
    if not _state.lambda_dynamics.get('theta', False):
        raise ValueError(
            "MSLD requires theta mode. Call enable_lambda_dynamics(theta=True) first. "
            "See CHARMM documentation: QTHETA must be previously assigned."
        )

    # MSLD requires at least 3 blocks (lambdadyn.F90:3081-3082)
    if _state.nblocks < 3:
        raise ValueError(
            f"MSLD requires at least 3 blocks, but only {_state.nblocks} initialized."
        )

    _state.msld['enabled'] = True

    cmd = 'MSLD'

    if site_assignments is not None:
        _state.msld['site_assignments'] = {
            i + 1: site for i, site in enumerate(site_assignments)
        }
        cmd += ' ' + ' '.join(str(s) for s in site_assignments)
    elif nsite is not None:
        cmd += f' NSITE {nsite}'

    # Normalize ffix to a consistent form
    ffix_blocks = []
    ffix_global = False
    if isinstance(ffix, (list, tuple)):
        ffix_blocks = list(ffix)
    elif isinstance(ffix, int) and not isinstance(ffix, bool):
        ffix_blocks = [ffix]
    elif ffix is True:
        ffix_global = True

    # Functional form
    if ffix_global:
        # Global FFIX: mutually exclusive with other functional forms
        _state.msld['functional_form'] = 'ffix'
        cmd += ' FFIX'
    elif fnxs is not None:
        _state.msld['functional_form'] = 'fnxs'
        _state.msld['fnxs'] = fnxs
        cmd += ' FNXS ' + ' '.join(str(v) for v in fnxs)
    elif fnsin:
        _state.msld['functional_form'] = 'fnsin'
        cmd += ' FNSIN'
    elif f2ex:
        _state.msld['functional_form'] = 'f2ex'
        cmd += ' F2EX'
    elif f2sin:
        _state.msld['functional_form'] = 'f2sin'
        cmd += ' F2SIN'
    else:
        _state.msld['functional_form'] = 'fnex'
        _state.msld['fnex'] = fnex
        cmd += f' FNEX {fnex}'

    # Partial FFIX: appended after the functional form keyword
    if ffix_blocks:
        _state.msld['ffix_blocks'] = ffix_blocks
        cmd += ' FFIX ' + ' '.join(str(b) for b in ffix_blocks)

    _execute_block_command(cmd)


def assign_block_to_site(block_id, site_id):
    """Assign a block to a site (when using NSITe).

    Maps to CHARMM command: BLASsign int_block int_site

    Parameters
    ----------
    block_id : int
        Block number
    site_id : int
        Site number
    """
    if not _state.active:
        raise ValueError("BLOCK facility not initialized. Call initialize() first.")

    _state.msld['site_assignments'][block_id] = site_id
    _execute_block_command(f'BLASGN {block_id} {site_id}')


def msmatrix():
    """Auto-populate MSLD coefficient matrix.

    Maps to CHARMM command: MSMAtrix
    """
    if not _state.active:
        raise ValueError("BLOCK facility not initialized. Call initialize() first.")

    _execute_block_command('MSMA')


def block_assign(block_id, site_id):
    """Assign a block to an MSLD site.

    Maps to CHARMM command: BLASsign int_block int_site

    Use this when there are too many blocks to assign in a single MSLD command
    (due to line length limitations). First call `msld(nsite=N)` to specify the
    number of sites, then use block_assign() to assign each block individually.

    Parameters
    ----------
    block_id : int
        Block number to assign (2 to nblocks, block 1 is always environment)
    site_id : int
        Site number to assign the block to (1 to nsite)

    Examples
    --------
    >>> # For systems with many blocks, use nsite mode + block_assign
    >>> block.initialize(20)
    >>> # ... call statements ...
    >>> block.msld(nsite=5)  # 5 sites, assignments to follow
    >>> block.block_assign(2, 1)   # Block 2 -> Site 1
    >>> block.block_assign(3, 1)   # Block 3 -> Site 1
    >>> block.block_assign(4, 2)   # Block 4 -> Site 2
    >>> # ... more assignments ...
    >>> block.msmatrix()

    See Also
    --------
    msld : Main MSLD setup (can assign all blocks at once for smaller systems)
    """
    if not _state.active:
        raise ValueError("BLOCK facility not initialized. Call initialize() first.")

    if block_id < 2 or block_id > _state.nblocks:
        raise ValueError(f"block_id must be between 2 and {_state.nblocks}")

    _execute_block_command(f'BLAS {block_id} {site_id}')

    # Update internal state
    _state.msld['site_assignments'][block_id] = site_id


def theta_bias(collective=True, site=None, values=None, auto=False):
    """Set theta-biasing for MSLD to improve sampling efficiency.

    Maps to CHARMM command: THBV COLL|INDE [int|ALL] real [real] | AUTO

    Theta biasing reduces time spent at intermediate λ values (λ ≈ 0.5),
    pushing the system toward pure states (λ ≈ 0 or 1). This is especially
    useful when simulating many substituents at multiple sites.

    **Works together with MSLD and fnex:**
    - `msld(..., fnex=5.5)` defines how λ is computed from θ
    - `theta_bias()` adds biasing potentials on θ to improve sampling

    Two modes are available:

    **Collective (COLL)**: Biases based on how many substituents are "high" vs "low"
        V = 0.5*coef*((nhigh-1)² + (nlow-nsub+1)²)
        where nhigh/nlow count substituents above/below λ=0.5

    **Independent (INDE)**: Per-substituent bias pushing toward pure states
        V = Σ -b*(-0.5*sin(θ)+0.5)⁴

    Parameters
    ----------
    collective : bool, optional
        If True (default), use collective mode (COLL).
        If False, use independent mode (INDE).
    site : int or 'all', optional
        Site number to apply bias to, or 'all' for all sites (default).
    values : list, optional
        For COLL mode: [coef, power] - coefficient (kT recommended) and power (2 recommended)
        For INDE mode: [coef] - bias coefficient (or use auto=True)
    auto : bool, optional
        For INDE mode only: automatically determine optimal bias coefficient.
        Uses formula: b = 0.5*ln(0.125*π*Ns²*b) where Ns = number of substituents

    Examples
    --------
    >>> # Typical MSLD setup with theta biasing
    >>> block.initialize(4)
    >>> block.call(2, sel_state1)
    >>> block.call(3, sel_state2)
    >>> block.call(4, sel_state3)
    >>> block.enable_lambda_dynamics(theta=True)
    >>> block.set_langevin(temp=298.15)
    >>> # ... ldin commands ...
    >>> block.msld([0, 1, 1, 1], fnex=5.5)
    >>> block.theta_bias(collective=True, site='all', values=[298.15, 2])  # kT, power=2
    >>> block.msmatrix()
    >>> block.end()

    >>> # Independent mode with auto coefficient
    >>> block.theta_bias(collective=False, auto=True)

    >>> # Apply to specific site only
    >>> block.theta_bias(collective=True, site=1, values=[298.15, 2])

    See Also
    --------
    msld : Set up Multi-Site Lambda Dynamics
    enable_lambda_dynamics : Enable theta-dynamics (required before MSLD)
    """
    if not _state.active:
        raise ValueError("BLOCK facility not initialized. Call initialize() first.")

    if auto:
        cmd = 'THBV INDE ALL AUTO'
    else:
        mode = 'COLL' if collective else 'INDE'
        cmd = f'THBV {mode}'

        if site is not None:
            if site == 'all':
                cmd += ' ALL'
            else:
                cmd += f' {site}'

        if values is not None:
            cmd += ' ' + ' '.join(str(v) for v in values)

    _execute_block_command(cmd)


def get_msld_state():
    """Query current MSLD settings.

    Returns
    -------
    dict
        MSLD state including:
        - 'enabled': bool - whether MSLD is active
        - 'site_assignments': dict - {block_id: site_id} mapping
        - 'fnex': float - FNEX parameter value
        - 'fnxs': list or None - per-site FNEX values
        - 'functional_form': str - which functional form is used

    Examples
    --------
    >>> block.msld([0, 1, 1, 1, 2, 2, 2], fnex=5.5)
    >>> state = block.get_msld_state()
    >>> print(state['enabled'])  # True
    >>> print(state['fnex'])     # 5.5
    """
    return _state.msld.copy()


# =============================================================================
# Advanced Functions - Soft Core
# =============================================================================

def soft_core(mode='on', w14=False):
    """MSLD soft core potentials (Hayes et al).

    Maps to CHARMM command: SOFT [W14 | ON] (no argument disables)

    Soft core potentials prevent singularities when atoms appear/disappear
    during alchemical transformations by modifying the Lennard-Jones potential.

    Parameters
    ----------
    mode : str, optional
        - 'on': Enable soft core (default) - recommended for MSLD
        - 'off': Disable soft core - use standard LJ potential
    w14 : bool, optional
        Enable 1-4 soft core interactions. Set True when 1-4 interactions
        involve disappearing atoms.

    Examples
    --------
    >>> block.soft_core('on')      # Enable soft core
    >>> block.soft_core('off')     # Disable soft core
    >>> block.soft_core(w14=True)  # Enable with 1-4 interactions

    See Also
    --------
    get_soft_core_state : Query current soft core settings
    soft_omm : OpenMM soft core for vdW
    pssp : Dual-topology soft core
    """
    if not _state.active:
        raise ValueError("BLOCK facility not initialized. Call initialize() first.")

    _state.soft_core['enabled'] = mode.lower() == 'on'
    _state.soft_core['mode'] = mode.lower()
    _state.soft_core['w14'] = w14

    if w14:
        _execute_block_command('SOFT W14')
    elif mode.lower() == 'on':
        _execute_block_command('SOFT ON')
    else:
        # SOFT without arguments disables soft core potentials
        # (CHARMM doesn't parse OFF, it just ignores non-ON/W14 keywords)
        _execute_block_command('SOFT')


def get_soft_core_state():
    """Query current soft core settings.

    Returns
    -------
    dict
        {'enabled': bool, 'mode': str, 'w14': bool}
    """
    return _state.soft_core.copy()


def soft_omm(enabled=True):
    """CHARMM/OpenMM soft core for vdW interactions.

    Maps to CHARMM command: SOMM / NOSOMM

    Enables or disables OpenMM-compatible soft core potentials for van der Waals
    interactions. This is specifically designed for use with CHARMM/OpenMM
    hybrid simulations.

    The OpenMM soft core differs slightly from CHARMM's native soft core
    in its functional form, providing compatibility when using OpenMM
    as the integrator.

    Parameters
    ----------
    enabled : bool, optional
        - True: Enable OpenMM soft core (default)
        - False: Disable OpenMM soft core

    Examples
    --------
    >>> block.soft_omm()        # Enable OpenMM soft core
    >>> block.soft_omm(True)    # Same as above
    >>> block.soft_omm(False)   # Disable OpenMM soft core

    See Also
    --------
    soft_core : Native CHARMM soft core for MSLD
    pssp : Dual-topology soft core
    get_soft_omm_state : Query current setting
    """
    if not _state.active:
        raise ValueError("BLOCK facility not initialized. Call initialize() first.")

    _state.soft_omm_enabled = enabled

    if enabled:
        _execute_block_command('SOMM')
    else:
        _execute_block_command('NOSOMM')


def get_soft_omm_state():
    """Query current OpenMM soft core setting.

    Returns
    -------
    bool
        True if OpenMM soft core is enabled

    Examples
    --------
    >>> block.soft_omm(True)
    >>> print(block.get_soft_omm_state())  # True
    """
    return _state.soft_omm_enabled


def pssp(alam=None, dlam=None):
    """Dual-topology soft core potentials for standard FEP.

    Maps to CHARMM command: PSSP [ALAM real] [DLAM real]

    Enables soft core potentials specifically designed for dual-topology
    free energy calculations. This prevents singularities in the potential
    when atoms appear or disappear during alchemical transformations.

    The soft core modifies the LJ potential near endpoints (lambda=0, 1)
    where appearing/disappearing atoms would otherwise cause infinite energies.

    Parameters
    ----------
    alam : float, optional
        Alpha parameter controlling the softness of the potential.
        Typical values: 0.5-1.0. Higher values = softer potential.
        - Default CHARMM value: 0.5
        - Recommended for small molecules: 0.5
        - For larger transformations: 0.8-1.0
    dlam : float, optional
        Delta lambda parameter for shifting the soft core region.
        Controls where soft core is applied in lambda space.
        - Default: 0.0 (symmetric around endpoints)
        - Positive values shift towards lambda=1

    Examples
    --------
    >>> block.pssp()                   # Enable with defaults
    >>> block.pssp(alam=0.5)           # Set alpha parameter
    >>> block.pssp(alam=0.5, dlam=0.1) # Set both parameters
    >>> block.no_pssp()                # Disable soft core

    See Also
    --------
    no_pssp : Disable dual-topology soft core
    get_pssp_state : Query current PSSP settings
    soft_core : Alternative soft core for MSLD
    """
    if not _state.active:
        raise ValueError("BLOCK facility not initialized. Call initialize() first.")

    _state.pssp_state['enabled'] = True
    _state.pssp_state['alam'] = alam
    _state.pssp_state['dlam'] = dlam

    cmd = 'PSSP'
    if alam is not None:
        cmd += f' ALAM {alam}'
    if dlam is not None:
        cmd += f' DLAM {dlam}'

    _execute_block_command(cmd)


def no_pssp():
    """Turn off dual-topology soft core.

    Maps to CHARMM command: NOPSsp

    Disables the PSSP soft core potentials, reverting to standard
    Lennard-Jones interactions. Use this when you want to switch
    from soft core to standard potentials mid-simulation.

    Examples
    --------
    >>> block.pssp(alam=0.5)  # Enable soft core
    >>> # ... run simulation ...
    >>> block.no_pssp()        # Disable soft core

    See Also
    --------
    pssp : Enable dual-topology soft core
    get_pssp_state : Query current PSSP settings
    """
    if not _state.active:
        raise ValueError("BLOCK facility not initialized. Call initialize() first.")

    _state.pssp_state['enabled'] = False
    _state.pssp_state['alam'] = None
    _state.pssp_state['dlam'] = None
    _execute_block_command('NOPSSP')


def get_pssp_state():
    """Query current PSSP (dual-topology soft core) settings.

    Returns
    -------
    dict
        {'enabled': bool, 'alam': float or None, 'dlam': float or None}

    Examples
    --------
    >>> block.pssp(alam=0.5)
    >>> state = block.get_pssp_state()
    >>> print(state['enabled'])  # True
    >>> print(state['alam'])     # 0.5
    """
    return _state.pssp_state.copy()


def pmel(mode):
    """MSLD PME electrostatics handling.

    Maps to CHARMM command: PMEL [NN | EX | ON | OFF]

    Controls how PME electrostatics are calculated in MSLD simulations.
    Different modes trade off accuracy vs computational cost.

    Parameters
    ----------
    mode : str
        PME mode:
        - 'on': Standard PME with MSLD (simple, less accurate)
        - 'off': Disable PME MSLD modifications
        - 'nn': Nearest-neighbor PME (faster, good for most cases)
        - 'ex': Exact PME treatment (most accurate, slower)

    Examples
    --------
    >>> block.pmel('on')   # Enable standard PME MSLD
    >>> block.pmel('ex')   # Switch to exact PME
    >>> block.pmel('nn')   # Switch to nearest-neighbor
    >>> block.pmel('off')  # Disable PME modifications

    See Also
    --------
    get_pmel_mode : Query current PME mode
    """
    if not _state.active:
        raise ValueError("BLOCK facility not initialized. Call initialize() first.")

    valid_modes = {'nn', 'ex', 'on', 'off'}
    if mode.lower() not in valid_modes:
        raise ValueError(f"Invalid mode '{mode}'. Valid: {valid_modes}")

    _state.pmel_mode = mode.lower()
    _execute_block_command(f'PMEL {mode.upper()}')


def get_pmel_mode():
    """Query current PME mode.

    Returns
    -------
    str or None
        Current PMEL mode ('on', 'off', 'nn', 'ex') or None if not set
    """
    return _state.pmel_mode


# =============================================================================
# Advanced Functions - Constrained Atoms
# =============================================================================

def constrained_atom_scaling(mode='on', k=None):
    """Scaling of constrained atoms during alchemical transformations.

    Maps to CHARMM command: SCAT [ON | K real | OFF]

    Controls how constrained atoms (e.g., SHAKE-constrained hydrogens)
    are handled during lambda scaling. This is important for maintaining
    proper behavior of constraints during alchemical transformations.

    Parameters
    ----------
    mode : str, optional
        - 'on': Enable scaling of constrained atoms (default)
                Constrained atoms participate in lambda scaling.
        - 'off': Disable scaling of constrained atoms
                Constrained atoms maintain constant interactions.
    k : float, optional
        Force constant for constraint scaling. When specified,
        overrides the mode parameter and sets a specific force
        constant value for the constraint scaling.
        - Higher k = stronger constraint maintenance
        - Lower k = more flexible during transformation

    Examples
    --------
    >>> block.constrained_atom_scaling('on')    # Enable scaling
    >>> block.constrained_atom_scaling('off')   # Disable scaling
    >>> block.constrained_atom_scaling(k=500.0) # Set force constant

    See Also
    --------
    define_constrained_atoms : Define which atoms are constrained
    get_constrained_atom_scaling : Query current settings
    """
    if not _state.active:
        raise ValueError("BLOCK facility not initialized. Call initialize() first.")

    if k is not None:
        _state.scat_state['enabled'] = True
        _state.scat_state['mode'] = 'k'
        _state.scat_state['k'] = k
        _execute_block_command(f'SCAT K {k}')
    else:
        _state.scat_state['enabled'] = mode.lower() == 'on'
        _state.scat_state['mode'] = mode.lower()
        _state.scat_state['k'] = None
        _execute_block_command(f'SCAT {mode.upper()}')


def get_constrained_atom_scaling():
    """Query current constrained atom scaling settings.

    Returns
    -------
    dict
        {'enabled': bool, 'mode': str or None, 'k': float or None}

    Examples
    --------
    >>> block.constrained_atom_scaling(k=500.0)
    >>> state = block.get_constrained_atom_scaling()
    >>> print(state['k'])  # 500.0
    """
    return _state.scat_state.copy()


def define_constrained_atoms(selection):
    """Define constrained atom group.

    Maps to CHARMM command: CATS atom-selection

    Parameters
    ----------
    selection : SelectAtoms, str, or selection-like
        Atom selection for constrained atoms
    """
    if not _state.active:
        raise ValueError("BLOCK facility not initialized. Call initialize() first.")

    sel_str = _build_selection_string(selection)
    _execute_block_command(f'CATS {sel_str}')


# =============================================================================
# Advanced Functions - Soft Bonds
# =============================================================================

def soft_bond_setup(n, ralf=None, lexp=None, blex=None):
    """Setup soft bonds for topology-changing TI.

    Maps to CHARMM command: NSOB int [RALF real] [LEXP real] [BLEX real]

    Parameters
    ----------
    n : int
        Number of soft bonds
    ralf : float, optional
        Saturation radius
    lexp : float, optional
        Lambda exponent
    blex : float, optional
        Bond lambda exponent
    """
    if not _state.active:
        raise ValueError("BLOCK facility not initialized. Call initialize() first.")

    cmd = f'NSOB {n}'
    if ralf is not None:
        cmd += f' RALF {ralf}'
    if lexp is not None:
        cmd += f' LEXP {lexp}'
    if blex is not None:
        cmd += f' BLEX {blex}'

    _execute_block_command(cmd)


def define_soft_bond(index, sel1, sel2):
    """Define individual soft bond.

    Maps to CHARMM command: SOBO int atom-selection1 atom-selection2

    Parameters
    ----------
    index : int
        Soft bond index
    sel1, sel2 : SelectAtoms, str, or selection-like
        Atom selections defining the bond endpoints
    """
    if not _state.active:
        raise ValueError("BLOCK facility not initialized. Call initialize() first.")

    sel_str1 = _build_selection_string(sel1)
    sel_str2 = _build_selection_string(sel2)
    _execute_block_command(f'SOBO {index} {sel_str1} {sel_str2}')


# =============================================================================
# Advanced Functions - Hybrid Hamiltonian
# =============================================================================

def hybrid_hamiltonian(lambda_val):
    """Enable hybrid Hamiltonian for free energy calculations.

    Maps to CHARMM command: HYBH real

    The hybrid Hamiltonian combines two states using a mixing parameter
    lambda: H_mix = (1-lambda)*H_A + lambda*H_B. This is used for
    thermodynamic integration and other free energy methods.

    When enabled, CHARMM calculates and tracks dE/dlambda, which can
    be output for post-processing.

    Parameters
    ----------
    lambda_val : float
        Lambda value for the hybrid Hamiltonian.
        - 0.0: Pure state A (reactant)
        - 1.0: Pure state B (product)
        - 0.0-1.0: Mixed state

    Examples
    --------
    >>> block.hybrid_hamiltonian(0.5)        # Set lambda=0.5
    >>> block.set_hybh_output(unit=50)       # Set output unit
    >>> # ... run dynamics ...
    >>> block.write_hybh()                    # Write dE/dlambda
    >>> block.clear_hybh()                    # Clear and disable

    See Also
    --------
    set_hybh_output : Set output unit for dE/dlambda
    print_hybh : Print dE/dlambda terms to output
    write_hybh : Write dE/dlambda to file
    clear_hybh : Disable hybrid Hamiltonian
    get_hybrid_hamiltonian_state : Query current settings
    """
    if not _state.active:
        raise ValueError("BLOCK facility not initialized. Call initialize() first.")

    _state.hybh['enabled'] = True
    _state.hybh['lambda'] = lambda_val
    _execute_block_command(f'HYBH {lambda_val}')


def set_hybh_output(unit):
    """Set output unit for dE/dl.

    Maps to CHARMM command: OUTH int

    Parameters
    ----------
    unit : int
        Fortran unit number for output
    """
    if not _state.active:
        raise ValueError("BLOCK facility not initialized. Call initialize() first.")

    _state.hybh['output_unit'] = unit
    _execute_block_command(f'OUTH {unit}')


def print_hybh():
    """Print dE/dl terms.

    Maps to CHARMM command: PRIN
    """
    if not _state.active:
        raise ValueError("BLOCK facility not initialized. Call initialize() first.")

    _execute_block_command('PRIN')


def write_hybh():
    """Write dE/dlambda to output unit.

    Maps to CHARMM command: PRDH
    """
    if not _state.active:
        raise ValueError("BLOCK facility not initialized. Call initialize() first.")

    _execute_block_command('PRDH')


def clear_hybh():
    """Clear and disable hybrid Hamiltonian.

    Maps to CHARMM command: CLHH

    Disables the hybrid Hamiltonian mode and clears accumulated
    dE/dlambda data. Use this to switch back to standard energy
    calculations or to reset before a new simulation.

    Examples
    --------
    >>> block.hybrid_hamiltonian(0.5)  # Enable
    >>> # ... run simulation ...
    >>> block.clear_hybh()              # Disable and clear

    See Also
    --------
    hybrid_hamiltonian : Enable hybrid Hamiltonian
    get_hybrid_hamiltonian_state : Query current settings
    """
    if not _state.active:
        raise ValueError("BLOCK facility not initialized. Call initialize() first.")

    _state.hybh['enabled'] = False
    _state.hybh['lambda'] = None
    _state.hybh['output_unit'] = None
    _execute_block_command('CLHH')


def get_hybrid_hamiltonian_state():
    """Query current hybrid Hamiltonian settings.

    Returns
    -------
    dict
        {'enabled': bool, 'lambda': float or None, 'output_unit': int or None}

    Examples
    --------
    >>> block.hybrid_hamiltonian(0.5)
    >>> state = block.get_hybrid_hamiltonian_state()
    >>> print(state['enabled'])  # True
    >>> print(state['lambda'])   # 0.5
    """
    return _state.hybh.copy()


# =============================================================================
# Advanced Functions - MC/MD
# =============================================================================

def enable_mc_md(mctemp=None, freq=None, mcstep=None, max_val=None):
    """Enable hybrid MC/MD for lambda space sampling.

    Maps to CHARMM command: QLMC [MCTEmperature real] [FREQ int] [MCSTep int] [MAX real]

    Hybrid Monte Carlo/Molecular Dynamics combines MD for configuration
    space sampling with MC for lambda space sampling. This can improve
    convergence for free energy calculations.

    Parameters
    ----------
    mctemp : float, optional
        MC temperature in Kelvin. Controls acceptance probability.
        - Higher temperature = more MC moves accepted
        - Lower temperature = more selective moves
        - Default: uses simulation temperature
    freq : int, optional
        Frequency of MC moves (every N MD steps).
        - Typical values: 10-100
        - Lower = more frequent MC attempts
    mcstep : int, optional
        Number of MC steps per attempt.
        - Typical values: 1-10
        - More steps = larger lambda changes per attempt
    max_val : float, optional
        Maximum allowed lambda step size.
        - Typical values: 0.1-0.3
        - Larger = bigger jumps in lambda space

    Examples
    --------
    >>> block.enable_mc_md()                                # Enable with defaults
    >>> block.enable_mc_md(mctemp=300.0, freq=50)          # Set temperature and frequency
    >>> block.enable_mc_md(mctemp=310.0, freq=100, max_val=0.2)  # Full specification
    >>> block.disable_mc_md()                               # Disable MC/MD

    See Also
    --------
    disable_mc_md : Disable MC/MD
    get_mc_md_state : Query current settings
    mc_intermediate : Define intermediate lambda states
    mc_step_size : Set uniform step size
    mc_clear : Clear MC/MD data
    """
    if not _state.active:
        raise ValueError("BLOCK facility not initialized. Call initialize() first.")

    _state.mc_md['enabled'] = True
    _state.mc_md['params'] = {
        'mctemp': mctemp,
        'freq': freq,
        'mcstep': mcstep,
        'max_val': max_val
    }

    cmd = 'QLMC'
    if mctemp is not None:
        cmd += f' MCTE {mctemp}'
    if freq is not None:
        cmd += f' FREQ {freq}'
    if mcstep is not None:
        cmd += f' MCST {mcstep}'
    if max_val is not None:
        cmd += f' MAX {max_val}'

    _execute_block_command(cmd)


def disable_mc_md():
    """Disable hybrid MC/MD.

    Maps to CHARMM command: NQLMC

    Disables the MC/MD hybrid method, reverting to pure MD or
    lambda-dynamics for lambda space sampling.

    Examples
    --------
    >>> block.enable_mc_md(mctemp=300.0)  # Enable
    >>> # ... run simulation ...
    >>> block.disable_mc_md()              # Disable

    See Also
    --------
    enable_mc_md : Enable MC/MD
    get_mc_md_state : Query current settings
    """
    if not _state.active:
        raise ValueError("BLOCK facility not initialized. Call initialize() first.")

    _state.mc_md['enabled'] = False
    _state.mc_md['params'] = {}
    _execute_block_command('NQLMC')


def get_mc_md_state():
    """Query current MC/MD settings.

    Returns
    -------
    dict
        {'enabled': bool, 'params': dict}
        params contains: mctemp, freq, mcstep, max_val

    Examples
    --------
    >>> block.enable_mc_md(mctemp=300.0, freq=50)
    >>> state = block.get_mc_md_state()
    >>> print(state['enabled'])              # True
    >>> print(state['params']['mctemp'])     # 300.0
    """
    return _state.mc_md.copy()


def mc_intermediate(n, *values):
    """Define intermediate lambda states for MC/MD.

    Maps to CHARMM command: MCIN int {real ... real}

    Parameters
    ----------
    n : int
        Number of intermediate states
    *values : float
        Lambda values for intermediate states
    """
    if not _state.active:
        raise ValueError("BLOCK facility not initialized. Call initialize() first.")

    cmd = f'MCIN {n} ' + ' '.join(str(v) for v in values)
    _execute_block_command(cmd)


def mc_step_size(value):
    """Define uniform lambda step size.

    Maps to CHARMM command: MCDI real

    Parameters
    ----------
    value : float
        Step size for lambda changes
    """
    if not _state.active:
        raise ValueError("BLOCK facility not initialized. Call initialize() first.")

    _execute_block_command(f'MCDI {value}')


def mc_ignore_restraint():
    """Ignore restraining forces in MC sampling.

    Maps to CHARMM command: MCRS
    """
    if not _state.active:
        raise ValueError("BLOCK facility not initialized. Call initialize() first.")

    _execute_block_command('MCRS')


def mc_clear():
    """Clear MC/MD data.

    Maps to CHARMM command: MCLEar
    """
    if not _state.active:
        raise ValueError("BLOCK facility not initialized. Call initialize() first.")

    _state.mc_md['enabled'] = False
    _state.mc_md['params'] = {}
    _execute_block_command('MCLE')


def mc_free(exfreq, fini, ffin, flat):
    """Simulated scaling (Wang-Landau) method.

    Maps to CHARMM command: MCFRee EXFReq int FINI real FFIN real FLAT real

    Parameters
    ----------
    exfreq : int
        Exchange frequency
    fini : float
        Initial flatness criterion
    ffin : float
        Final flatness criterion
    flat : float
        Flatness parameter
    """
    if not _state.active:
        raise ValueError("BLOCK facility not initialized. Call initialize() first.")

    _execute_block_command(f'MCFR EXFR {exfreq} FINI {fini} FFIN {ffin} FLAT {flat}')


def mc_lambda(n, *lambda_values):
    """Define lambda values for simulated scaling.

    Maps to CHARMM command: MCLAmd int LAMD0 real LAMD1 real ...

    Parameters
    ----------
    n : int
        Number of lambda values
    *lambda_values : float
        Lambda values (LAMD0, LAMD1, ...)
    """
    if not _state.active:
        raise ValueError("BLOCK facility not initialized. Call initialize() first.")

    cmd = f'MCLA {n}'
    for i, val in enumerate(lambda_values):
        cmd += f' LAMD{i} {val}'

    _execute_block_command(cmd)


# =============================================================================
# Advanced Functions - Other
# =============================================================================

def save_decomposed():
    """Save decomposed energy for TSM post-processing.

    Maps to CHARMM command: SAVE
    """
    if not _state.active:
        raise ValueError("BLOCK facility not initialized. Call initialize() first.")

    _execute_block_command('SAVE')


def unsave_decomposed():
    """Remove SAVE traces.

    Maps to CHARMM command: UNSAve
    """
    if not _state.active:
        raise ValueError("BLOCK facility not initialized. Call initialize() first.")

    _execute_block_command('UNSA')


def read_lambda_from_restart(enabled=True):
    """Control reading lambda from restart file.

    Maps to CHARMM command: RLFR [ON | OFF]

    Parameters
    ----------
    enabled : bool, optional
        If True, read lambda from restart (default: True)
    """
    if not _state.active:
        raise ValueError("BLOCK facility not initialized. Call initialize() first.")

    mode = 'ON' if enabled else 'OFF'
    _execute_block_command(f'RLFR {mode}')


def save_lambda_force(varname, block_id):
    """Save lambda force to parameter variable.

    Maps to CHARMM command: FLAM string int

    Parameters
    ----------
    varname : str
        Variable name to store force
    block_id : int
        Block number
    """
    if not _state.active:
        raise ValueError("BLOCK facility not initialized. Call initialize() first.")

    _execute_block_command(f'FLAM {varname} {block_id}')


# =============================================================================
# Direct Memory Access (ctypes interface)
# =============================================================================

# These functions provide direct access to CHARMM internal data structures
# via the C API, offering faster reads than command-based queries.
# They require CHARMM to be compiled with KEY_BLOCK=1.


def _check_lambdata_available():
    """Check if lambda data API is available and active."""
    try:
        return bool(lib.lambdata_is_active())
    except AttributeError:
        return False


def _check_msldata_available():
    """Check if MSLD data API is available and active."""
    try:
        return bool(lib.msldata_is_active())
    except AttributeError:
        return False


def get_nblock_direct():
    """Get number of blocks directly from CHARMM memory.

    This uses ctypes to read directly from CHARMM's internal data,
    which is faster than parsing command output.

    Returns
    -------
    int or None
        Number of blocks, or None if blockdata API not available

    Examples
    --------
    >>> nblocks = block.get_nblock_direct()
    >>> if nblocks is not None:
    ...     print(f"Direct read: {nblocks} blocks")

    See Also
    --------
    get_nblocks : Command-based query (uses Python state)
    """
    if not _check_blockdata_available():
        return None
    try:
        return int(lib.blockdata_get_nblock())
    except AttributeError:
        return None


def get_nbiasv_direct():
    """Get number of biasing potentials directly from CHARMM memory.

    Returns
    -------
    int or None
        Number of biases, or None if API not available

    See Also
    --------
    get_biases : Command-based query with full bias info
    """
    try:
        return int(lib.lambdata_get_nbiasv())
    except AttributeError:
        return None


def get_lambda_squared_direct():
    """Get lambda squared values directly from CHARMM memory.

    Returns current bixlam^2 values for all blocks from the last
    dynamics step. This provides fast access to lambda state without
    command parsing.

    Returns
    -------
    pandas.DataFrame or None
        DataFrame with columns ['STEP', 'TIME', 'XLM2'] for each block,
        or None if API not available or not active

    Examples
    --------
    >>> df = block.get_lambda_squared_direct()
    >>> if df is not None:
    ...     print(df['XLM2'].values)  # Lambda^2 for each block

    See Also
    --------
    get_lambda_dynamics_state : Command-based query
    """
    if not _check_lambdata_available():
        return None

    try:
        nblock = lib.lambdata_get_nblock()
        ncols = lib.lambdata_bixlamsq_get_ncols()
        nelts = nblock * ncols

        name_size = lib.dataframe_get_name_size()
        labels_bufs = [ctypes.create_string_buffer(name_size) for _ in range(ncols)]
        labels_ptrs = (ctypes.c_char_p * ncols)(*map(ctypes.addressof, labels_bufs))
        data = (ctypes.c_double * nelts)()

        lib.lambdata_bixlamsq_get(labels_ptrs, data)

        labels = [buf.value.decode(errors='ignore').strip() for buf in labels_bufs]
        data_rows = [data[i:i+ncols] for i in range(0, nelts, ncols)]

        return pd.DataFrame(data_rows, columns=labels)
    except Exception:
        return None


def get_bias_data_direct():
    """Get bias potential data directly from CHARMM memory.

    Returns bias information from the last dynamics step including
    block indices, class, reference values, and force constants.

    Returns
    -------
    pandas.DataFrame or None
        DataFrame with bias data, or None if API not available

    See Also
    --------
    get_biases : Command-based query (Python state)
    """
    if not _check_lambdata_available():
        return None

    try:
        nbiasv = lib.lambdata_get_nbiasv()
        ncols = lib.lambdata_bias_get_ncols()
        nelts = nbiasv * ncols

        name_size = lib.dataframe_get_name_size()
        labels_bufs = [ctypes.create_string_buffer(name_size) for _ in range(ncols)]
        labels_ptrs = (ctypes.c_char_p * ncols)(*map(ctypes.addressof, labels_bufs))
        data = (ctypes.c_double * nelts)()

        lib.lambdata_bias_get(labels_ptrs, data)

        labels = [buf.value.decode(errors='ignore').strip() for buf in labels_bufs]
        data_rows = [data[i:i+ncols] for i in range(0, nelts, ncols)]

        return pd.DataFrame(data_rows, columns=labels)
    except Exception:
        return None


def get_msld_blocks_direct(nblocks=None):
    """Get MSLD block data directly from CHARMM memory.

    Provides fast access to site indices and lambda values for all blocks.

    Parameters
    ----------
    nblocks : int, optional
        Number of blocks. If None, uses get_nblock_direct().

    Returns
    -------
    pandas.DataFrame or None
        DataFrame with columns ['ISITE', 'BIELAM', 'BIXLAM'],
        or None if API not available

    See Also
    --------
    get_msld_state : Command-based query (Python state)
    """
    if not _check_msldata_available():
        return None

    try:
        if nblocks is None:
            nblocks = get_nblock_direct()
            if nblocks is None:
                nblocks = _state.nblocks

        nsteps = lib.msldata_get_nsteps()
        ncols = lib.msldata_block_get_ncols()
        nelts = nblocks * nsteps * ncols

        name_size = lib.dataframe_get_name_size()
        labels_bufs = [ctypes.create_string_buffer(name_size) for _ in range(ncols)]
        labels_ptrs = (ctypes.c_char_p * ncols)(*map(ctypes.addressof, labels_bufs))
        data = (ctypes.c_double * nelts)()

        lib.msldata_block_get(labels_ptrs, data)

        labels = [buf.value.decode(errors='ignore').strip() for buf in labels_bufs]
        data_rows = [data[i:i+ncols] for i in range(0, nelts, ncols)]

        return pd.DataFrame(data_rows, columns=labels)
    except Exception:
        return None


def get_msld_bias_direct():
    """Get MSLD bias data directly from CHARMM memory.

    Provides fast access to MSLD biasing potential information.

    Returns
    -------
    pandas.DataFrame or None
        DataFrame with MSLD bias data, or None if API not available

    See Also
    --------
    get_bias_data_direct : Lambda-dynamics bias data
    get_biases : Command-based query (Python state)
    """
    if not _check_msldata_available():
        return None

    try:
        nrows = lib.msldata_bias_get_nrows()
        ncols = lib.msldata_bias_get_ncols()
        nelts = nrows * ncols

        name_size = lib.dataframe_get_name_size()
        labels_bufs = [ctypes.create_string_buffer(name_size) for _ in range(ncols)]
        labels_ptrs = (ctypes.c_char_p * ncols)(*map(ctypes.addressof, labels_bufs))
        data = (ctypes.c_double * nelts)()

        lib.msldata_bias_get(labels_ptrs, data)

        labels = [buf.value.decode(errors='ignore').strip() for buf in labels_bufs]
        data_rows = [data[i:i+ncols] for i in range(0, nelts, ncols)]

        return pd.DataFrame(data_rows, columns=labels)
    except Exception:
        return None


def enable_lambda_data_collection():
    """Enable lambda data collection for the next dynamics run.

    When enabled, CHARMM will collect lambda dynamics data during
    the next dynamics run, which can then be retrieved using
    get_lambda_squared_direct() and get_bias_data_direct().

    Returns
    -------
    bool
        True if collection was already enabled, False otherwise

    See Also
    --------
    disable_lambda_data_collection : Disable collection
    get_lambda_squared_direct : Retrieve collected data
    """
    try:
        was_on = lib.lambdata_on()
        return bool(was_on)
    except AttributeError:
        return False


def disable_lambda_data_collection():
    """Disable lambda data collection.

    Returns
    -------
    bool
        True if collection was enabled before, False otherwise

    See Also
    --------
    enable_lambda_data_collection : Enable collection
    """
    try:
        was_on = lib.lambdata_off()
        return bool(was_on)
    except AttributeError:
        return False


def enable_msld_data_collection():
    """Enable MSLD data collection for the next dynamics run.

    When enabled, CHARMM will collect MSLD data during the next
    dynamics run, which can then be retrieved using the direct
    access functions.

    Returns
    -------
    bool
        True if collection was already enabled, False otherwise

    See Also
    --------
    disable_msld_data_collection : Disable collection
    get_msld_blocks_direct : Retrieve collected block data
    get_msld_step_direct : Retrieve collected step data
    """
    try:
        was_on = lib.msldata_on()
        return bool(was_on)
    except AttributeError:
        return False


def disable_msld_data_collection():
    """Disable MSLD data collection.

    Returns
    -------
    bool
        True if collection was enabled before, False otherwise

    See Also
    --------
    enable_msld_data_collection : Enable collection
    """
    try:
        was_on = lib.msldata_off()
        return bool(was_on)
    except AttributeError:
        return False


def get_msld_step_direct():
    """Get MSLD step data directly from CHARMM memory.

    Returns step-level MSLD information including NBLOCKS, NBIASV,
    NSITES, TBLD (temperature), and FCNFORM (functional form).

    Returns
    -------
    pandas.DataFrame or None
        DataFrame with columns ['STEP', 'TIME', 'NBLOCKS', 'NBIASV',
        'NSITES', 'TBLD', 'FCNFORM'], or None if API not available

    See Also
    --------
    get_msld_blocks_direct : Per-block data
    get_msld_state : Command-based query (Python state)
    """
    if not _check_msldata_available():
        return None

    try:
        nsteps = lib.msldata_get_nsteps()
        ncols = lib.msldata_step_get_ncols()
        nelts = nsteps * ncols

        name_size = lib.dataframe_get_name_size()
        labels_bufs = [ctypes.create_string_buffer(name_size) for _ in range(ncols)]
        labels_ptrs = (ctypes.c_char_p * ncols)(*map(ctypes.addressof, labels_bufs))
        data = (ctypes.c_double * nelts)()

        lib.msldata_step_get(labels_ptrs, data)

        labels = [buf.value.decode(errors='ignore').strip() for buf in labels_bufs]
        data_rows = [data[i:i+ncols] for i in range(0, nelts, ncols)]

        return pd.DataFrame(data_rows, columns=labels)
    except Exception:
        return None


def get_msld_theta_direct(nsites=None):
    """Get MSLD theta values directly from CHARMM memory.

    Returns theta values for all MSLD sites from the dynamics run.

    Parameters
    ----------
    nsites : int, optional
        Number of sites. If None, attempts to determine from step data.

    Returns
    -------
    numpy.ndarray or None
        Array of theta values, or None if API not available

    See Also
    --------
    get_msld_state : Command-based query (Python state)
    theta_bias : Set theta biasing
    """
    if not _check_msldata_available():
        return None

    try:
        nsteps = lib.msldata_get_nsteps()
        ncols = lib.msldata_theta_get_ncols()
        nelts = nsteps * ncols

        data = (ctypes.c_double * nelts)()
        lib.msldata_theta_get(data)

        # Reshape to (nsteps, ncols)
        arr = np.array(list(data)).reshape(nsteps, ncols)
        return arr
    except Exception:
        return None


def get_msld_nsubs_direct():
    """Get MSLD number of substituents per site directly from CHARMM memory.

    Returns
    -------
    numpy.ndarray or None
        Array of nsubs values per site, or None if API not available

    See Also
    --------
    get_msld_state : Command-based query (Python state)
    """
    if not _check_msldata_available():
        return None

    try:
        nsteps = lib.msldata_get_nsteps()
        # nsubs has (nsitemld - 1) columns per step
        step_df = get_msld_step_direct()
        if step_df is None or len(step_df) == 0:
            return None
        nsites = int(step_df['NSITES'].iloc[0]) - 1
        nelts = nsteps * nsites

        data = (ctypes.c_double * nelts)()
        lib.msldata_nsubs_get(data)

        arr = np.array(list(data)).reshape(nsteps, nsites)
        return arr
    except Exception:
        return None


def is_direct_access_available():
    """Check if direct memory access functions are available.

    The direct access functions require CHARMM to be compiled with
    KEY_BLOCK=1 and the appropriate API functions to be exposed.

    Returns
    -------
    dict
        {'lambdata': bool, 'msldata': bool} indicating availability
        of each API

    Examples
    --------
    >>> avail = block.is_direct_access_available()
    >>> if avail['lambdata']:
    ...     # Use fast direct access
    ...     df = block.get_lambda_squared_direct()
    ... else:
    ...     # Fall back to command-based
    ...     state = block.get_lambda_dynamics_state()
    """
    return {
        'lambdata': _check_lambdata_available(),
        'msldata': _check_msldata_available(),
        'blockdata': _check_blockdata_available()
    }


# -----------------------------------------------------------------------------
# Block Configuration Data (api_blockdata) - Read/Write Access
# -----------------------------------------------------------------------------

def _check_blockdata_available():
    """Check if blockdata API is available and BLOCK is active."""
    try:

        lib.blockdata_is_active.restype = ctypes.c_int
        return bool(lib.blockdata_is_active())
    except (AttributeError, OSError, RuntimeError):
        return False


def _get_blockdata_nblock():
    """Get the current number of blocks from CHARMM."""
    if not _check_blockdata_available():
        return 0

    try:
        return max(int(lib.blockdata_get_nblock()), 0)
    except Exception:
        return 0


def _get_blockdata_nbiasv():
    """Get the number of currently allocated bias slots from CHARMM."""
    if not _check_blockdata_available():
        return 0

    try:
        return max(int(lib.blockdata_get_nbiasv()), 0)
    except Exception:
        return 0


def _normalize_positive_index(value):
    """Convert a user-facing 1-based index to int, or return None."""
    try:
        index = int(value)
    except (TypeError, ValueError):
        return None

    return index if index > 0 else None


def _valid_block_index(block_id, nblock=None):
    """Check whether a block index is valid for the current BLOCK state."""
    if nblock is None:
        nblock = _get_blockdata_nblock()

    index = _normalize_positive_index(block_id)
    return index is not None and index <= nblock


def _check_msld_active():
    """Check if MSLD is active (fullblcoep array allocated).

    This checks the Fortran qmld flag via the blockdata API.
    fullblcoep is only allocated when MSLD is enabled.

    IMPORTANT: Always check CHARMM state first because Python state can be
    stale after clear() is called (which deallocates CHARMM arrays but may
    not reset all Python state fields).

    Returns
    -------
    bool
        True if MSLD is active and fullblcoep is available
    """
    # ALWAYS check CHARMM state first - Python state can be stale after clear()
    try:

        lib.blockdata_msld_active.restype = ctypes.c_int
        return bool(lib.blockdata_msld_active())
    except (AttributeError, OSError, RuntimeError):
        # API not available, fall back to Python state
        return _state.msld.get('enabled', False)


def _check_fullblcoep_allocated():
    """Check if fullblcoep array is allocated in CHARMM.

    Returns
    -------
    bool
        True if fullblcoep is allocated
    """
    try:

        lib.blockdata_fullblcoep_allocated.restype = ctypes.c_int
        return bool(lib.blockdata_fullblcoep_allocated())
    except (AttributeError, OSError, RuntimeError):
        # API not available, assume not allocated unless MSLD enabled
        return _state.msld.get('enabled', False)


def _get_coefficient_matrix_cached():
    """Get coefficient matrix from Python state cache.

    This is a fallback for when fullblcoep is not allocated
    (i.e., MSLD is not enabled). Returns the coefficients
    as tracked by Python state.

    Returns
    -------
    numpy.ndarray or None
        2D array of coefficients (nblock x nblock), or None if not available
    """
    n = _state.nblocks
    if n <= 0:
        return None

    # Create matrix filled with 1.0 (default coefficient)
    matrix = np.ones((n, n))

    # Fill in any modified coefficients
    # Coefficients are stored as dicts with 'default' key, or as floats
    for (i, j), val in _state.coefficients.items():
        if 1 <= i <= n and 1 <= j <= n:
            # Extract value - could be dict with 'default' key or plain float
            if isinstance(val, dict):
                coef_val = val.get('default', 1.0)
            else:
                coef_val = val
            matrix[i - 1, j - 1] = coef_val
            matrix[j - 1, i - 1] = coef_val  # Symmetric

    return matrix


def set_coefficient_direct(i, j, value):
    """Set a coefficient value directly in CHARMM memory.

    Bypasses Python state tracking and command execution for fast updates.
    Use with caution - Python state will be out of sync.

    This function writes to fullblcoep which is ONLY allocated when MSLD
    is enabled. For non-MSLD BLOCK setups, use the modify() context manager
    with coef() instead.

    Parameters
    ----------
    i : int
        First block index (1-based)
    j : int
        Second block index (1-based)
    value : float
        Coefficient value

    Returns
    -------
    bool
        True if successful, False if MSLD not active or other error

    See Also
    --------
    coef : Command-based coefficient setting with state tracking
    modify : Context manager for re-entering BLOCK to modify parameters
    """
    if not _check_blockdata_available():
        return False

    # fullblcoep only exists with MSLD
    if not _check_msld_active() or not _check_fullblcoep_allocated():
        return False

    try:
        lib.blockdata_coef_set(
            ctypes.c_int(i),
            ctypes.c_int(j),
            ctypes.c_double(value)
        )
        # Update Python state to stay in sync (use dict format like coef())
        key = (min(i, j), max(i, j))
        if key not in _state.coefficients:
            _state.coefficients[key] = {}
        _state.coefficients[key]['default'] = value
        return True
    except Exception:
        return False


def set_coefficient_live(i, j, value):
    """Set a coefficient value in a running BLOCK system.

    This is the recommended way to modify coefficients after end() has been
    called. It automatically chooses the best method:
    - For MSLD: Uses direct ctypes access to fullblcoep (fastest)
    - For basic BLOCK: Uses modify() context manager with CHARMM commands

    Parameters
    ----------
    i : int
        First block index (1-based)
    j : int
        Second block index (1-based)
    value : float
        Coefficient value to set

    Returns
    -------
    bool
        True if successful

    Examples
    --------
    >>> block.initialize(3)
    >>> block.coef(1, 2, 0.5)
    >>> block.end()
    >>>
    >>> # Later, modify the coefficient
    >>> block.set_coefficient_live(1, 2, 0.8)
    True

    See Also
    --------
    coef : Set coefficient during initial BLOCK setup
    set_coefficient_direct : Low-level ctypes access (MSLD only)
    modify : Context manager for re-entering BLOCK
    """
    if not _state.active:
        return False

    # Try direct access first (fastest, but requires MSLD)
    if _check_msld_active() and _check_fullblcoep_allocated():
        return set_coefficient_direct(i, j, value)

    # Fall back to CHARMM command via modify() context
    try:
        with modify():
            coef(i, j, value)
        return True
    except Exception:
        return False


def get_lambda_values_direct():
    """Get current lambda values for all blocks directly from CHARMM.

    Returns
    -------
    numpy.ndarray or None
        Array of lambda values (size nblock), or None if not available

    See Also
    --------
    get_lambda_squared_direct : Lambda^2 values from dynamics data
    """
    if not _check_blockdata_available():
        return None

    try:
        nblock = lib.blockdata_get_nblock()
        if nblock <= 0:
            return None

        data = (ctypes.c_double * nblock)()
        lib.blockdata_lambda_get(data)

        return np.array(list(data))
    except Exception:
        return None


def get_ph_rex_state_direct():
    """Read the compact CHARMM state used by pH/MSLD label exchange.

    The core refreshes lambda occupancies from the current theta coordinates
    before returning. This matters for BLaDE runs that do not write an LMD
    frame at every exchange attempt.

    Returns
    -------
    dict or None
        ``ph``, ``temperature``, ``lambdas``, ``biases`` (BIELAM), and the
        static protocol arrays ``masses``, ``frictions``, ``fixed``, and
        ``sites``; or None when MSLD state is unavailable.
    """
    if not _check_blockdata_available():
        return None

    try:
        nblock = int(lib.blockdata_get_nblock())
        if nblock < 1:
            return None

        ph = ctypes.c_double()
        temperature = ctypes.c_double()
        lambdas = (ctypes.c_double * nblock)()
        biases = (ctypes.c_double * nblock)()
        masses = (ctypes.c_double * nblock)()
        frictions = (ctypes.c_double * nblock)()
        fixed = (ctypes.c_int * nblock)()
        sites = (ctypes.c_int * nblock)()
        status = ctypes.c_int()
        lib.blockdata_ph_rex_get(
            ctypes.byref(ph),
            ctypes.byref(temperature),
            lambdas,
            biases,
            masses,
            frictions,
            fixed,
            sites,
            ctypes.c_int(nblock),
            ctypes.byref(status),
        )
        if status.value != 0:
            return None
        return {
            'ph': ph.value,
            'temperature': temperature.value,
            'lambdas': np.array(lambdas, dtype=np.float64),
            'biases': np.array(biases, dtype=np.float64),
            'masses': np.array(masses, dtype=np.float64),
            'frictions': np.array(frictions, dtype=np.float64),
            'fixed': np.array(fixed, dtype=np.bool_),
            'sites': np.array(sites, dtype=np.int32),
        }
    except Exception:
        return None


def set_ph_rex_label_direct(ph, biases):
    """Apply a pH label and its complete BIELAM vector.

    Dynamical state (coordinates, theta, velocities, masses, friction, and
    FFIX) is deliberately not changed. An active BLaDE MSLD context receives
    the new bias vector before this function reports success.
    """
    if not _check_blockdata_available():
        return False

    try:
        ph = float(ph)
        values = np.asarray(biases, dtype=np.float64)
        nblock = int(lib.blockdata_get_nblock())
    except (TypeError, ValueError):
        return False
    if (not np.isfinite(ph) or values.ndim != 1 or
            values.size != nblock or not np.all(np.isfinite(values))):
        return False

    try:
        data = (ctypes.c_double * nblock)(*values.tolist())
        status = ctypes.c_int()
        lib.blockdata_ph_rex_set(
            ctypes.c_double(ph),
            data,
            ctypes.c_int(nblock),
            ctypes.byref(status),
        )
        return status.value == 0
    except Exception:
        return False


def _get_ldin_params_from_charmm_internal(block_id):
    """Internal: Get LDIN parameters from CHARMM memory.

    LDIN arrays are only allocated when lambda dynamics is enabled (QLDM).
    Returns None if lambda dynamics is not enabled.
    """
    if not _check_blockdata_available():
        return None

    # Check if lambda dynamics is enabled - LDIN arrays not allocated otherwise
    qldm = is_lambda_dynamics_enabled_direct()
    if not qldm:
        return None

    try:
        lam0 = ctypes.c_double()
        vel = ctypes.c_double()
        mass = ctypes.c_double()
        bias = ctypes.c_double()
        friction = ctypes.c_double()

        lib.blockdata_ldin_get(
            ctypes.c_int(block_id),
            ctypes.byref(lam0),
            ctypes.byref(vel),
            ctypes.byref(mass),
            ctypes.byref(bias),
            ctypes.byref(friction)
        )

        return {
            'lambda_sq': lam0.value,
            'velocity': vel.value,
            'mass': mass.value,
            'bias': bias.value,
            'friction': friction.value
        }
    except Exception:
        return None


def set_ldin_params_direct(block_id, lambda_sq, velocity, mass, bias,
                           friction=0.0):
    """Set LDIN parameters directly in CHARMM memory.

    Bypasses Python state tracking for fast updates.

    Parameters
    ----------
    block_id : int
        Block number (1-based)
    lambda_sq : float
        Lambda squared value
    velocity : float
        Lambda velocity
    mass : float
        Lambda mass
    bias : float
        Biasing energy (bielam)
    friction : float, optional
        Friction coefficient (biblam). Default 0.0.

    Returns
    -------
    bool
        True if successful

    See Also
    --------
    ldin : Command-based LDIN setting with state tracking
    """
    if not _check_blockdata_available():
        return False

    if not _valid_block_index(block_id):
        return False

    if is_lambda_dynamics_enabled_direct() is not True:
        return False

    try:
        lambda_sq = float(lambda_sq)
        velocity = float(velocity)
        mass = float(mass)
        bias = float(bias)
        friction = float(friction)
    except (TypeError, ValueError):
        return False

    if lambda_sq < 0.0:
        return False

    try:
        lib.blockdata_ldin_set(
            ctypes.c_int(block_id),
            ctypes.c_double(lambda_sq),
            ctypes.c_double(velocity),
            ctypes.c_double(mass),
            ctypes.c_double(bias),
            ctypes.c_double(friction)
        )
        return True
    except Exception:
        return False


def get_ffix_direct():
    """Get FFIX (fixed lambda) flags for all blocks from CHARMM memory.

    Returns
    -------
    list of bool or None
        List of flags (True=fixed, False=dynamic) for each block,
        or None if qlfix array is not allocated.

    See Also
    --------
    set_ffix_direct : Set FFIX flag for a specific block
    msld : Set FFIX during MSLD initialization
    """
    if not _check_blockdata_available():
        return None

    try:
        nblock = lib.blockdata_get_nblock()
        if nblock <= 0:
            return None

        flags = (ctypes.c_int * nblock)()
        status = ctypes.c_int()

        lib.blockdata_ffix_get(flags, ctypes.byref(status))

        if status.value != 0:
            return None

        return [bool(flags[i]) for i in range(nblock)]
    except Exception:
        return None


def set_ffix_direct(block_id, is_fixed):
    """Set FFIX (fixed lambda) flag for a specific block in CHARMM memory.

    Parameters
    ----------
    block_id : int
        Block number (1-based)
    is_fixed : bool
        True to fix this block's lambda, False for dynamic.

    Returns
    -------
    bool
        True if successful, False if qlfix not allocated.

    See Also
    --------
    get_ffix_direct : Get FFIX flags for all blocks
    """
    if not _check_blockdata_available():
        return False

    flags = get_ffix_direct()
    if flags is None:
        return False

    if not _valid_block_index(block_id, len(flags)):
        return False

    try:
        status = ctypes.c_int()
        lib.blockdata_ffix_set(
            ctypes.c_int(block_id),
            ctypes.c_int(1 if is_fixed else 0),
            ctypes.byref(status)
        )
        return status.value == 0
    except Exception:
        return False


def get_friction_direct():
    """Get per-block friction coefficients (biblam) from CHARMM memory.

    Returns
    -------
    numpy.ndarray or None
        Array of friction values (size nblock), or None if not allocated.

    See Also
    --------
    set_friction_direct : Set friction for a specific block
    ldin : Set friction during LDIN initialization
    """
    if not _check_blockdata_available():
        return None

    try:
        nblock = lib.blockdata_get_nblock()
        if nblock <= 0:
            return None

        frictions = (ctypes.c_double * nblock)()
        status = ctypes.c_int()

        lib.blockdata_friction_get(frictions, ctypes.byref(status))

        if status.value != 0:
            return None

        return np.array(list(frictions))
    except Exception:
        return None


def set_friction_direct(block_id, friction):
    """Set friction coefficient for a specific block in CHARMM memory.

    Parameters
    ----------
    block_id : int
        Block number (1-based)
    friction : float
        Friction coefficient value.

    Returns
    -------
    bool
        True if successful.

    See Also
    --------
    get_friction_direct : Get friction for all blocks
    """
    if not _check_blockdata_available():
        return False

    frictions = get_friction_direct()
    if frictions is None:
        return False

    if not _valid_block_index(block_id, len(frictions)):
        return False

    try:
        friction = float(friction)
    except (TypeError, ValueError):
        return False

    try:
        lib.blockdata_friction_set(
            ctypes.c_int(block_id),
            ctypes.c_double(friction)
        )
        return True
    except Exception:
        return False


def get_bias_params_direct(bias_id):
    """Get bias parameters directly from CHARMM memory.

    Parameters
    ----------
    bias_id : int
        Bias index (1-based)

    Returns
    -------
    dict or None
        {'block_i': int, 'block_j': int, 'cls': int, 'ref_up': float,
         'ref_low': float, 'force_const': float, 'power': int}
        or None if not available

    See Also
    --------
    get_biases : Python state-based query
    """
    if not _check_blockdata_available():
        return None

    try:
        block_i = ctypes.c_int()
        block_j = ctypes.c_int()
        cls = ctypes.c_int()
        reup = ctypes.c_double()
        rlow = ctypes.c_double()
        kbias = ctypes.c_double()
        pbias = ctypes.c_int()

        lib.blockdata_bias_get(
            ctypes.c_int(bias_id),
            ctypes.byref(block_i),
            ctypes.byref(block_j),
            ctypes.byref(cls),
            ctypes.byref(reup),
            ctypes.byref(rlow),
            ctypes.byref(kbias),
            ctypes.byref(pbias)
        )

        return {
            'block_i': block_i.value,
            'block_j': block_j.value,
            'cls': cls.value,
            'ref_up': reup.value,
            'ref_low': rlow.value,
            'force_const': kbias.value,
            'power': pbias.value
        }
    except Exception:
        return None


def get_temperature_direct():
    """Get lambda dynamics temperature directly from CHARMM memory.

    Returns
    -------
    float or None
        Temperature (tbld), or None if not available

    See Also
    --------
    get_langevin_state : Python state query including temperature
    """
    if not _check_blockdata_available():
        return None

    try:

        lib.blockdata_get_temperature.restype = ctypes.c_double
        return float(lib.blockdata_get_temperature())
    except Exception:
        return None


def set_temperature_direct(temp):
    """Set lambda dynamics temperature directly in CHARMM memory.

    Parameters
    ----------
    temp : float
        Temperature value

    Returns
    -------
    bool
        True if successful
    """
    if not _check_blockdata_available():
        return False

    try:

        lib.blockdata_set_temperature.argtypes = [ctypes.c_double]
        lib.blockdata_set_temperature(temp)
        return True
    except Exception:
        return False


def is_lambda_dynamics_enabled_direct():
    """Check if lambda dynamics is enabled directly from CHARMM.

    Returns
    -------
    bool or None
        True if enabled, False if disabled, None if not available
    """
    if not _check_blockdata_available():
        return None

    try:
        return bool(lib.blockdata_qldm_enabled())
    except Exception:
        return None


def is_theta_enabled_direct():
    """Check if theta mode is enabled directly from CHARMM.

    Returns
    -------
    bool or None
        True if enabled, False if disabled, None if not available
    """
    if not _check_blockdata_available():
        return None

    try:
        return bool(lib.blockdata_theta_enabled())
    except Exception:
        return None


def is_langevin_enabled_direct():
    """Check if Langevin coupling is enabled directly from CHARMM.

    Returns
    -------
    bool or None
        True if enabled, False if disabled, None if not available
    """
    if not _check_blockdata_available():
        return None

    try:
        return bool(lib.blockdata_langevin_enabled())
    except Exception:
        return None


def get_nsites_direct():
    """Get number of MSLD sites directly from CHARMM.

    Returns
    -------
    int or None
        Number of MSLD sites, or None if not available
    """
    if not _check_blockdata_available():
        return None

    try:
        return int(lib.blockdata_get_nsites())
    except Exception:
        return None


def get_site_assignments_direct():
    """Get site assignment for each block directly from CHARMM.

    Site assignments (isitemld array) are only available when MSLD is enabled.
    Returns None if MSLD is not active.

    Returns
    -------
    numpy.ndarray or None
        Array of site IDs for each block (1-indexed), or None if not available
    """
    if not _check_blockdata_available():
        return None

    # Site assignments only exist with MSLD
    if not _check_msld_active():
        return None

    try:
        nblock = lib.blockdata_get_nblock()
        if nblock <= 0:
            return None

        data = (ctypes.c_int * nblock)()
        lib.blockdata_sites_get(data)

        return np.array(list(data))
    except Exception:
        return None


def get_softcore_mode_direct():
    """Get soft-core mode directly from CHARMM.

    Returns
    -------
    str or None
        'off', 'on', or 'w14', or None if not available
    """
    if not _check_blockdata_available():
        return None

    try:
        mode = int(lib.blockdata_softcore_mode())
        return {0: 'off', 1: 'on', 2: 'w14'}.get(mode, 'unknown')
    except Exception:
        return None


def get_pme_mode_direct():
    """Get PME handling mode for MSLD directly from CHARMM.

    Returns
    -------
    str or None
        'off', 'nn', 'ex', or 'on', or None if not available
    """
    if not _check_blockdata_available():
        return None

    try:
        mode = int(lib.blockdata_pme_mode())
        return {0: 'off', 1: 'nn', 2: 'ex', 3: 'on'}.get(mode, 'unknown')
    except Exception:
        return None


# =============================================================================
# Basic BLOCK Direct Access (BLCOEP - triangular matrix)
# =============================================================================

def _check_qblock_active():
    """Check if basic BLOCK is active (QBLOCK flag).

    Returns True if BLOCK has been initialized (even without MSLD).
    """
    try:

        lib.blockdata_qblock_active.restype = ctypes.c_int
        return bool(lib.blockdata_qblock_active())
    except (AttributeError, OSError, RuntimeError):
        return False


def _check_blcoep_allocated():
    """Check if BLCOEP triangular matrix is allocated."""
    try:

        lib.blockdata_blcoep_allocated.restype = ctypes.c_int
        return bool(lib.blockdata_blcoep_allocated())
    except (AttributeError, OSError, RuntimeError):
        return False


def _get_coefficient_from_charmm_internal(i, j):
    """Internal: Get coefficient from CHARMM's BLCOEP array."""
    if not _check_blockdata_available():
        return None

    try:

        lib.blockdata_blcoep_get_ij.restype = ctypes.c_double
        lib.blockdata_blcoep_get_ij.argtypes = [ctypes.c_int, ctypes.c_int]

        val = lib.blockdata_blcoep_get_ij(ctypes.c_int(i), ctypes.c_int(j))

        # -999.0 is the error sentinel
        if val < -998.0:
            return None
        return float(val)
    except Exception:
        return None


def _get_coefficient_matrix_from_charmm_internal():
    """Internal: Get coefficient matrix from CHARMM's BLCOEP array."""
    if not _check_blockdata_available():
        return None

    if not _check_qblock_active() or not _check_blcoep_allocated():
        return None

    try:

        lib.blockdata_get_nblock.restype = ctypes.c_int
        lib.blockdata_get_ninter.restype = ctypes.c_int

        nblock = lib.blockdata_get_nblock()
        ninter = lib.blockdata_get_ninter()

        if nblock <= 0 or ninter <= 0:
            return None

        # Get triangular matrix data
        data = (ctypes.c_double * ninter)()
        status = ctypes.c_int()

        lib.blockdata_blcoep_get.argtypes = [
            ctypes.POINTER(ctypes.c_double),
            ctypes.POINTER(ctypes.c_int)
        ]
        lib.blockdata_blcoep_get(data, ctypes.byref(status))

        if status.value != 0:
            return None

        # Reconstruct full symmetric matrix from triangular storage
        # BLCOEP indexing: idx = i*(i-1)/2 + j where i >= j (1-based)
        matrix = np.zeros((nblock, nblock))
        for i in range(1, nblock + 1):
            for j in range(1, i + 1):
                idx = i * (i - 1) // 2 + j - 1  # Convert to 0-based array index
                val = data[idx]
                matrix[i - 1, j - 1] = val
                matrix[j - 1, i - 1] = val  # Symmetric

        return matrix
    except Exception:
        return None


def _get_nblock_from_charmm_internal():
    """Internal: Get number of blocks from CHARMM."""
    if not _check_blockdata_available():
        return None

    try:

        lib.blockdata_get_nblock.restype = ctypes.c_int
        return int(lib.blockdata_get_nblock())
    except Exception:
        return None


def _is_block_active_in_charmm_internal():
    """Internal: Check if BLOCK is active in CHARMM."""
    if not _check_blockdata_available():
        return None
    return _check_qblock_active()


def _get_atom_blocks_from_charmm_internal():
    """Get per-atom block assignments directly from CHARMM's IBLCKP array.

    Returns
    -------
    list or None
        List of block IDs (1-based) for each atom, indexed by atom index (0-based).
        Returns None if BLOCK not active or API unavailable.

    Notes
    -----
    This reads IBLCKP(i) for each atom from CHARMM's internal state,
    which works regardless of whether BLOCK was set up via Python API
    or charmm_script().
    """
    if not _check_blockdata_available():
        return None

    try:
        import pycharmm.psf as psf
        natom = psf.get_natom()
        if natom <= 0:
            return None


        assignments = (ctypes.c_int * natom)()
        status = ctypes.c_int()

        lib.blockdata_get_atom_blocks(assignments, ctypes.c_int(natom), ctypes.byref(status))

        if status.value != 0:
            return None

        return list(assignments)
    except Exception:
        return None


def get_atom_block_assignments_direct():
    """Get per-atom block assignments directly from CHARMM memory.

    This reads from CHARMM's internal IBLCKP array, which works for
    both Python API and charmm_script() setup methods.

    Returns
    -------
    dict or None
        {atom_index (1-based): block_id (1-based)} for all atoms,
        or None if BLOCK not active or API unavailable.

    Examples
    --------
    >>> block.initialize(3)
    >>> block.call(2, 'resname TIP3')
    >>> assignments = block.get_atom_block_assignments_direct()
    >>> # atoms in TIP3 residues will have value 2, others 1
    """
    atom_blocks = _get_atom_blocks_from_charmm_internal()
    if atom_blocks is None:
        return None

    # Convert to dict with 1-based atom indices
    return {i + 1: block_id for i, block_id in enumerate(atom_blocks)}


def verify_coefficients_with_charmm():
    """Verify Python-cached coefficients match CHARMM's internal state.

    Compares Python's cached coefficient values with the actual values
    stored in CHARMM's BLCOEP array.

    Returns
    -------
    dict
        {
            'match': bool,  # True if all match
            'python_cache': dict,  # Python cached values
            'charmm_values': dict,  # CHARMM internal values
            'differences': list  # List of mismatches
        }

    Examples
    --------
    >>> block.initialize(3)
    >>> block.coef(1, 2, 0.5)
    >>> block.end()
    >>> result = block.verify_coefficients_with_charmm()
    >>> print(result['match'])  # True if Python and CHARMM agree
    """
    result = {
        'match': False,
        'python_cache': {},
        'charmm_values': {},
        'differences': []
    }

    n = _state.nblocks
    if n <= 0:
        return result

    # Get Python cached values
    for i in range(1, n + 1):
        for j in range(1, i + 1):
            key = (j, i) if j < i else (i, j)
            val = _state.coefficients.get(key, 1.0)
            if isinstance(val, dict):
                val = val.get('default', 1.0)
            result['python_cache'][(i, j)] = val

    # Get CHARMM values (use internal function to avoid deprecation warning)
    for i in range(1, n + 1):
        for j in range(1, i + 1):
            charmm_val = _get_coefficient_from_charmm_internal(i, j)
            if charmm_val is not None:
                result['charmm_values'][(i, j)] = charmm_val

    # Compare
    all_match = True
    for key, py_val in result['python_cache'].items():
        charmm_val = result['charmm_values'].get(key)
        if charmm_val is None:
            result['differences'].append({
                'indices': key,
                'python': py_val,
                'charmm': None,
                'reason': 'CHARMM value unavailable'
            })
            all_match = False
        elif abs(py_val - charmm_val) > 1e-6:
            result['differences'].append({
                'indices': key,
                'python': py_val,
                'charmm': charmm_val,
                'reason': f'Values differ: {py_val} vs {charmm_val}'
            })
            all_match = False

    result['match'] = all_match
    return result


def get_fnex_direct():
    """Get FNEX factor directly from CHARMM memory.

    Returns
    -------
    float or None
        FNEX exponential factor, or None if not available
    """
    if not _check_blockdata_available():
        return None

    try:
        return float(lib.blockdata_get_fnex())
    except Exception:
        return None


def get_ph_direct():
    """Get pH value directly from CHARMM memory.

    For constant-pH MD simulations.

    Returns
    -------
    float or None
        pH value, or None if not available
    """
    if not _check_blockdata_available():
        return None

    try:
        lib.blockdata_get_ph.restype = ctypes.c_double
        return float(lib.blockdata_get_ph())
    except Exception:
        return None


def set_ph_direct(ph):
    """Set pH value directly in CHARMM memory.

    For constant-pH MD simulations.

    Parameters
    ----------
    ph : float
        pH value

    Returns
    -------
    bool
        True if successful
    """
    if not _check_blockdata_available():
        return False

    try:
        lib.blockdata_set_ph(ctypes.c_double(ph))
        return True
    except Exception:
        return False


def set_atom_blocks_direct(atom_indices, block_id):
    """Set block assignment for multiple atoms directly in CHARMM memory.

    This is the Pythonic way to assign atoms to blocks - no need for
    initialize() or end() calls. Block count auto-expands as needed.

    Parameters
    ----------
    atom_indices : array-like
        Array of atom indices (1-based) to assign
    block_id : int
        Block number to assign atoms to

    Returns
    -------
    bool
        True if successful, False otherwise

    See Also
    --------
    call : High-level function that uses this internally
    """
    if not _check_blockdata_available():
        return False

    try:
        import numpy as np
        indices = np.array(atom_indices, dtype=np.int32)
        count = len(indices)
        status = ctypes.c_int()

        lib.blockdata_set_atom_blocks(
            indices.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
            ctypes.c_int(block_id),
            ctypes.c_int(count),
            ctypes.byref(status)
        )

        if status.value != 0:
            return False

        return True
    except Exception:
        return False


def set_bias_direct(bias_id, block_i, block_j, cls, ref, cforce, npower):
    """Set bias parameters directly in CHARMM memory.

    This updates an existing bias slot in CHARMM memory. Bias storage must
    already be allocated through the standard BLOCK/LDBI path.

    Parameters
    ----------
    bias_id : int
        Bias index (1-based)
    block_i : int
        First block index
    block_j : int
        Second block index
    cls : int
        Bias class (1-12)
    ref : float
        Reference value
    cforce : float
        Force constant
    npower : int
        Power/exponent

    Returns
    -------
    bool
        True if successful

    See Also
    --------
    add_bias : High-level function that uses this internally
    """
    if not _check_blockdata_available():
        return False

    if is_lambda_dynamics_enabled_direct() is not True:
        return False

    nblock = _get_blockdata_nblock()
    nbiasv = _get_blockdata_nbiasv()
    if not _valid_block_index(block_i, nblock):
        return False
    if not _valid_block_index(block_j, nblock):
        return False

    bias_index = _normalize_positive_index(bias_id)
    if bias_index is None or bias_index > nbiasv:
        return False

    try:
        cls = int(cls)
        npower = int(npower)
        ref = float(ref)
        cforce = float(cforce)
    except (TypeError, ValueError):
        return False

    if not 1 <= cls <= 12:
        return False
    if npower < 0:
        return False

    try:
        lib.blockdata_bias_set(
            ctypes.c_int(bias_id),
            ctypes.c_int(block_i),
            ctypes.c_int(block_j),
            ctypes.c_int(cls),
            ctypes.c_double(ref),
            ctypes.c_double(cforce),
            ctypes.c_int(npower)
        )
        return True
    except Exception:
        return False


def set_sites_direct(site_assignments):
    """Set MSLD site assignments directly in CHARMM memory.

    MSLD site arrays must already be allocated by an active MSLD setup.

    Parameters
    ----------
    site_assignments : array-like
        Array of site IDs for each block (1-based indexing)

    Returns
    -------
    bool
        True if successful

    See Also
    --------
    msld : High-level function that uses this internally
    """
    if not _check_blockdata_available():
        return False

    if not _check_msld_active():
        return False

    try:
        sites = np.asarray(site_assignments, dtype=np.int32)
    except Exception:
        return False

    if sites.ndim != 1:
        return False

    count = len(sites)
    if count != _get_blockdata_nblock():
        return False

    if np.any(sites < 0):
        return False

    try:
        lib.blockdata_sites_set(
            sites.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
            ctypes.c_int(count)
        )
        return True
    except Exception:
        return False


def sync_state_from_charmm():
    """Synchronize Python state from CHARMM memory.

    Updates the Python-side state tracking to match CHARMM's current
    internal state. Useful after direct memory modifications or when
    Python state may be out of sync.

    Returns
    -------
    bool
        True if synchronization was successful
    """
    if not _check_blockdata_available():
        return False

    try:
        # Sync basic info
        _state.nblocks = int(lib.blockdata_get_nblock())
        _state.active = _state.nblocks > 0

        if not _state.active:
            return True

        # Sync coefficient matrix
        coef_matrix = get_coefficient_matrix(direct=True)
        if coef_matrix is not None:
            _state.coefficients.clear()
            for i in range(_state.nblocks):
                for j in range(_state.nblocks):
                    _state.coefficients[(i+1, j+1)] = {'default': coef_matrix[i, j]}

        # Sync lambda dynamics state
        qldm = is_lambda_dynamics_enabled_direct()
        if qldm is not None:
            _state.lambda_dynamics['enabled'] = qldm
            _state.lambda_dynamics['theta'] = is_theta_enabled_direct() or False

        # Sync Langevin state
        lang = is_langevin_enabled_direct()
        if lang is not None:
            _state.langevin_enabled = lang
            temp = get_temperature_direct()
            if temp is not None:
                _state.lambda_dynamics['langevin_temp'] = temp

        # Sync MSLD state
        nsites = get_nsites_direct()
        if nsites is not None and nsites > 0:
            _state.msld['enabled'] = True
            sites = get_site_assignments_direct()
            if sites is not None:
                _state.msld['site_assignments'] = {
                    i+1: int(sites[i]) for i in range(len(sites))
                }
            fnex = get_fnex_direct()
            if fnex is not None:
                _state.msld['fnex'] = fnex

        # Sync soft-core
        sc_mode = get_softcore_mode_direct()
        if sc_mode is not None:
            _state.soft_core['enabled'] = sc_mode != 'off'
            _state.soft_core['mode'] = sc_mode

        # Sync PME mode
        pme_mode = get_pme_mode_direct()
        if pme_mode is not None:
            _state.pmel_mode = pme_mode

        # Sync pH
        ph = get_ph_direct()
        if ph is not None:
            _state.phmd_ph = ph

        return True
    except Exception:
        return False


# =============================================================================
# Convenience Functions
# =============================================================================

def setup_dual_topology(reactant_selection, product_selection, lambda_value=0.0):
    """Quick setup for standard 3-block dual-topology FEP.

    Creates:
    - Block 1: Environment (all atoms not in reactant/product)
    - Block 2: Reactant atoms
    - Block 3: Product atoms

    Parameters
    ----------
    reactant_selection : SelectAtoms, str, or selection-like
        Atom selection for reactant state
    product_selection : SelectAtoms, str, or selection-like
        Atom selection for product state
    lambda_value : float, optional
        Initial lambda value (default: 0.0 = reactant state)

    Examples
    --------
    >>> block.setup_dual_topology('segid LIG1', 'segid LIG2', lambda_value=0.5)
    """
    initialize(3)
    call(2, reactant_selection)
    call(3, product_selection)
    set_lambda(lambda_value)
    end()


def get_lambda_schedule(n_windows, method='linear'):
    """Generate lambda values for FEP windows.

    Parameters
    ----------
    n_windows : int
        Number of windows
    method : str, optional
        Spacing method: 'linear', 'gaussian', or 'trapezoidal'

    Returns
    -------
    list
        Lambda values from 0.0 to 1.0
    """
    if method == 'linear':
        return list(np.linspace(0.0, 1.0, n_windows))
    elif method == 'gaussian':
        # Gaussian quadrature-like spacing (denser at endpoints)
        x = np.linspace(0, np.pi, n_windows)
        return list((1 - np.cos(x)) / 2)
    elif method == 'trapezoidal':
        # Trapezoidal rule spacing
        return list(np.linspace(0.0, 1.0, n_windows))
    else:
        raise ValueError(f"Unknown method '{method}'. Use 'linear', 'gaussian', or 'trapezoidal'")
