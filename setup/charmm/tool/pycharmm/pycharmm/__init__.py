# pycharmm: molecular dynamics in python with CHARMM
# Copyright (C) 2018 Josh Buckner

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

from .atom_info import get_atom_table
from .charmm_file import CharmmFile
from .coor import Coordinates
from .custom import CustomDynam
from .dynamics import DynamicsScript
from .energy_func import EnergyFunc
from .energy_mlpot import MLpot
from .safeguards import CharmmError, CharmmScriptError, CharmmWarning
from .loader import lib, print_lib_path, set_mpi_comm, is_initialized, initialize
from .lingo import (charmm_script,
                    get_charmm_variable,
                    get_energy_value,
                    set_charmm_variable,
                    set_default_error_handling,
                    get_default_error_handling,
                    strict_mode,
                    permissive_mode)

from .script import (NonBondedScript, UpdateNonBondedScript, PatchScript,
                     script_factory)

from .select_atoms import (SelectAtoms, parse_selection,
                            protein, backbone, water, ions, nucleic, sidechain,
                            PROTEIN_RESIDUES, BACKBONE_ATOMS, WATER_RESIDUES,
                            ION_RESIDUES, NUCLEIC_RESIDUES)

# Import the Dimens object for pre-initialization configuration
from .dimens import dimens

import pycharmm.cdocker as cdocker
import pycharmm.cons_fix as cons_fix
import pycharmm.cons_harm as cons_harm
import pycharmm.cons_methods as cons_methods
import pycharmm.coor as coor
import pycharmm.correl as correl
import pycharmm.crystal as crystal
import pycharmm.domdec as domdec
import pycharmm.dynamics as dyn
import pycharmm.energy as energy
import pycharmm.generate as gen
import pycharmm.grid as grid
import pycharmm.ic as ic
import pycharmm.image as image
import pycharmm.keywords as keywords
# A subpackage rather than a single module; bound here for the same reason as
# the others, so `pycharmm.ligandff` and `from pycharmm import *` both work.
import pycharmm.ligandff as ligandff
import pycharmm.minimize as minimize
import pycharmm.nbonds as nbonds
import pycharmm.nxm as nxm
import pycharmm.omm as omm
import pycharmm.blade as blade
import pycharmm.psf as psf
import pycharmm.read as read
# Imported explicitly rather than relied on as a side effect of
# dynamics.py importing it: `import *` must expose every submodule as a
# module (tests/test_package_exports.py), and now that __all__ is
# declared, an accidental binding no longer counts.
import pycharmm.replica_exchange as replica_exchange
import pycharmm.reset as reset
import pycharmm.rtf as rtf
import pycharmm.select as select
import pycharmm.settings as settings
import pycharmm.shake as shake
import pycharmm.trace as trace
from .settings import set_api_echo, get_api_echo
import pycharmm.write as write
import pycharmm.block as block
import pycharmm.restraints as restraints
import pycharmm.safeguards as safeguards

import pycharmm.fict as fict
import pycharmm.implicit_solvent as implicit_solvent
import pycharmm.param as param
import pycharmm.scalar as scalar

# Modules whose public symbols are re-exported above via `from .X import ...`.
# Also bind the module itself so every submodule is uniformly reachable as
# `pycharmm.X` (and via `from pycharmm import *`), matching the other
# submodules. `dimens` is intentionally excluded: the name `dimens` is the
# pre-initialization singleton object, not the module.
import pycharmm.atom_info as atom_info
import pycharmm.charmm_file as charmm_file
import pycharmm.custom as custom
import pycharmm.energy_func as energy_func
import pycharmm.energy_mlpot as energy_mlpot
import pycharmm.lingo as lingo
import pycharmm.loader as loader
import pycharmm.script as script
import pycharmm.select_atoms as select_atoms


name = 'pycharmm'
try:
    from importlib.metadata import version as _get_version
    __version__ = _get_version('pycharmm')
except Exception:
    __version__ = '0.0.0'

# Public API. Every name below is a deliberate re-export, so this list
# is what `from pycharmm import *` gives you -- and it is also what
# tells pyflakes (and tool/lint/charmm-lint, which CI runs on changed
# files) that these imports are used rather than dead. Without it the
# whole module lints as ~90 "imported but unused" errors, so any MR
# that touches this file fails the lint stage.
__all__ = [
    'BACKBONE_ATOMS', 'CharmmError', 'CharmmFile', 'CharmmScriptError',
    'CharmmWarning', 'Coordinates', 'CustomDynam', 'DynamicsScript',
    'EnergyFunc', 'ION_RESIDUES', 'MLpot', 'NUCLEIC_RESIDUES',
    'NonBondedScript', 'PROTEIN_RESIDUES', 'PatchScript', 'SelectAtoms',
    'UpdateNonBondedScript', 'WATER_RESIDUES', 'atom_info', 'backbone',
    'blade', 'block', 'cdocker', 'charmm_file', 'charmm_script',
    'cons_fix', 'cons_harm', 'cons_methods', 'coor', 'correl', 'crystal',
    'custom', 'dimens', 'domdec', 'dyn', 'dynamics', 'energy',
    'energy_func', 'energy_mlpot', 'fict', 'gen', 'generate',
    'get_api_echo', 'get_atom_table', 'get_charmm_variable',
    'get_default_error_handling', 'get_energy_value', 'grid', 'ic',
    'image', 'implicit_solvent', 'initialize', 'ions', 'is_initialized',
    'keywords', 'lib', 'ligandff', 'lingo', 'loader', 'minimize', 'name', 'nbonds',
    'nucleic', 'nxm', 'omm', 'param', 'parse_selection',
    'permissive_mode', 'print_lib_path', 'protein', 'psf', 'read',
    'replica_exchange', 'reset', 'restraints', 'rtf', 'safeguards', 'scalar', 'script',
    'script_factory', 'select', 'select_atoms', 'set_api_echo',
    'set_charmm_variable', 'set_default_error_handling', 'set_mpi_comm',
    'settings', 'shake', 'sidechain', 'strict_mode', 'trace', 'water',
    'write'
]

