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

"""Functions to parse other languages relevant to CHARMM

Functions
=========
- `charmm_script` -- evaluate a line of native CHARMM script
- `get_energy_value` -- get the value of CHARMM substitution parameters
- `get_charmm_variable` -- get the value of a variable in CHARMM
- `set_charmm_variable` -- set a variable in CHARMM


Examples
========
>>> import pycharmm.lingo as lingo

Evaluate one line of native CHARMM script in pyCHARMM
>>> lingo.charmm_script('stream ./toppar_water_ions.str')

Get value of CHARMM substitution parameters
>>> pi=lingo.get_energy_value('PI')
>>> print(pi)
3.141592653589793

Both ``get_energy_value`` and ``get_charmm_variable`` raise ``KeyError``
when the name is not found.  Pass *default* to suppress the error:
>>> lingo.get_energy_value('MISSING', default=0)
0

The following command is equivalent to CHARMM command `set M = 3`
>>> lingo.set_charmm_variable('M', 3)
>>> var=lingo.get_charmm_variable('M')
>>> print(var)
3

"""

import ctypes
import warnings
from contextlib import contextmanager

import pycharmm
from pycharmm.loader import lib
import pycharmm.atom_info as atom_info
from pycharmm.safeguards import CharmmScriptError


# Global configuration for error handling
_raise_on_error_default = False


def set_default_error_handling(raise_on_error: bool) -> bool:
    """Set the default error handling behavior for charmm_script().

    Parameters
    ----------
    raise_on_error : bool
        If True, charmm_script() will raise CharmmScriptError on failure.
        If False (default), it will only return the status code.

    Returns
    -------
    old_value : bool
        The previous setting, useful for restoring later.

    Examples
    --------
    >>> import pycharmm.lingo as lingo
    >>> old = lingo.set_default_error_handling(True)
    >>> # Now all charmm_script calls will raise on error
    >>> lingo.set_default_error_handling(old)  # Restore
    """
    global _raise_on_error_default
    old_value = _raise_on_error_default
    _raise_on_error_default = raise_on_error
    return old_value


def get_default_error_handling() -> bool:
    """Get the current default error handling setting.

    Returns
    -------
    raise_on_error : bool
        True if charmm_script() raises on error by default.
    """
    return _raise_on_error_default


@contextmanager
def strict_mode():
    """Context manager to enable strict error handling.

    Within this context, charmm_script() will raise CharmmScriptError
    on any failure, regardless of the global default.

    Examples
    --------
    >>> import pycharmm.lingo as lingo
    >>> with lingo.strict_mode():
    ...     lingo.charmm_script('read rtf card name top.rtf')
    ...     lingo.charmm_script('read param card name par.prm')
    """
    old_value = set_default_error_handling(True)
    try:
        yield
    finally:
        set_default_error_handling(old_value)


@contextmanager
def permissive_mode():
    """Context manager to disable error raising.

    Within this context, charmm_script() will never raise on errors,
    regardless of the global default. Use this when you want to handle
    errors manually via return status.

    Examples
    --------
    >>> import pycharmm.lingo as lingo
    >>> lingo.set_default_error_handling(True)  # Global strict mode
    >>> with lingo.permissive_mode():
    ...     status = lingo.charmm_script('maybe fails')
    ...     if status != 1:
    ...         print("Command failed, but we continue")
    """
    old_value = set_default_error_handling(False)
    try:
        yield
    finally:
        set_default_error_handling(old_value)


def _invalidate_cache():
    """Invalidate atom cache after potential PSF modifications."""
    atom_info.invalidate_atom_cache()

_SENTINEL = object()


def _charmm_script_line(script):
    """Evaluate a line of native CHARMM script

    Returns
    -------
    status : integer
             1 indicates success
    """
    c_script = ctypes.create_string_buffer(script.encode())
    len_script = ctypes.c_int(len(script))
    status = lib.eval_charmm_script(c_script, len_script)
    return status


def _clean_charmm_script(script_lines):
    """Remove comment lines, remove blank lines and join lines ending in -

    Returns
    -------
    reduction : list
                a list of non-blank non-comment lines that do not end in -
    """
    clean_lines = list()
    script_lines = [line.strip() for line in script_lines if line.strip()]
    script_lines = [line for line in script_lines if not line.startswith('!')]
    iter_lines = iter(script_lines)
    for sline in iter_lines:
        to_join = list()
        to_join.append(sline)
        while sline.endswith('-'):
            sline = next(iter_lines)
            to_join.append(sline)

        to_join = [line.rstrip('- ').strip() for line in to_join
                   if line.rstrip('- ').strip()]
        to_join = ' '.join(to_join)
        clean_lines.append(to_join)

    return clean_lines


# def charmm_script(script):
#     """evaluate one or several lines of native CHARMM script

#     Returns
#     -------
#     success : boolean
#               True indicates success
#     """
#     success = True
#     script_lines = _clean_charmm_script(script.splitlines())
#     for script_line in script_lines:
#         line_success = _charmm_script_line(script_line)
#         success = success and (1 == line_success)

#     return success


def charmm_script(script, raise_on_error=None):
    """Evaluate one or several lines of native CHARMM script.

    Parameters
    ----------
    script : str
        One or more lines of CHARMM script to execute.
    raise_on_error : bool, optional
        If True, raise CharmmScriptError when the script fails.
        If False, just return the status code.
        If None (default), use the global setting from set_default_error_handling().

    Note
    ----
    Since arbitrary CHARMM commands can modify the PSF structure,
    the atom cache is invalidated after each script execution.

    If safeguards are enabled (via pycharmm.safeguards.enable()),
    commands are logged and pre-validated for debugging purposes.
    Safeguards are informative only and never block operations.

    Returns
    -------
    status : int
        1 indicates success, other values indicate failure.

    Raises
    ------
    CharmmScriptError
        If the script fails and raise_on_error is True (or global default is True).

    Examples
    --------
    >>> import pycharmm.lingo as lingo

    # Default behavior (backward compatible)
    >>> status = lingo.charmm_script('print coor')

    # Raise on error for this call
    >>> lingo.charmm_script('read rtf card name top.rtf', raise_on_error=True)

    # Set global default to raise
    >>> lingo.set_default_error_handling(True)
    >>> try:
    ...     lingo.charmm_script('invalid command')
    ... except CharmmScriptError as e:
    ...     print(f"Failed: {e}")
    """
    # Run safeguards pre-execution hook (if enabled)
    # Import here to avoid circular imports
    from pycharmm import safeguards
    safeguards._pre_execute_hook(script)

    # One command per eval_charmm_script call. The KEY_LIBRARY path uses
    # maincomx directly (no scratch stream), so a multiline buffer would be
    # one illegal command and would desync cooperative MPI READ.
    script_lines = _clean_charmm_script(script.splitlines())
    if not script_lines:
        script_lines = [script.strip()] if script.strip() else []
    status = 1
    for script_line in script_lines:
        line_status = _charmm_script_line(script_line)
        if line_status != 1:
            status = line_status
    _invalidate_cache()

    # Run safeguards post-execution hook (if enabled)
    safeguards._post_execute_hook(script, status)

    # Determine whether to raise
    should_raise = raise_on_error if raise_on_error is not None else _raise_on_error_default

    if should_raise and status != 1:
        raise CharmmScriptError(script, status)

    return status


class FoundValue(ctypes.Structure):
    _fields_ = [('is_found', ctypes.c_int),
                ('int_val', ctypes.c_int),
                ('bool_val', ctypes.c_int),
                ('real_val', ctypes.c_double)]


(NotFound, FoundInt, FoundReal, FoundBool) = (0, 1, 2, 3)
(BoolFalse, BoolTrue) = (0, 1)


def charmm_echo(string):
    """Echos output to the CHARMM output stream to print python
    outputs

    Input
    -----

    string: str - String to print to charmm output
    """
    echo_command = ' '.join(['echo', 'pyCHARMM>',string])

    echo_script = pycharmm.script.CommandScript(echo_command)
    echo_script.run()

    

def get_energy_value(name, default=_SENTINEL):
    """Get the value of a substitution parameter in CHARMM

    See CHARMM documentation [subst](<https://academiccharmm.org/documentation/version/c47b1/subst>)
    for more information

    Parameters
    ----------
    name : string
           name of the CHARMM substitution parameter
    default : optional
           value to return if *name* is not found.  If omitted, a
           KeyError is raised when the parameter does not exist.

    Returns
    -------
    ret_val : numeric, boolean, or *default*

    Raises
    ------
    KeyError
        If *name* is not found and no *default* was given.
    """
    c_name = ctypes.create_string_buffer(name.encode())
    n = ctypes.c_int(len(name))
    get_energy_val = lib.eval_get_energy_value
    get_energy_val.restype = FoundValue
    val = get_energy_val(c_name, n)

    if val.is_found == FoundInt:
        return int(val.int_val)
    elif val.is_found == FoundReal:
        return float(val.real_val)
    elif val.is_found == FoundBool:
        return val.bool_val == BoolTrue

    if default is not _SENTINEL:
        warnings.warn(
            f"CHARMM substitution parameter '{name}' not found; "
            f"returning default value {default!r}")
        return default

    raise KeyError(
        f"CHARMM substitution parameter '{name}' not found")


def get_charmm_variable(name, default=_SENTINEL):
    """Get the value of a variable in CHARMM

    Parameters
    ----------
    name : string
           name of the variable
    default : optional
           value to return if *name* is not found.  If omitted, a
           KeyError is raised when the variable does not exist.

    Returns
    -------
    ret_val : int, float, string, or *default*

    Raises
    ------
    KeyError
        If *name* is not found and no *default* was given.
    """
    len_name = ctypes.c_int(len(name))
    c_name = ctypes.create_string_buffer(name.encode())

    len_val = ctypes.c_int(128)
    c_val = ctypes.create_string_buffer(128)

    is_found = lib.eval_get_param(c_name, len_name, c_val, len_val)

    if is_found == BoolTrue:
        try:
            return int(c_val.value)
        except ValueError:
            try:
                return float(c_val.value)
            except ValueError:
                return c_val.value

    if default is not _SENTINEL:
        warnings.warn(
            f"CHARMM variable '{name}' not found; "
            f"returning default value {default!r}")
        return default

    raise KeyError(f"CHARMM variable '{name}' not found")


def set_charmm_variable(name, val):
    """Set the value of variable in CHARMM

    Parameters
    ----------
    name : string
         name of the variable
    val : string, float or int
         value of the variable

    Returns
    -------
    ret_val : string or None
              if name found, then value in CHARMM, otherwise None
    """
    len_name = ctypes.c_int(len(name))
    c_name = ctypes.create_string_buffer(name.encode())

    val = str(val)
    len_val = ctypes.c_int(len(val))
    c_val = ctypes.create_string_buffer(val.encode())

    is_found = lib.eval_set_param(c_name, len_name, c_val, len_val)

    ret_val = False
    if is_found == BoolTrue:
        ret_val = True

    return ret_val

def _retype_param(value):
    value = str(value)
    if value.isnumeric():
        try:
            new_value = int(value)
        except ValueError:
            new_value = float(value)
    else:
        new_value = value.strip()

    return new_value


def get_charmm_params():
    """Get the @ substitution parameter table from CHARMM
    """
    n = lib.eval_num_params()

    max_name = lib.eval_max_param_name()
    name_buffers = [ctypes.create_string_buffer(max_name) for _ in range(n)]
    name_pointers = (ctypes.c_char_p * n)(*map(ctypes.addressof,
                                               name_buffers))

    max_value_length = lib.eval_max_param_val()
    value_buffers = [ctypes.create_string_buffer(max_value_length) for _ in range(n)]
    value_pointers = (ctypes.c_char_p * n)(*map(ctypes.addressof,
                                                value_buffers))

    lib.eval_get_all_params(name_pointers, value_pointers)
    names = [name.value.decode(errors='ignore').strip() for name in name_buffers[0:n]]
    values = [_retype_param(v.value.decode(errors='ignore')) for v in value_buffers[0:n]]

    return dict(zip(names, values))


def _charmm_builtins_reals_get():
    n = lib.builtins_reals_num()
    max_name = lib.builtins_names_max()

    name_buffers = [ctypes.create_string_buffer(max_name) for _ in range(n)]
    name_pointers = (ctypes.c_char_p * n)(*map(ctypes.addressof,
                                               name_buffers))

    values = (ctypes.c_double * n)()

    lib.builtins_reals_get(name_pointers, values)

    names = [name.value.decode(errors='ignore').strip() for name in name_buffers[0:n]]
    values = [float(v) for v in values[0:n]]

    real_builtins = dict(zip(names, values))
    return real_builtins


def _charmm_builtins_ints_get():
    n = lib.builtins_ints_num()
    max_name = lib.builtins_names_max()

    name_buffers = [ctypes.create_string_buffer(max_name) for _ in range(n)]
    name_pointers = (ctypes.c_char_p * n)(*map(ctypes.addressof,
                                               name_buffers))

    values = (ctypes.c_int * n)()

    lib.builtins_ints_get(name_pointers, values)

    names = [name.value.decode(errors='ignore').strip() for name in name_buffers[0:n]]
    values = [int(v) for v in values[0:n]]

    int_builtins = dict(zip(names, values))
    return int_builtins


def _charmm_builtins_strs_get():
    n = lib.builtins_strs_num()
    max_name = lib.builtins_names_max()

    name_buffers = [ctypes.create_string_buffer(max_name) for _ in range(n)]
    name_pointers = (ctypes.c_char_p * n)(*map(ctypes.addressof,
                                               name_buffers))

    values_buffers = [ctypes.create_string_buffer(max_name) for _ in range(n)]
    values_pointers = (ctypes.c_char_p * n)(*map(ctypes.addressof,
                                                 values_buffers))

    lib.builtins_strs_get(name_pointers, values_pointers)

    names = [name.value.decode(errors='ignore').strip() for name in name_buffers[0:n]]
    values = [v.value.decode(errors='ignore').strip() for v in values_buffers[0:n]]

    str_builtins = dict(zip(names, values))
    return str_builtins


def get_charmm_builtins():
    """Get the ? substitution parameters from CHARMM
    """
    return (_charmm_builtins_reals_get()
            | _charmm_builtins_ints_get()
            | _charmm_builtins_strs_get())
