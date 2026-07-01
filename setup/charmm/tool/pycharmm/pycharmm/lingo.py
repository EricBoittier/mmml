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

The following command is equivalent to CHARMM command `set M = 3`
>>> lingo.set_charmm_variable('M', 3)
>>> var=lingo.get_charmm_variable('M')
>>> print(var)
3

"""

import ctypes

import pandas

import pycharmm.lib as lib


def _charmm_script_line(script):
    """Evaluate a line of native CHARMM script

    Returns
    -------
    status : integer
             1 indicates success
    """
    c_script = ctypes.create_string_buffer(script.upper().encode())
    len_script = ctypes.c_int(len(script))

    status = lib.charmm.eval_charmm_script(c_script,
                                           ctypes.byref(len_script))
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


def charmm_script(script):
    """Evaluate one or several lines of native CHARMM script

    Returns
    -------
    success : boolean
              True indicates success
    """
    c_script = ctypes.create_string_buffer(script.encode())
    len_script = ctypes.c_int(len(script))
    status = lib.charmm.eval_charmm_script(c_script, len_script)
    return status


class FoundValue(ctypes.Structure):
    _fields_ = [('is_found', ctypes.c_int),
                ('int_val', ctypes.c_int),
                ('bool_val', ctypes.c_int),
                ('real_val', ctypes.c_double)]


(NotFound, FoundInt, FoundReal, FoundBool) = (0, 1, 2, 3)
(BoolFalse, BoolTrue) = (0, 1)


def get_energy_value(name):
    """Get the value of a substitution parameter in CHARMM
    
    See CHARMM documentation [subst](<https://academiccharmm.org/documentation/version/c47b1/subst>)
    for more information

    Parameters
    ----------
    name : string
           name of the CHARMM substitution parameter

    Returns
    -------
    ret_val : None or numeric or boolean
              if name found, then value in CHARMM, otherwise None
    """
    c_name = ctypes.create_string_buffer(name.encode())
    n = ctypes.c_int(len(name))
    get_energy_val = lib.charmm.eval_get_energy_value
    get_energy_val.restype = FoundValue
    val = get_energy_val(c_name, n)

    if val.is_found == FoundInt:
        ret_val = int(val.int_val)
    elif val.is_found == FoundReal:
        ret_val = float(val.real_val)
    elif val.is_found == FoundBool:
        ret_val = False
        if val.bool_val == BoolTrue:
            ret_val = True
    else:  # consider throwing an error here
        ret_val = None

    return ret_val


def get_charmm_variable(name):
    """Get the value of a variable in CHARMM

    Parameters
    ----------
    name : string
           name of the variable

    Returns
    -------
    ret_val : string or None
              if name found, then value in CHARMM, otherwise None
    """
    len_name = ctypes.c_int(len(name))
    c_name = ctypes.create_string_buffer(name.encode())

    len_val = ctypes.c_int(128)
    c_val = ctypes.create_string_buffer(128)

    is_found = lib.charmm.eval_get_param(c_name, len_name, c_val, len_val)

    ret_val = None
    if is_found == BoolTrue:
        try:
            ret_val = int(c_val.value)
        except ValueError:
            try:
                ret_val = float(c_val.value)
            except ValueError:
                ret_val = c_val.value

    return ret_val


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

    is_found = lib.charmm.eval_set_param(c_name, len_name, c_val, len_val)

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
    n = lib.charmm.eval_num_params()

    max_name = lib.charmm.eval_max_param_name()
    name_buffers = [ctypes.create_string_buffer(max_name) for _ in range(n)]
    name_pointers = (ctypes.c_char_p * n)(*map(ctypes.addressof,
                                               name_buffers))

    max_value_length = lib.charmm.eval_max_param_val()
    value_buffers = [ctypes.create_string_buffer(max_value_length) for _ in range(n)]
    value_pointers = (ctypes.c_char_p * n)(*map(ctypes.addressof,
                                                value_buffers))

    lib.charmm.eval_get_all_params(name_pointers, value_pointers)
    names = [name.value.decode(errors='ignore').strip() for name in name_buffers[0:n]]
    values = [_retype_param(v.value.decode(errors='ignore')) for v in value_buffers[0:n]]

    return dict(zip(names, values))


def _charmm_builtins_reals_get():
    n = lib.charmm.builtins_reals_num()
    max_name = lib.charmm.builtins_names_max()

    name_buffers = [ctypes.create_string_buffer(max_name) for _ in range(n)]
    name_pointers = (ctypes.c_char_p * n)(*map(ctypes.addressof,
                                               name_buffers))

    values = (ctypes.c_double * n)()

    lib.charmm.builtins_reals_get(name_pointers, values)

    names = [name.value.decode(errors='ignore').strip() for name in name_buffers[0:n]]
    values = [float(v) for v in values[0:n]]

    real_builtins = dict(zip(names, values))
    return real_builtins


def _charmm_builtins_ints_get():
    n = lib.charmm.builtins_ints_num()
    max_name = lib.charmm.builtins_names_max()

    name_buffers = [ctypes.create_string_buffer(max_name) for _ in range(n)]
    name_pointers = (ctypes.c_char_p * n)(*map(ctypes.addressof,
                                               name_buffers))

    values = (ctypes.c_int * n)()

    lib.charmm.builtins_ints_get(name_pointers, values)

    names = [name.value.decode(errors='ignore').strip() for name in name_buffers[0:n]]
    values = [int(v) for v in values[0:n]]

    int_builtins = dict(zip(names, values))
    return int_builtins


def _charmm_builtins_strs_get():
    n = lib.charmm.builtins_strs_num()
    max_name = lib.charmm.builtins_names_max()

    name_buffers = [ctypes.create_string_buffer(max_name) for _ in range(n)]
    name_pointers = (ctypes.c_char_p * n)(*map(ctypes.addressof,
                                               name_buffers))

    values_buffers = [ctypes.create_string_buffer(max_name) for _ in range(n)]
    values_pointers = (ctypes.c_char_p * n)(*map(ctypes.addressof,
                                                 values_buffers))

    lib.charmm.builtins_strs_get(name_pointers, values_pointers)

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
