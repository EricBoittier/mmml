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

"""A class to simulate a Python dict accessible from Fortran and C"""

import ctypes
import os
import os.path
import platform

from functools import singledispatch, update_wrapper


def fictvaldispatch(func):
    """A simple decorator to do single dispatch on the 3rd arg"""
    
    dispatcher = singledispatch(func)

    def wrapper(*args, **kw):
        return dispatcher.dispatch(args[2].__class__)(*args, **kw)

    wrapper.register = dispatcher.register
    update_wrapper(wrapper, func)
    return wrapper


def fictkeydispatch(func):
    """A simple decorator to do single dispatch on the 2nd arg"""
    dispatcher = singledispatch(func)

    def wrapper(*args, **kw):
        return dispatcher.dispatch(args[1].__class__)(*args, **kw)

    wrapper.register = dispatcher.register
    update_wrapper(wrapper, func)
    return wrapper


def _open_lib(fict_lib_dir=None):
    fict_lib_name = 'libfict'

    if fict_lib_dir:
        fict_home = fict_lib_dir
    else:
        fict_home = os.environ.get('FICT_LIB_DIR', '')

    if fict_home:
        fict_lib_name = os.path.join(fict_home, fict_lib_name)

    sys_name = platform.system()
    if sys_name == 'Darwin':
        fict_lib_name += '.dylib'
    elif sys_name == 'Linux':
        fict_lib_name += '.so'
    elif sys_name == 'Windows':
        fict_lib_name += '.dll'

    return ctypes.CDLL(fict_lib_name)


class Fict:
    """
    Like a python dict but accessible also from Fortran and C

    Methods
    -------

    set(key, val) : 
        associates key with val in the dictionary

    get(key, default=None) :
        gets the val associated with key.
        If default is provided and key not found, returns default.
        Otherwise, raises KeyError

    remove(key) :
        removes key from the dictionary.
        Returns True if key was found.
        Otherwise returns False
    """
    def __init__(self, fict_lib_dir=None, fict_lib=None):
        if fict_lib:
            self.__lib = fict_lib
        else:
            self.__lib = _open_lib(fict_lib_dir)
            self.__init_lib_signatures()
            self.__fdict = self.__create_dict()

    def __del__(self):
        self.__free_dict()

    def __init_lib_signatures(self):
        # constructors & destructors
        self.__lib.dict_new.argtypes = list()
        self.__lib.dict_new.restype = ctypes.c_void_p

        self.__lib.dict_free.argtypes = [ctypes.c_void_p]
        self.__lib.dict_free.restype = None

        # setters
        self.__lib.dict_set_bool.argtypes = [ctypes.c_void_p,
                                             ctypes.c_char_p,
                                             ctypes.c_bool]
        self.__lib.dict_set_bool.restype = ctypes.c_bool

        self.__lib.dict_set_double.argtypes = [ctypes.c_void_p,
                                               ctypes.c_char_p,
                                               ctypes.c_double]
        self.__lib.dict_set_double.restype = ctypes.c_bool

        self.__lib.dict_set_int.argtypes = [ctypes.c_void_p,
                                            ctypes.c_char_p,
                                            ctypes.c_int]
        self.__lib.dict_set_int.restype = ctypes.c_bool

        self.__lib.dict_set_string.argtypes = [ctypes.c_void_p,
                                               ctypes.c_char_p,
                                               ctypes.c_char_p]
        self.__lib.dict_set_string.restype = ctypes.c_bool

        # getters
        self.__lib.dict_get_bool.argtypes = [ctypes.c_void_p,
                                             ctypes.c_char_p,
                                             ctypes.POINTER(ctypes.c_bool)]
        self.__lib.dict_get_bool.restype = ctypes.c_bool

        self.__lib.dict_get_double.argtypes = [ctypes.c_void_p,
                                               ctypes.c_char_p,
                                               ctypes.POINTER(ctypes.c_bool)]
        self.__lib.dict_get_double.restype = ctypes.c_double

        self.__lib.dict_get_int.argtypes = [ctypes.c_void_p,
                                            ctypes.c_char_p,
                                            ctypes.POINTER(ctypes.c_bool)]
        self.__lib.dict_get_int.restype = ctypes.c_int

        self.__lib.dict_get_string.argtypes = [ctypes.c_void_p,
                                               ctypes.c_char_p,
                                               ctypes.POINTER(ctypes.c_bool)]
        self.__lib.dict_get_string.restype = ctypes.c_char_p

        # remove
        self.__lib.dict_remove.argtypes = [ctypes.c_void_p,
                                           ctypes.c_char_p]
        self.__lib.dict_remove.restype = ctypes.c_bool

    def __create_dict(self):
        new_dict = self.__lib.dict_new()
        return new_dict

    def __free_dict(self):
        fdict = ctypes.c_void_p(self.__fdict)
        self.__lib.dict_free(fdict)

    @fictvaldispatch
    def set(self, key, val):
        """Associates key with val in the dictionary

        Parameters
        ----------
        key : str
            associate this name with val in the dictionary
        val : bool, double, int, or str
            this value will be associate with key for retrieval

        Raises
        ------
        TypeError
            if val is not bool, float, int, or str, raise TypeError

        Returns
        -------
        bool
            True if key's val was replaced, False if key is new
        """
        raise TypeError('fict not implemented for values of type '
                        + str(type(val)))

    @set.register(bool)
    def _(self, key, val):
        replaced = self.__lib.dict_set_bool(self.__fdict,
                                            key.encode('utf-8'), val)
        return replaced

    @set.register(float)
    def _(self, key, val):
        replaced = self.__lib.dict_set_double(self.__fdict,
                                              key.encode('utf-8'), val)
        return replaced

    @set.register(int)
    def _(self, key, val):
        replaced = self.__lib.dict_set_int(self.__fdict,
                                           key.encode('utf-8'), val)
        return replaced

    @set.register(bytes)
    def _(self, key, val):
        replaced = self.__lib.dict_set_string(self.__fdict,
                                              key.encode('utf-8'), val)
        return replaced

    @set.register(str)
    def _(self, key, val):
        replaced = self.__lib.dict_set_string(self.__fdict,
                                              key.encode('utf-8'),
                                              val.encode('utf-8'))
        return replaced

    @fictkeydispatch
    def get(self, key, default=None):
        """Gets the val associated with key

        If default is provided and key not found, returns default.
        Otherwise, raises KeyError

        Parameters
        ----------
        key : str
            look for this name in dictionary
        default : optional
            return this if key not found in dictionary

        Raises
        ------
        KeyError
            if default is not provided and key not found, raise KeyError

        Returns
        -------
        bool, real, int, or str
            if key is in dictionary, then return associated val
        """
        getters = [self.__lib.dict_get_bool,
                   self.__lib.dict_get_double,
                   self.__lib.dict_get_int,
                   self.__lib.dict_get_string]
        val = None
        found = False
        for getter in getters:
            found = ctypes.c_bool(False)
            val = getter(self.__fdict, key.encode('utf-8'),
                         ctypes.byref(found))
            if found:
                break

        result = val
        if (not found) and default:
            result = default

        if (not found) and (not default):
            raise KeyError('key ->' + str(key) + '<- not in bool dictionary')

        return result

    @get.register(list)
    def _(self, keys, default=None):
        return [self.get(k, default) for k in keys]

    def remove(self, key):
        """Removes key from the dictionary

        Parameters
        ----------
        key : str
            remove this name in dictionary

        Returns
        -------
        bool
            True if key in dictionary, False otherwise
        """
        found = self.__lib.dict_remove(self.__fdict, key.encode('utf-8'))
        return found
