# pycharmm: molecular dynamics in python with CHARMM
# Copyright (C) 2018 Josh Buckner
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

"""Inspect which CHARMM `pref.dat` keywords were compiled into this build.

Every CHARMM binary is configured at build time by enabling or
disabling a set of preprocessor keywords (``OPENMM``, ``BLADE``,
``DOMDEC``, ``FFTDOCK``, ...) defined in ``pref.dat``. The exact set
that ended up in the running binary is fixed once compilation
completes, so any code that needs to behave differently depending on
which optional features are present (test-suite skip-if logic,
diagnostic banners, fallback selection) needs a way to query that
list at runtime.

This module exposes the list. Internally it makes a single bulk fetch
to CHARMM the first time it's queried, builds a ``frozenset`` of
keyword strings, and caches it for the rest of the session.

Functions
=========
- `all` -- Return every pref keyword compiled into this CHARMM binary
- `has` -- Test whether a single keyword is present (case-insensitive)

Examples
========
>>> from pycharmm import keywords

Check whether a backend is available before invoking it:

>>> if keywords.has('BLADE'):
...     blade.enable()

List everything compiled in (for a startup banner or bug report):

>>> sorted(keywords.all())
['CMAP', 'DOMDEC', 'FFTDOCK', 'OPENMM', ...]

Use as a pytest skip-gate:

>>> import pytest
>>> @pytest.mark.skipif(not keywords.has('BLADE'),
...                     reason="requires BLaDE")
... def test_blade_thing():
...     ...

Notes
=====
The returned set is *immutable* (``frozenset``), so it's safe to share
across modules without worrying about accidental mutation. The cache
is filled on first call and is not invalidated; CHARMM's compiled-in
feature set cannot change without restarting the process, so this is
fine.

The names returned are the canonical uppercase forms used in
``pref.dat`` (e.g. ``BLADE``, ``OPENMM``, ``OMMTORCH``). `has` is
case-insensitive for ergonomics: ``keywords.has('blade')`` and
``keywords.has('BLADE')`` give the same answer.
"""

from __future__ import annotations

import ctypes
from typing import FrozenSet

from pycharmm.loader import lib


_cached_keywords: FrozenSet[str] | None = None


def _fetch() -> FrozenSet[str]:
    """Pull the keyword list out of CHARMM via the C-binding shims.

    Internal helper. Calls
    ``api_keywords_count`` / ``api_keywords_max_len`` to size the
    buffers, then ``api_keywords_get_all`` to fill them. Strings are
    decoded as UTF-8 (CHARMM's keyword names are pure ASCII), stripped
    of trailing whitespace and NULs, uppercased, and frozen into a
    set.

    Returns
    -------
    frozenset of str
        Canonical uppercase keyword names compiled into this binary.
    """
    n = lib.api_keywords_count()
    max_len = lib.api_keywords_max_len()

    # Allocate one buffer per keyword (+1 byte for the NUL terminator).
    # This mirrors the bulk-fetch pattern in lingo.get_charmm_params.
    buffers = [ctypes.create_string_buffer(max_len + 1) for _ in range(n)]
    pointers = (ctypes.c_char_p * n)(
        *map(ctypes.addressof, buffers)
    )

    lib.api_keywords_get_all(pointers)

    return frozenset(
        buf.value.decode("utf-8", errors="ignore").strip().upper()
        for buf in buffers
        if buf.value
    )


def all() -> FrozenSet[str]:
    """Return the set of pref keywords compiled into this CHARMM build.

    The first call queries CHARMM and caches the result; subsequent
    calls are O(1).

    Returns
    -------
    frozenset of str
        Every pref keyword present in this binary, as canonical
        uppercase strings (e.g. ``{'OPENMM', 'OMMTORCH', 'DOMDEC',
        'FFTDOCK', ...}``). The set is immutable; do not attempt to
        mutate it.

    Examples
    --------
    >>> from pycharmm import keywords
    >>> 'OPENMM' in keywords.all()
    True
    """
    global _cached_keywords
    if _cached_keywords is None:
        _cached_keywords = _fetch()
    return _cached_keywords


def has(name: str) -> bool:
    """Return True if `name` is a pref keyword in this CHARMM build.

    Case-insensitive: ``has('blade')``, ``has('BLADE')``, and
    ``has('Blade')`` all behave identically.

    Parameters
    ----------
    name : str
        The keyword to look up. Whitespace is stripped before
        comparison.

    Returns
    -------
    bool
        ``True`` if `name` (uppercased) is present in
        :func:`all`'s result, ``False`` otherwise.

    Examples
    --------
    >>> from pycharmm import keywords
    >>> keywords.has('OPENMM')
    True
    >>> keywords.has('definitely_not_a_real_keyword')
    False
    """
    return name.strip().upper() in all()
