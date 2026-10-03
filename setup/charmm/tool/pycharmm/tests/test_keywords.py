import ctypes
import re

from pycharmm.loader import lib


def test_keyword_buffers_are_nul_terminated():
    count = int(lib.api_keywords_count())
    width = int(lib.api_keywords_max_len())
    assert count > 0
    assert width > 0

    buffers = [ctypes.create_string_buffer(width + 1) for _ in range(count)]
    for buffer in buffers:
        ctypes.memset(ctypes.addressof(buffer), 0x7F, width + 1)
    pointers = (ctypes.c_char_p * count)(*map(ctypes.addressof, buffers))

    lib.api_keywords_get_all(pointers)

    names = [buffer.raw.split(b"\0", 1)[0] for buffer in buffers]
    assert all(re.fullmatch(rb"[A-Z][A-Z0-9_]*", name) for name in names)
    assert b"UNIX" in names
