"""Safe ``READ SEQUENCE`` for long residue lists.

CHARMM's ``read_sequence_string`` API (api_read.F90) stores every residue name
in ``res(nres + i)`` before it calls ``resize_psf``, so one call with more
residues than the current PSF capacity silently keeps only the first ~1000
(``TIP3:1092`` became 1001 residues / 3003 atoms). Each call appends to the
pending sequence and resizes at its end, so feeding the names in chunks gives
the full sequence for a single ``GENERATE``.
"""

from __future__ import annotations

DEFAULT_CHUNK = 200


def read_sequence_chunked(read_module, sequence: str | list[str], chunk: int = DEFAULT_CHUNK) -> None:
    """Call ``read_module.sequence_string`` on at most ``chunk`` residues at a time."""
    items = sequence.split() if isinstance(sequence, str) else [str(s) for s in sequence]
    for start in range(0, len(items), int(chunk)):
        read_module.sequence_string(" ".join(items[start:start + int(chunk)]))
