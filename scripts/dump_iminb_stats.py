#!/usr/bin/env python3
"""Print CHARMM image nonbond exclusion buffer stats (MKIMNB/UPIMNB).

Requires a live PyCHARMM session and a CHARMM build exporting
``image_get_iminb_stats`` (MMML ``api_image.F90``).

Example (after PSF+crystal are loaded in an interactive session):

  uv run python scripts/dump_iminb_stats.py

Or trigger a rebuild first:

  uv run python scripts/dump_iminb_stats.py --update-bimag
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--update-bimag",
        action="store_true",
        help="call pycharmm.image.update_bimag() before reading stats",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="emit JSON instead of a one-line summary",
    )
    args = parser.parse_args()

    import mmml.interfaces.pycharmmInterface.import_pycharmm  # noqa: F401
    import pycharmm.image as charmm_image

    from mmml.interfaces.pycharmmInterface.charmm_image_geometry import (
        fetch_charmm_image_nb_stats,
        format_charmm_image_nb_stats,
    )

    if args.update_bimag:
        charmm_image.update_bimag()
    stats = fetch_charmm_image_nb_stats()
    if stats is None:
        print(
            "image_get_iminb_stats unavailable (rebuild CHARMM lib from MMML api_image.F90)",
            file=sys.stderr,
        )
        return 1
    if args.json:
        print(json.dumps(stats.__dict__, indent=2, sort_keys=True))
    else:
        print(format_charmm_image_nb_stats(stats))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
