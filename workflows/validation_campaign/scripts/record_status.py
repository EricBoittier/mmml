"""Write status.json / provenance.json from inside a running job.

Called by the generated job scripts before and after the driver command. Keeping
this separate from the driver means a driver that crashes hard still leaves an
honest terminal receipt behind.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import campaign_lib as lib


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--phase", required=True, choices=["running", "finished"])
    parser.add_argument("--environment", required=True)
    parser.add_argument("--exit-code", type=int, default=None)
    args = parser.parse_args()

    out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    environment = lib.load_environment(args.environment)

    if args.phase == "running":
        lib.write_provenance(out_dir, environment)
        lib.write_status(out_dir, phase="running", environment=environment.name)
    else:
        lib.write_status(
            out_dir,
            phase="finished",
            exit_code=args.exit_code,
            environment=environment.name,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
