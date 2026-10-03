#!/usr/bin/env python3
"""Generate the pyCHARMM API documentation with pdoc, without letting a
documentation failure abort the whole CHARMM build.

Documentation is the last and least critical step of `make install` /
`ninja install`.  If pdoc cannot import the pyCHARMM package -- most often
because a stray or outdated .py file is lingering in the package directory
(github bucknerj/dev #25, #27) -- we print a clear warning and drop a small
placeholder page in place of the API pages, then exit successfully so the
rest of the install completes.  CHARMM itself is unaffected by whether the
Python API pages were produced.

Pass --strict to restore the old behaviour and treat a pdoc failure as a
build error; QA/CI builds use this so a genuine regression in a real module
is not quietly hidden behind the placeholder.
"""

import argparse
import os
import subprocess
import sys

_PLACEHOLDER_HTML = """<!doctype html>
<html>
<head><meta charset="utf-8"><title>pyCHARMM API documentation unavailable</title></head>
<body>
<h1>pyCHARMM API documentation was not generated</h1>
<p>The documentation generator (pdoc) could not import the pyCHARMM package
during this build, so the Python API reference pages were skipped.  CHARMM
itself built and installed normally.</p>
<p>The most common cause is a stray or outdated Python file left behind in the
pyCHARMM package directory.  The specific import error is shown in the build
log, just above where this page was written.</p>
</body>
</html>
"""

_WARNING = """
========================================================================
WARNING: pyCHARMM API documentation could not be generated.

pdoc failed to import the pyCHARMM package (see the error just above).
A common cause is a stray or outdated .py file left in the pyCHARMM
package directory.  CHARMM itself built and installed normally; only the
pyCHARMM API html pages were skipped, and a placeholder was written in
their place.
========================================================================
"""


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_dir", help="directory to write html into")
    parser.add_argument("package_dir", help="path to the pyCHARMM package")
    parser.add_argument(
        "--strict",
        action="store_true",
        help="treat a pdoc failure as a build error (QA/CI)",
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    result = subprocess.run(
        [sys.executable, "-m", "pdoc", "-o", args.output_dir, args.package_dir]
    )
    if result.returncode == 0:
        return 0

    sys.stderr.write(_WARNING)
    sys.stderr.flush()

    if args.strict:
        # QA/CI: surface the failure so a real regression is not masked.
        return result.returncode

    # Leave a placeholder where pdoc would have written the package page, so
    # the downstream rename and index steps still succeed and the overall
    # install completes.
    placeholder = os.path.join(args.output_dir, "pycharmm.html")
    with open(placeholder, "w") as handle:
        handle.write(_PLACEHOLDER_HTML)
    return 0


if __name__ == "__main__":
    sys.exit(main())
