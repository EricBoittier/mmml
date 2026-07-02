#!/usr/bin/env python3
"""Report CHARMM bonded energies for ACE–ALA×3–CT3 (protein toppar).

Run on a CHARMM node::

    ./scripts/mmml-charmm-mpirun.sh python tests/functionality/aaa_ama/report_charmm_bonded.py

This is a **reference MM** build (42 atoms).  Compare NPZ labels only after
building a PSF that matches ``dataset_aaa.npz`` ``Z`` (34 atoms).
"""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]


def main() -> int:
    from mmml.interfaces.pycharmmInterface.import_pycharmm import (
        CHARMM_HOME,
        crystal_free_charmm_for_param_append,
        ensure_pycharmm_loaded,
    )

    ensure_pycharmm_loaded()
    import pycharmm.generate as generate
    import pycharmm.lingo as lingo
    import pycharmm.read as read
    from mmml.interfaces.pycharmmInterface import setupRes
    from mmml.interfaces.pycharmmInterface.charmm_levels import charmm_relaxed_bomlev
    from mmml.interfaces.pycharmmInterface.cgenff_bonded_reference import (
        charmm_bonded_energy_components_kcalmol,
        run_charmm_bonded_ener_force,
        setup_bonded_only_charmm,
    )

    base = Path(CHARMM_HOME) / "toppar"
    rtf = base / "top_all36_prot.rtf"
    prm = base / "par_all36m_prot.prm"
    if not prm.is_file():
        prm = base / "par_all36_prot.prm"
    if not rtf.is_file() or not prm.is_file():
        print(f"Missing protein toppar under {base}", file=sys.stderr)
        return 1

    workdir = Path(tempfile.mkdtemp(prefix="aaa_ama_bonded_"))
    prev = os.getcwd()
    try:
        os.chdir(workdir)
        crystal_free_charmm_for_param_append()
        lingo.charmm_script("DELETE ATOM SELE ALL END")
        with charmm_relaxed_bomlev():
            read.rtf(str(rtf))
            read.prm(str(prm))
        read.sequence_string("ALA ALA ALA")
        generate.new_segment(
            seg_name="PEPT",
            first_patch="ACE",
            last_patch="CT3",
            setup_ic=True,
        )
        setupRes.generate_coordinates(skip_energy_show=True, validate=True)

        import pycharmm.psf as psf

        setup_bonded_only_charmm()
        run_charmm_bonded_ener_force(silent=True)
        terms = charmm_bonded_energy_components_kcalmol()
        n_atoms = int(psf.get_natom())

        print("CHARMM bonded energies (kcal/mol) — ACE–ALA×3–CT3, seg PEPT")
        for key in ("bond", "angl", "dihe", "impr", "cmap", "total"):
            if key in terms:
                print(f"  {key:6s} {terms[key]:12.6f}")
        print()
        print(f"NPZ dataset_aaa.npz uses 34 atoms; this PSF has {n_atoms} atoms.")
        print("Align topology with training PSF before comparing to NPZ E/F labels.")
        return 0
    finally:
        os.chdir(prev)


if __name__ == "__main__":
    raise SystemExit(main())
