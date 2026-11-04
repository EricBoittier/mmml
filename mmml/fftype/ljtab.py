from pathlib import Path
from mmml.fftype.cgenff_envtyper import TOP  # $MMML_TOPPAR or <repo>/setup/charmm/toppar
def lj_table(files=("par_all36_cgenff.prm", "toppar_water_ions.str", "par_all36m_prot.prm", "par_all36_na.prm", "par_all36_lipid.prm", "par_all36_carb.prm")):
    """type -> (eps kcal/mol >=0, Rmin/2 A) from CHARMM NONBONDED sections (first definition wins)."""
    tab = {}
    for f in files:
        p = TOP / f
        if not p.exists(): continue
        on = False
        for l in open(p, errors="ignore"):
            s = l.split("!")[0].split()
            if not s: continue
            k = s[0].upper()
            if k.startswith("NONBONDED"): on = True; continue
            if k in ("NBFIX", "HBOND", "END", "BONDS", "ANGLES", "DIHEDRALS", "IMPROPER", "IMPROPERS", "CMAP", "THETAS", "PHI", "IMPHI"): on = False; continue
            if on and len(s) >= 4 and s[0][0].isalpha() and s[0].lower() != "cutnb":
                try: tab.setdefault(s[0], (abs(float(s[2])), float(s[3])))
                except ValueError: pass
    return tab
