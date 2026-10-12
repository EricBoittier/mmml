"""The CGenFF LJ table read by the hybrid MM tail must include TIP3P (OT/HT): their prm lines carry no comment."""
from pathlib import Path


def _parse(prm, min_fields):
    d = {}
    for p in open(prm).readlines():
        if len(p) > 5 and len(p.split()) >= min_fields and p.split()[1] == "0.0" and p[0] != "!":
            res, _, ep, sig = p.split()[:4]
            d[res] = (float(ep), float(sig))
    return d


def test_tip3p_lj_parsed():
    prm = Path(__file__).resolve().parents[2] / "mmml" / "data" / "charmm" / "par_all36_cgenff.prm"
    d = _parse(prm, 4)
    assert d["OT"] == (-0.1521, 1.7682)
    assert d["HT"] == (-0.046, 0.2245)
    src = (Path(__file__).resolve().parents[2] / "mmml" / "interfaces" / "pycharmmInterface" / "mm_energy_forces.py").read_text()
    assert "len(p.split()) >= 4 and p.split()[1] == \"0.0\"" in src
