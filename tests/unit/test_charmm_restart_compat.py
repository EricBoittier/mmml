"""c49 restarts (8 RNG seeds) must not overflow c52a1 READYN (nrand=4)."""

from __future__ import annotations

from mmml.interfaces.pycharmmInterface.mlpot.charmm_restart_compat import restart_with_seed_count

_HEAD = " !NATOM,NPRIV,NSTEP,NSAVC,NSAVV,JHSTRT,NDEGF,SEED,NSAVL\n"


def _seed_line(seeds):
    fixed = "".join(f"{v:12d}" for v in (1540, 20, 20, 1, 1, 20, 4617)) + f"{0.314159e6:22.15E}".replace("E", "D") + f"{0:12d}"
    return fixed + f"{1:22d}{len(seeds):22d}" + "".join(f"{s:22d}" for s in seeds) + "\n"


def _write(tmp_path, seeds):
    p = tmp_path / "nve.res"
    p.write_text("REST    48     1  CUBI\n\n" + _HEAD + _seed_line(seeds) + "\n !XOLD, YOLD, ZOLD\n 0.1D+01\n")
    return p


def test_c49_restart_cut_to_c52_nrand(tmp_path):
    seeds = [1539871992, 1640919914, -1824406032, 1903922810, 143131128, -1064598337, -2126112516, 871045183]
    src = _write(tmp_path, seeds)
    out = restart_with_seed_count(src, 4, verbose=False)
    assert out != src and out.name == "nve.res"
    old_lines, new_lines = src.read_text().splitlines(), out.read_text().splitlines()
    assert len(old_lines) == len(new_lines)
    diff = [i for i, (a, b) in enumerate(zip(old_lines, new_lines)) if a != b]
    assert diff == [3]
    assert new_lines[3] == _seed_line(seeds[:4]).rstrip("\n")
    assert new_lines[3][:118] == old_lines[3][:118]


def test_matching_or_fewer_seeds_untouched(tmp_path):
    src = _write(tmp_path, [1, 2, 3, 4])
    assert restart_with_seed_count(src, 4, verbose=False) == src
    assert restart_with_seed_count(src, 8, verbose=False) == src


def test_unknown_nrand_or_missing_file(tmp_path):
    src = _write(tmp_path, list(range(1, 9)))
    assert restart_with_seed_count(src, 0, verbose=False) == src
    missing = tmp_path / "none.res"
    assert restart_with_seed_count(missing, 4, verbose=False) == missing
