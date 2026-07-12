#!/usr/bin/env python3
"""Run counterpoise-corrected HF/MP2 dimer scans via ORCA (subprocess).

For each requested pair, reuses the same adaptive per-pair distance grid as
``run_dimer_scan_campaign.py`` (``build_pair_distance_grid`` — anchored to
where fragment atoms actually clear contact, optionally extended inward via
``--close-floor``), so the output plugs directly into the same
``plot_2d_pes.py`` / ``plot_1d_slices_by_offset.py`` pipeline as every other
backend.

Counterpoise (Boys-Bernardi) correction: for each geometry, three single
points are run — the full dimer, fragment A with fragment B's basis
functions as ghost atoms, and fragment B with fragment A's basis functions
as ghost atoms:

    E_int(CP) = E_dimer(AB basis) - E_A(AB basis) - E_B(AB basis)

Ghost atoms are ORCA's ``Element:`` syntax (basis functions only, no nuclear
charge/electrons). This is ~3x the single points of an uncorrected scan but
avoids basis-set superposition error inflating exactly the close-contact
region this scan is meant to probe (see ``--no-counterpoise`` to skip it).

Usage
-----
    python scripts/run_orca_hf_mp2_scan.py --pairs TIP3:TIP3 BENZ:BENZ \\
        --methods HF MP2 --basis def2-SVP --close-floor 1.2 \\
        --output-dir results/dimer_scan_campaign/orca_hf_mp2

Requires the ``orca`` executable on PATH (or set ``$ORCA`` / pass
``--orca-exe``), e.g. on scicore: ``module load ORCA/6.1.0-gompi-2023b``.
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT))

from mmml.analysis.dimer_molecules import PAIR_SCAN_CONFIG, make_oriented_scan_geometries
from mmml.analysis.dimer_scans import min_fragment_contact_distance
from scripts.run_dimer_scan_campaign import build_pair_distance_grid

HARTREE_TO_EV = 27.211386245988
EV_TO_KCAL_MOL = 23.060548867
HARTREE_TO_KCAL_MOL = HARTREE_TO_EV * EV_TO_KCAL_MOL

# Deliberately not imported from mmml.interfaces.qc_backends.orca_qm: that
# module transitively imports mmml.interfaces.pycharmmInterface.import_pycharmm,
# which does a *module-level* `import pycharmm` (+ MPI init) whenever
# libcharmm.so is present on the system — completely unrelated to this
# ORCA-only script, but it would silently drag in the same CHARMM/MPI
# dependency (and its Slurm-PMI failure mode) that this script has nothing
# to do with. Kept minimal and self-contained instead.
_ENERGY_PATTERNS = (
    re.compile(r"FINAL SINGLE POINT ENERGY\s+(-?\d+\.\d+)"),
    re.compile(r"Total Energy\s+:\s+(-?\d+\.\d+)"),
)


def parse_orca_out_energy(text: str) -> float | None:
    """Extract the final single-point energy (Hartree) from ORCA stdout/output."""
    for pattern in _ENERGY_PATTERNS:
        matches = pattern.findall(text)
        if matches:
            return float(matches[-1])
    return None


def _xyz_block(symbols: list[str], positions: np.ndarray, ghost_mask: np.ndarray, charge: int, multiplicity: int) -> str:
    lines = [f"* xyz {charge} {multiplicity}"]
    for sym, (x, y, z), is_ghost in zip(symbols, positions, ghost_mask):
        tag = f"{sym}:" if is_ghost else sym
        lines.append(f"  {tag} {x:.8f} {y:.8f} {z:.8f}")
    lines.append("*")
    return "\n".join(lines)


def _render_input(
    method: str, basis: str, pal: int, maxcore: int, xyz_block: str,
    *, aux_basis: str | None = None, dispersion: str | None = None, rijcosx: bool = False,
) -> str:
    # Double-hybrids (B2PLYP, DSD-BLYP, PWPB95, ...) are a DFT-SCF plus a
    # perturbative MP2-like correlation correction — RIJCOSX + an auxiliary
    # ("/C") basis is the standard way to make that correction cheap (often
    # cheaper than canonical MP2), and an empirical dispersion correction
    # (D3BJ) matters a lot for these weak-interaction scans since the bare
    # double-hybrid correlation alone tends to underbind.
    simple_line = f"! {method} {basis} TightSCF"
    if aux_basis:
        simple_line += f" {aux_basis}"
    if rijcosx:
        simple_line += " RIJCOSX"
    if dispersion and dispersion.lower() != "none":
        simple_line += f" {dispersion}"
    lines = [simple_line, f"%pal nprocs {pal} end", f"%maxcore {maxcore}", xyz_block, ""]
    return "\n".join(lines)


def _run_orca_energy(orca_exe: str, inp_text: str, workdir: Path) -> float:
    """Run one ORCA single point in *workdir*, return the energy in Hartree."""
    inp_path = workdir / "job.inp"
    inp_path.write_text(inp_text)
    proc = subprocess.run(
        [orca_exe, str(inp_path.name)],
        cwd=str(workdir),
        capture_output=True,
        text=True,
        check=False,
    )
    out_text = proc.stdout + proc.stderr
    (workdir / "job.out").write_text(out_text)
    if proc.returncode != 0:
        raise RuntimeError(f"ORCA failed (exit {proc.returncode}) in {workdir}:\n{out_text[-1500:]}")
    energy = parse_orca_out_energy(out_text)
    if energy is None:
        raise RuntimeError(f"Could not parse ORCA energy from {workdir / 'job.out'}:\n{out_text[-1500:]}")
    return energy


def _energy_for(
    symbols: list[str], positions: np.ndarray, ghost_mask: np.ndarray,
    *, method: str, basis: str, pal: int, maxcore: int, orca_exe: str, tmp_prefix: str,
    aux_basis: str | None = None, dispersion: str | None = None, rijcosx: bool = False,
) -> float:
    inp_text = _render_input(
        method, basis, pal, maxcore,
        _xyz_block(symbols, positions, ghost_mask, charge=0, multiplicity=1),
        aux_basis=aux_basis, dispersion=dispersion, rijcosx=rijcosx,
    )
    with tempfile.TemporaryDirectory(prefix=tmp_prefix) as tmp:
        return _run_orca_energy(orca_exe, inp_text, Path(tmp))


def _cache_key(label_a: str, label_b: str, backend_name: str, distance: float, offset: float) -> tuple:
    return (label_a, label_b, backend_name, round(float(distance), 6), round(float(offset), 6))


def load_cached_results(csv_path: Path) -> tuple[list[dict], set[tuple]]:
    """Load a previous run's output CSV (if any) as resumable cache state."""
    if not csv_path.is_file():
        return [], set()
    df = pd.read_csv(csv_path)
    rows = df.to_dict("records")
    seen = {
        _cache_key(r["molecule_a"], r["molecule_b"], r["backend"], r["distance_angstrom"], r["offset_angstrom"])
        for r in rows
    }
    print(f"Resuming from {csv_path}: {len(rows)} points already computed, will be skipped.")
    return rows, seen


def evaluate_orca_scan(
    geometries, label_a: str, label_b: str, *,
    method: str, basis: str, counterpoise: bool,
    pal: int, maxcore: int, orca_exe: str, backend_name: str,
    skip_keys: set[tuple] | None = None,
    aux_basis: str | None = None, dispersion: str | None = None, rijcosx: bool = False,
) -> list[dict]:
    rows: list[dict] = []
    isolated_cache: dict[str, float] = {}
    skip_keys = skip_keys or set()

    for geom in geometries:
        cache_key = _cache_key(label_a, label_b, backend_name, geom.distance_angstrom, geom.offset_angstrom)
        if cache_key in skip_keys:
            continue

        symbols = geom.atoms.get_chemical_symbols()
        positions = geom.atoms.get_positions()
        idx_a, idx_b = geom.fragments
        n = len(symbols)
        mask_a = np.zeros(n, dtype=bool)
        mask_a[idx_a] = True

        tag = f"{label_a}_{label_b}_{geom.distance_angstrom:.3f}_{geom.offset_angstrom:.3f}_"
        try:
            common = dict(
                method=method, basis=basis, pal=pal, maxcore=maxcore, orca_exe=orca_exe,
                aux_basis=aux_basis, dispersion=dispersion, rijcosx=rijcosx,
            )
            e_dimer = _energy_for(
                symbols, positions, np.zeros(n, dtype=bool),
                tmp_prefix=f"orca_{tag}dimer_", **common,
            )

            if counterpoise:
                # Fragment A real, fragment B as ghost basis (and vice versa) —
                # both computed *in the full dimer geometry* so the ghost
                # basis functions sit exactly where the real atoms are.
                e_a = _energy_for(
                    symbols, positions, ~mask_a,
                    tmp_prefix=f"orca_{tag}ghostB_", **common,
                )
                e_b = _energy_for(
                    symbols, positions, mask_a,
                    tmp_prefix=f"orca_{tag}ghostA_", **common,
                )
            else:
                # Isolated monomer energies don't depend on separation/offset
                # (rigid monomers) — compute once per pair and reuse.
                if "A" not in isolated_cache:
                    isolated_cache["A"] = _energy_for(
                        [symbols[i] for i in idx_a], positions[idx_a],
                        np.zeros(len(idx_a), dtype=bool),
                        tmp_prefix=f"orca_{label_a}_{label_b}_isoA_", **common,
                    )
                if "B" not in isolated_cache:
                    isolated_cache["B"] = _energy_for(
                        [symbols[i] for i in idx_b], positions[idx_b],
                        np.zeros(len(idx_b), dtype=bool),
                        tmp_prefix=f"orca_{label_a}_{label_b}_isoB_", **common,
                    )
                e_a = isolated_cache["A"]
                e_b = isolated_cache["B"]

            e_int_hartree = e_dimer - e_a - e_b
            rows.append(
                {
                    "molecule_a": label_a,
                    "molecule_b": label_b,
                    "distance_angstrom": geom.distance_angstrom,
                    "offset_angstrom": geom.offset_angstrom,
                    "energy_ev": e_int_hartree * HARTREE_TO_EV,
                    "energy_kcal_mol": e_int_hartree * HARTREE_TO_KCAL_MOL,
                    "comp_Edimer_hartree": e_dimer,
                    "comp_EfragA_hartree": e_a,
                    "comp_EfragB_hartree": e_b,
                    "min_contact_angstrom": min_fragment_contact_distance(geom.atoms, geom.fragments),
                    "backend": backend_name,
                }
            )
            print(
                f"    d={geom.distance_angstrom:.2f} off={geom.offset_angstrom:.2f}  "
                f"E_int={e_int_hartree * HARTREE_TO_KCAL_MOL:+.3f} kcal/mol"
            )
        except Exception as e:
            print(f"    Warning: {method} failed at d={geom.distance_angstrom:.2f} off={geom.offset_angstrom:.2f}: {e}")

    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--pairs", nargs="+", default=["TIP3:TIP3", "BENZ:BENZ"],
        help="Pairs to scan as LABEL_A:LABEL_B (default: TIP3:TIP3 BENZ:BENZ)",
    )
    parser.add_argument(
        "--methods", nargs="+", default=["HF", "MP2"],
        help=(
            "ORCA method keywords (default: HF MP2). Double-hybrid DFT methods work "
            "the same way, e.g. --methods B2PLYP DSD-BLYP — pair with --aux-basis, "
            "--rijcosx, and --dispersion D3BJ for efficiency/accuracy."
        ),
    )
    parser.add_argument("--basis", default="def2-SVP", help="ORCA basis set (default: def2-SVP)")
    parser.add_argument(
        "--aux-basis", default=None,
        help=(
            "Auxiliary ('/C') basis for RI-MP2 / double-hybrid correlation, e.g. "
            "def2-SVP/C. Recommended whenever --rijcosx is set or a double-hybrid "
            "method is used — makes the perturbative correlation step much cheaper."
        ),
    )
    parser.add_argument(
        "--rijcosx", action="store_true",
        help="Add RIJCOSX (RI-J + COSX exchange approximation) — standard for efficient double-hybrid/RI-MP2 runs.",
    )
    parser.add_argument(
        "--dispersion", default="none", choices=["none", "D3BJ", "D3ZERO"],
        help=(
            "Empirical dispersion correction (default: none). Recommended for double-hybrid "
            "DFT methods on weak-interaction scans like this one (D3BJ) — not meaningful for "
            "HF/MP2, which already include (or lack) dispersion through their own physics."
        ),
    )
    parser.add_argument(
        "--no-counterpoise", action="store_true",
        help="Skip Boys-Bernardi counterpoise correction (cheaper, but BSSE will distort close-range results)",
    )
    parser.add_argument("--pal", type=int, default=4, help="ORCA %%pal nprocs (default 4)")
    parser.add_argument("--maxcore", type=int, default=2000, help="ORCA %%maxcore MB per core (default 2000)")
    parser.add_argument("--orca-exe", default=None, help="Path to orca executable (default: $ORCA or 'orca' on PATH)")
    parser.add_argument("--min-contact", type=float, default=1.5, help="Grid-anchoring contact distance (default 1.5)")
    parser.add_argument(
        "--close-floor", type=float, default=None,
        help="Extend the grid inward to this centre-to-centre distance (Å), same as run_dimer_scan_campaign.py",
    )
    parser.add_argument("--n-close", type=int, default=6, help="Extra points between --close-floor and grid start")
    parser.add_argument("--output-dir", type=Path, default=Path("results/dimer_scan_campaign/orca_hf_mp2"))
    parser.add_argument(
        "--no-resume", action="store_true",
        help="Ignore any existing scan_results.csv in --output-dir and recompute everything from scratch",
    )
    args = parser.parse_args()

    orca_exe_requested = args.orca_exe or __import__("os").environ.get("ORCA", "orca")
    import shutil
    orca_exe = shutil.which(orca_exe_requested)
    if orca_exe is None:
        print(f"ORCA executable not found: {orca_exe_requested!r}. Load the ORCA module or pass --orca-exe.")
        sys.exit(1)
    # ORCA requires the *full absolute pathname* when running in parallel
    # (%pal nprocs > 1) — it re-execs itself using this exact string for the
    # worker processes, so a bare "orca" resolved only for existence-checking
    # isn't enough; every single point would fail with "ORCA_MAIN: For
    # parallel runs ORCA has to be called with full pathname".
    orca_exe = str(Path(orca_exe).resolve())
    print(f"Using ORCA executable: {orca_exe}")

    basis_slug = args.basis.lower().replace("-", "").replace("*", "s").replace("(", "").replace(")", "").replace(",", "")
    cp_suffix = "_cp" if not args.no_counterpoise else ""

    args.output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.output_dir / "scan_results.csv"
    if args.no_resume:
        results, seen_keys = [], set()
    else:
        results, seen_keys = load_cached_results(csv_path)

    for pair_str in args.pairs:
        label_a, label_b = pair_str.split(":")
        if (label_a, label_b) not in PAIR_SCAN_CONFIG:
            print(f"Skipping {pair_str}: not in PAIR_SCAN_CONFIG")
            continue
        cfg = PAIR_SCAN_CONFIG[(label_a, label_b)]
        offsets = cfg["offsets_angstrom"]
        distances, safe_d = build_pair_distance_grid(
            label_a, label_b, min_contact=args.min_contact,
            close_floor=args.close_floor, n_close=args.n_close,
        )
        geometries = list(make_oriented_scan_geometries(label_a, label_b, distances, offsets))
        print(f"{label_a}+{label_b}: {cfg['description']}")
        print(
            f"  safe contact clears at d≈{safe_d:.2f} Å — grid spans "
            f"{distances.min():.2f}–{distances.max():.2f} Å ({len(distances)} distances × {len(offsets)} offsets)"
        )

        for method in args.methods:
            backend_name = f"{method.lower()}_{basis_slug}{cp_suffix}"
            print(f"  Evaluating {method}/{args.basis}{' (counterpoise)' if not args.no_counterpoise else ''}...")
            rows = evaluate_orca_scan(
                geometries, label_a, label_b,
                method=method, basis=args.basis, counterpoise=not args.no_counterpoise,
                pal=args.pal, maxcore=args.maxcore, orca_exe=orca_exe, backend_name=backend_name,
                skip_keys=seen_keys,
            )
            results.extend(rows)
            seen_keys.update(
                _cache_key(r["molecule_a"], r["molecule_b"], r["backend"], r["distance_angstrom"], r["offset_angstrom"])
                for r in rows
            )

            # Save incrementally so partial progress survives a crash/timeout
            # and can be resumed with the same command later.
            pd.DataFrame(results).to_csv(csv_path, index=False)

    print(f"Results saved to {csv_path}")


if __name__ == "__main__":
    main()
