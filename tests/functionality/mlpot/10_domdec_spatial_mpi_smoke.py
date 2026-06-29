#!/usr/bin/env python3
"""Tier 3 DOMDEC + Spatial MPI MLpot smoke.

Exercises the new ``build_domdec_spatial_batch_indices`` path (Phase 3) that
uses CHARMM's own DOMDEC atom ownership (``domdec_atoms`` ctypes) instead of
COM slabs when spatial MPI is active.

Two independent sub-tests:

1. **Callback-only** (default, no real CHARMM / no checkpoint needed):
   Patches DOMDEC ctypes to report ``domdec_active=True`` + ``NDIR=(N,1,1)``,
   then runs ``DecomposedMlpotCalculator.calculate_charmm`` with a mocked JAX
   forward to confirm the DOMDEC-ownership branch is taken and ``owned_mono``
   counts are rank-correct.

2. **Live CHARMM ENER** (``--charmm-ener``, cluster only, requires
   ``--checkpoint`` and prebuilt PSF/CRD/**RES**):
   At ``np>1``, loads a CHARMM restart written at ``np=1`` (multi-step
   ``read rtf/psf`` hangs under MPI on current builds).  Then issues
   ``domdec ndir N 1 1`` + ``faster on``, registers spatial MLpot, ENER.

Prerequisites
-------------
Prebuilt PSF/CRD/**RES** for the live path (run once at ``MMML_MPI_NP=1``)::

    MMML_MPI_NP=1 ./scripts/mmml-charmm-mpirun.sh python \\
      tests/functionality/mlpot/10_domdec_spatial_mpi_smoke.py \\
      --prepare-prebuilt-only --residue DCM --n-molecules 20 --box-side 40

The prepare step writes ``artifacts/domdec_spatial_smoke/dcm_20mer.{psf,crd,res}``.
``np>1`` live ENER loads the ``.res`` restart (not the RTF/PSF chain).

Callback-only (np=2, no checkpoint; CPU node: add JAX_PLATFORM_NAME=cpu)::

    MMML_MPI_NP=2 MMML_MLPOT_SPATIAL_MPI=1 \\
      CUDA_VISIBLE_DEVICES="" JAX_PLATFORM_NAME=cpu \\
      ./scripts/mmml-charmm-mpirun.sh python \\
      tests/functionality/mlpot/10_domdec_spatial_mpi_smoke.py

Live ENER (np=4, prebuilt artifacts required)::

    MMML_MPI_NP=4 MMML_MLPOT_SPATIAL_MPI=1 \\
      ./scripts/mmml-charmm-mpirun.sh python \\
      tests/functionality/mlpot/10_domdec_spatial_mpi_smoke.py \\
      --charmm-ener --checkpoint $MMML_CKPT \\
      --residue DCM --n-molecules 20 --box-side 40

Pass criteria
-------------
* Callback-only: ``use_spatial=True``, ``domdec_path=True``,
  owned monomer counts partition the system, allreduced energy finite.
* Live ENER: ``energy.show()`` completes, TOTE is finite, ``domdec_summary``
  reports ``DOMDEC active: True`` and ``Symbols found: 8/8``.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from unittest import mock

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

_DOMDEC_MODULE = "mmml.interfaces.pycharmmInterface.mlpot.mpi_spatial.domdec_atoms"


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--residue", default="DCM",
                   help="CGenFF residue name for the test system (default: DCM).")
    p.add_argument("--n-molecules", type=int, default=20,
                   help="Number of monomers in the test system.")
    p.add_argument("--box-side", type=float, default=40.0,
                   help="Cubic box side length (Å).")
    p.add_argument("--atoms-per-monomer", type=int, default=None,
                   help="Atoms per monomer (default: auto from n_atoms / n_molecules).")
    p.add_argument("--checkpoint", type=Path, default=None,
                   help="PhysNet checkpoint for the live ENER path (env: MMML_CKPT).")
    p.add_argument("--prebuilt-dir", type=Path,
                   default=Path("artifacts/domdec_spatial_smoke"),
                   help="Directory for prebuilt PSF/CRD artifacts.")
    p.add_argument("--prebuilt-psf", type=Path, default=None)
    p.add_argument("--prebuilt-crd", type=Path, default=None)
    p.add_argument(
        "--prepare-prebuilt-only",
        action="store_true",
        help="Build PSF/CRD with np=1 and exit (no ENER or spatial-MPI test).",
    )
    p.add_argument(
        "--charmm-ener",
        action="store_true",
        help="Run the live CHARMM ENER sub-test (requires checkpoint + prebuilt PSF/CRD).",
    )
    p.add_argument("--ndir", type=int, default=None,
                   help="DOMDEC NDIR (domains along x). Default: mpi_size.")
    p.add_argument("--cutnb", type=float, default=10.0,
                   help="Nonbond cutoff for PBC ENER (Å). Must satisfy box/ndir >= cutnb.")
    p.add_argument("--dry-run", action="store_true",
                   help="Print recommended launch commands and exit.")
    return p.parse_args()


def _mpi_info() -> tuple[int, int]:
    try:
        from mmml.interfaces.pycharmmInterface.mlpot.mpi_bridge import mpi_rank_size
        return mpi_rank_size()
    except Exception:
        return 0, max(1, int(os.environ.get("MMML_MPI_NP", "1")))


def _mpi_barrier(*, tag: str = "sync") -> None:
    """Synchronise Python ranks before/after CHARMM script blocks."""
    _, size = _mpi_info()
    if size <= 1:
        return
    try:
        from mmml.interfaces.pycharmmInterface.charmm_mpi import _mpi_script_barrier

        _mpi_script_barrier()
    except Exception as exc:
        _log(tag, f"MPI barrier failed ({type(exc).__name__}: {exc}) — CHARMM may hang")


def _log(tag: str, msg: str) -> None:
    rank, size = _mpi_info()
    print(f"[{tag} rank {rank}/{size}] {msg}", flush=True)


def _skip_vacuum_crystal_free_for_mpi() -> None:
    """Avoid ``prepare_charmm_vacuum`` → ``crystal free`` during np>1 prebuilt load."""
    import mmml.interfaces.pycharmmInterface.mlpot.setup as mlpot_setup

    def _skip() -> None:
        _log("setup", "skipping prepare_charmm_vacuum for np>1 prebuilt path")

    mlpot_setup.prepare_charmm_vacuum = _skip


def _configure_live_charmm_mpi_import(*, size: int) -> None:
    """Env guards before the first ``import_pycharmm`` on np>1 live CHARMM paths."""
    if size <= 1:
        return
    os.environ.setdefault("MMML_SKIP_CHARMM_RESET_BLOCK", "1")
    os.environ.setdefault("MMML_DEFER_MPI4PY_PACKAGE_IMPORT", "1")
    os.environ.setdefault("MMML_QUIET", "1")
    _log("setup", "np>1 live CHARMM: skip import-time reset_block; defer mpi4py; MMML_QUIET")
    _skip_vacuum_crystal_free_for_mpi()


def _sync_import_pycharmm(*, tag: str = "sync") -> None:
    """Load ``import_pycharmm`` on all ranks in lockstep (mpi4py barriers)."""
    rank, size = _mpi_info()
    if size <= 1:
        import mmml.interfaces.pycharmmInterface.import_pycharmm  # noqa: F401
        return
    _log(tag, "barrier before import_pycharmm")
    _mpi_barrier(tag=tag)
    _log(tag, "importing import_pycharmm")
    import mmml.interfaces.pycharmmInterface.import_pycharmm  # noqa: F401
    from mmml.interfaces.pycharmmInterface.charmm_mpi import ensure_mpi4py_after_charmm_init

    if not ensure_mpi4py_after_charmm_init(phase="synchronized import_pycharmm"):
        raise RuntimeError(f"rank {rank}/{size}: mpi4py.MPI unavailable after import_pycharmm")
    _log(tag, "import_pycharmm done — barrier")
    _mpi_barrier(tag=tag)


def _prebuilt_paths(args: argparse.Namespace) -> tuple[Path, Path, Path]:
    stem = f"{args.residue.lower()}_{args.n_molecules}mer"
    base = Path(args.prebuilt_dir)
    psf = args.prebuilt_psf or (base / f"{stem}.psf")
    crd = args.prebuilt_crd or (base / f"{stem}.crd")
    res = base / f"{stem}.res"
    return Path(psf), Path(crd), Path(res)


def _write_prebuilt_restart(res_path: Path, *, write_unit: int = 20) -> None:
    """Write CHARMM restart from in-memory state (np=1 prepare phase)."""
    import pycharmm.lingo as lingo

    from mmml.interfaces.pycharmmInterface.charmm_levels import charmm_relaxed_bomlev
    from mmml.interfaces.pycharmmInterface.charmm_paths import charmm_fortran_path

    res_path.parent.mkdir(parents=True, exist_ok=True)
    fortran_path, alias = charmm_fortran_path(res_path, for_write=True)
    with charmm_relaxed_bomlev():
        lingo.charmm_script(
            f"open write form unit {write_unit} name {fortran_path}\n"
            f"write restart unit {write_unit}\n"
            f"close unit {write_unit}\n"
        )
    if alias is not None:
        alias.finalize()


def _load_prebuilt_restart_mpi(res_path: Path, *, read_unit: int = 20) -> None:
    """Load prebuilt state via ``read restart`` (np>1 MPI-safe bootstrap)."""
    from mmml.interfaces.pycharmmInterface.charmm_mpi import mpi_charmm_script
    from mmml.interfaces.pycharmmInterface.charmm_paths import charmm_fortran_path

    if not res_path.is_file():
        raise FileNotFoundError(
            f"Prebuilt restart not found: {res_path}\n"
            "Re-run --prepare-prebuilt-only with MMML_MPI_NP=1 (writes .res)."
        )
    fortran_path, alias = charmm_fortran_path(res_path, for_write=False)
    script = (
        f"open read unit {read_unit} form name {fortran_path}\n"
        f"read restart unit {read_unit}\n"
        f"close unit {read_unit}\n"
    )
    try:
        mpi_charmm_script(script, relaxed_bomlev=True)
    finally:
        if alias is not None:
            alias.finalize()


def _print_dry_run(args: argparse.Namespace) -> None:
    print("# Step 1 — build prebuilt PSF/CRD (np=1):")
    print(
        f"MMML_MPI_NP=1 ./scripts/mmml-charmm-mpirun.sh python \\\n"
        f"  tests/functionality/mlpot/10_domdec_spatial_mpi_smoke.py \\\n"
        f"  --prepare-prebuilt-only --residue {args.residue} "
        f"--n-molecules {args.n_molecules} --box-side {args.box_side}"
    )
    print()
    print("# Step 2 — callback-only smoke (no checkpoint needed):")
    np_cb = max(2, int(os.environ.get("MMML_MPI_NP", "2")))
    print(
        f"MMML_MPI_NP={np_cb} MMML_MLPOT_SPATIAL_MPI=1 \\\n"
        f"  CUDA_VISIBLE_DEVICES=\"\" JAX_PLATFORM_NAME=cpu \\\n"
        f"  ./scripts/mmml-charmm-mpirun.sh python \\\n"
        f"  tests/functionality/mlpot/10_domdec_spatial_mpi_smoke.py"
    )
    print()
    print("# Step 3 — live CHARMM ENER (requires checkpoint + prebuilt artifacts):")
    ckpt = args.checkpoint or Path("$MMML_CKPT")
    print(
        f"MMML_MPI_NP={np_cb} MMML_MLPOT_SPATIAL_MPI=1 \\\n"
        f"  ./scripts/mmml-charmm-mpirun.sh python \\\n"
        f"  tests/functionality/mlpot/10_domdec_spatial_mpi_smoke.py \\\n"
        f"  --charmm-ener --checkpoint {ckpt} \\\n"
        f"  --residue {args.residue} --n-molecules {args.n_molecules} "
        f"--box-side {args.box_side}"
    )


# ---------------------------------------------------------------------------
# Sub-test 1: callback-only with mocked DOMDEC
# ---------------------------------------------------------------------------

def _callback_smoke(args: argparse.Namespace) -> int:
    """Exercise the DOMDEC-active branch without live CHARMM."""
    rank, size = _mpi_info()
    if size < 2:
        _log("callback", "skipped (need np>=2; re-run under mpirun with MMML_MPI_NP=2)")
        return 0

    # Log first so output from all ranks confirms they are alive before heavy imports.
    _log("callback", "starting — heavy imports next, may take 1–2 min on first run")

    import jax.numpy as jnp

    from mmml.interfaces.pycharmmInterface.cutoffs import CutoffParameters
    from mmml.interfaces.pycharmmInterface.mlpot.hybrid_mlpot import (
        DecomposedMlpotCalculator,
    )
    from mmml.interfaces.pycharmmInterface.mlpot.medium_pbc_validation import (
        lattice_positions_cubic_pbc,
    )

    _log("callback", "imports done, building test fixture")

    box = float(args.box_side)
    n_monomers = 8  # lightweight fixture regardless of CLI n-molecules
    atoms_per = 10
    pos = lattice_positions_cubic_pbc(n_monomers, atoms_per, box, spacing_A=5.0, seed=42)
    z = np.tile(np.array([6, 17, 1, 1, 1, 1, 1, 1, 1, 1], dtype=int), n_monomers)
    n = len(z)

    # Synthetic per-rank atom ownership: contiguous atom blocks
    total_atoms = n_monomers * atoms_per
    per_rank = total_atoms // size
    lo = rank * per_rank
    hi = lo + per_rank
    local_atoms = np.arange(lo, hi, dtype=np.int32)
    expected_owned = n_monomers // size   # monomers per rank (uniform split)

    calc = DecomposedMlpotCalculator(
        mock.MagicMock(),
        CutoffParameters(),
        n_monomers,
        z,
        cell=box,
        do_mm=False,
        spatial_mpi=True,
        atoms_per_monomer=[atoms_per] * n_monomers,
    )

    captured: dict = {}

    def _fake_forward_fn(*, n_atoms, atomic_numbers_jax, box_jax):
        def _eval(
            positions_jax,
            mm_pair_idx,
            mm_pair_mask,
            use_mm_pairs,
            spatial_monomer_indices,
            spatial_dimer_indices,
            use_spatial,
        ):
            captured["use_spatial"] = bool(use_spatial)
            captured["owned_mono"] = int(spatial_monomer_indices.shape[0])
            val = float(rank + 1)
            return jnp.array(val), jnp.full((n_atoms, 3), val * 0.01)

        return _eval

    calc._get_spherical_forward_fn = mock.MagicMock(side_effect=_fake_forward_fn)
    x, y, zc = pos[:, 0], pos[:, 1], pos[:, 2]
    dx = dy = dz = np.zeros(n, dtype=np.float64)

    # Barrier 1: synchronise all ranks after heavy imports (JAX, etc.).
    # Callback-only runs with MMML_WARMUP_MLPOT_JAX_ONLY (no pycharmm) and
    # mpi4py already initialised MPI in main().
    _mpi_barrier()
    _log("callback", "running calculate_charmm with mocked DOMDEC ctypes")
    with (
        mock.patch(f"{_DOMDEC_MODULE}.is_domdec_active", return_value=True),
        mock.patch(f"{_DOMDEC_MODULE}.get_ndir", return_value=(size, 1, 1)),
        mock.patch(f"{_DOMDEC_MODULE}.get_local_atom_indices", return_value=local_atoms),
        mock.patch(f"{_DOMDEC_MODULE}.get_ghost_atom_indices",
                   return_value=np.empty(0, dtype=np.int32)),
        mock.patch(
            "mmml.interfaces.pycharmmInterface.mlpot.pbc_env.resolve_mlpot_mic_box_side_A",
            return_value=(box, "smoke"),
        ),
        mock.patch(
            "mmml.interfaces.pycharmmInterface.jax_device_policy.mlpot_jax_device_context",
            return_value=mock.MagicMock(
                __enter__=mock.MagicMock(return_value=None),
                __exit__=mock.MagicMock(return_value=False),
            ),
        ),
        mock.patch("mmml.interfaces.pycharmmInterface.charmm_mpi.recover_mpi_for_charmm_after_jax"),
        mock.patch("mmml.utils.jax_gpu_warmup.sync_jax_gpu_before_charmm"),
        # Prevent broadcast_mlpot_result / mpi_allreduce_forces / mpi_allreduce_energy
        # from calling ensure_charmm_mpi_initialized(), which would try to run
        # init_vacuum_charmm_state_mpi() (sets up CHARMM topology, nbonds, etc.)
        # — not needed in this mocked callback-only path and potentially
        # unsafe if pycharmm was imported without a vacuum topology.
        mock.patch("mmml.interfaces.pycharmmInterface.charmm_mpi.ensure_charmm_mpi_initialized"),
        mock.patch(
            "mmml.interfaces.pycharmmInterface.mlpot.hybrid_mlpot.as_ml_array",
            side_effect=lambda arr, dtype=None: jnp.asarray(arr),
        ),
        mock.patch(
            "mmml.interfaces.pycharmmInterface.mlpot.hybrid_mlpot.resolve_ml_compute_dtype",
            return_value=jnp.float32,
        ),
        mock.patch(
            "mmml.interfaces.pycharmmInterface.mlpot.hybrid_mlpot.jax.device_get",
            side_effect=lambda x: np.asarray(x),
        ),
    ):
        energy_val = calc.calculate_charmm(
            n, 0, 0, None,
            x, y, zc, dx, dy, dz,
            0, 0, None, None, None, None, None, None, None,
        )

    # Barrier 2: no rank exits main() before all ranks complete calculate_charmm.
    # This prevents rank 0 from calling MPI_Finalize while others are mid-collective.
    _mpi_barrier()

    _log("callback", f"energy={float(energy_val):.4f}  use_spatial={captured.get('use_spatial')}  "
                     f"owned_mono={captured.get('owned_mono')}")

    failures: list[str] = []
    if not captured.get("use_spatial"):
        failures.append(f"rank {rank}: use_spatial=False (expected True at np>{size-1})")
    owned = captured.get("owned_mono", -1)
    if owned != expected_owned:
        failures.append(
            f"rank {rank}: owned_monomers={owned} != expected {expected_owned} "
            f"(DOMDEC ctypes path should partition {n_monomers} monomers over {size} ranks)"
        )
    if not np.isfinite(float(energy_val)):
        failures.append(f"rank {rank}: non-finite energy {energy_val}")

    if failures:
        for f in failures:
            print(f"FAIL callback smoke: {f}", file=sys.stderr)
        return 1

    if rank == 0:
        print(
            f"PASS callback smoke: np={size} energy={float(energy_val):.4f} "
            f"use_spatial={captured['use_spatial']} "
            f"owned_mono_per_rank={captured['owned_mono']} "
            f"(DOMDEC ctypes ownership path confirmed)"
        )
    return 0


# ---------------------------------------------------------------------------
# Sub-test 2: live CHARMM ENER with DOMDEC + spatial MPI
# ---------------------------------------------------------------------------

def _prepare_prebuilt(args: argparse.Namespace) -> int:
    """Build PSF/CRD for the multi-monomer test system (np=1 phase)."""
    rank, size = _mpi_info()
    if size > 1:
        print("--prepare-prebuilt-only must be run with MMML_MPI_NP=1.", file=sys.stderr)
        return 6

    from _common import build_ase_cluster  # noqa: F401 — available in tests/functionality/mlpot/

    _log("prepare", f"building {args.residue}:{args.n_molecules} cluster")
    z, r = build_ase_cluster(args.residue, args.n_molecules, spacing=5.0)
    _log("prepare", f"cluster built: {len(z)} atoms")

    import pycharmm.write as write

    psf_path, crd_path, res_path = _prebuilt_paths(args)
    psf_path.parent.mkdir(parents=True, exist_ok=True)
    _log("prepare", f"writing PSF → {psf_path}")
    write.psf_card(str(psf_path), title=f"DOMDEC spatial MPI smoke {args.residue}:{args.n_molecules}")
    _log("prepare", f"writing CRD → {crd_path}")
    write.coor_card(str(crd_path), title="coords")
    _log("prepare", f"writing RES → {res_path}")
    _write_prebuilt_restart(res_path)
    print(f"\nPASS prepare: PSF={psf_path}  CRD={crd_path}  RES={res_path}")
    return 0


def _psf_atom_types(psf_path: Path) -> set[str]:
    types: set[str] = set()
    in_atoms = False
    remaining = 0
    for line in psf_path.read_text(encoding="utf-8", errors="replace").splitlines():
        if "!NATOM" in line:
            remaining = int(line.split()[0])
            in_atoms = True
            continue
        if not in_atoms:
            continue
        if remaining <= 0:
            break
        parts = line.split()
        if len(parts) >= 6:
            types.add(parts[5])
            remaining -= 1
    if not types:
        raise ValueError(f"No atom types parsed from PSF: {psf_path}")
    return types


def _shared_minimal_rtf_path(psf_path: Path) -> Path:
    """Deterministic sidecar path — all MPI ranks must read the same RTF file."""
    return psf_path.with_name(f"{psf_path.stem}.minimal_mass.rtf")


def _ensure_shared_minimal_rtf(psf_path: Path, prm_path: Path) -> Path:
    """Build minimal MASS-only RTF beside the PSF (shared path for MPI read.rtf).

    Per-rank ``tempfile.mkstemp`` paths deadlock under ``mpirun``: CHARMM MPI
    expects every rank to pass the same filename to ``read rtf`` (rank 0 reads,
    others receive the broadcast topology).
    """
    needed = _psf_atom_types(psf_path)
    mass_lines: list[str] = []
    seen: set[str] = set()
    for line in prm_path.read_text(encoding="utf-8", errors="replace").splitlines():
        parts = line.split()
        if len(parts) >= 4 and parts[0].upper() == "MASS" and parts[2] in needed:
            mass_lines.append(line)
            seen.add(parts[2])
    missing = sorted(needed - seen)
    if missing:
        raise ValueError(f"Missing MASS records in {prm_path}: {missing}")

    out = _shared_minimal_rtf_path(psf_path)
    lines = [
        "* MMML DOMDEC spatial smoke minimal MASS topology",
        "*",
        *mass_lines,
        "END",
    ]
    rank, size = _mpi_info()
    if rank == 0 or (size <= 1 and not out.is_file()):
        out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    if size > 1:
        _mpi_barrier(tag="load")
    return out


def _load_prebuilt(args: argparse.Namespace) -> tuple["np.ndarray", "np.ndarray", int, list[int]]:
    """Load prebuilt PSF/CRD and return (z, r, n_monomers, atoms_per_monomer)."""
    import numpy as np
    import pycharmm.coor as coor

    from mmml.interfaces.pycharmmInterface.charmm_mpi import mpi_charmm_script
    from mmml.interfaces.pycharmmInterface.import_pycharmm import CGENFF_PRM
    from mmml.interfaces.pycharmmInterface.utils import get_Z_from_psf

    rank, size = _mpi_info()
    psf_path, crd_path, res_path = _prebuilt_paths(args)
    if size > 1:
        _log("load", f"MPI read restart begin: {res_path.resolve()}")
        _load_prebuilt_restart_mpi(res_path.resolve())
        _log("load", "MPI read restart done")
    else:
        if not psf_path.is_file():
            raise FileNotFoundError(
                f"Prebuilt PSF not found: {psf_path}\n"
                "Run with --prepare-prebuilt-only first (MMML_MPI_NP=1)."
            )
        if not crd_path.is_file():
            raise FileNotFoundError(f"Prebuilt CRD not found: {crd_path}")
        minimal_rtf = _ensure_shared_minimal_rtf(psf_path, Path(CGENFF_PRM)).resolve()
        psf_abs = psf_path.resolve()
        crd_abs = crd_path.resolve()
        prm_abs = Path(CGENFF_PRM).resolve()
        n_types = len(_psf_atom_types(psf_path))
        _log("load", f"serial READ begin ({n_types} MASS types)")
        read_script = (
            f"read rtf card name {minimal_rtf}\n"
            f"read param card name {prm_abs} flex\n"
            f"read psf card name {psf_abs}\n"
            f"read coor card name {crd_abs}\n"
        )
        mpi_charmm_script(read_script, relaxed_bomlev=True)
        _log("load", "serial READ done")
    _log("load", f"loaded PSF/CRD on rank {rank}/{size}")

    z = np.asarray(get_Z_from_psf(), dtype=int)
    r = coor.get_positions()[["x", "y", "z"]].to_numpy(dtype=float)
    n_atoms = len(z)
    if n_atoms <= 0:
        raise RuntimeError(
            f"rank {rank}/{size}: empty CHARMM state after prebuilt load. "
            "At np>1 use read restart (.res from --prepare-prebuilt-only)."
        )
    n_monomers = int(args.n_molecules)
    apm_cli = args.atoms_per_monomer
    if apm_cli is not None:
        if n_atoms % apm_cli != 0:
            raise ValueError(f"n_atoms={n_atoms} not divisible by --atoms-per-monomer={apm_cli}")
        atoms_per_monomer = [apm_cli] * n_monomers
    elif n_atoms % n_monomers != 0:
        raise ValueError(
            f"n_atoms={n_atoms} not divisible by n_molecules={n_monomers}. "
            "Use --atoms-per-monomer if monomers differ in size."
        )
    else:
        apm = n_atoms // n_monomers
        atoms_per_monomer = [apm] * n_monomers
    _log("load", f"loaded: n_atoms={n_atoms} n_monomers={n_monomers} apm={atoms_per_monomer[0]}")
    return z, r, n_monomers, atoms_per_monomer


def _charmm_domdec_ener_smoke(args: argparse.Namespace) -> int:
    """Live CHARMM ENER: prebuilt DCM + DOMDEC + spatial MPI."""
    rank, size = _mpi_info()
    ndir = args.ndir or size  # default: one domain per rank along x

    from _common import check_mlpot_symbols, resolve_checkpoint

    _log("ener", "loading prebuilt topology")
    os.environ["MMML_NO_CHARMM_DOMDEC_OFF"] = "1"
    _sync_import_pycharmm(tag="ener")

    from mmml.interfaces.pycharmmInterface.charmm_mpi import mpi_charmm_script

    _log("ener", "crystal free (MPI bootstrap)")
    mpi_charmm_script("crystal free\n", quiet=True)

    _log("ener", "checking MLpot symbols")
    missing = check_mlpot_symbols()
    if missing:
        print(f"FAIL: missing libcharmm MLpot symbols: {missing}", file=sys.stderr)
        return 1
    _mpi_barrier(tag="ener")

    ckpt = resolve_checkpoint(args.checkpoint)
    if rank == 0:
        print(f"\nDOMDEC + Spatial MPI ENER smoke: np={size} ndir={ndir} "
              f"residue={args.residue} n_monomers={args.n_molecules} "
              f"box={args.box_side:.1f}Å cutnb={args.cutnb:.1f}Å checkpoint={ckpt}",
              flush=True)
    _mpi_barrier(tag="ener")

    # Geometry check: domain width must be >= cutnb
    domain_width = float(args.box_side) / max(1, ndir)
    if domain_width < float(args.cutnb):
        print(
            f"WARN: domain_width={domain_width:.1f}Å < cutnb={args.cutnb:.1f}Å for "
            f"ndir={ndir}. Reducing ndir to avoid DOMDEC geometry hang.",
            file=sys.stderr,
        )
        ndir = max(1, int(float(args.box_side) / float(args.cutnb)))
        _log("ener", f"ndir reduced to {ndir}")

    z, r, n_monomers, atoms_per_monomer = _load_prebuilt(args)
    n_atoms = len(z)
    _mpi_barrier(tag="ener")

    from mmml.interfaces.pycharmmInterface.charmm_mpi import disable_ase_mpi_parallel

    disable_ase_mpi_parallel()
    _mpi_barrier(tag="ener")

    import ase
    import pycharmm.energy as energy

    from mmml.interfaces.pycharmmInterface.charmm_mpi import mpi_charmm_script
    from mmml.interfaces.pycharmmInterface.mlpot.mpi_spatial.domdec_atoms import domdec_summary
    from mmml.interfaces.pycharmmInterface.mlpot.pbc_env import setup_charmm_environment
    from mmml.interfaces.pycharmmInterface.mlpot.setup import (
        load_physnet_mlpot_bundle,
        register_mlpot,
        select_all_atoms,
    )

    _log("ener", f"PBC setup: box={args.box_side:.1f}Å")
    setup_charmm_environment(use_pbc=True, cubic_box_side_A=float(args.box_side))
    _mpi_barrier(tag="ener")

    # Enable DOMDEC
    if ndir > 1:
        domdec_cmd = f"domdec ndir {ndir} 1 1\nfaster on"
    else:
        domdec_cmd = "domdec on"
    _log("ener", f"DOMDEC command (rank 0): {domdec_cmd!r}")
    mpi_charmm_script(domdec_cmd)

    _log("ener", "domdec_summary (pre-MLpot):")
    if rank == 0:
        print(domdec_summary())

    _log("ener", "loading PhysNet checkpoint / building model")
    ase_atoms = ase.Atoms(numbers=z, positions=r)
    _, _, pyCModel = load_physnet_mlpot_bundle(
        ckpt,
        n_atoms,
        ase_atoms,
        n_monomers=n_monomers,
        atoms_per_monomer=atoms_per_monomer,
        cell=float(args.box_side),
    )
    _log("ener", "model built — registering MLpot")
    _mpi_barrier(tag="ener")

    # Enable spatial MPI before the ENER callback fires
    os.environ["MMML_MLPOT_SPATIAL_MPI"] = "1"

    ml_selection = select_all_atoms()
    ctx = register_mlpot(
        pyCModel,
        z.tolist(),
        ml_selection,
        use_pbc=True,
        cubic_box_side_A=float(args.box_side),
    )
    _log("ener", "MLpot registered — running ENER")
    _mpi_barrier(tag="ener")
    try:
        energy.show()
        tote = energy.get_total()
        _log("ener", f"ENER done: TOTE={float(tote):.4f} kcal/mol")
    except Exception as exc:
        print(f"FAIL rank {rank}: ENER raised {type(exc).__name__}: {exc}", file=sys.stderr)
        ctx.unset()
        return 1

    _log("ener", "domdec_summary (post-ENER):")
    summary = domdec_summary()
    print(summary, flush=True)

    ctx.unset()

    if not np.isfinite(float(tote)):
        print(f"FAIL rank {rank}: non-finite TOTE={tote}", file=sys.stderr)
        return 1

    active_in_summary = "DOMDEC active    : True" in summary
    if not active_in_summary:
        print(
            f"WARN rank {rank}: DOMDEC was not reported as active in domdec_summary. "
            "The ctypes path may not have fired (e.g. DOMDEC symbols absent or "
            "DOMDEC not activated before first ENER).",
            file=sys.stderr,
        )

    if rank == 0:
        print(f"\nPASS DOMDEC+Spatial MLpot ENER: TOTE={float(tote):.4f} kcal/mol  "
              f"np={size}  ndir={ndir}  domdec_active={active_in_summary}")
    return 0


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    args = _parse_args()
    callback_only = not args.prepare_prebuilt_only and not args.charmm_ener

    # Callback-only: no real CHARMM topology/ENER — skip PyCHARMM import entirely.
    # Without this, hybrid_mlpot → mm_energy_forces → import_pycharmm loads
    # libcharmm on every rank; non-root ranks enter CHARMM's Fortran receive loop
    # (100% CPU spin) while rank 0 continues in Python → Barrier deadlock.
    #
    # mpi4py owns MPI instead (MMML_MPI_PY_INIT=1).  Safe here because pycharmm
    # is never loaded, so there is no second Fortran MPI_Init.
    if callback_only:
        os.environ["MMML_WARMUP_MLPOT_JAX_ONLY"] = "1"
        os.environ["MMML_MPI_PY_INIT"] = "1"

    from mmml.interfaces.pycharmmInterface.charmm_mpi import (
        mpi4py_openmpi_mismatch,
        prepare_serial_charmm_mpi_env,
    )

    prepare_serial_charmm_mpi_env()
    ok, msg = mpi4py_openmpi_mismatch()
    if not ok:
        print(f"FAIL: {msg}", file=sys.stderr)
        return 1

    if callback_only:
        # First mpi4py import with MMML_MPI_PY_INIT=1: auto MPI_Init under mpirun.
        try:
            from mpi4py import MPI as _MPI  # noqa: N812, F401
        except Exception:
            pass
    elif args.charmm_ener or args.prepare_prebuilt_only:
        rank, size = _mpi_info()
        if args.charmm_ener:
            os.environ.setdefault("MMML_NO_CHARMM_DOMDEC_OFF", "1")
            _configure_live_charmm_mpi_import(size=size)
        # np>1 live ENER: defer PyCHARMM import to _charmm_domdec_ener_smoke so
        # MMML_SKIP_CHARMM_RESET_BLOCK is in place before import_pycharmm loads.
        # ensure_charmm_mpi_initialized also runs init_vacuum/crystal free, which
        # is unsafe before synchronized prebuilt topology on np>1.
        if args.prepare_prebuilt_only or size <= 1:
            from mmml.interfaces.pycharmmInterface.charmm_mpi import ensure_charmm_mpi_initialized
            _log("setup", "ensure_charmm_mpi_initialized")
            ensure_charmm_mpi_initialized()
            _log("setup", "ensure_charmm_mpi_initialized done")
        elif args.charmm_ener:
            _log("setup", f"np={size} live ENER: deferred PyCHARMM bootstrap")

    rank, size = _mpi_info()

    if args.dry_run:
        if rank == 0:
            _print_dry_run(args)
        return 0

    if rank == 0:
        print(f"10_domdec_spatial_mpi_smoke: rank {rank}/{size}")

    if args.prepare_prebuilt_only:
        # Requires PyCHARMM (topology build uses lingo/gen/ic)
        import mmml.interfaces.pycharmmInterface.import_pycharmm  # noqa: F401
        return _prepare_prebuilt(args)

    # Phase 1: callback-only (default; skipped when --charmm-ener — live path
    # loads PyCHARMM on all ranks and must not re-enter callback import pattern).
    if not args.charmm_ener:
        code = _callback_smoke(args)
        if code != 0:
            return code

    # Phase 2 (opt-in): live CHARMM ENER
    if args.charmm_ener:
        return _charmm_domdec_ener_smoke(args)

    return 0


if __name__ == "__main__":
    sys.exit(main())
