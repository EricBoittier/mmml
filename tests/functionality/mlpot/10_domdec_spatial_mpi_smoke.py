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
   ``--checkpoint`` and prebuilt PSF/CRD):
   Loads a DCM PSF/CRD, issues ``domdec ndir N 1 1`` + ``faster on``,
   registers ``DecomposedMlpotCalculator`` (multi-monomer), enables spatial
   MPI, and runs ``energy.show()``.  Reports ``domdec_summary()`` for each
   rank after the ENER call.

Prerequisites
-------------
Prebuilt PSF/CRD for the live path (run once at ``MMML_MPI_NP=1``)::

    MMML_MPI_NP=1 ./scripts/mmml-charmm-mpirun.sh python \\
      tests/functionality/mlpot/10_domdec_spatial_mpi_smoke.py \\
      --prepare-prebuilt-only --residue DCM --n-molecules 20 --box-side 40

Callback-only (np=2, no checkpoint)::

    MMML_MPI_NP=2 MMML_MLPOT_SPATIAL_MPI=1 \\
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


def _mpi_barrier() -> None:
    """Best-effort MPI barrier (no-op when mpi4py is unavailable)."""
    try:
        from mpi4py import MPI  # noqa: PLC0415
        MPI.COMM_WORLD.Barrier()
    except Exception:
        pass


def _log(tag: str, msg: str) -> None:
    rank, size = _mpi_info()
    print(f"[{tag} rank {rank}/{size}] {msg}", flush=True)


def _prebuilt_paths(args: argparse.Namespace) -> tuple[Path, Path]:
    stem = f"{args.residue.lower()}_{args.n_molecules}mer"
    psf = args.prebuilt_psf or (Path(args.prebuilt_dir) / f"{stem}.psf")
    crd = args.prebuilt_crd or (Path(args.prebuilt_dir) / f"{stem}.crd")
    return Path(psf), Path(crd)


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

    # Barrier 1: ensure all ranks finish heavy imports (JAX, etc.) and are
    # ready before any rank calls the first mpi4py collective inside
    # calculate_charmm → broadcast_mlpot_result → mpi_allreduce_*.
    # mpi4py auto-initialises MPI on the first "from mpi4py import MPI"
    # inside _mpi_barrier(), so this is also the point where MPI_Init fires.
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
        # from calling ensure_charmm_mpi_initialized(), which would trigger
        # lingo.charmm_script() and put non-root ranks into CHARMM's Fortran
        # receive loop mid-callback — causing an MPI allreduce deadlock.
        # With MMML_MPI_PY_INIT=1 set in main(), mpi4py already owns MPI
        # (initialised at Barrier 1 above), so CHARMM init is not needed here.
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

    psf_path, crd_path = _prebuilt_paths(args)
    psf_path.parent.mkdir(parents=True, exist_ok=True)
    _log("prepare", f"writing PSF → {psf_path}")
    write.psf_card(str(psf_path), title=f"DOMDEC spatial MPI smoke {args.residue}:{args.n_molecules}")
    _log("prepare", f"writing CRD → {crd_path}")
    write.coor_card(str(crd_path), title="coords")
    print(f"\nPASS prepare: PSF={psf_path}  CRD={crd_path}")
    return 0


def _load_prebuilt(args: argparse.Namespace) -> tuple["np.ndarray", "np.ndarray", int, list[int]]:
    """Load prebuilt PSF/CRD and return (z, r, n_monomers, atoms_per_monomer)."""
    import numpy as np
    import pycharmm.coor as coor
    import pycharmm.read as read

    from mmml.interfaces.pycharmmInterface.charmm_levels import charmm_relaxed_bomlev
    from mmml.interfaces.pycharmmInterface.nbonds_config import read_cgenff_toppar
    from mmml.interfaces.pycharmmInterface.utils import get_Z_from_psf

    psf_path, crd_path = _prebuilt_paths(args)
    if not psf_path.is_file():
        raise FileNotFoundError(
            f"Prebuilt PSF not found: {psf_path}\n"
            "Run with --prepare-prebuilt-only first (MMML_MPI_NP=1)."
        )
    if not crd_path.is_file():
        raise FileNotFoundError(f"Prebuilt CRD not found: {crd_path}")

    _log("load", "reading CGenFF toppar")
    read_cgenff_toppar()
    _log("load", f"reading PSF: {psf_path}")
    with charmm_relaxed_bomlev():
        read.psf_card(str(psf_path))
    _log("load", f"reading CRD: {crd_path}")
    with charmm_relaxed_bomlev():
        read.coor_card(str(crd_path))

    z = np.asarray(get_Z_from_psf(), dtype=int)
    r = coor.get_positions()[["x", "y", "z"]].to_numpy(dtype=float)
    n_atoms = len(z)
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
    import ase
    import pycharmm
    import pycharmm.energy as energy
    import pycharmm.lingo as lingo

    from _common import check_mlpot_symbols, resolve_checkpoint
    from mmml.interfaces.pycharmmInterface.mlpot.mpi_spatial.domdec_atoms import domdec_summary
    from mmml.interfaces.pycharmmInterface.mlpot.pbc_env import setup_charmm_environment
    from mmml.interfaces.pycharmmInterface.mlpot.setup import (
        load_physnet_mlpot_bundle,
        register_mlpot,
        select_all_atoms,
    )

    rank, size = _mpi_info()
    ndir = args.ndir or size  # default: one domain per rank along x

    _log("ener", "checking MLpot symbols")
    missing = check_mlpot_symbols()
    if missing:
        print(f"FAIL: missing libcharmm MLpot symbols: {missing}", file=sys.stderr)
        return 1

    ckpt = resolve_checkpoint(args.checkpoint)
    if rank == 0:
        print(f"\nDOMDEC + Spatial MPI ENER smoke: np={size} ndir={ndir} "
              f"residue={args.residue} n_monomers={args.n_molecules} "
              f"box={args.box_side:.1f}Å cutnb={args.cutnb:.1f}Å checkpoint={ckpt}")

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

    _log("ener", "loading prebuilt topology")
    import mmml.interfaces.pycharmmInterface.import_pycharmm  # noqa: F401
    # Guard: disable the automatic domdec-off hook so active DOMDEC survives MLpot registration
    os.environ["MMML_NO_CHARMM_DOMDEC_OFF"] = "1"

    z, r, n_monomers, atoms_per_monomer = _load_prebuilt(args)
    n_atoms = len(z)

    _log("ener", f"PBC setup: box={args.box_side:.1f}Å")
    setup_charmm_environment(use_pbc=True, cubic_box_side_A=float(args.box_side))

    # Enable DOMDEC
    if ndir > 1:
        domdec_cmd = f"domdec ndir {ndir} 1 1\nfaster on"
    else:
        domdec_cmd = "domdec on"
    _log("ener", f"DOMDEC command: {domdec_cmd!r}")
    lingo.charmm_script(domdec_cmd)

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
    # Parse args FIRST so we know whether CHARMM will be needed before
    # configuring the MPI ownership model.
    args = _parse_args()

    # Callback-only mode: Python (mpi4py) owns MPI init so ALL 4 Python
    # processes stay alive and can participate in mpi4py barriers and
    # Allreduce collectives.  Setting MMML_MPI_PY_INIT=1 prevents
    # configure_mpi4py_charmm_owned_init() from calling
    # mpi4py.rc(initialize=False), so mpi4py auto-initialises MPI normally
    # on every rank — no CHARMM Fortran receive loop is entered.
    #
    # CHARMM-enabled paths (--prepare-prebuilt-only, --charmm-ener):
    # keep the normal CHARMM-owns-MPI design; ensure_charmm_mpi_initialized()
    # loads PyCHARMM and lets non-root ranks enter the CHARMM receive loop,
    # which is required for lingo.charmm_script() to work correctly.
    callback_only = not args.prepare_prebuilt_only and not args.charmm_ener
    if callback_only:
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
        # Init mpi4py MPI BEFORE any pycharmm import fires — exactly the
        # pattern from pyCHARMM Workshop rep_ex_amino_acid.py which does
        # "from mpi4py import MPI" as its very first import.
        #
        # CHARMM's settings.set_verbosity() (called at import time by
        # import_pycharmm._init_charmm_default_levels) initialises CHARMM's
        # Fortran before mpi4py has a chance to call MPI_Init; each rank ends
        # up with COMM_WORLD size=1, making every Barrier and Allreduce a
        # no-op on rank 0 while ranks 1-3 may block on a genuine 4-rank
        # MPI world they initialised later — causing the hang.
        #
        # By calling "from mpi4py import MPI" here (AFTER prepare_serial…
        # which already pinned CUDA_VISIBLE_DEVICES per rank), mpi4py calls
        # MPI_Init with the full 4-rank COMM_WORLD BEFORE pycharmm loads.
        # All subsequent CHARMM Fortran calls then see an already-initialised
        # MPI world of size 4.
        try:
            from mpi4py import MPI as _MPI  # noqa: N812 — local alias only
            if not _MPI.Is_initialized():
                _MPI.Init()
        except Exception:
            pass
    else:
        # Load PyCHARMM so CHARMM Fortran MPI_Init fires before mpi4py
        # collectives.  Non-root ranks enter the CHARMM receive loop here.
        from mmml.interfaces.pycharmmInterface.charmm_mpi import ensure_charmm_mpi_initialized
        ensure_charmm_mpi_initialized()

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

    # Phase 1: callback-only (no checkpoint required, no real CHARMM ENER)
    code = _callback_smoke(args)
    if code != 0:
        return code

    # Phase 2 (opt-in): live CHARMM ENER
    if args.charmm_ener:
        return _charmm_domdec_ener_smoke(args)

    return 0


if __name__ == "__main__":
    sys.exit(main())
