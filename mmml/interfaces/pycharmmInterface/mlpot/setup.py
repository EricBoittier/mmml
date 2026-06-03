"""Register PhysNet (or other) models with ``pycharmm.MLpot``."""

from __future__ import annotations

import ctypes
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Optional, Sequence, Union

import numpy as np
import pandas as pd

PathLike = Union[str, Path]


def _positions_xyzw_dataframe(arr: np.ndarray) -> pd.DataFrame:
    """Build a CHARMM coor dataframe with ``x,y,z,w`` (``w`` = 1)."""
    n = arr.shape[0]
    return pd.DataFrame(
        {
            "x": arr[:, 0],
            "y": arr[:, 1],
            "z": arr[:, 2],
            "w": np.ones(n, dtype=float),
        }
    )


def sync_charmm_positions(positions: np.ndarray) -> None:
    """Push ``(N, 3)`` into CHARMM main and auxiliary coordinate sets."""
    import pycharmm.coor as coor

    arr = np.asarray(positions, dtype=float)
    if arr.ndim != 2 or arr.shape[1] != 3:
        raise ValueError(f"positions must be (N, 3), got {arr.shape}")
    n = coor.get_natom()
    if arr.shape[0] != n:
        raise ValueError(f"positions rows {arr.shape[0]} != CHARMM natom {n}")

    xyz = pd.DataFrame(arr, columns=["x", "y", "z"])
    xyzw = _positions_xyzw_dataframe(arr)
    coor.set_positions(xyz)
    coor.set_main(xyzw)
    coor.set_comparison(xyzw)

    check = get_charmm_positions_array()
    if np.allclose(check, 0.0) and not np.allclose(arr, 0.0):
        raise RuntimeError(
            "sync_charmm_positions: CHARMM coordinates still zero after set_main/set_positions"
        )


def get_charmm_positions_array() -> np.ndarray:
    """Read CHARMM coordinates as ``(N, 3)`` (main set, then positions, then comparison)."""
    import pycharmm.coor as coor

    for getter in (coor.get_main, coor.get_positions, coor.get_comparison):
        df = getter()
        pos = df[["x", "y", "z"]].to_numpy(dtype=float)
        if pos.shape[0] and not np.allclose(pos, 0.0):
            return pos
    n = coor.get_natom()
    return np.zeros((n, 3), dtype=float)


def resolve_export_positions(
    *,
    pyCModel: Any = None,
    reference_positions: Optional[np.ndarray] = None,
) -> Optional[np.ndarray]:
    """Best-effort positions for file export after minimization."""
    if pyCModel is not None:
        calc = pyCModel.get_pycharmm_calculator()
        cached = getattr(calc, "last_full_positions", None)
        if cached is not None:
            cached = np.asarray(cached, dtype=float)
            if cached.size and not np.allclose(cached, 0.0):
                return cached

    charmm = get_charmm_positions_array()
    if charmm.size and not np.allclose(charmm, 0.0):
        return charmm

    if reference_positions is not None:
        ref = np.asarray(reference_positions, dtype=float)
        if ref.size and not np.allclose(ref, 0.0):
            return ref
    return None


@dataclass
class MlpotContext:
    """Active MLpot registration (call :meth:`unset` when finished)."""

    mlpot: Any
    pyCModel: Any
    params: Any
    model: Any
    ml_selection: Any = None
    block_tag: str = "all"
    ml_Z: np.ndarray | None = None
    use_pbc: bool = False
    cubic_box_side_A: float | None = None
    ml_charge: float = 0.0
    ml_fq: bool = True

    def unset(self) -> None:
        self.mlpot.unset_mlpot()
        from mmml.interfaces.pycharmmInterface.mlpot.block_terms import (
            apply_charmm_mm_block,
            clear_mlpot_energy_block,
        )

        if self.ml_selection is not None:
            clear_mlpot_energy_block(self.ml_selection, block_tag=self.block_tag)
        apply_charmm_mm_block()

    def reregister_mlpot(self) -> None:
        """Re-attach MLpot + ML BLOCK after temporary MM-only work."""
        from mmml.interfaces.pycharmmInterface.mlpot.block_terms import (
            apply_mlpot_energy_block,
        )

        if self.ml_selection is None or self.ml_Z is None:
            raise RuntimeError("MlpotContext missing ml_selection or ml_Z for reregister")
        self.block_tag = apply_mlpot_energy_block(self.ml_selection)
        reattach = getattr(self.mlpot, "reattach_mlpot", None)
        if callable(reattach):
            # Do not construct a new MLpot(): __init__ rebuilds iblo/inb via update_bnbnd
            # (upinb), which segfaults after long MD. Re-enable the existing callback.
            reattach()
            return

        self._reattach_mlpot_compat()

    def _reattach_mlpot_compat(self) -> None:
        """Compatibility path for PyCHARMM MLpot builds without ``reattach_mlpot``."""
        required = ("energy_func", "ml_indices", "ml_Z", "ml_Natoms")
        missing = [name for name in required if not hasattr(self.mlpot, name)]
        if missing:
            raise RuntimeError(
                "PyCHARMM MLpot cannot be reattached; missing attributes: "
                + ", ".join(missing)
            )

        pycharmm = _import_pycharmm()
        pycharmm.lib.charmm.mlpot_set_func(self.mlpot.energy_func)
        ml_indices = np.asarray(self.mlpot.ml_indices, dtype=int)
        ml_z = np.asarray(self.mlpot.ml_Z, dtype=int)
        n_ml = int(self.mlpot.ml_Natoms)
        mlidx = (ctypes.c_int * n_ml)(*(ml_indices + 1))
        mlidz = (ctypes.c_int * n_ml)(*ml_z)
        nml = (ctypes.c_int * 1)(n_ml)
        pycharmm.lib.charmm.mlpot_set_properties(nml, mlidx, mlidz)
        if hasattr(self.mlpot, "is_set"):
            self.mlpot.is_set = True


def _import_pycharmm():
    import mmml.interfaces.pycharmmInterface.import_pycharmm  # noqa: F401 — CHARMM env
    import pycharmm

    return pycharmm


def select_all_atoms():
    """CHARMM selection of all atoms."""
    return _import_pycharmm().SelectAtoms().all_atoms()


def select_by_seg_id(seg_id: str):
    """CHARMM selection by segment ID (e.g. ``'AMM1'`` for an ML region)."""
    return _import_pycharmm().SelectAtoms(seg_id=seg_id)


def select_by_resid(resid: int | str):
    """CHARMM selection by residue ID (e.g. ``1`` for the first residue)."""
    return _import_pycharmm().SelectAtoms(res_id=str(resid))


def select_by_resids(resids: Sequence[int | str]) -> Any:
    """Union selection over multiple residue IDs (one CGenFF residue = one monomer)."""
    ids = [str(r).strip() for r in resids if str(r).strip()]
    if not ids:
        raise ValueError("select_by_resids: empty resid list")
    sel = select_by_resid(ids[0])
    for rid in ids[1:]:
        sel = sel | select_by_resid(rid)
    return sel


def apply_charmm_verbosity(
    *,
    prnlev: int = 5,
    warnlev: int = 5,
    bomlev: int = -2,
) -> dict[str, int]:
    """Raise CHARMM console output (``PRNLev``, ``WRNLev``, ``BOMBlev``).

    Returns the previous levels as ``{"prnlev", "warnlev", "bomlev"}``.
    Higher ``prnlev`` / ``warnlev`` (up to ~5) print more from the Fortran core.
    """
    import pycharmm.settings as settings

    pycharmm = _import_pycharmm()
    old = {
        "prnlev": int(settings.set_verbosity(int(prnlev))),
        "warnlev": int(settings.set_warn_level(int(warnlev))),
        "bomlev": int(settings.set_bomb_level(int(bomlev))),
    }
    pycharmm.lingo.charmm_script(f"bomlev {int(bomlev)}")
    return old


def write_charmm_psf(path: PathLike) -> Path:
    """Write the current in-memory PSF (connectivity as in CHARMM)."""
    import pycharmm.write as write

    p = Path(path).expanduser().resolve()
    p.parent.mkdir(parents=True, exist_ok=True)
    write.psf_card(str(p))
    return p


def resolve_topology_psf_for_mlpot_reload(
    psf_path: PathLike,
    *,
    tag: str | None = None,
) -> Path:
    """Return a PSF safe for ``read.psf_card`` before MLpot re-registration.

    ``mini_full_mlpot_*.psf`` is written after MLpot and embeds large ML–ML
    exclusion lists; CHARMM then aborts with "Maximum number of nonbond
    exclusions exceeded". Prefer ``cluster_for_vmd_<tag>.psf`` (saved pre-MLpot).
    """
    psf = Path(psf_path).expanduser().resolve()
    if "mini_full_mlpot_" not in psf.name:
        return psf

    tags: list[str] = []
    if tag:
        tags.append(str(tag))
    derived = psf.name.replace("mini_full_mlpot_", "").replace(".psf", "")
    if derived and derived not in tags:
        tags.append(derived)

    for t in tags:
        vmd = psf.parent / f"cluster_for_vmd_{t}.psf"
        if vmd.is_file():
            return vmd.resolve()

    raise FileNotFoundError(
        f"Cannot reload topology from {psf.name}: it contains MLpot nonbond "
        f"exclusions that exceed CHARMM PSF read limits. Provide "
        f"{psf.parent / f'cluster_for_vmd_{derived or tag or '<tag>'}.psf'} "
        f"(from the initial build, pre-MLpot) via --from-psf and keep using the "
        f"mini CRD for coordinates."
    )


def save_cluster_topology_for_vmd(
    out_dir: PathLike,
    positions: np.ndarray,
    *,
    stem: str = "cluster_for_vmd",
    title: str = "cluster",
) -> dict[str, Path]:
    """Save PSF + PDB for VMD (connectivity preserved; MLpot uses BLOCK, not PSF deletes).

    Load in VMD with: ``vmd cluster_for_vmd.psf cluster_for_vmd.pdb`` (or a trajectory).
    """
    import pycharmm.write as write

    sync_charmm_positions(positions)
    out = Path(out_dir).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    psf_path = write_charmm_psf(out / f"{stem}.psf")
    pdb_path = out / f"{stem}.pdb"
    write.coor_pdb(str(pdb_path), title=title)
    return {"psf": psf_path, "pdb": pdb_path.resolve()}


def disable_charmm_domdec() -> None:
    """Turn off domdec once (``domdec dlb off`` would leave domdec on)."""
    from mmml.interfaces.pycharmmInterface.import_pycharmm import disable_charmm_domdec as _disable

    _disable()


def prepare_charmm_vacuum() -> None:
    """Vacuum: domdec off (once), crystal free."""
    from mmml.interfaces.pycharmmInterface.import_pycharmm import (
        crystal_free_charmm,
        disable_charmm_domdec,
    )

    disable_charmm_domdec()
    crystal_free_charmm()


def setup_default_nbonds(*, nbxmod: int = 5) -> None:
    """Vacuum nonbonds (same kwargs as ``md_pbc_suite/ase._run_charmm_minimize``)."""
    from mmml.interfaces.pycharmmInterface.nbonds_config import apply_vacuum_nbonds

    apply_vacuum_nbonds(nbxmod=nbxmod)


def refresh_nbonds_after_mlpot(*, nbxmod: int = 5) -> None:
    """Rebuild nonbond lists after :class:`pycharmm.MLpot` changes exclusions."""
    from mmml.interfaces.pycharmmInterface.nbonds_config import vacuum_nbond_kwargs

    prepare_charmm_vacuum()
    pycharmm = _import_pycharmm()
    pycharmm.nbonds.update_bnbnd()
    pycharmm.UpdateNonBondedScript(**vacuum_nbond_kwargs(nbxmod=nbxmod)).run()


DEFAULT_WORKFLOW_NBXMOD = 5
RECOVERY_NBXMOD = 2
# NBXMod controls VDW/ELEC exclusion lists (CHARMM nbonds):
#   2 = exclude only 1-2 (bonded) pairs — milder exclusions during rescue SD
#   5 = exclude 1-2, 1-3, and special 1-4 (normal production MD)


def apply_recovery_nbonds(ctx: MlpotContext, *, nbxmod: int = RECOVERY_NBXMOD) -> None:
    """Temporary nonbond settings for bonded rescue SD (``NBXMOD 2``, VDW on in BLOCK)."""
    from mmml.interfaces.pycharmmInterface.nbonds_config import (
        pbc_nbond_kwargs,
        vacuum_nbond_kwargs,
    )

    pycharmm = _import_pycharmm()
    pycharmm.nbonds.update_bnbnd()
    if ctx.use_pbc and ctx.cubic_box_side_A is not None:
        cutnb = 18.0
        cutim = cutnb + 4.0
        pycharmm.UpdateNonBondedScript(
            **pbc_nbond_kwargs(nbxmod=nbxmod, cutnb=cutnb, cutim=cutim)
        ).run()
    else:
        # Dynamics may leave imgfrq>0; clear before rescue SD (inbfrq=0 is invalid then).
        pycharmm.nbonds.set_imgfrq(-1)
        pycharmm.UpdateNonBondedScript(**vacuum_nbond_kwargs(nbxmod=nbxmod)).run()


def restore_workflow_nbonds(
    ctx: MlpotContext,
    *,
    nbxmod: int = DEFAULT_WORKFLOW_NBXMOD,
) -> None:
    """No-op after overlap rescue when MLpot is (re-)registered.

    Rescue minimization temporarily uses ``NBXMOD 2`` via
    :func:`apply_recovery_nbonds`. Switching back to production ``NBXMOD`` rebuilds
    exclusion lists through ``update_bnbnd`` / ``UpdateNonBondedScript`` → ``upinb``,
    which segfaults once MLpot has established ML exclusions (even after
    ``unset_mlpot``). Hybrid MLpot MD keeps CHARMM VDW/ELEC off on ML atoms via
    BLOCK, so staying on ``NBXMOD 2`` after rescue is safe.
    """
    del ctx, nbxmod  # API kept for callers; intentionally no CHARMM nbond rebuild.


def refresh_nbonds_after_mlpot_pbc(
    *,
    cubic_box_side_A: float,
    nbxmod: int = 5,
    cutnb: float = 18.0,
    force: bool = False,
) -> None:
    """Rebuild PBC nonbond lists after MLpot registration.

    With ``force=False`` (default), skip when the live CHARMM box already matches
    ``cubic_box_side_A``, or when box lengths are unavailable — rebuilding
    crystal/nbonds with MLpot active mid-workflow can segfault in ``upinb``.
    Pass ``force=True`` once immediately after initial MLpot registration.
    """
    from mmml.interfaces.pycharmmInterface.mlpot.pbc_env import (
        _is_cubic_box_sides,
        _read_charmm_box_sides_A,
        apply_pbc_nbonds,
        prepare_charmm_pbc,
    )

    side = float(cubic_box_side_A)
    if not force:
        try:
            lx, ly, lz = _read_charmm_box_sides_A()
            if _is_cubic_box_sides(lx, ly, lz):
                mean = (lx + ly + lz) / 3.0
                tol = max(1e-3, 1e-4 * side)
                if abs(mean - side) <= tol:
                    return
            if min(lx, ly, lz) <= 0.0:
                return
        except Exception:
            return

    pycharmm = _import_pycharmm()
    from mmml.interfaces.pycharmmInterface.charmm_levels import charmm_relaxed_bomlev

    with charmm_relaxed_bomlev():
        prepare_charmm_pbc(side)
        pycharmm.nbonds.update_bnbnd()
        apply_pbc_nbonds(nbxmod=nbxmod, cutnb=cutnb)


def load_cluster_from_artifacts(
    args: Any,
) -> tuple[np.ndarray, np.ndarray, int, str]:
    """Load PSF + CRD (and optional coordinates from ``--restart-from``)."""
    from mmml.interfaces.pycharmmInterface.mlpot.dynamics import load_minimized_coordinates
    from mmml.interfaces.pycharmmInterface.nbonds_config import read_cgenff_toppar
    from mmml.interfaces.pycharmmInterface.utils import get_Z_from_psf

    psf = getattr(args, "from_psf", None)
    crd = getattr(args, "from_crd", None)
    if (psf is None or crd is None) and getattr(args, "output_dir", None):
        out = Path(args.output_dir).expanduser().resolve()
        tag_guess = getattr(args, "tag", None)
        if not tag_guess and getattr(args, "composition", None):
            from mmml.cli.run.md_pbc_suite.ase import _parse_composition
            from mmml.interfaces.pycharmmInterface.mlpot.cli_common import composition_tag

            comp = _parse_composition(args.composition)
            n_from_comp = sum(c for _, c in comp)
            tag_guess = composition_tag(
                comp,
                str(getattr(args, "residue", "ACO")).upper(),
                n_from_comp,
            )
        if tag_guess:
            psf = psf or out / f"mini_full_mlpot_{tag_guess}.psf"
            crd = crd or out / f"mini_full_mlpot_{tag_guess}.crd"
        elif psf is None and crd is None:
            psf_candidates = sorted(out.glob("mini_full_mlpot_*.psf"))
            if len(psf_candidates) == 1:
                psf = psf_candidates[0]
                crd = out / psf.name.replace(".psf", ".crd")
    if psf is None or crd is None:
        raise ValueError(
            "skip-cluster-build requires --from-psf and --from-crd "
            "(or mini artifacts under --output-dir with --tag)"
        )
    psf_path = Path(psf).expanduser().resolve()
    crd_path = Path(crd).expanduser().resolve()
    if not psf_path.is_file():
        raise FileNotFoundError(f"PSF not found: {psf_path}")
    if not crd_path.is_file():
        raise FileNotFoundError(f"CRD not found: {crd_path}")

    pycharmm = _import_pycharmm()
    import pycharmm.read as read

    from mmml.interfaces.pycharmmInterface.charmm_levels import charmm_relaxed_bomlev

    tag_guess = str(getattr(args, "tag", None) or psf_path.stem.replace("mini_full_mlpot_", ""))
    topology_psf = resolve_topology_psf_for_mlpot_reload(psf_path, tag=tag_guess)
    if topology_psf != psf_path and not getattr(args, "quiet", False):
        print(
            f"Reload topology from {topology_psf.name} "
            f"(not {psf_path.name}; mini PSF embeds ML exclusions)",
            flush=True,
        )

    read_cgenff_toppar()
    with charmm_relaxed_bomlev():
        read.psf_card(str(topology_psf))
        load_minimized_coordinates(crd_path)
    z = np.asarray(get_Z_from_psf(), dtype=int)
    r = get_charmm_positions_array()

    n_mol = int(getattr(args, "n_molecules", 0) or 0)
    if getattr(args, "composition", None):
        from mmml.cli.run.md_pbc_suite.ase import _parse_composition

        n_mol = sum(c for _, c in _parse_composition(args.composition))
    if n_mol <= 0:
        n_mol = max(1, int(getattr(args, "n_molecules", 1) or 1))

    tag = str(getattr(args, "tag", None) or psf_path.stem.replace("mini_full_mlpot_", ""))
    if not getattr(args, "quiet", False):
        print(
            f"Loaded cluster from {topology_psf.name} + {crd_path.name}",
            flush=True,
        )
    return z, r, n_mol, tag


def physnet_ml_atomic_numbers(z: Sequence[int]) -> list[int]:
    """PSF/ASE atomic numbers for MLpot (must match ``setup_calculator`` inputs)."""
    return [int(x) for x in z]


def load_physnet_mlpot_bundle(
    checkpoint: PathLike,
    n_atoms: int,
    ase_atoms: Any,
    *,
    n_monomers: int = 1,
    atoms_per_monomer: Sequence[int] | None = None,
    ml_batch_size: Optional[int] = None,
    ml_gpu_count: Optional[int] = None,
    ml_max_active_dimers: Optional[int] = None,
    cell: float | None = None,
    verbose: bool = False,
    args: Any | None = None,
) -> tuple[Any, Any, Any]:
    """Load PhysNet for MLpot. Multi-monomer clusters use monomer/dimer batches."""
    ckpt = Path(checkpoint).expanduser().resolve()
    z = np.asarray(ase_atoms.get_atomic_numbers(), dtype=int)

    if int(n_monomers) > 1:
        from mmml.interfaces.pycharmmInterface.mlpot.hybrid_mlpot import (
            build_decomposed_mlpot_model,
        )

        if atoms_per_monomer is None:
            if int(n_atoms) % int(n_monomers) != 0:
                raise ValueError(
                    f"atom count {n_atoms} not divisible by n_monomers={n_monomers}"
                )
            per = int(n_atoms) // int(n_monomers)
            atoms_per_monomer = [per] * int(n_monomers)
        pyCModel = build_decomposed_mlpot_model(
            ckpt,
            z,
            atoms_per_monomer,
            int(n_monomers),
            ml_batch_size=ml_batch_size,
            ml_gpu_count=ml_gpu_count,
            ml_max_active_dimers=ml_max_active_dimers,
            cell=float(cell) if cell is not None else False,
            verbose=verbose,
            args=args,
        )
        return None, None, pyCModel

    from mmml.cli.base import load_physnet_params_and_ef_model
    from mmml.models.physnetjax.physnetjax.calc.helper_mlp import get_pyc

    params, model = load_physnet_params_and_ef_model(ckpt, natoms=n_atoms)
    model.natoms = n_atoms
    pyCModel = get_pyc(params, model, ase_atoms)
    return params, model, pyCModel


def register_mlpot(
    pyCModel: Any,
    ml_Z: Sequence[int],
    ml_selection: Any,
    *,
    ml_charge: float = 0,
    ml_fq: bool = True,
    mlmm_ctonnb: Optional[float] = None,
    mlmm_ctofnb: Optional[float] = None,
    preserve_psf_internals: bool = True,
    use_pbc: bool = False,
    **kwargs: Any,
) -> MlpotContext:
    """Register ``pycharmm.MLpot`` and return a context manager-like handle."""
    from mmml.interfaces.pycharmmInterface.mlpot.block_terms import apply_mlpot_energy_block
    from mmml.interfaces.pycharmmInterface.mlpot.mlpot_limits import validate_mlpot_system_size

    pycharmm = _import_pycharmm()
    z_ml = physnet_ml_atomic_numbers(ml_Z)
    n_ml = len(ml_selection.get_atom_indexes())
    validate_mlpot_system_size(n_ml)
    from mmml.interfaces.pycharmmInterface.charmm_levels import charmm_relaxed_bomlev

    with charmm_relaxed_bomlev():
        block_tag = apply_mlpot_energy_block(ml_selection)
        mlpot = pycharmm.MLpot(
            ml_model=pyCModel,
            ml_Z=z_ml,
            ml_selection=ml_selection,
            ml_charge=ml_charge,
            ml_fq=ml_fq,
            mlmm_ctonnb=mlmm_ctonnb,
            mlmm_ctofnb=mlmm_ctofnb,
            preserve_psf_internals=preserve_psf_internals,
            **kwargs,
        )
        if not use_pbc:
            # MLpot.__init__ already set iblo/inb and ran update_bnbnd (upinb).
            # Re-running prepare_charmm_vacuum + update_bnbnd here segfaults in upinb
            # for large clusters (e.g. DCM:90) after JAX GPU warmup.
            from mmml.interfaces.pycharmmInterface.nbonds_config import (
                vacuum_nbond_kwargs,
            )

            pycharmm.UpdateNonBondedScript(**vacuum_nbond_kwargs(nbxmod=5)).run()
    ml_z = np.asarray(ml_Z, dtype=int)
    return MlpotContext(
        mlpot=mlpot,
        pyCModel=pyCModel,
        params=None,
        model=None,
        ml_selection=ml_selection,
        block_tag=block_tag,
        ml_Z=ml_z,
        use_pbc=bool(use_pbc),
        cubic_box_side_A=None,
        ml_charge=float(ml_charge),
        ml_fq=bool(ml_fq),
    )
