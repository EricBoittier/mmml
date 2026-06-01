"""Register PhysNet (or other) models with ``pycharmm.MLpot``."""

from __future__ import annotations

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

    def unset(self) -> None:
        self.mlpot.unset_mlpot()
        from mmml.interfaces.pycharmmInterface.mlpot.block_terms import (
            apply_charmm_mm_block,
            clear_mlpot_energy_block,
        )

        if self.ml_selection is not None:
            clear_mlpot_energy_block(self.ml_selection, block_tag=self.block_tag)
        apply_charmm_mm_block()


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
    bomlev: int = 0,
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
    verbose: bool = False,
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
            verbose=verbose,
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
    **kwargs: Any,
) -> MlpotContext:
    """Register ``pycharmm.MLpot`` and return a context manager-like handle."""
    from mmml.interfaces.pycharmmInterface.mlpot.block_terms import apply_mlpot_energy_block
    from mmml.interfaces.pycharmmInterface.mlpot.mlpot_limits import validate_mlpot_system_size

    pycharmm = _import_pycharmm()
    z_ml = physnet_ml_atomic_numbers(ml_Z)
    n_ml = len(ml_selection.get_atom_indexes())
    validate_mlpot_system_size(n_ml)
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
    refresh_nbonds_after_mlpot()
    return MlpotContext(
        mlpot=mlpot,
        pyCModel=pyCModel,
        params=None,
        model=None,
        ml_selection=ml_selection,
        block_tag=block_tag,
    )
