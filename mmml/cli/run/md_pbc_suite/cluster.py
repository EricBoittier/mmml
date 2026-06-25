"""CHARMM PSF-ordered cluster construction for md_pbc_suite."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

import mmml.interfaces.pycharmmInterface.import_pycharmm as pyci
from mmml.interfaces.pycharmmInterface.import_pycharmm import (
    coor,
    pycharmm,
    reset_block,
)
from mmml.interfaces.pycharmmInterface.utils import get_Z_from_psf

import pycharmm.generate as gen
import pycharmm.ic as ic
import pycharmm.param as param
import pycharmm.psf as psf
import pycharmm.read as read
import pycharmm.settings as settings

pyci.read = read
pyci.settings = settings
pyci.psf = psf


def _load_template_pdb_coords(template_pdb: Path) -> dict[str, np.ndarray]:
    """Load atom-name keyed coordinates from a PDB template."""
    coords: dict[str, np.ndarray] = {}
    for line in template_pdb.read_text(encoding="utf-8").splitlines():
        if not (line.startswith("ATOM") or line.startswith("HETATM")):
            continue
        atom_name = line[12:16].strip()
        x = float(line[30:38])
        y = float(line[38:46])
        z = float(line[46:54])
        coords[atom_name] = np.array([x, y, z], dtype=float)
    if not coords:
        raise ValueError(f"No ATOM/HETATM coordinates found in template PDB: {template_pdb}")
    return coords


def _build_psf_ordered_cluster(
    residue: str,
    n_molecules: int,
    spacing: float,
    template_pdb: Path | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    residue = residue.upper()
    if template_pdb is None:
        template_pdb = _default_template_pdb_for_residue(residue)
    if template_pdb is None:
        from mmml.cli.run.md_pbc_suite.ase import _build_cluster_from_composition

        z, shifted, _, _ = _build_cluster_from_composition(
            composition=[(residue, n_molecules)],
            spacing=spacing,
        )
        span = np.ptp(shifted, axis=0)
        if float(span[1]) < 0.3 or float(span[2]) < 0.3:
            raise RuntimeError(
                f"Cluster geometry not 3D (spans Å x={span[0]:.3f} y={span[1]:.3f} z={span[2]:.3f})"
            )
        return z, shifted

    sequence = " ".join([residue] * n_molecules)

    from mmml.interfaces.pycharmmInterface.mlpot.setup import prepare_charmm_vacuum
    from mmml.interfaces.pycharmmInterface.nbonds_config import read_cgenff_toppar

    pycharmm.lingo.charmm_script("DELETE ATOM SELE ALL END")
    reset_block()
    prepare_charmm_vacuum()
    read_cgenff_toppar(enable_drude=False)

    read.sequence_string(sequence)
    gen.new_segment(seg_name="CLST", setup_ic=True)
    ic.prm_fill(replace_all=True)
    ic.build()

    pos_df = coor.get_positions()
    positions = pos_df[["x", "y", "z"]].to_numpy(dtype=float)
    n_atoms = positions.shape[0]
    if n_atoms % n_molecules != 0:
        raise RuntimeError(
            f"Atom count {n_atoms} not divisible by n_molecules={n_molecules}; "
            "cannot form equal same-residue chunks."
        )
    atoms_per_res = n_atoms // n_molecules

    n_side = int(np.ceil(np.sqrt(n_molecules)))
    shifted = positions.copy()
    atom_names = np.asarray(psf.get_atype())
    if len(atom_names) != n_atoms:
        raise RuntimeError(f"PSF atom-name count mismatch: {len(atom_names)} vs positions {n_atoms}")

    if template_pdb is not None:
        tmpl = _load_template_pdb_coords(template_pdb)
        for i in range(n_molecules):
            start = i * atoms_per_res
            end = (i + 1) * atoms_per_res
            local_names = atom_names[start:end]
            local_coords = []
            for nm in local_names:
                if nm not in tmpl:
                    raise KeyError(
                        f"Template {template_pdb} missing atom '{nm}' (PSF order). "
                        f"Have: {sorted(tmpl.keys())}"
                    )
                local_coords.append(tmpl[nm])
            shifted[start:end] = np.asarray(local_coords, dtype=float)

    for i in range(n_molecules):
        start = i * atoms_per_res
        end = (i + 1) * atoms_per_res
        com = shifted[start:end].mean(axis=0)
        shift = np.array([(i % n_side) * spacing, (i // n_side) * spacing, 0.0], dtype=float)
        shifted[start:end] = shifted[start:end] - com + shift

    coor.set_positions(pd.DataFrame(shifted, columns=["x", "y", "z"]))
    try:
        from mmml.interfaces.pycharmmInterface.mlpot.setup import sync_charmm_positions

        sync_charmm_positions(shifted)
    except Exception:
        pass

    span = np.ptp(shifted, axis=0)
    if float(span[1]) < 0.3 or float(span[2]) < 0.3:
        raise RuntimeError(
            f"Cluster geometry not 3D (spans Å x={span[0]:.3f} y={span[1]:.3f} z={span[2]:.3f})"
        )

    z = np.asarray(get_Z_from_psf(), dtype=int)
    return z, shifted


def _default_template_pdb_for_residue(residue: str) -> Path | None:
    """Bundled 3D monomer templates keyed by CGenFF residue name."""
    residue = residue.upper()
    from mmml.paths import default_aco_template_pdb, default_meoh_template_pdb

    if residue == "ACO":
        path = default_aco_template_pdb()
        return path if path.is_file() else None
    if residue == "MEOH":
        path = default_meoh_template_pdb()
        return path if path.is_file() else None
    return None


def _monomer_geometry_is_3d(coords: np.ndarray, *, min_axis_span: float = 0.3) -> bool:
    span = np.max(coords, axis=0) - np.min(coords, axis=0)
    return float(span[1]) >= min_axis_span and float(span[2]) >= min_axis_span


def ensure_monomer_3d_coords(
    coords: np.ndarray,
    *,
    amplitude: float = 0.8,
) -> np.ndarray:
    """Break collinear/planar monomer IC coordinates with a deterministic 3D spread."""
    out = np.asarray(coords, dtype=np.float64).copy()
    if out.ndim != 2 or out.shape[1] != 3:
        raise ValueError(f"coords must be (N, 3), got {out.shape}")
    n = int(out.shape[0])
    if n < 2:
        return out
    com = out.mean(axis=0)
    out -= com
    span = np.ptp(out, axis=0)
    if float(span[1]) < 0.3:
        out[min(1, n - 1), 1] += float(amplitude)
    if float(span[2]) < 0.3:
        out[min(2, n - 1), 2] += float(amplitude)
    out += com
    return out


def relax_monomer_geometry_for_cluster(
    residue: str,
) -> tuple[np.ndarray, list[str], np.ndarray]:
    """Build a 3D monomer from IC tables without CHARMM minimization.

    Cluster assembly only needs a non-flat placement geometry. SD/ABNR minimization
    can segfault in ``ebondfs`` on some cluster CHARMM builds (notably DCM in notebooks).
    For production geometries use ``build_cluster_from_reference_npz`` instead.
    """
    from mmml.cli.run.md_pbc_suite.ase import _read_cgenff_toppar, _reset_pycharmm_system
    from mmml.interfaces.pycharmmInterface.mlpot.setup import prepare_charmm_vacuum

    residue = residue.upper()
    _reset_pycharmm_system()
    prepare_charmm_vacuum()
    _read_cgenff_toppar()
    read.sequence_string(residue)
    gen.new_segment(seg_name="TMP", setup_ic=True)
    ic.prm_fill(replace_all=True)
    ic.build()

    atom_names = [str(x) for x in np.asarray(psf.get_atype(), dtype=str)]
    coords = coor.get_positions()[["x", "y", "z"]].to_numpy(dtype=np.float64)
    if int(coords.shape[0]) != len(atom_names):
        raise RuntimeError(
            f"PSF atom-name count mismatch while generating {residue}: "
            f"{len(atom_names)} vs positions {coords.shape[0]}"
        )

    if not _monomer_geometry_is_3d(coords):
        coords = ensure_monomer_3d_coords(coords)

    if not _monomer_geometry_is_3d(coords):
        span = np.ptp(coords, axis=0)
        raise RuntimeError(
            f"Monomer {residue} IC geometry is not 3D (spans Å x={span[0]:.2f} "
            f"y={span[1]:.2f} z={span[2]:.2f}). "
            "Pass reference_npz= to build_ase_cluster with a PSF-order reference NPZ."
        )

    z = np.asarray(get_Z_from_psf(), dtype=int)
    if int(z.shape[0]) != len(atom_names):
        raise RuntimeError(
            f"PSF Z count mismatch for {residue}: {z.shape[0]} vs {len(atom_names)} atom names"
        )
    return coords, atom_names, z


def build_cluster_from_reference_npz(
    residue: str,
    n_molecules: int,
    reference_npz: str | Path,
    *,
    frame: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Build CHARMM PSF for ``residue``×``n_molecules`` and load positions from reference NPZ."""
    from mmml.cli.run.md_pbc_suite.ase import _build_cluster_psf_topology_only
    from mmml.interfaces.pycharmmInterface.cluster_geometry import reference_frame_geometry
    from mmml.interfaces.pycharmmInterface.mlpot.setup import sync_charmm_positions

    ref_z, ref_r = reference_frame_geometry(reference_npz, frame=frame)
    n_atoms = int(len(ref_z))
    if n_atoms % int(n_molecules) != 0:
        raise ValueError(
            f"Reference frame has {n_atoms} atoms, not divisible by n_molecules={n_molecules}"
        )
    atoms_per = n_atoms // int(n_molecules)
    residue = residue.upper()
    composition = [(residue, int(n_molecules))]
    atoms_per_list = [atoms_per] * int(n_molecules)
    residue_labels = [residue] * int(n_molecules)

    z = _build_cluster_psf_topology_only(
        composition,
        expected_atoms=n_atoms,
        atoms_per_list=atoms_per_list,
        residue_labels=residue_labels,
    )
    if not np.array_equal(z, ref_z):
        raise ValueError(
            f"Reference atomic numbers {ref_z.tolist()} != PSF order {z.tolist()}. "
            "Use a PSF-order reference NPZ (e.g. *_psf_order.npz) matching the composition."
        )
    coor.set_positions(pd.DataFrame(ref_r, columns=["x", "y", "z"]))
    sync_charmm_positions(ref_r)
    return z, ref_r


def build_minimized_monomer_for_packmol(
    residue: str,
    *,
    nstep_sd: int = 50,
    nstep_abnr: int = 100,
    tolenr: float = 1e-3,
    tolgrd: float = 1e-3,
    verbose: bool = True,
) -> tuple[np.ndarray, list[str], np.ndarray]:
    """Build and CHARMM-minimize an isolated monomer before Packmol (MM only, no MLpot)."""
    from mmml.cli.run.md_pbc_suite.ase import _generate_residue_with_make_res_recipe
    from mmml.interfaces.pycharmmInterface.mlpot.dynamics import (
        CharmmMmMinimizeConfig,
        minimize_charmm_mm_only,
    )
    from mmml.interfaces.pycharmmInterface.mlpot.setup import (
        get_charmm_positions_array,
        sync_charmm_positions,
    )

    residue = residue.upper()
    coords, atom_names, z = _generate_residue_with_make_res_recipe(residue)

    from mmml.interfaces.pycharmmInterface.mlpot.setup import prepare_charmm_vacuum
    from mmml.interfaces.pycharmmInterface.nbonds_config import read_cgenff_toppar

    pycharmm.lingo.charmm_script("DELETE ATOM SELE ALL END")
    reset_block()
    prepare_charmm_vacuum()
    read_cgenff_toppar(enable_drude=False)
    read.sequence_string(residue)
    gen.new_segment(seg_name="CLST", setup_ic=True)
    ic.prm_fill(replace_all=True)
    ic.build()

    psf_names = [str(x) for x in np.asarray(psf.get_atype(), dtype=str)]
    if psf_names != atom_names:
        raise RuntimeError(
            f"Atom order mismatch for {residue}: PSF {psf_names} vs relaxed {atom_names}"
        )
    sync_charmm_positions(coords)

    if verbose and (nstep_sd > 0 or nstep_abnr > 0):
        print(
            f"Packmol monomer {residue}: CHARMM MM minimize (SD={nstep_sd}, ABNR={nstep_abnr})",
            flush=True,
        )
    if nstep_sd > 0 or nstep_abnr > 0:
        minimize_charmm_mm_only(
            CharmmMmMinimizeConfig(
                nstep_sd=int(nstep_sd),
                nstep_abnr=int(nstep_abnr),
                nprint=10,
                tolenr=float(tolenr),
                tolgrd=float(tolgrd),
                verbose=verbose,
                show_energy=False,
            )
        )
        coords = get_charmm_positions_array()

    if not _monomer_geometry_is_3d(coords):
        span = np.ptp(coords, axis=0)
        raise RuntimeError(
            f"Monomer {residue} not 3D after minimization "
            f"(spans Å x={span[0]:.2f} y={span[1]:.2f} z={span[2]:.2f})"
        )
    z = np.asarray(get_Z_from_psf(), dtype=int)
    if int(z.shape[0]) != len(atom_names):
        raise RuntimeError(
            f"Atom count mismatch for {residue}: PSF {z.shape[0]} vs {len(atom_names)} names"
        )
    return coords, atom_names, z


def build_packmol_composition_cluster(
    *,
    composition: list[tuple[str, int]],
    placement: str = "cube",
    center: tuple[float, float, float],
    cube_side: float | None = None,
    radius: float | None = None,
    tolerance: float = 2.0,
    seed: int | None = None,
    charmm_sd_steps: int = 50,
    charmm_abnr_steps: int = 100,
    charmm_tolenr: float = 1e-3,
    charmm_tolgrd: float = 1e-3,
    scratch_dir: str | Path | None = None,
    verbose: bool = True,
    reuse_packmol_cache: bool = True,
    packmol_cache_dir: str | Path | None = None,
    force_rebuild_packmol_cache: bool = False,
) -> tuple[np.ndarray, np.ndarray, list[int], list[str]]:
    """CHARMM-minimize monomers, Packmol cube/sphere pack, cluster PSF, then cluster MM relax."""
    from mmml.cli.run.md_pbc_suite.ase import (
        _build_cluster_psf_from_composition,
        _load_packmol_sphere_positions,
    )
    from mmml.interfaces.pycharmmInterface import packmol_cache, packmol_placement
    from mmml.interfaces.pycharmmInterface.mlpot.dynamics import (
        CharmmMmMinimizeConfig,
        minimize_charmm_mm_only,
    )

    scratch_root = Path(scratch_dir) if scratch_dir is not None else Path("pdb/packmol_cluster")
    cache_root = packmol_cache.packmol_cache_root(
        output_dir=scratch_root.parent if scratch_dir is not None else None,
        override=packmol_cache_dir,
    )
    if reuse_packmol_cache and not force_rebuild_packmol_cache:
        cached = packmol_cache.try_load_packmol_cluster_cache(
            composition=composition,
            placement=str(placement),
            center=center,
            cube_side=cube_side,
            radius=radius,
            tolerance=float(tolerance),
            seed=seed,
            charmm_sd_steps=int(charmm_sd_steps),
            charmm_abnr_steps=int(charmm_abnr_steps),
            charmm_tolenr=float(charmm_tolenr),
            charmm_tolgrd=float(charmm_tolgrd),
            cache_root=cache_root,
        )
        if cached is not None:
            if verbose:
                key = cached["manifest"].get("cache_key", "?")
                print(
                    f"[cluster] Packmol cache hit ({key}): skip monomer/Packmol/MM build",
                    flush=True,
                )
            z = cached["z"]
            shifted = cached["positions"]
            atoms_per_list = cached["atoms_per_list"]
            ordered_residue_names = cached["residue_names"]
            residue_geometries = cached.get("residue_geometries")
            if residue_geometries is None:
                residue_geometries = {}
                for residue, _count in composition:
                    key = residue.upper()
                    if key not in residue_geometries:
                        residue_geometries[key] = build_minimized_monomer_for_packmol(
                            key,
                            nstep_sd=int(charmm_sd_steps),
                            nstep_abnr=int(charmm_abnr_steps),
                            tolenr=float(charmm_tolenr),
                            tolgrd=float(charmm_tolgrd),
                            verbose=False,
                        )
            _psf_z, atom_names, _, _ = _build_cluster_psf_from_composition(
                composition,
                residue_geometries=residue_geometries,
            )
            if len(_psf_z) != len(z) or not np.all(_psf_z == z):
                raise RuntimeError(
                    "Packmol cache Z does not match rebuilt PSF; "
                    "use --rebuild-packmol or delete the cache entry"
                )
            coor.set_positions(
                pd.DataFrame(shifted, columns=["x", "y", "z"])
            )
            span = np.ptp(shifted, axis=0)
            print(
                f"Packmol cluster (cached): {len(atom_names)} atoms, "
                f"span Å x={span[0]:.1f} y={span[1]:.1f} z={span[2]:.1f}",
                flush=True,
            )
            return z, shifted, atoms_per_list, ordered_residue_names

    if verbose:
        print(
            "[cluster] 1/4 CHARMM MM minimize isolated monomer(s) (before Packmol)",
            flush=True,
        )

    residue_geometries: dict[str, tuple[np.ndarray, list[str], np.ndarray]] = {}
    for residue, _count in composition:
        key = residue.upper()
        if key not in residue_geometries:
            residue_geometries[key] = build_minimized_monomer_for_packmol(
                key,
                nstep_sd=int(charmm_sd_steps),
                nstep_abnr=int(charmm_abnr_steps),
                tolenr=float(charmm_tolenr),
                tolgrd=float(charmm_tolgrd),
                verbose=verbose,
            )

    if verbose:
        print("[cluster] 2/4 Write monomer PDBs for Packmol", flush=True)

    packmol_dir = scratch_root / "monomers"
    packmol_blocks: list[tuple[Path, int]] = []
    for residue, count in composition:
        key = residue.upper()
        coords, atom_names, monomer_z = residue_geometries[key]
        pdb_path = packmol_dir / f"{key.lower()}.pdb"
        packmol_placement.write_monomer_pdb_for_packmol(
            pdb_path,
            coords,
            monomer_z,
            atom_names=atom_names,
            resname=key,
        )
        packmol_blocks.append((pdb_path, int(count)))

    output_pdb = scratch_root / "init-packmol-sphere.pdb"
    if output_pdb.exists():
        output_pdb.unlink()

    if verbose:
        packed_counts = ", ".join(
            f"{path.stem.upper()}:{count}" for path, count in packmol_blocks
        )
        if placement == "sphere":
            print(
                f"[cluster] 3/4 Packmol sphere placement ({packed_counts})",
                flush=True,
            )
        else:
            print(
                f"[cluster] 3/4 Packmol cube placement ({packed_counts}, "
                f"side={float(cube_side):.1f} Å)",
                flush=True,
            )

    inp_name = "packmol_sphere.inp" if placement == "sphere" else "packmol_cube.inp"
    if placement == "sphere":
        packmol_placement.run_packmol_sphere_mixed(
            packmol_blocks,
            center=center,
            radius=float(radius),
            output_pdb=output_pdb,
            input_path=scratch_root / inp_name,
            tolerance=float(tolerance),
            seed=seed,
        )
    else:
        packmol_placement.run_packmol_cube_mixed(
            packmol_blocks,
            center=center,
            cube_side=float(cube_side),
            output_pdb=output_pdb,
            input_path=scratch_root / inp_name,
            tolerance=float(tolerance),
            seed=seed,
        )

    if verbose:
        print("[cluster] 4/4 Build cluster PSF and CHARMM MM relax", flush=True)

    z, atom_names, atoms_per_list, ordered_residue_names = _build_cluster_psf_from_composition(
        composition,
        residue_geometries=residue_geometries,
    )
    shifted = _load_packmol_sphere_positions(
        output_pdb, atoms_per_list, psf_atom_names=atom_names
    )
    coor.set_positions(pd.DataFrame(shifted, columns=["x", "y", "z"]))

    if int(charmm_sd_steps) > 0 or int(charmm_abnr_steps) > 0:
        if verbose:
            print(
                f"Packmol cluster: CHARMM MM minimize "
                f"SD={charmm_sd_steps} ABNR={charmm_abnr_steps}",
                flush=True,
            )
        minimize_charmm_mm_only(
            CharmmMmMinimizeConfig(
                nstep_sd=int(charmm_sd_steps),
                nstep_abnr=int(charmm_abnr_steps),
                tolenr=float(charmm_tolenr),
                tolgrd=float(charmm_tolgrd),
                verbose=verbose,
            )
        )
        shifted = coor.get_positions()[["x", "y", "z"]].to_numpy(dtype=float)
        if verbose:
            from mmml.interfaces.pycharmmInterface.mlpot.cli_common import charmm_grms

            pycharmm.lingo.charmm_script("ENER")
            print(
                f"Packmol cluster post-MM GRMS: {charmm_grms():.4f} kcal/mol/Å",
                flush=True,
            )

    span = np.ptp(shifted, axis=0)
    print(
        f"Packmol cluster: {len(atom_names)} atoms, "
        f"span Å x={span[0]:.1f} y={span[1]:.1f} z={span[2]:.1f}",
        flush=True,
    )
    if reuse_packmol_cache:
        cache_key = packmol_cache.packmol_cache_key(
            composition=composition,
            placement=str(placement),
            center=center,
            cube_side=cube_side,
            radius=radius,
            tolerance=float(tolerance),
            seed=seed,
            charmm_sd_steps=int(charmm_sd_steps),
            charmm_abnr_steps=int(charmm_abnr_steps),
            charmm_tolenr=float(charmm_tolenr),
            charmm_tolgrd=float(charmm_tolgrd),
        )
        entry = cache_root / cache_key
        manifest = {
            "version": packmol_cache.CACHE_VERSION,
            "cache_key": cache_key,
            "composition": [[r, n] for r, n in composition],
            "placement": str(placement),
            "center": list(center),
            "cube_side": None if cube_side is None else float(cube_side),
            "radius": None if radius is None else float(radius),
            "tolerance": float(tolerance),
            "seed": seed,
        }
        packmol_cache.save_packmol_cluster_cache(
            entry,
            manifest=manifest,
            z=z,
            positions=shifted,
            atoms_per_list=atoms_per_list,
            residue_names=ordered_residue_names,
            packmol_pdb=output_pdb,
            residue_geometries=residue_geometries,
        )
        if verbose:
            print(f"[cluster] Packmol cache saved: {entry}", flush=True)
    return z, shifted, atoms_per_list, ordered_residue_names
