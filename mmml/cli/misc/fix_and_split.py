#!/usr/bin/env python3
"""
CLI tool to fix units and create train/valid/test splits from molecular NPZ data.

Supports NPZ files with or without ESP grid data.

Usage:
    # PySCF / atomic-units NPZ → ASE-style training splits (default)
    mmml fix-and-split \\
        --efd energies_forces_dipoles.npz \\
        --grid grids_esp.npz \\
        --output-dir ./training_data_fixed

    # NPZ already in training units (eV, eV/Å, e·Å, Å): split only, no conversion
    mmml fix-and-split \\
        --efd data_ev.npz \\
        --output-dir ./splits \\
        --coords-in angstrom --coords-out same \\
        --energy-in ev --energy-out same \\
        --force-in ev-angstrom --force-out same \\
        --dipole-in e-angstrom --dipole-out same \\
        --grid-coords-in angstrom --grid-coords-out same

    # Explicit per-field control (see --help for all *-in / *-out flags)
    mmml fix-and-split --efd data.npz -o out \\
        --energy-in hartree --energy-out ev \\
        --force-in hartree-bohr --force-out ev-angstrom

Default conversions (when *-out is not ``same``):
- R: auto-detect Bohr vs Å → Å (use ``--coords-in`` to override auto)
- E: Hartree → eV
- F: Hartree/Bohr → eV/Å (optional ``--flip-forces`` for ∂E/∂R gradients)
- Dxyz: Debye → e·Å
- ESP grid: Bohr or cube indices → Å (``--grid-coords-in auto``)

Writes ``units_manifest.json`` documenting input/output units for each array.
"""

import json
import sys
import argparse
from pathlib import Path
import numpy as np
from scipy.spatial.distance import cdist
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Literal, Optional, Tuple, Union

# Add parent directory to path
repo_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(repo_root.resolve()))

ATOMIC_REF_PATH = Path(__file__).parent.parent.parent / "data" / "qcml" / "atomic_reference_energies.json"


def create_splits(n_samples: int, train_frac=0.8, valid_frac=0.1, test_frac=0.1, seed=42):
    """Create train/valid/test split indices."""
    assert abs(train_frac + valid_frac + test_frac - 1.0) < 1e-6, "Fractions must sum to 1"
    
    rng = np.random.default_rng(seed)
    indices = np.arange(n_samples)
    rng.shuffle(indices)
    
    n_train = int(n_samples * train_frac)
    n_valid = int(n_samples * valid_frac)
    
    return {
        'train': indices[:n_train],
        'valid': indices[n_train:n_train + n_valid],
        'test': indices[n_train + n_valid:]
    }


def convert_energy_hartree_to_ev(E_hartree: np.ndarray) -> np.ndarray:
    """Convert energies from Hartree to eV."""
    from mmml.data.units import HARTREE_TO_EV
    return E_hartree * HARTREE_TO_EV


def convert_energy_ev_to_hartree(E_ev: np.ndarray) -> np.ndarray:
    """Convert energies from eV to Hartree."""
    from mmml.data.units import EV_TO_HARTREE
    return E_ev * EV_TO_HARTREE


def convert_forces_hartree_bohr_to_ev_angstrom(F_hartree_bohr: np.ndarray) -> np.ndarray:
    """Convert forces from Hartree/Bohr to eV/Angstrom."""
    from mmml.data.units import HARTREE_BOHR_TO_EV_ANGSTROM
    return F_hartree_bohr * HARTREE_BOHR_TO_EV_ANGSTROM


def convert_forces_ev_angstrom_to_hartree_bohr(F_ev_ang: np.ndarray) -> np.ndarray:
    """Convert forces from eV/Angstrom to Hartree/Bohr."""
    from mmml.data.units import HARTREE_BOHR_TO_EV_ANGSTROM
    return F_ev_ang / HARTREE_BOHR_TO_EV_ANGSTROM


def convert_dipole_debye_to_eA(D_debye: np.ndarray) -> np.ndarray:
    """Convert dipole moments from Debye to e·Å."""
    from mmml.data.units import DEBYE_TO_EANGSTROM
    return D_debye * DEBYE_TO_EANGSTROM


def convert_dipole_eA_to_debye(D_eA: np.ndarray) -> np.ndarray:
    """Convert dipole moments from e·Å to Debye."""
    from mmml.data.units import EANGSTROM_TO_DEBYE
    return D_eA * EANGSTROM_TO_DEBYE


CoordsIn = Literal["auto", "bohr", "angstrom"]
CoordsOut = Literal["angstrom", "bohr", "same"]
EnergyIn = Literal["hartree", "ev"]
EnergyOut = Literal["ev", "hartree", "same"]
ForceIn = Literal["hartree_bohr", "ev_angstrom"]
ForceOut = Literal["ev_angstrom", "hartree_bohr", "same"]
DipoleIn = Literal["debye", "e_angstrom"]
DipoleOut = Literal["e_angstrom", "debye", "same"]
GridCoordsIn = Literal["auto", "bohr", "angstrom", "index"]
GridCoordsOut = Literal["angstrom", "bohr", "same"]


@dataclass
class UnitsManifest:
    """Recorded in units_manifest.json for downstream loaders."""

    coords_in: str
    coords_out: str
    coords_detected: Optional[str]
    energy_in: str
    energy_out: str
    force_in: str
    force_out: str
    dipole_in: Optional[str]
    dipole_out: Optional[str]
    grid_coords_in: Optional[str]
    grid_coords_out: Optional[str]
    esp_values: str
    flip_forces: bool
    preserve_units: bool
    notes: List[str]


def _mean_shortest_interatomic_distance(R: np.ndarray, max_samples: int = 100) -> Optional[float]:
    """Mean shortest interatomic distance over up to max_samples structures."""
    min_dists: List[float] = []
    for i in range(min(max_samples, len(R))):
        r = R[i]
        valid = np.any(r != 0, axis=1)
        vpos = r[valid]
        if len(vpos) < 2:
            continue
        d = vpos[:, np.newaxis, :] - vpos[np.newaxis, :, :]
        norms = np.linalg.norm(d, axis=2)
        norms[np.triu_indices_from(norms, k=0)] = np.inf
        min_dists.append(float(norms.min()))
    if not min_dists:
        return None
    return float(np.mean(min_dists))


def detect_coords_unit(R: np.ndarray) -> str:
    """
    Infer whether coordinates are in Angstrom or Bohr from bond lengths.

    Returns ``angstrom`` or ``bohr``.
    """
    d_mean = _mean_shortest_interatomic_distance(R)
    if d_mean is None:
        return "angstrom"
    if 0.8 < d_mean < 2.5:
        return "angstrom"
    if 1.8 < d_mean < 2.9:
        return "bohr"
    return "angstrom"


def convert_coords_array(
    R: np.ndarray,
    coords_in: CoordsIn,
    coords_out: CoordsOut,
    *,
    verbose: bool = False,
) -> Tuple[np.ndarray, str, Optional[str]]:
    """
    Convert atomic coordinates between unit conventions.

    Returns (R_out, effective_input_unit, detected_unit_if_auto).
    """
    from mmml.data.units import ANGSTROM_TO_BOHR, BOHR_TO_ANGSTROM

    detected: Optional[str] = None
    if coords_in == "auto":
        detected = detect_coords_unit(R)
        effective_in = detected
        if verbose:
            print(f"  Auto-detected coordinates: {effective_in}")
    else:
        effective_in = coords_in

    if coords_out == "same":
        return np.asarray(R, dtype=np.float64).copy(), effective_in, detected

    if effective_in == coords_out:
        return np.asarray(R, dtype=np.float64).copy(), effective_in, detected

    R_out = np.asarray(R, dtype=np.float64).copy()
    if effective_in == "bohr" and coords_out == "angstrom":
        R_out *= BOHR_TO_ANGSTROM
    elif effective_in == "angstrom" and coords_out == "bohr":
        R_out *= ANGSTROM_TO_BOHR
    else:
        raise ValueError(
            f"Cannot convert coordinates from {effective_in!r} to {coords_out!r}. "
            "Use --coords-in / --coords-out or --preserve-units."
        )
    return R_out, effective_in, detected


def convert_energy_array(
    E: np.ndarray,
    energy_in: EnergyIn,
    energy_out: EnergyOut,
) -> np.ndarray:
    """Convert total energies between supported unit conventions."""
    E_work = np.asarray(E, dtype=np.float64).copy()
    if energy_out == "same":
        return E_work
    if energy_in == energy_out:
        return E_work
    if energy_in == "hartree" and energy_out == "ev":
        return convert_energy_hartree_to_ev(E_work)
    if energy_in == "ev" and energy_out == "hartree":
        return convert_energy_ev_to_hartree(E_work)
    raise ValueError(
        f"Cannot convert energy from {energy_in!r} to {energy_out!r}. "
        "Supported: hartree↔ev, or use --energy-out same."
    )


def convert_force_array(
    F: np.ndarray,
    force_in: ForceIn,
    force_out: ForceOut,
) -> np.ndarray:
    """Convert forces between supported unit conventions."""
    F_work = np.asarray(F, dtype=np.float64).copy()
    if force_out == "same":
        return F_work
    if force_in == force_out:
        return F_work
    if force_in == "hartree_bohr" and force_out == "ev_angstrom":
        return convert_forces_hartree_bohr_to_ev_angstrom(F_work)
    if force_in == "ev_angstrom" and force_out == "hartree_bohr":
        return convert_forces_ev_angstrom_to_hartree_bohr(F_work)
    raise ValueError(
        f"Cannot convert forces from {force_in!r} to {force_out!r}. "
        "Supported: hartree_bohr↔ev_angstrom, or use --force-out same."
    )


def convert_dipole_array(
    D: np.ndarray,
    dipole_in: DipoleIn,
    dipole_out: DipoleOut,
) -> np.ndarray:
    """Convert dipole vectors between Debye and e·Å."""
    D_work = np.asarray(D, dtype=np.float64).copy()
    if dipole_out == "same":
        return D_work
    if dipole_in == dipole_out:
        return D_work
    if dipole_in == "debye" and dipole_out == "e_angstrom":
        return convert_dipole_debye_to_eA(D_work)
    if dipole_in == "e_angstrom" and dipole_out == "debye":
        return convert_dipole_eA_to_debye(D_work)
    raise ValueError(
        f"Cannot convert dipole from {dipole_in!r} to {dipole_out!r}. "
        "Supported: debye↔e_angstrom, or use --dipole-out same."
    )


def _grid_key_candidates(grid_data: Dict[str, Any]) -> List[str]:
    for key in ("vdw_surface", "vdw_grid", "esp_grid"):
        if key in grid_data:
            return [key]
    return []


def convert_grid_surface_array(
    grid_data: Dict[str, Any],
    grid_coords_in: GridCoordsIn,
    grid_coords_out: GridCoordsOut,
    *,
    cube_spacing_bohr: float = 0.25,
    verbose: bool = False,
) -> Tuple[Optional[np.ndarray], str, List[str]]:
    """
    Convert ESP grid point coordinates to the requested output units.

    Returns (grid_coords_or_none, effective_input_unit, log_notes).
    """
    from mmml.data.units import BOHR_TO_ANGSTROM

    notes: List[str] = []
    keys = _grid_key_candidates(grid_data)
    if not keys:
        return None, "none", notes

    key = keys[0]
    vdw_raw = np.asarray(grid_data[key], dtype=np.float64)

    if grid_coords_out == "same":
        effective_in = grid_coords_in
        if grid_coords_in == "auto":
            required = ["vdw_grid", "grid_origin", "grid_axes", "grid_dims"]
            missing = [k for k in required if k not in grid_data]
            grid_from_pyscf = "esp" in grid_data and bool(keys)
            if not missing:
                effective_in = "index"
            elif grid_from_pyscf:
                effective_in = "bohr"
            else:
                effective_in = "angstrom"
            notes.append(f"auto (preserve): grid input assumed {effective_in}")
        if verbose:
            print(
                f"  Grid ({key}): preserving coordinates "
                f"(--grid-coords-out same, input={effective_in})"
            )
        return vdw_raw.copy(), effective_in, notes

    effective_in = grid_coords_in
    if grid_coords_in == "auto":
        required = ["vdw_grid", "grid_origin", "grid_axes", "grid_dims"]
        missing = [k for k in required if k not in grid_data]
        grid_from_pyscf = "esp" in grid_data and bool(keys)
        if not missing:
            effective_in = "index"
        elif grid_from_pyscf:
            effective_in = "bohr"
            notes.append("auto: pyscf-style esp+grid → input assumed Bohr")
        else:
            effective_in = "angstrom"
            notes.append("auto: no cube metadata → input assumed Angstrom")
        if verbose:
            print(f"  Auto-detected grid coordinates: {effective_in}")

    if effective_in == grid_coords_out:
        return vdw_raw.copy(), effective_in, notes

    if effective_in == "index" and grid_coords_out == "angstrom":
        required = ["grid_origin", "grid_axes", "grid_dims"]
        missing = [k for k in required if k not in grid_data]
        if missing:
            raise ValueError(
                f"Grid coords are index space but missing keys {missing} for conversion to Angstrom. "
                "Provide cube metadata or set --grid-coords-in angstrom/bohr."
            )
        vdw_ang = convert_grid_indices_to_angstrom(
            grid_data.get("vdw_grid", vdw_raw),
            grid_data["grid_origin"],
            grid_data["grid_axes"],
            grid_data["grid_dims"],
            cube_spacing_bohr=cube_spacing_bohr,
        )
        notes.append(f"index→angstrom via cube spacing {cube_spacing_bohr} Bohr")
        return vdw_ang, "index", notes

    if effective_in == "bohr" and grid_coords_out == "angstrom":
        return vdw_raw * BOHR_TO_ANGSTROM, "bohr", notes
    if effective_in == "angstrom" and grid_coords_out == "bohr":
        from mmml.data.units import ANGSTROM_TO_BOHR
        return vdw_raw * ANGSTROM_TO_BOHR, "angstrom", notes
    if effective_in == "angstrom" and grid_coords_out == "angstrom":
        return vdw_raw.copy(), "angstrom", notes
    if effective_in == "bohr" and grid_coords_out == "bohr":
        return vdw_raw.copy(), "bohr", notes

    raise ValueError(
        f"Cannot convert grid coordinates from {effective_in!r} to {grid_coords_out!r}."
    )


def _unit_label_energy(u: str) -> str:
    return {"hartree": "Hartree", "ev": "eV", "same": "(unchanged)"}.get(u, u)


def _unit_label_force(u: str) -> str:
    return {
        "hartree_bohr": "Hartree/Bohr",
        "ev_angstrom": "eV/Å",
        "same": "(unchanged)",
    }.get(u, u)


def _unit_label_dipole(u: str) -> str:
    return {"debye": "Debye", "e_angstrom": "e·Å", "same": "(unchanged)"}.get(u, u)


def _unit_label_coords(u: str) -> str:
    return {"bohr": "Bohr", "angstrom": "Å", "same": "(unchanged)", "auto": "auto"}.get(u, u)


def _scale_ndarrays_in_dict(
    data: Dict[str, Any],
    keys: Tuple[str, ...],
    factor: float,
) -> List[str]:
    """In-place multiply numeric ndarrays for keys by factor if factor != 1. Skips missing keys and non-numeric dtypes."""
    f = float(factor)
    if f == 1.0:
        return []
    out: List[str] = []
    for k in keys:
        if k not in data:
            continue
        v = data[k]
        if not isinstance(v, np.ndarray) or v.dtype.kind not in "biufc":
            continue
        data[k] = np.asarray(v, dtype=np.float64) * f
        out.append(f"{k}×{f:g}")
    return out


def load_atomic_reference_energies(scheme: str) -> Dict[str, float]:
    """Load per-atom reference energies (Hartree) for a given scheme from atomic_reference_energies.json."""
    with open(ATOMIC_REF_PATH) as f:
        all_refs = json.load(f)
    if scheme not in all_refs:
        raise ValueError(
            f"Unknown atomic ref scheme '{scheme}'. "
            f"Available: {list(all_refs.keys())[:10]}..."
        )
    return all_refs[scheme]


def _species_entry_to_atomic_number(zi: Any) -> int:
    """Map one Z entry (atomic number or element symbol) to atomic number; 0 = padding."""
    if zi is None:
        return 0
    if isinstance(zi, (bytes, np.bytes_)):
        try:
            zi = zi.decode("ascii").strip()
        except Exception:
            return 0
    if isinstance(zi, str) or isinstance(zi, np.str_):
        s = str(zi).strip()
        if not s or s.lower() in ("none", "nan"):
            return 0
        try:
            n = int(float(s))
            return n if n > 0 else 0
        except ValueError:
            pass
        from ase.data import atomic_numbers

        sym = s[0].upper() + s[1:].lower() if len(s) > 1 else s.upper()
        return int(atomic_numbers.get(sym, 0))
    try:
        n = int(np.asarray(zi).item())
        return n if n > 0 else 0
    except (ValueError, TypeError, OverflowError):
        return 0


def npz_z_array_to_atomic_numbers(Z: np.ndarray) -> np.ndarray:
    """
    Convert NPZ Z array to int64 atomic numbers.

    Integer/float Z uses a fast path; object / Unicode arrays (e.g. 'C', 'H') are
    mapped via ASE. Non-positive and unknown species become 0 (padding).
    """
    Z = np.asarray(Z)
    if Z.size == 0:
        return np.zeros(Z.shape, dtype=np.int64)
    if Z.dtype.kind in "iuf" and Z.dtype != object:
        Zn = np.asarray(Z, dtype=np.int64)
        return np.where(Zn > 0, Zn, 0).astype(np.int64)
    flat = Z.ravel()
    out = np.fromiter(
        (_species_entry_to_atomic_number(x) for x in flat),
        dtype=np.int64,
        count=flat.size,
    )
    return out.reshape(Z.shape)


def subtract_atomic_references(
    E_hartree: np.ndarray,
    Z: np.ndarray,
    scheme: str,
    ref_units: str = "hartree",
) -> np.ndarray:
    """
    Subtract per-atom reference energies from total energies.
    E_corrected = E - sum(E_ref[Z_i]) for each molecule.
    Returns corrected energies in Hartree.

    ref_units: "hartree" (default) or "ev". If "ev", refs are converted to Hartree
    before subtraction (E_ref_Ha = E_ref_eV / 27.211386).
    """
    from ase.data import chemical_symbols

    HARTREE_TO_EV = 27.211386
    refs = load_atomic_reference_energies(scheme)
    E_ref_per_sample = np.zeros(len(E_hartree), dtype=np.float64)
    Z = npz_z_array_to_atomic_numbers(np.asarray(Z))

    for i in range(len(E_hartree)):
        z = Z[i] if Z.ndim > 1 else Z
        for zi in z:
            zn = int(zi)
            if zn <= 0:
                continue
            sym = chemical_symbols[zn]
            key = f"{sym}:0"
            if key not in refs:
                raise ValueError(f"No reference for {key} in scheme '{scheme}'")
            val = refs[key]
            if ref_units.lower() == "ev":
                val = val / HARTREE_TO_EV
            E_ref_per_sample[i] += val

    return E_hartree - E_ref_per_sample


def atomic_ref_sum_hartree(
    Z: np.ndarray,
    scheme: str,
    ref_units: str = "hartree",
) -> float:
    """Mean per-sample sum of atomic reference energies (Hartree) from Z."""
    Z = np.asarray(Z)
    n = int(Z.shape[0]) if Z.ndim > 1 else 1
    zero_e = np.zeros(n, dtype=np.float64)
    corrected = subtract_atomic_references(zero_e, Z, scheme, ref_units=ref_units)
    return float(abs(np.mean(corrected)))


def diagnose_energy_unit_for_atomic_refs(
    E: np.ndarray,
    Z: np.ndarray,
    scheme: str,
    ref_units: str,
    declared_energy_in: str,
) -> Optional[str]:
    """
    Detect when E is labeled Hartree but magnitudes match eV totals for the Z composition.
    """
    from mmml.data.units import EV_TO_HARTREE, HARTREE_TO_EV

    ref_ha = abs(atomic_ref_sum_hartree(Z, scheme, ref_units))
    if ref_ha <= 0.0:
        return None
    e_mean = abs(float(np.mean(np.asarray(E, dtype=np.float64))))
    if declared_energy_in == "hartree" and e_mean > 5.0 * ref_ha:
        e_as_ha_if_ev = e_mean * EV_TO_HARTREE
        if 0.7 * ref_ha <= e_as_ha_if_ev <= 1.3 * ref_ha:
            return (
                f"|E| mean ({e_mean:.4g}) is ~{e_mean / ref_ha:.0f}× the atomic-ref sum implied by Z "
                f"({ref_ha:.4g} Ha), but matches input energies in eV converted to Hartree "
                f"({e_as_ha_if_ev:.4g} Ha). Use --energy-in ev "
                f"(and --force-in ev-angstrom if forces are not Hartree/Bohr)."
            )
    if declared_energy_in == "ev" and e_mean > 5.0 * ref_ha * HARTREE_TO_EV:
        e_as_ha_if_hartree = e_mean * EV_TO_HARTREE
        if 0.7 * ref_ha <= e_as_ha_if_hartree <= 1.3 * ref_ha:
            return None
        if 0.7 * ref_ha <= e_mean <= 1.3 * ref_ha:
            return (
                f"|E| mean ({e_mean:.4g}) matches atomic-ref sum in Hartree ({ref_ha:.4g} Ha) "
                f"but you passed --energy-in ev. Try --energy-in hartree."
            )
    return None


def expected_atomic_ref_units(scheme: str) -> str:
    """
    Infer native units of per-atom references in atomic_reference_energies.json.

    Schemes with total atomic energies (e.g. pbe0/def2-tzvp) store C:0 around -37 Hartree.
    Schemes with small deltas (e.g. pbe0/sz) store C:0 around -0.1 eV.
    """
    refs = load_atomic_reference_energies(scheme)
    probe_keys = ("C:0", "H:0", "N:0", "O:0")
    for key in probe_keys:
        if key in refs:
            return "hartree" if abs(float(refs[key])) > 1.0 else "ev"
    vals = [abs(float(v)) for k, v in refs.items() if k.endswith(":0")]
    if vals and max(vals) > 1.0:
        return "hartree"
    return "ev"


def check_atomic_ref_subtraction(
    E_before_hartree: np.ndarray,
    E_after_hartree: np.ndarray,
    *,
    scheme: str,
    ref_units: str,
    energy_in_declared: Optional[str] = None,
    Z: Optional[np.ndarray] = None,
) -> None:
    """
    Raise ValueError when atomic-reference subtraction barely changed total-like energies.
    """
    e_before = np.asarray(E_before_hartree, dtype=np.float64)
    e_after = np.asarray(E_after_hartree, dtype=np.float64)
    mean_before = float(np.mean(e_before))
    mean_after = float(np.mean(e_after))
    shift = abs(mean_before - mean_after)
    if abs(mean_before) < 500.0 or shift >= 0.05 * abs(mean_before):
        return

    expected_units = expected_atomic_ref_units(scheme)
    units = ref_units.lower()
    hint = (
        f"Atomic reference subtraction only shifted mean |E| by {shift:.4g} Hartree "
        f"(before={mean_before:.6f}, after={mean_after:.6f}). "
        f"For scheme '{scheme}', refs in JSON look like {expected_units}."
    )
    if units != expected_units:
        hint += (
            f" You passed --atomic-ref-units {units}; try --atomic-ref-units {expected_units}."
        )
    elif energy_in_declared and Z is not None:
        unit_hint = diagnose_energy_unit_for_atomic_refs(
            e_before, Z, scheme, ref_units, energy_in_declared
        )
        if unit_hint:
            hint += f" {unit_hint}"
        else:
            hint += " Check that Z contains real atomic numbers for all non-padded atoms."
    else:
        hint += " Check that Z contains real atomic numbers for all non-padded atoms."
    raise ValueError(hint)


def reduce_esp_grid(
    esp: np.ndarray,
    esp_grid: np.ndarray,
    R: np.ndarray,
    n_grid_points: int = 3000,
    esp_sd_sigma: float = 3.0,
    esp_max_abs_kcal_mol: float = 100.0,
    min_dist_to_atoms: float = 1.0,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Reduce ESP grid to fixed number of points per sample.
    Excludes points beyond ±esp_sd_sigma standard deviations from the mean (ignores
    distribution tails), points with |esp| > esp_max_abs_kcal_mol (kcal/mol/e), and
    points too close to atomic centers.
    """
    rng = np.random.default_rng(seed)
    n_samples = esp.shape[0]
    R.shape[1]

    esp_out = np.zeros((n_samples, n_grid_points), dtype=esp.dtype)
    esp_grid_out = np.full((n_samples, n_grid_points, 3), 1e6, dtype=esp_grid.dtype)

    for i in range(n_samples):
        esp_i = esp[i]
        grid_i = esp_grid[i]
        r_i = R[i]

        # Mask padding (esp_grid uses 1e6 for padding)
        valid = np.all(np.abs(grid_i) < 1e5, axis=1)

        # Exclude points too close to any atom
        atoms_valid = np.any(r_i != 0, axis=1)
        if np.any(atoms_valid):
            dists = cdist(grid_i, r_i[atoms_valid])
            min_d = dists.min(axis=1)
            valid &= min_d > min_dist_to_atoms

        # Exclude distribution tails: keep only points within ±esp_sd_sigma SD of mean
        valid_esp = esp_i[valid]
        if len(valid_esp) > 0:
            mean_esp = np.mean(valid_esp)
            std_esp = np.std(valid_esp)
            if std_esp > 0:
                valid &= np.abs(esp_i - mean_esp) <= esp_sd_sigma * std_esp

        # Exclude points with |esp| > esp_max_abs_kcal_mol (kcal/mol/e)
        from mmml.data.units import KCAL_MOL_TO_HARTREE
        esp_max_hartree = esp_max_abs_kcal_mol * KCAL_MOL_TO_HARTREE
        valid &= np.abs(esp_i) <= esp_max_hartree

        idx = np.where(valid)[0]
        if len(idx) >= n_grid_points:
            chosen = rng.choice(idx, size=n_grid_points, replace=False)
            esp_out[i] = esp_i[chosen]
            esp_grid_out[i] = grid_i[chosen]
        elif len(idx) > 0:
            n_take = len(idx)
            esp_out[i, :n_take] = esp_i[idx]
            esp_grid_out[i, :n_take, :] = grid_i[idx]
            # Rest stays as padding (zeros for esp, 1e6 for grid)

    return esp_out, esp_grid_out


def verify_reduction_preserves_alignment(
    esp_raw: np.ndarray,
    grid_raw: np.ndarray,
    esp_reduced: np.ndarray,
    grid_reduced: np.ndarray,
    R: np.ndarray,
    n_grid_points: int = 3000,
    esp_sd_sigma: float = 3.0,
    esp_max_abs_kcal_mol: float = 100.0,
    min_dist_to_atoms: float = 1.0,
    seed: int = 42,
    n_spot_check: int = 3,
) -> bool:
    """
    Verify that reduce_esp_grid used the same indices for esp and grid (alignment preserved).
    Recomputes the selection logic and checks that output pairs match input pairs.
    """
    from mmml.data.units import KCAL_MOL_TO_HARTREE
    rng = np.random.default_rng(seed)
    for i in range(min(n_spot_check, esp_raw.shape[0])):
        esp_i = esp_raw[i]
        grid_i = grid_raw[i]
        r_i = R[i]
        valid = np.all(np.abs(grid_i) < 1e5, axis=1)
        atoms_valid = np.any(r_i != 0, axis=1)
        if np.any(atoms_valid):
            dists = cdist(grid_i, r_i[atoms_valid])
            valid &= dists.min(axis=1) > min_dist_to_atoms
        valid_esp = esp_i[valid]
        if len(valid_esp) > 0:
            mean_esp = np.mean(valid_esp)
            std_esp = np.std(valid_esp)
            if std_esp > 0:
                valid &= np.abs(esp_i - mean_esp) <= esp_sd_sigma * std_esp
        esp_max_hartree = esp_max_abs_kcal_mol * KCAL_MOL_TO_HARTREE
        valid &= np.abs(esp_i) <= esp_max_hartree
        idx = np.where(valid)[0]
        if len(idx) < 3:
            continue
        if len(idx) >= n_grid_points:
            chosen = rng.choice(idx, size=n_grid_points, replace=False)
            n_check = n_grid_points
        else:
            chosen = idx
            n_check = len(idx)
        for k in range(n_check):
            if not np.isclose(esp_reduced[i, k], esp_i[chosen[k]]):
                return False
            if not np.allclose(grid_reduced[i, k], grid_i[chosen[k]]):
                return False
    return True


def convert_grid_indices_to_angstrom(
    vdw_grid_indices: np.ndarray,
    origin: np.ndarray,
    axes: np.ndarray,
    dims: np.ndarray,
    cube_spacing_bohr: float = 0.25
) -> np.ndarray:
    """
    Convert ESP grid from index space to physical Angstrom coordinates.
    
    The vdw_grid currently contains values like 0-49 which are grid indices.
    We need to convert to physical coordinates using the cube metadata.
    """
    _ = dims  # Reserved for future non-cubic grid handling.
    n_samples = vdw_grid_indices.shape[0]
    bohr_to_angstrom = 0.529177
    
    vdw_grid_angstrom = np.zeros_like(vdw_grid_indices)
    
    for i in range(n_samples):
        grid_indices = vdw_grid_indices[i] - origin[i]  # Remove origin offset
        coord_bohr = origin[i] + grid_indices * cube_spacing_bohr
        vdw_grid_angstrom[i] = coord_bohr * bohr_to_angstrom
    
    return vdw_grid_angstrom


def check_esp_grid_alignment(
    esp: np.ndarray,
    grid: np.ndarray,
    R: np.ndarray,
    n_check: int = 5,
) -> Tuple[bool, float]:
    """
    Verify that esp[i,j] corresponds to grid[i,j] (same physical point).
    
    Uses spatial consistency: |esp| tends to be larger near atoms (1/r decay).
    If esp and grid are misaligned (wrong index order), this correlation drops.
    
    Returns
    -------
    ok : bool
        True if alignment appears correct (mean correlation > 0.3)
    mean_corr : float
        Mean Pearson correlation across checked samples
    """
    from scipy.stats import pearsonr
    BOHR_TO_ANGSTROM = 0.529177
    1.0 / BOHR_TO_ANGSTROM
    correlations = []
    n_samples = min(n_check, esp.shape[0])
    for i in range(n_samples):
        esp_i = esp[i]
        grid_i = grid[i]
        r_i = R[i]
        valid = np.all(np.abs(grid_i) < 1e5, axis=1)
        if np.sum(valid) < 10:
            continue
        esp_valid = np.abs(esp_i[valid])
        grid_valid = grid_i[valid]
        atoms_valid = np.any(r_i != 0, axis=1)
        if not np.any(atoms_valid):
            continue
        dists = cdist(grid_valid, r_i[atoms_valid])
        min_dist = dists.min(axis=1)
        inv_dist = 1.0 / (min_dist + 0.5)
        try:
            corr, _ = pearsonr(esp_valid, inv_dist)
            if not np.isnan(corr):
                correlations.append(float(corr))
        except Exception:
            pass
    if not correlations:
        return True, 0.0
    mean_corr = float(np.mean(correlations))
    ok = mean_corr > 0.2
    return ok, mean_corr


def validate_fixed_data(
    R_ang,
    E_ev,
    F_ev_ang,
    vdw_grid_ang,
    Z,
    N,
    has_grid: bool = True,
    verbose: bool = True,
    *,
    coords_unit: str = "angstrom",
    energy_unit: str = "ev",
    force_unit: str = "ev_angstrom",
):
    """Validate converted data; checks adapt to declared output units."""
    if verbose:
        print(f"\n{'='*70}")
        print("POST-FIX VALIDATION")
        print(f"{'='*70}")
    
    # Check atomic coordinates (shortest interatomic distance)
    min_dists = []
    for i in range(min(100, len(R_ang))):
        r = R_ang[i]
        valid = np.any(r != 0, axis=1)
        vpos = r[valid]
        if len(vpos) < 2:
            continue
        d = vpos[:, np.newaxis, :] - vpos[np.newaxis, :, :]
        norms = np.linalg.norm(d, axis=2)
        norms[np.triu_indices_from(norms, k=0)] = np.inf
        min_dists.append(norms.min())
    min_dists = np.array(min_dists) if min_dists else np.array([])

    coords_ok = False
    energy_ok = False
    force_ok = False
    grid_ok = True  # Default to True if no grid
    spatial_ok = True  # Default to True if no grid

    if len(min_dists) > 0:
        if verbose:
            print("\nAtomic Coordinates (up to 100 samples):")
            print(f"  Shortest distance: mean={min_dists.mean():.4f} Å, "
                  f"range=[{min_dists.min():.4f}, {min_dists.max():.4f}]")
        if coords_unit == "bohr":
            coords_ok = 0.9 <= min_dists.mean() <= 5.5
        else:
            coords_ok = 0.5 <= min_dists.mean() <= 3.0
        if verbose and coords_ok:
            print("  ✓ Coordinates in reasonable range")
        elif verbose:
            print("  ⚠️  Coordinates outside expected range")
    else:
        coords_ok = True
    
    # Check energies
    if verbose:
        print("\nEnergies (sample 0):")
        e_label = _unit_label_energy(energy_unit)
        print(f"  Value: {E_ev[0]:.6f} {e_label}")
        print(f"  Dataset mean: {E_ev.mean():.6f} {e_label}")

    if energy_unit == "hartree":
        energy_ok = -500 < E_ev.mean() < 50
        ok_msg = "reasonable range for Hartree"
    else:
        energy_ok = -10000 < E_ev.mean() < 1000
        ok_msg = "reasonable range for eV"
    if energy_ok:
        if verbose:
            print(f"  ✓ Energies in {ok_msg}")
    elif verbose:
        print("  ⚠️  Energy range unexpected (may be fine for your chemistry/units)")
    
    # Check forces
    f_sample = F_ev_ang[0, :min(3, F_ev_ang.shape[1]), :]  # First sample, first atoms
    f_norm = np.linalg.norm(f_sample.reshape(-1, 3), axis=1).mean()
    
    if verbose:
        print("\nForces (sample 0):")
        print(f"  Mean norm: {f_norm:.6e} {_unit_label_force(force_unit)}")

    f_max = 5000.0 if force_unit == "hartree_bohr" else 1000.0
    if 1e-6 < f_norm < f_max:
        if verbose:
            print("  ✓ Force magnitudes in reasonable range")
        force_ok = True
    else:
        if verbose:
            print("  ⚠️  Force magnitudes outside expected range")
        force_ok = False
    
    # Check ESP grid (only if grid data exists)
    if has_grid and vdw_grid_ang is not None:
        grid0 = vdw_grid_ang[0]
        # Mask out padding (e.g. 1e6 used for variable-length grids)
        valid_mask = np.all(np.abs(grid0) < 1e5, axis=1)
        grid0_valid = grid0[valid_mask] if np.any(valid_mask) else grid0
        grid_extent = (grid0_valid.max(axis=0) - grid0_valid.min(axis=0)).mean()
        
        if verbose:
            print("\nESP Grid Coordinates:")
            print(f"  Average extent: {grid_extent:.4f} Angstrom")
            print(f"  X range: [{grid0_valid[:, 0].min():.4f}, {grid0_valid[:, 0].max():.4f}]")
            print(f"  Y range: [{grid0_valid[:, 1].min():.4f}, {grid0_valid[:, 1].max():.4f}]")
            print(f"  Z range: [{grid0_valid[:, 2].min():.4f}, {grid0_valid[:, 2].max():.4f}]")
        
        # Expect reasonable grid extent for molecular systems (2-20 Angstroms)
        if 2.0 < grid_extent < 50.0:
            if verbose:
                print("  ✓ Grid extent in reasonable range")
            grid_ok = True
        else:
            if verbose:
                print("  ⚠️  Grid extent outside expected range")
            grid_ok = False
        
        # Check spatial relationship
        r0 = R_ang[0]
        z0 = np.asarray(Z[0] if Z.ndim > 1 else Z)
        zn0 = npz_z_array_to_atomic_numbers(z0)
        valid = zn0 > 0
        valid_pos = r0[valid]
        
        if len(valid_pos) > 0:
            mol_center = valid_pos.mean(axis=0)
            grid_min = grid0_valid.min(axis=0)
            grid_max = grid0_valid.max(axis=0)
            
            if verbose:
                print("\nSpatial relationship:")
                print(f"  Molecule center: [{mol_center[0]:.2f}, {mol_center[1]:.2f}, {mol_center[2]:.2f}]")
                print(f"  Grid bounds: X[{grid_min[0]:.2f}, {grid_max[0]:.2f}], "
                      f"Y[{grid_min[1]:.2f}, {grid_max[1]:.2f}], "
                      f"Z[{grid_min[2]:.2f}, {grid_max[2]:.2f}]")
            
            # Check if molecule is within or near grid bounds
            max_min_dist = max([np.min(np.linalg.norm(grid0_valid - atom_pos, axis=1)) 
                               for atom_pos in valid_pos])
            
            if max_min_dist < 10.0:
                if verbose:
                    print(f"  ✓ Grid points within {max_min_dist:.2f} Å of molecule")
                spatial_ok = True
            else:
                if verbose:
                    print(f"  ⚠️  Grid too far from molecule ({max_min_dist:.2f} Å)")
                spatial_ok = False
        else:
            if verbose:
                print("\n⚠️  Could not validate spatial relationship")
            spatial_ok = True
    else:
        if verbose:
            print("\nESP Grid: Skipped (no grid data provided)")
    
    overall_ok = coords_ok and energy_ok and force_ok and grid_ok and spatial_ok
    
    if verbose:
        print(f"\n{'='*70}")
        print("VALIDATION SUMMARY")
        print(f"{'='*70}")
        print(f"  Coordinates: {'✓' if coords_ok else '❌'}")
        print(f"  Energies:    {'✓' if energy_ok else '❌'}")
        print(f"  Forces:      {'✓' if force_ok else '❌'}")
        if has_grid:
            print(f"  ESP Grid:    {'✓' if grid_ok else '❌'}")
            print(f"  Spatial:     {'✓' if spatial_ok else '❌'}")
        
        if overall_ok:
            print("\n✅ ALL VALIDATIONS PASSED - Data ready for training!")
        else:
            print("\n⚠️  SOME VALIDATIONS FAILED - Review above")
        print(f"{'='*70}")
    
    return overall_ok


def _normalize_for_concat(arr: np.ndarray, n_samples: int, key: str) -> np.ndarray:
    """Ensure array has at least 1 dim for concatenation. Scalar/0-d -> (n_samples,)."""
    arr = np.asarray(arr)
    if arr.ndim == 0:
        return np.full(n_samples, arr.flat[0], dtype=arr.dtype)
    if arr.ndim == 1 and arr.shape[0] == 1 and key in ('N',):
        return np.full(n_samples, arr[0], dtype=arr.dtype)
    if key in ('Ef', 'efield_Ef', 'efield_scf_Ef'):
        if arr.shape[0] == n_samples:
            return arr
        if arr.size == 3 or (arr.ndim == 1 and arr.shape[0] == 3):
            v = np.asarray(arr, dtype=np.float64).reshape(1, 3)
            return np.broadcast_to(v, (n_samples, 3))
        if arr.ndim == 2 and arr.shape[0] == 1 and arr.shape[1] == 3:
            v = np.asarray(arr, dtype=np.float64).reshape(1, 3)
            return np.broadcast_to(v, (n_samples, 3))
        raise ValueError(
            f"Cannot merge {key}: need (n_samples, 3) with n_samples={n_samples}, "
            f"or a single (3,) / (1, 3) field to broadcast; got {arr.shape}"
        )
    return arr


def _normalize_z_for_concat(arr: np.ndarray, n_samples: int, n_atoms: int) -> np.ndarray:
    """
    Normalize Z for concatenation to shape (n_samples, n_atoms) when needed.

    Supports common encodings:
    - (n_atoms,) shared composition for all samples -> broadcast
    - (n_samples, n_atoms) already aligned -> pass through
    - scalar/0-d -> broadcast to (n_samples, n_atoms)
    """
    arr = np.asarray(arr)
    if arr.ndim == 0:
        return np.full((n_samples, n_atoms), int(arr.flat[0]), dtype=np.int32)
    if arr.ndim == 1:
        if arr.shape[0] == n_atoms:
            return np.broadcast_to(arr.reshape(1, n_atoms), (n_samples, n_atoms))
        if arr.shape[0] == n_samples and n_atoms == 1:
            return arr.reshape(n_samples, 1)
    if arr.ndim == 2 and arr.shape[0] == n_samples:
        return arr
    raise ValueError(
        f"Cannot merge Z: expected scalar, (n_atoms,), or (n_samples, n_atoms). "
        f"Got shape {arr.shape} with n_samples={n_samples}, n_atoms={n_atoms}"
    )


def _pad_concat_axis1(arr: np.ndarray, target_size: int, key: str, pad_value: float) -> np.ndarray:
    """
    Pad axis=1 to ``target_size`` for variable-length ESP/grid arrays before concat.

    ``esp`` uses 0 padding; grid coordinate arrays use a large sentinel (1e6).
    """
    arr = np.asarray(arr)
    if arr.ndim < 2:
        raise ValueError(f"Cannot pad {key}: expected at least 2D array, got shape {arr.shape}")
    if arr.shape[1] > target_size:
        raise ValueError(
            f"Cannot pad {key}: current axis-1 size {arr.shape[1]} exceeds target {target_size}"
        )
    if arr.shape[1] == target_size:
        return arr
    pad_width = [(0, 0)] * arr.ndim
    pad_width[1] = (0, target_size - arr.shape[1])
    return np.pad(arr, pad_width, mode="constant", constant_values=pad_value)


def _load_and_merge_efd(efd_files: Union[Path, List[Path]]) -> Dict:
    """Load one or more EFD npz files and concatenate along sample dimension."""
    if isinstance(efd_files, (str, Path)):
        efd_files = [Path(efd_files)]
    else:
        efd_files = [Path(f) for f in efd_files]

    if len(efd_files) == 1:
        return dict(np.load(efd_files[0], allow_pickle=True))

    # Concatenate multiple files
    parts = [dict(np.load(f, allow_pickle=True)) for f in efd_files]
    concat_keys = ['R', 'E', 'F', 'N', 'Z', 'D', 'Q', 'Dxyz', 'esp', 'Ef', 'efield_Ef', 'efield_scf_Ef']
    grid_key = 'esp_grid' if 'esp_grid' in parts[0] else ('vdw_surface' if 'vdw_surface' in parts[0] else None)
    if grid_key:
        concat_keys.append(grid_key)
    variable_grid_like_keys = {'esp', 'esp_grid', 'vdw_surface', 'vdw_grid'}
    variable_atom_like_keys = {'R', 'F', 'Z'}
    # For EFD merges we expect the same atom padding across files for R/F/Z.
    # Mixed atom-axis sizes usually means mixed systems were provided accidentally.
    for k in variable_atom_like_keys:
        sizes = [
            np.asarray(p[k]).shape[1]
            for p in parts
            if k in p and np.asarray(p[k]).ndim >= 2
        ]
        if sizes and len(set(sizes)) > 1:
            raise ValueError(
                f"Inconsistent atom-axis size for key '{k}' across EFD files: {sorted(set(sizes))}. "
                "All merged EFD files must have the same padded natoms. "
                "Split by composition/system first, or re-pad consistently before merging."
            )
    axis1_targets = {}
    for k in variable_grid_like_keys:
        sizes = [
            np.asarray(p[k]).shape[1]
            for p in parts
            if k in p and np.asarray(p[k]).ndim >= 2
        ]
        if sizes:
            axis1_targets[k] = max(sizes)

    all_keys = set()
    for p in parts:
        all_keys.update(p.keys())
    merged = {}
    for k in all_keys:
        if k in concat_keys and all(k in p for p in parts):
            # Normalize shapes: pyscf-evaluate may have N as scalar (0-d), fix-and-split has (n,)
            to_concat = []
            for p in parts:
                arr = np.asarray(p[k])
                n_samples = p['R'].shape[0]  # sample count from R
                n_atoms = p['R'].shape[1] if np.asarray(p['R']).ndim >= 2 else 1
                if arr.ndim == 0 or (arr.ndim == 1 and arr.shape[0] != n_samples and k == 'N'):
                    arr = _normalize_for_concat(arr, n_samples, k)
                elif k in ('Ef', 'efield_Ef', 'efield_scf_Ef') and arr.shape[0] != n_samples:
                    arr = _normalize_for_concat(arr, n_samples, k)
                elif k == 'Z':
                    arr = _normalize_z_for_concat(arr, n_samples, n_atoms)
                if k in axis1_targets and arr.ndim >= 2:
                    # ESP-like arrays use 0/1e6 semantics.
                    if k == 'esp':
                        pad_value = 0.0
                    else:
                        pad_value = 1e6
                    arr = _pad_concat_axis1(arr, axis1_targets[k], k, pad_value=pad_value)
                to_concat.append(arr)
            merged[k] = np.concatenate(to_concat, axis=0)
        else:
            # Key in some but not all parts, or not concat-able: use first part that has it
            for p in parts:
                if k in p:
                    merged[k] = np.asarray(p[k])
                    break
    return merged


def fix_and_split_data(
    efd_file: Union[Path, List[Path]],
    grid_file: Optional[Path] = None,
    output_dir: Path = None,
    train_frac: float = 0.8,
    valid_frac: float = 0.1,
    test_frac: float = 0.1,
    seed: int = 42,
    cube_spacing_bohr: float = 0.25,
    skip_validation: bool = False,
    atomic_ref: Optional[str] = None,
    atomic_ref_units: str = "hartree",
    n_grid_points: int = 3000,
    esp_sd_sigma: float = 3.0,
    esp_max_abs_kcal_mol: float = 100.0,
    min_dist_to_atoms: float = 1.0,
    flip_forces: bool = False,
    energy_scale: float = 1.0,
    force_scale: float = 1.0,
    dipole_scale: float = 1.0,
    efield_scale: float = 1.0,
    esp_scale: float = 1.0,
    charge_scale: float = 1.0,
    zscale_energies: bool = False,
    coords_in: CoordsIn = "auto",
    coords_out: CoordsOut = "angstrom",
    energy_in: EnergyIn = "hartree",
    energy_out: EnergyOut = "ev",
    force_in: ForceIn = "hartree_bohr",
    force_out: ForceOut = "ev_angstrom",
    dipole_in: DipoleIn = "debye",
    dipole_out: DipoleOut = "e_angstrom",
    grid_coords_in: GridCoordsIn = "auto",
    grid_coords_out: GridCoordsOut = "angstrom",
    preserve_units: bool = False,
    verbose: bool = True,
) -> bool:
    """
    Main workflow to fix units and create splits.
    
    Parameters
    ----------
    efd_file : Path
        Path to energies_forces_dipoles.npz file
    grid_file : Path, optional
        Path to grids_esp.npz file (optional)
    output_dir : Path
        Directory to save output files
    train_frac : float
        Fraction of data for training (default 0.8)
    valid_frac : float
        Fraction of data for validation (default 0.1)
    test_frac : float
        Fraction of data for testing (default 0.1)
    seed : int
        Random seed for reproducible splits (default 42)
    cube_spacing_bohr : float
        Grid spacing in Bohr from original cube files (default 0.25)
    skip_validation : bool
        Skip validation checks (default False)
    atomic_ref : str, optional
        Subtract per-atom reference energies using scheme from atomic_reference_energies.json
        (e.g. "pbe0/sz" for PBE0/SZ, "pbe0/def2-tzvp" for Hartree). Default: None.
    atomic_ref_units : str
        Units of refs in JSON: "hartree" (default) or "ev". If "ev", refs are converted
        before subtraction. Schemes like pbe0/sz may use eV; pbe0/def2-tzvp uses Hartree.
    n_grid_points : int
        Target number of ESP grid points per sample (default 3000). Points beyond ±esp_sd_sigma
        SD from the mean or too close to atoms are excluded, then subsampled.
    esp_sd_sigma : float
        Exclude grid points beyond ±this many standard deviations from the mean (default 3.0).
    esp_max_abs_kcal_mol : float
        Exclude grid points with |esp| > this in kcal/mol/e (default 100.0).
    min_dist_to_atoms : float
        Exclude grid points closer than this to any atom in Å (default 1.0).
    flip_forces : bool
        If True, negate ``F`` before converting units (use when NPZ stores PySCF-style
        energy gradient ∂E/∂R in Ha/Bohr instead of forces F = −∇E). Default False.
    energy_scale : float
        Multiply ``E`` after Hartree→eV (and after atomic-ref). Also ``efield_energy``, ``efield_scf_energy`` if present. Default 1.0.
    force_scale : float
        Multiply ``F`` after Ha/Bohr→eV/Å. Also ``efield_scf_F`` if present. Default 1.0.
    dipole_scale : float
        Multiply ``Dxyz`` after Debye→e·Å (ignored if no Dxyz). Same factor on ``D``, ``efield_scf_D``, ``efield_D`` (Debye) if present. Not applied to ``efield_D_au``. Default 1.0.
    efield_scale : float
        Multiply external-field vectors ``Ef``, ``efield_Ef``, ``efield_scf_Ef`` if present. Default 1.0.
    esp_scale : float
        Multiply ``esp`` (Hartree/e) on grid output and on EFD if that key is present (combined NPZ). Default 1.0.
    charge_scale : float
        Multiply NPZ key ``Q`` if present. Intended for **total molecular charge** when your files
        use ``Q`` for that. Note: PySCF ``dens_esp`` export may store **quadrupole** under ``Q``;
        only use this scale when ``Q`` matches your intended quantity. Default 1.0.
    zscale_energies : bool
        If True, replace ``E`` with ``(E - mean_train) / std_train`` after unit conversion,
        optional scaling, validation, and split creation. Mean/std are computed from the
        training split only and saved to ``energy_zscale_stats.json``. Default False.
    coords_in, coords_out : str
        Input/output units for ``R``. ``coords_in=auto`` infers Bohr vs Å from bond lengths.
        ``coords_out=same`` leaves coordinates unchanged (after resolving ``auto``).
    energy_in, energy_out : str
        Input/output units for ``E`` (``hartree``, ``ev``, or ``same`` for output).
    force_in, force_out : str
        Input/output units for ``F`` (``hartree_bohr``, ``ev_angstrom``, or ``same``).
    dipole_in, dipole_out : str
        Input/output units for ``Dxyz`` (``debye``, ``e_angstrom``, or ``same``).
    grid_coords_in, grid_coords_out : str
        Input/output units for ESP grid positions (``auto``, ``bohr``, ``angstrom``, ``index``).
    preserve_units : bool
        If True, equivalent to ``*-out same`` for R, E, F, Dxyz, and grid coordinates.
    verbose : bool
        Print detailed progress (default True)
        
    Returns
    -------
    bool
        True if successful, False otherwise
    """
    manifest_notes: List[str] = []
    if preserve_units:
        coords_out = "same"
        energy_out = "same"
        force_out = "same"
        dipole_out = "same"
        grid_coords_out = "same"
        manifest_notes.append("--preserve-units: no unit conversions on R, E, F, Dxyz, or grid")

    effective_energy_out = energy_out if energy_out != "same" else energy_in
    effective_force_out = force_out if force_out != "same" else force_in
    effective_dipole_out = dipole_out if dipole_out != "same" else dipole_in
    effective_grid_coords_out = (
        grid_coords_out if grid_coords_out != "same" else grid_coords_in
    )

    if verbose:
        print("\n" + "="*70)
        print("Molecular Data Unit Conversion and Splitting")
        print("="*70)
        print("\nUnit policy:")
        print(f"  R:     {_unit_label_coords(coords_in)} → {_unit_label_coords(coords_out)}")
        print(f"  E:     {_unit_label_energy(energy_in)} → {_unit_label_energy(energy_out)}")
        print(f"  F:     {_unit_label_force(force_in)} → {_unit_label_force(force_out)}")
        print(f"  Dxyz:  {_unit_label_dipole(dipole_in)} → {_unit_label_dipole(dipole_out)}")
        print(f"  Grid:  {grid_coords_in} → {grid_coords_out}")
        print("\nInput files:")
        if isinstance(efd_file, (list, tuple)):
            print(f"  EFD:  {[str(p) for p in efd_file]}")
        else:
            print(f"  EFD:  {efd_file}")
        if grid_file:
            print(f"  Grid: {grid_file}")
        else:
            print("  Grid: (not provided)")
        print(f"Output directory: {output_dir}")
    
    # Create output directory
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # =========================================================================
    # Load data
    # =========================================================================
    if verbose:
        print(f"\n{'#'*70}")
        print("# Step 1: Loading Data")
        print(f"{'#'*70}")
    
    try:
        efd_data = _load_and_merge_efd(efd_file)
        grid_data = None
        has_grid = False

        grid_from_efd = False
        if grid_file is not None and grid_file.exists():
            grid_data = dict(np.load(grid_file, allow_pickle=True))
            has_grid = True
            grid_from_efd = False
        elif 'esp' in efd_data and 'esp_grid' in efd_data:
            # Combined EFD+grid file (e.g. from pyscf-evaluate --esp)
            grid_data = {
                'esp': efd_data['esp'],
                'vdw_surface': efd_data['esp_grid'],
                'R': efd_data['R'],
                'Z': efd_data['Z'],
                'N': efd_data['N'],
            }
            if 'Dxyz' in efd_data:
                grid_data['Dxyz'] = efd_data['Dxyz']
            has_grid = True
            grid_from_efd = True
            if verbose:
                print("  Using esp/esp_grid from EFD file (combined format)")
    except Exception as e:
        print(f"\n❌ Error loading data: {e}")
        return False
    
    n_samples = efd_data['R'].shape[0]

    # Normalize Z: pyscf-evaluate outputs Z as (n_atoms,) for same molecule; broadcast to (n_samples, n_atoms)
    Z_raw = efd_data['Z']
    if Z_raw.ndim == 1:
        Z_expanded = np.broadcast_to(Z_raw[np.newaxis, :], (n_samples, Z_raw.shape[0]))
    else:
        Z_expanded = Z_raw

    if verbose:
        print(f"\nLoaded {n_samples} samples")
        print(f"  Keys in EFD: {list(efd_data.keys())}")
        if has_grid:
            print(f"  Keys in Grid: {list(grid_data.keys())}")
        print("\nShapes:")
        for k in ['R', 'Z', 'E', 'F', 'Dxyz']:
            if k in efd_data:
                v = efd_data[k]
                print(f"  {k}: {v.shape}")
        if has_grid and grid_data:
            if 'esp' in grid_data:
                print(f"  esp: {grid_data['esp'].shape}")
            if 'vdw_surface' in grid_data:
                print(f"  esp_grid (vdw_surface): {grid_data['vdw_surface'].shape}")
        print("\nSummary statistics:")
        print(f"  E:  mean={efd_data['E'].mean():.6f}, std={efd_data['E'].std():.6f}, "
              f"range=[{efd_data['E'].min():.6f}, {efd_data['E'].max():.6f}]")
        f_norms = np.linalg.norm(efd_data['F'].reshape(-1, 3), axis=1)
        print(f"  F:  mean_norm={f_norms.mean():.6e}, max_norm={f_norms.max():.6e}")
        if 'Dxyz' in efd_data:
            d_norms = np.linalg.norm(efd_data['Dxyz'].reshape(-1, 3), axis=1)
            print(
                f"  Dxyz: mean_norm={d_norms.mean():.4f}, max_norm={d_norms.max():.4f} "
                f"({_unit_label_dipole(dipole_in)} per --dipole-in)"
            )
        if has_grid and 'esp' in (grid_data or {}):
            esp_flat = grid_data['esp'].flatten()
            valid_esp = esp_flat[np.abs(esp_flat) < 1e5]  # exclude padding
            if len(valid_esp) > 0:
                print(f"  esp: mean={valid_esp.mean():.6e}, range=[{valid_esp.min():.6e}, {valid_esp.max():.6e}]")

    # =========================================================================
    # Atomic coordinates
    # =========================================================================
    if verbose:
        print(f"\n{'#'*70}")
        print("# Step 2: Atomic Coordinates (R)")
        print(f"{'#'*70}")

    d_mean = _mean_shortest_interatomic_distance(efd_data["R"])
    if verbose and d_mean is not None:
        print(f"\nShortest interatomic distance (mean over samples): {d_mean:.4f} Å-equivalent scale")

    R_out, coords_effective_in, coords_detected = convert_coords_array(
        efd_data["R"], coords_in, coords_out, verbose=verbose
    )
    effective_coords_out = coords_effective_in if coords_out == "same" else coords_out
    if verbose:
        print(f"✓ R output units: {_unit_label_coords(effective_coords_out)}")

    # =========================================================================
    # Energies (optional atomic references, then unit conversion)
    # =========================================================================
    if verbose:
        print(f"\n{'#'*70}")
        print(f"# Step 3: Energies (E): {_unit_label_energy(energy_in)} → {_unit_label_energy(energy_out)}")
        print(f"{'#'*70}")

    E_work = np.asarray(efd_data["E"], dtype=np.float64).copy()
    if atomic_ref:
        if verbose:
            print(f"\nSubtracting atomic reference energies (scheme: {atomic_ref}, ref units: {atomic_ref_units})")
        try:
            expected_units = expected_atomic_ref_units(atomic_ref)
            if atomic_ref_units.lower() != expected_units and verbose:
                print(
                    f"  ⚠️  Scheme '{atomic_ref}' refs look like {expected_units} in JSON; "
                    f"you passed --atomic-ref-units {atomic_ref_units}"
                )
            unit_hint = diagnose_energy_unit_for_atomic_refs(
                E_work, Z_expanded, atomic_ref, atomic_ref_units, energy_in
            )
            if unit_hint:
                raise ValueError(unit_hint)
            E_before_ref = E_work.copy()
            if energy_in == "ev":
                E_ha = convert_energy_ev_to_hartree(E_work)
                E_ha = subtract_atomic_references(E_ha, Z_expanded, atomic_ref, ref_units=atomic_ref_units)
                check_atomic_ref_subtraction(
                    convert_energy_ev_to_hartree(E_before_ref),
                    E_ha,
                    scheme=atomic_ref,
                    ref_units=atomic_ref_units,
                    energy_in_declared=energy_in,
                    Z=Z_expanded,
                )
                E_work = convert_energy_hartree_to_ev(E_ha)
            else:
                E_work = subtract_atomic_references(E_work, Z_expanded, atomic_ref, ref_units=atomic_ref_units)
                check_atomic_ref_subtraction(
                    E_before_ref,
                    E_work,
                    scheme=atomic_ref,
                    ref_units=atomic_ref_units,
                    energy_in_declared=energy_in,
                    Z=Z_expanded,
                )
        except ValueError as e:
            print(f"\n❌ Atomic reference subtraction failed: {e}")
            return False
        if verbose:
            print(f"  E after ref subtraction: mean={E_work.mean():.6f}, range=[{E_work.min():.6f}, {E_work.max():.6f}]")

    E_out = convert_energy_array(E_work, energy_in, energy_out)
    if verbose:
        print(f"  Input ({_unit_label_energy(energy_in)}): mean={efd_data['E'].mean():.6f}")
        print(f"  Output ({_unit_label_energy(effective_energy_out)}): mean={E_out.mean():.6f}")
        if energy_out != "same" and energy_in != energy_out:
            print("✓ Energies converted")
        else:
            print("✓ Energies unchanged (per unit flags)")

    # =========================================================================
    # Forces
    # =========================================================================
    if verbose:
        print(f"\n{'#'*70}")
        print(f"# Step 4: Forces (F): {_unit_label_force(force_in)} → {_unit_label_force(force_out)}")
        print(f"{'#'*70}")

    F_work = np.asarray(efd_data["F"], dtype=np.float64).copy()
    if flip_forces:
        F_work = -F_work
        if verbose:
            print("  --flip-forces: negating F before unit conversion (gradient ∂E/∂R → force −∇E)")

    F_out = convert_force_array(F_work, force_in, force_out)
    if verbose:
        f_in_norms = np.linalg.norm(F_work.reshape(-1, 3), axis=1)[:10]
        f_out_norms = np.linalg.norm(F_out.reshape(-1, 3), axis=1)[:10]
        print(f"  Input mean |F| (sample): {f_in_norms.mean():.6e}")
        print(f"  Output mean |F| (sample): {f_out_norms.mean():.6e}")
        if force_out != "same" and force_in != force_out:
            print("✓ Forces converted")
        else:
            print("✓ Forces unchanged (per unit flags)")

    # =========================================================================
    # Dipoles
    # =========================================================================
    D_out: Optional[np.ndarray] = None
    if "Dxyz" in efd_data:
        D_out = convert_dipole_array(efd_data["Dxyz"], dipole_in, dipole_out)
        if verbose:
            d_norms_before = np.linalg.norm(efd_data["Dxyz"].reshape(-1, 3), axis=1)
            d_norms_after = np.linalg.norm(D_out.reshape(-1, 3), axis=1)
            print(f"\n{'#'*70}")
            print(
                f"# Step 4b: Dipoles (Dxyz): {_unit_label_dipole(dipole_in)} → "
                f"{_unit_label_dipole(dipole_out)}"
            )
            print(f"{'#'*70}")
            print(f"  Input:  mean |D|={d_norms_before.mean():.4f}, max={d_norms_before.max():.4f}")
            print(f"  Output: mean |D|={d_norms_after.mean():.4f}, max={d_norms_after.max():.4f}")
            if dipole_out != "same" and dipole_in != dipole_out:
                print("✓ Dipoles converted")
            else:
                print("✓ Dipoles unchanged (per unit flags)")

    # =========================================================================
    # Optional extra scales (after standard conversions; before validation / ESP)
    # =========================================================================
    es = float(energy_scale)
    fs = float(force_scale)
    ds = float(dipole_scale)
    if es != 1.0:
        E_out = E_out * es
    if fs != 1.0:
        F_out = F_out * fs
    if D_out is not None and ds != 1.0:
        D_out = D_out * ds
    ef_s = float(efield_scale)
    esp_s = float(esp_scale)
    q_s = float(charge_scale)
    if verbose and (es != 1.0 or fs != 1.0 or ds != 1.0 or ef_s != 1.0 or esp_s != 1.0 or q_s != 1.0):
        print(f"\n{'#'*70}")
        print("# Extra property scales (applied after standard unit conversion)")
        print(f"{'#'*70}")
        if es != 1.0:
            print(f"  E:     × {es}  ({_unit_label_energy(effective_energy_out)}); also efield_energy, efield_scf_energy when present")
        if fs != 1.0:
            print(f"  F:     × {fs}  ({_unit_label_force(effective_force_out)}); also efield_scf_F when present")
        if D_out is not None and ds != 1.0:
            print(f"  Dxyz:  × {ds}  ({_unit_label_dipole(effective_dipole_out)}); also D, efield_scf_D, efield_D when present")
        elif ds != 1.0:
            print(f"  D/efield dipoles: × {ds}  (Debye vectors, if present)")
        if ef_s != 1.0:
            print(f"  Ef:    × {ef_s}  (Ef, efield_Ef, efield_scf_Ef)")
        if esp_s != 1.0:
            print(f"  esp:   × {esp_s}  (Hartree/e on grid and on EFD if present)")
        if q_s != 1.0:
            print(f"  Q:     × {q_s}  (NPZ key Q; use for total charge—PySCF may use Q for quadrupole)")

    # =========================================================================
    # ESP grid coordinates (if grid data exists)
    # =========================================================================
    vdw_surface_out: Optional[np.ndarray] = None
    grid_coords_effective_in: Optional[str] = None
    grid_conversion_notes: List[str] = []

    if has_grid and grid_data is not None:
        if verbose:
            print(f"\n{'#'*70}")
            print(
                f"# Step 5: ESP grid coordinates: {grid_coords_in} → {grid_coords_out}"
            )
            print(f"{'#'*70}")
        try:
            vdw_surface_out, grid_coords_effective_in, grid_conversion_notes = convert_grid_surface_array(
                grid_data,
                grid_coords_in,
                grid_coords_out,
                cube_spacing_bohr=cube_spacing_bohr,
                verbose=verbose,
            )
            manifest_notes.extend(grid_conversion_notes)
            if grid_coords_out != "same" and grid_coords_effective_in != effective_grid_coords_out:
                if verbose:
                    print(f"✓ Grid output units: {_unit_label_coords(effective_grid_coords_out)}")
            elif verbose:
                print(f"✓ Grid coordinates unchanged ({grid_coords_effective_in})")
        except Exception as e:
            print(f"\n❌ Grid unit conversion failed: {e}")
            return False
    elif verbose:
        print(f"\n{'#'*70}")
        print("# Step 5: ESP Grid (skipped - no grid data provided)")
        print(f"{'#'*70}")

    if grid_coords_out == "same" and grid_coords_effective_in is not None:
        effective_grid_coords_out = grid_coords_effective_in
    
    # =========================================================================
    # Reduce ESP grid to fixed number of points (if grid exists)
    # =========================================================================
    if has_grid and effective_coords_out != "angstrom" and not skip_validation:
        manifest_notes.append(
            "ESP grid filtering (--min-dist-to-atoms, etc.) assumes R in Å; "
            f"output R is in {effective_coords_out}"
        )
        if verbose:
            print(
                "\n⚠️  R is not in Angstrom: ESP distance filters use Å by default; "
                "adjust --min-dist-to-atoms or convert coordinates."
            )

    if has_grid and vdw_surface_out is not None:
        esp_raw = grid_data['esp']
        # Sanity check: esp and grid must have matching shapes (esp[i,j] pairs with grid[i,j])
        if esp_raw.shape[0] != vdw_surface_out.shape[0]:
            raise ValueError(
                f"ESP/grid sample count mismatch: esp {esp_raw.shape[0]} vs grid {vdw_surface_out.shape[0]}. "
                "Check that EFD and grid files have the same samples in the same order."
            )
        if esp_raw.shape[1] != vdw_surface_out.shape[1]:
            raise ValueError(
                f"ESP/grid point count mismatch: esp {esp_raw.shape[1]} vs grid {vdw_surface_out.shape[1]}. "
                "esp[i,j] must correspond to grid[i,j] for correct pairing."
            )
        # Check index alignment only for separate grid files (combined EFD format is trusted)
        if not grid_from_efd:
            align_ok_raw, align_corr_raw = check_esp_grid_alignment(
                esp_raw, vdw_surface_out, R_out, n_check=min(5, esp_raw.shape[0])
            )
            if verbose:
                print(f"  ESP-grid alignment (raw): correlation={align_corr_raw:.3f} {'✓' if align_ok_raw else '⚠️'}")
            if not align_ok_raw:
                print("  ⚠️  WARNING: Low esp-grid correlation suggests index misalignment. "
                      "esp[i,j] may not correspond to grid[i,j]. Check that esp and grid come from the same source.")
        if verbose:
            print(f"\n{'#'*70}")
            print("# Step 5b: Reducing ESP Grid")
            print(f"{'#'*70}")
            print(f"  Target points: {n_grid_points}")
            print(f"  Exclude points beyond ±{esp_sd_sigma} SD from mean")
            print(f"  Exclude |esp| > {esp_max_abs_kcal_mol} kcal/mol/e")
            print(f"  Exclude points < {min_dist_to_atoms} Å from atoms")
        esp_reduced, grid_reduced = reduce_esp_grid(
            esp_raw,
            vdw_surface_out,
            R_out,
            n_grid_points=n_grid_points,
            esp_sd_sigma=esp_sd_sigma,
            esp_max_abs_kcal_mol=esp_max_abs_kcal_mol,
            min_dist_to_atoms=min_dist_to_atoms,
            seed=seed,
        )
        # Verify reduction preserved esp-grid alignment (same indices used for both)
        reduction_ok = verify_reduction_preserves_alignment(
            esp_raw, vdw_surface_out, esp_reduced, grid_reduced, R_out,
            n_grid_points=n_grid_points, esp_sd_sigma=esp_sd_sigma,
            esp_max_abs_kcal_mol=esp_max_abs_kcal_mol, min_dist_to_atoms=min_dist_to_atoms,
            seed=seed, n_spot_check=3,
        )
        if not reduction_ok:
            raise RuntimeError(
                "reduce_esp_grid verification failed: esp and grid output pairs do not match input. "
                "This indicates a bug in the reduction logic."
            )
        if verbose:
            print("  ✓ Reduction preserves esp-grid alignment (verified)")
        vdw_surface_out = grid_reduced
        if verbose:
            n_valid_per_sample = np.sum(np.all(np.abs(grid_reduced) < 1e5, axis=2), axis=1)
            print(f"  Reduced to shape: esp {esp_reduced.shape}, grid {grid_reduced.shape}")
            print(f"  Valid points per sample: min={n_valid_per_sample.min()}, max={n_valid_per_sample.max()}, mean={n_valid_per_sample.mean():.0f}")
        # Verify esp-grid alignment only for separate grid files
        if not grid_from_efd:
            align_ok, align_corr = check_esp_grid_alignment(esp_reduced, grid_reduced, R_out, n_check=5)
            if verbose:
                print(f"  ESP-grid alignment check: correlation={align_corr:.3f} {'✓' if align_ok else '⚠️'}")
            if not align_ok and verbose:
                print("  ⚠️  Low esp-grid correlation may indicate index misalignment. "
                      "Ensure esp and grid come from the same source with matching point order.")
        # Update grid_data for saving - we need esp and vdw_surface
        grid_data = dict(grid_data)
        grid_data['esp'] = esp_reduced
        grid_data['vdw_surface'] = grid_reduced
        grid_data['esp_grid'] = grid_reduced
    
    # =========================================================================
    # Validate fixed data
    # =========================================================================
    if not skip_validation:
        if verbose:
            print(f"\n{'#'*70}")
            print("# Step 6: Validating Fixed Data")
            print(f"{'#'*70}")
        
        is_valid = validate_fixed_data(
            R_out,
            E_out,
            F_out,
            vdw_surface_out,
            Z_expanded,
            efd_data['N'],
            has_grid=has_grid,
            verbose=verbose,
            coords_unit=effective_coords_out,
            energy_unit=effective_energy_out,
            force_unit=effective_force_out,
        )
        
        if not is_valid:
            print("\n❌ Validation failed! Not proceeding with splits.")
            return False
    
    # =========================================================================
    # Create splits
    # =========================================================================
    if verbose:
        print(f"\n{'#'*70}")
        print("# Step 7: Creating Train/Valid/Test Splits")
        print(f"{'#'*70}")
    
    splits = create_splits(n_samples, train_frac=train_frac, valid_frac=valid_frac, 
                          test_frac=test_frac, seed=seed)
    
    if verbose:
        print(f"\nTotal samples: {n_samples}")
        print(f"  Train: {len(splits['train'])} ({len(splits['train'])/n_samples*100:.1f}%)")
        print(f"  Valid: {len(splits['valid'])} ({len(splits['valid'])/n_samples*100:.1f}%)")
        print(f"  Test:  {len(splits['test'])} ({len(splits['test'])/n_samples*100:.1f}%)")

    energy_zscale_stats: Optional[Dict[str, Any]] = None
    if zscale_energies:
        train_E = np.asarray(E_out[splits['train']], dtype=np.float64)
        if train_E.size == 0:
            raise ValueError("Cannot Z-scale energies: training split is empty.")
        energy_mean = float(np.mean(train_E))
        energy_std = float(np.std(train_E))
        if not np.isfinite(energy_std) or energy_std == 0.0:
            raise ValueError(
                f"Cannot Z-scale energies: training energy std must be finite and non-zero, got {energy_std}."
            )
        E_out = (np.asarray(E_out, dtype=np.float64) - energy_mean) / energy_std
        energy_zscale_stats = {
            "enabled": True,
            "property": "E",
            "mean": energy_mean,
            "std": energy_std,
            "units_before_zscale": effective_energy_out,
            "train_samples": int(train_E.size),
            "std_ddof": 0,
        }
        if verbose:
            print("\nEnergy Z-scaling enabled:")
            print(f"  mean_train = {energy_mean:.12g} {_unit_label_energy(effective_energy_out)}")
            print(f"  std_train  = {energy_std:.12g} {_unit_label_energy(effective_energy_out)}")
    
    # =========================================================================
    # Prepare datasets with fixed units
    # =========================================================================
    if verbose:
        print(f"\n{'#'*70}")
        print("# Step 8: Preparing Fixed Datasets")
        print(f"{'#'*70}")
    
    # Update EFD data with fixed/converted values
    efd_fixed = efd_data.copy()
    efd_fixed['R'] = R_out
    efd_fixed['E'] = E_out
    efd_fixed['F'] = F_out
    if D_out is not None:
        efd_fixed['Dxyz'] = D_out
    # PhysNet expects N (n_samples,) and Z (n_samples, n_atoms); pyscf-evaluate outputs scalar N and 1D Z
    N_raw = efd_data['N']
    if (np.isscalar(N_raw) or (isinstance(N_raw, np.ndarray) and N_raw.size == 1) or
            (isinstance(N_raw, np.ndarray) and N_raw.shape[0] != n_samples)):
        n_atoms = int(np.asarray(N_raw).flat[0])
        efd_fixed['N'] = np.full(n_samples, n_atoms, dtype=np.int32)
    Z_raw = efd_fixed['Z']
    if Z_raw.ndim == 1:
        efd_fixed['Z'] = np.broadcast_to(Z_raw[np.newaxis, :], (n_samples, Z_raw.shape[0]))
    efd_fixed['Z'] = npz_z_array_to_atomic_numbers(np.asarray(efd_fixed['Z']))

    for key in ("efield_energy", "efield_scf_energy"):
        if key in efd_fixed:
            efd_fixed[key] = convert_energy_array(efd_fixed[key], energy_in, energy_out)
    if "efield_scf_F" in efd_fixed:
        efd_fixed["efield_scf_F"] = convert_force_array(
            efd_fixed["efield_scf_F"], force_in, force_out
        )
    for key in ("efield_scf_D", "efield_D", "D"):
        if key in efd_fixed:
            efd_fixed[key] = convert_dipole_array(efd_fixed[key], dipole_in, dipole_out)

    # Update grid data with fixed coordinates (if grid exists)
    grid_fixed = None
    if has_grid and grid_data is not None:
        grid_fixed = grid_data.copy()
        grid_fixed['R'] = R_out
        if vdw_surface_out is not None:
            grid_fixed['vdw_surface'] = vdw_surface_out
            grid_fixed['vdw_grid'] = vdw_surface_out  # Backward compatibility
        if D_out is not None and 'Dxyz' in grid_fixed:
            grid_fixed['Dxyz'] = D_out
        # Align Z and N with EFD (per-sample shapes for PhysNet)
        if 'N' in grid_fixed:
            N_g = grid_fixed['N']
            if (np.isscalar(N_g) or (isinstance(N_g, np.ndarray) and N_g.size == 1) or
                    (isinstance(N_g, np.ndarray) and N_g.ndim == 0) or
                    (isinstance(N_g, np.ndarray) and N_g.shape[0] != n_samples)):
                n_a = int(np.asarray(N_g).flat[0]) if np.asarray(N_g).size else R_out.shape[1]
                grid_fixed['N'] = np.full(n_samples, n_a, dtype=np.int32)
        if 'Z' in grid_fixed:
            Z_g = grid_fixed['Z']
            if Z_g.ndim == 1:
                grid_fixed['Z'] = np.broadcast_to(Z_g[np.newaxis, :], (n_samples, Z_g.shape[0]))
            grid_fixed['Z'] = npz_z_array_to_atomic_numbers(np.asarray(grid_fixed['Z']))

    # Same extra factors on auxiliary NPZ keys (pass-through arrays not unit-converted here)
    aux_scaled: List[str] = []
    aux_scaled.extend(_scale_ndarrays_in_dict(efd_fixed, ("Ef", "efield_Ef", "efield_scf_Ef"), efield_scale))
    aux_scaled.extend(_scale_ndarrays_in_dict(efd_fixed, ("efield_energy", "efield_scf_energy"), energy_scale))
    aux_scaled.extend(_scale_ndarrays_in_dict(efd_fixed, ("efield_scf_F",), force_scale))
    aux_scaled.extend(_scale_ndarrays_in_dict(efd_fixed, ("efield_scf_D", "efield_D", "D"), dipole_scale))
    aux_scaled.extend(_scale_ndarrays_in_dict(efd_fixed, ("Q",), charge_scale))
    aux_scaled.extend(_scale_ndarrays_in_dict(efd_fixed, ("esp",), esp_scale))
    if grid_fixed is not None:
        aux_scaled.extend(_scale_ndarrays_in_dict(grid_fixed, ("esp",), esp_scale))
    if verbose and aux_scaled:
        print(f"\n  Auxiliary/grid arrays scaled: {', '.join(dict.fromkeys(aux_scaled))}")

    units_manifest = UnitsManifest(
        coords_in=coords_in,
        coords_out=effective_coords_out,
        coords_detected=coords_detected,
        energy_in=energy_in,
        energy_out=effective_energy_out,
        force_in=force_in,
        force_out=effective_force_out,
        dipole_in=dipole_in if "Dxyz" in efd_data else None,
        dipole_out=effective_dipole_out if "Dxyz" in efd_data else None,
        grid_coords_in=grid_coords_in if has_grid else None,
        grid_coords_out=effective_grid_coords_out if has_grid else None,
        esp_values="hartree/e",
        flip_forces=flip_forces,
        preserve_units=preserve_units,
        notes=manifest_notes,
    )
    from mmml.data.units import UnitsManifestV2

    manifest_v2 = UnitsManifestV2.from_dict(asdict(units_manifest))
    manifest_v2.schema_version = 2
    manifest_v2.canonical = {
        "energy": "ev",
        "force": "ev_angstrom",
        "coords": "angstrom",
    }
    manifest_payload = manifest_v2.to_dict()
    units_embed = np.array(json.dumps(manifest_v2.arrays))

    # =========================================================================
    # Save split datasets
    # =========================================================================
    if verbose:
        print(f"\n{'#'*70}")
        print("# Step 9: Saving Split Datasets")
        print(f"{'#'*70}")
    
    def _index_if_sample_dim(v, indices):
        """Index array by split if it has n_samples in first dim; else pass through."""
        if not isinstance(v, np.ndarray):
            return v
        if v.ndim == 0:
            return v
        if v.shape[0] == n_samples:
            return v[indices]
        return v

    for split_name, split_indices in splits.items():
        if verbose:
            print(f"\nSaving {split_name} split ({len(split_indices)} samples)...")
        
        # Create EFD split
        efd_split = {k: _index_if_sample_dim(v, split_indices) for k, v in efd_fixed.items()}
        efd_split["_mmml_units"] = units_embed
        efd_out = output_dir / f"energies_forces_dipoles_{split_name}.npz"
        np.savez_compressed(efd_out, **efd_split)
        
        if verbose:
            size_mb = efd_out.stat().st_size / 1024 / 1024
            print(f"  ✓ {efd_out.name} ({size_mb:.1f} MB)")
        
        # Create grid split (only if grid data exists)
        if has_grid and grid_fixed is not None:
            grid_split = {k: _index_if_sample_dim(v, split_indices) for k, v in grid_fixed.items()}
            grid_split["_mmml_units"] = units_embed
            grid_out = output_dir / f"grids_esp_{split_name}.npz"
            np.savez_compressed(grid_out, **grid_split)
            
            if verbose:
                size_mb = grid_out.stat().st_size / 1024 / 1024
                print(f"  ✓ {grid_out.name} ({size_mb:.1f} MB)")
    
    # Save split indices
    indices_out = output_dir / "split_indices.npz"
    np.savez(indices_out, **splits)
    if verbose:
        print(f"\n✓ Split indices saved to {indices_out.name}")

    if energy_zscale_stats is not None:
        stats_out = output_dir / "energy_zscale_stats.json"
        with open(stats_out, 'w') as f:
            json.dump(energy_zscale_stats, f, indent=2)
            f.write("\n")
        if verbose:
            print(f"✓ Energy Z-scale stats saved to {stats_out.name}")

    manifest_out = output_dir / "units_manifest.json"
    with open(manifest_out, "w") as f:
        json.dump(manifest_payload, f, indent=2)
        f.write("\n")
    if verbose:
        print(f"✓ Units manifest saved to {manifest_out.name}")

    # =========================================================================
    # Create documentation
    # =========================================================================
    if verbose:
        print(f"\n{'#'*70}")
        print("# Step 10: Creating Documentation")
        print(f"{'#'*70}")
    
    def _label_for(prop: str, unit: str) -> str:
        if prop == "R":
            return _unit_label_coords(unit)
        if prop == "E":
            return _unit_label_energy(unit)
        if prop == "F":
            return _unit_label_force(unit)
        return _unit_label_dipole(unit)

    def _conv_line(name: str, u_in: str, u_out: str) -> str:
        if u_in == u_out:
            return f"- **{name}**: {_label_for(name, u_in)} (unchanged)"
        return f"- **{name}**: {_label_for(name, u_in)} → {_label_for(name, u_out)}"

    grid_section = ""
    if has_grid:
        grid_section = f"""
### 4. ESP Grid Coordinates (vdw_surface / vdw_grid)
- **Input policy**: `{grid_coords_in}` (effective: `{grid_coords_effective_in or 'n/a'}`)
- **Output units**: {_unit_label_coords(effective_grid_coords_out)}
- See `units_manifest.json` for full conversion log
"""

    forces_sign_readme = ""
    if flip_forces:
        forces_sign_readme = (
            "- **Sign**: `F` was negated (`--flip-forces`) so output is force −∇E, "
            "not raw gradient ∂E/∂R.\n"
        )

    extra_scales_readme = ""
    _es_r, _fs_r, _ds_r = float(energy_scale), float(force_scale), float(dipole_scale)
    _ef_r, _esp_r, _q_r = float(efield_scale), float(esp_scale), float(charge_scale)
    _scale_lines = []
    if _es_r != 1.0:
        _scale_lines.append(
            f"- **E**: × {_es_r:g} after Hartree→eV; same factor on `efield_energy`, `efield_scf_energy` if present"
        )
    if _fs_r != 1.0:
        _scale_lines.append(
            f"- **F**: × {_fs_r:g} after Ha/Bohr→eV/Å; same factor on `efield_scf_F` if present"
        )
    if _ds_r != 1.0:
        _scale_lines.append(
            f"- **Dxyz**: × {_ds_r:g} after Debye→e·Å; same factor on `D`, `efield_scf_D`, `efield_D` if present"
        )
    if _ef_r != 1.0:
        _scale_lines.append(f"- **Ef** (field vectors): × {_ef_r:g} on `Ef`, `efield_Ef`, `efield_scf_Ef` if present")
    if _esp_r != 1.0:
        _scale_lines.append(f"- **esp**: × {_esp_r:g} [Hartree/e] on grid NPZ and EFD NPZ if present")
    if _q_r != 1.0:
        _scale_lines.append(
            f"- **Q** (NPZ key): × {_q_r:g} — for total charge in your convention; PySCF may use Q for quadrupole"
        )
    if _scale_lines:
        extra_scales_readme = "\n### 3b. Extra property scales\n" + "\n".join(_scale_lines) + "\n"

    energy_zscale_readme = ""
    energy_unit_label = effective_energy_out
    energy_file_note = f"Energies [{_unit_label_energy(effective_energy_out)}]"
    if energy_in != effective_energy_out:
        energy_file_note += f" ← converted from {_unit_label_energy(energy_in)}"
    energy_usage_comment = _unit_label_energy(effective_energy_out)
    energy_usage_intro = (
        "Units match `units_manifest.json` (default pipeline targets ASE-style training units)."
        if not preserve_units
        else "Units preserved from input (`--preserve-units`); see `units_manifest.json`."
    )
    if energy_zscale_stats is not None:
        energy_unit_label = "dimensionless"
        energy_file_note = (
            f"Energies [Z-scaled] ← (E in {_unit_label_energy(effective_energy_out)} - train_mean) / train_std"
        )
        energy_usage_comment = "Z-scaled, dimensionless"
        energy_usage_intro = "E is Z-scaled; see `units_manifest.json` for other properties."
        energy_zscale_readme = (
            "\n### 3c. Energy Z-scaling\n"
            "- **Applied**: `E = (E - mean_train) / std_train`\n"
            f"- **Training mean**: {energy_zscale_stats['mean']:.12g} {_unit_label_energy(effective_energy_out)}\n"
            f"- **Training std**: {energy_zscale_stats['std']:.12g} {_unit_label_energy(effective_energy_out)}\n"
            "- **Stats file**: `energy_zscale_stats.json`\n"
        )

    dipole_file_line = ""
    if "Dxyz" in efd_data:
        dipole_file_line = (
            f"- `Dxyz`: Dipole moments [{_unit_label_dipole(effective_dipole_out)}]"
        )
        if dipole_in != effective_dipole_out:
            dipole_file_line += f" ← converted from {_unit_label_dipole(dipole_in)}"
        dipole_file_line += "\n"

    readme_content = f"""# Training Data (Unit-Corrected)

This directory contains molecular data prepared for DCMnet/PhysnetJax training.
**Always read `units_manifest.json`** before training or evaluation.

## Unit conversions applied

{_conv_line('R', coords_effective_in if coords_in == 'auto' else coords_in, effective_coords_out)}
{_conv_line('E', energy_in, effective_energy_out)}
{_conv_line('F', force_in, effective_force_out)}
{f"{_conv_line('Dxyz', dipole_in, effective_dipole_out)}" if "Dxyz" in efd_data else ""}
{forces_sign_readme}{extra_scales_readme}{energy_zscale_readme}{grid_section}
## Data Splits

- **Train**: {len(splits['train'])} samples ({train_frac*100:.0f}%)
- **Valid**: {len(splits['valid'])} samples ({valid_frac*100:.0f}%)
- **Test**: {len(splits['test'])} samples ({test_frac*100:.0f}%)
- **Seed**: {seed} (reproducible)

## Files

### Energy, Forces, and Dipoles
- `energies_forces_dipoles_train.npz`
- `energies_forces_dipoles_valid.npz`
- `energies_forces_dipoles_test.npz`

Each contains:
- `R`: Atomic coordinates [{_unit_label_coords(effective_coords_out)}]
- `Z`: Atomic numbers [int]
- `N`: Number of atoms [int]
- `E`: {energy_file_note}
- `F`: Forces [{_unit_label_force(effective_force_out)}]
{dipole_file_line}"""
    
    if has_grid:
        readme_content += """
### ESP Grids
- `grids_esp_train.npz`
- `grids_esp_valid.npz`
- `grids_esp_test.npz`

Each contains:
- `R`: Atomic coordinates [{_unit_label_coords(effective_coords_out)}]
- `Z`: Atomic numbers [int]
- `N`: Number of atoms [int]
- `esp`: ESP values [Hartree/e]
- `vdw_surface`: Grid coordinates [{_unit_label_coords(effective_grid_coords_out)}]
- `vdw_grid`: Same as vdw_surface (backward compatibility)
- `grid_dims`: Original cube dimensions (if available)
- `grid_origin`: Original cube origins [Bohr] (if available)
- `grid_axes`: Original cube axes (if available)
{dipole_file_line if "Dxyz" in efd_data else ""}"""
    
    readme_content += f"""
## Units Summary

| Property | Unit | Notes |
|----------|------|-------|
| R (coordinates) | {_unit_label_coords(effective_coords_out)} | see manifest |
| E (energy) | {energy_unit_label} | {'Z-scaled' if energy_zscale_stats is not None else 'see manifest'} |
| F (forces) | {_unit_label_force(effective_force_out)} | see manifest |
"""
    
    if "Dxyz" in efd_data:
        readme_content += f"| Dxyz (dipoles) | {_unit_label_dipole(effective_dipole_out)} | see manifest |\n"

    if has_grid:
        readme_content += f"""| esp (values) | Hartree/e | unchanged |
| vdw_surface | {_unit_label_coords(effective_grid_coords_out)} | see manifest |
"""
    
    readme_content += f"""
## Usage

```python
import numpy as np

# Load training data
train_props = np.load('energies_forces_dipoles_train.npz')

# {energy_usage_intro}
R = train_props['R']  # {_unit_label_coords(effective_coords_out)}
E = train_props['E']  # {energy_usage_comment}
F = train_props['F']  # {_unit_label_force(effective_force_out)}
# Dxyz: see units_manifest.json
"""
    
    if has_grid:
        readme_content += """
# Load grid data (if available)
train_grids = np.load('grids_esp_train.npz')
esp = train_grids['esp']  # Hartree/e
vdw_surface = train_grids['vdw_surface']  # Angstroms
"""
    
    readme_content += """
Generated by: mmml.cli.fix_and_split
"""
    
    readme_path = output_dir / "README.md"
    with open(readme_path, 'w') as f:
        f.write(readme_content)
    
    if verbose:
        print(f"✓ Created {readme_path.name}")
    
    # =========================================================================
    # Final summary
    # =========================================================================
    if verbose:
        print(f"\n{'='*70}")
        print("✅ DATA PREPARATION COMPLETE!")
        print(f"{'='*70}")
        print(f"\nOutput files in: {output_dir}")
        print("\nTrain/Valid/Test splits:")
        print("  - energies_forces_dipoles_{train,valid,test}.npz")
        if has_grid:
            print("  - grids_esp_{train,valid,test}.npz")
        print("  - split_indices.npz")
        if energy_zscale_stats is not None:
            print("  - energy_zscale_stats.json")
        print("  - README.md")
        if energy_zscale_stats is not None:
            print("\n✅ IMPORTANT: Outputs are ready for training with Z-scaled energies!")
            print("   - Energies: Z-scaled from eV using training-set mean/std")
        else:
            print("\n✅ IMPORTANT: All units are now ASE-standard compliant!")
            print("   - Energies: eV (converted from Hartree)")
        print("   - Forces: eV/Angstrom (converted from Hartree/Bohr)")
        print("   - Coordinates: Angstrom")
        if has_grid:
            print("   - ESP grid: Angstrom (converted from grid indices)")
        print("\nArray shapes (per split):")
        for split_name in splits:
            efd_path = output_dir / f"energies_forces_dipoles_{split_name}.npz"
            with np.load(efd_path, allow_pickle=True) as f:
                for k in sorted(f.keys()):
                    v = f[k]
                    sh = v.shape if hasattr(v, 'shape') else 'scalar'
                    print(f"  {split_name}: {k} {sh}")
            if has_grid:
                grid_path = output_dir / f"grids_esp_{split_name}.npz"
                with np.load(grid_path, allow_pickle=True) as g:
                    for k in sorted(g.keys()):
                        v = g[k]
                        sh = v.shape if hasattr(v, 'shape') else 'scalar'
                        print(f"  {split_name} (grid): {k} {sh}")
        print(f"{'='*70}\n")
    
    return True


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Fix units and create train/valid/test splits from molecular NPZ data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage with default 8:1:1 split (with grid)
  %(prog)s --efd data.npz --grid grids.npz --output-dir ./training_data
  
  # Without grid data (EFD only)
  %(prog)s --efd data.npz --output-dir ./training_data
  
  # Custom split ratios
  %(prog)s --efd data.npz --grid grids.npz --output-dir ./training_data \\
    --train-frac 0.7 --valid-frac 0.15 --test-frac 0.15
  
  # Different cube spacing (e.g., 0.5 Bohr)
  %(prog)s --efd data.npz --grid grids.npz --output-dir ./training_data \\
    --cube-spacing 0.5
  
  # Skip validation for speed
  %(prog)s --efd data.npz --output-dir ./training_data \\
    --skip-validation

  # Concatenate multiple NPZ files (e.g. extend training set with MD samples)
  %(prog)s --efd train.npz md_evaluated.npz --output-dir ./splits_extended

  # NPZ has raw PySCF gradient in F (not forces): negate before converting to eV/Å
  %(prog)s --efd raw.npz --output-dir ./out --flip-forces

  # Correct a systematic factor after normal conversion (e.g. duplicate unit fix upstream)
  %(prog)s --efd data.npz --output-dir ./out --energy-scale 0.5 --force-scale 1.0

  # Z-scale energies with training-set statistics and save the mean/std
  %(prog)s --efd data.npz --output-dir ./out --zscale-energies

  # Already in training units (eV, eV/Å, e·Å, Å): split only
  %(prog)s --efd data.npz -o ./splits --preserve-units

  # Explicit: PySCF Hartree/Bohr in, ASE units out (same as default)
  %(prog)s --efd pyscf.npz -o ./out \\
    --energy-in hartree --energy-out ev \\
    --force-in hartree-bohr --force-out ev-angstrom \\
    --dipole-in debye --dipole-out e-angstrom
"""
    )

    units_group = parser.add_argument_group(
        "Unit conversion",
        "Declare units in the NPZ files. Defaults assume PySCF/atomic units on input "
        "and ASE-style training units on output. Use --preserve-units or *-out same "
        "to avoid converting fields that are already correct.",
    )
    units_group.add_argument(
        "--coords-in",
        choices=["auto", "bohr", "angstrom"],
        default="auto",
        help="Units of R in the input NPZ (default: auto = infer from bond lengths)",
    )
    units_group.add_argument(
        "--coords-out",
        choices=["angstrom", "bohr", "same"],
        default="angstrom",
        help="Units of R in output NPZ (default: angstrom; same = no conversion)",
    )
    units_group.add_argument(
        "--energy-in",
        choices=["hartree", "ev"],
        default="hartree",
        help="Units of E in the input NPZ (default: hartree)",
    )
    units_group.add_argument(
        "--energy-out",
        choices=["ev", "hartree", "same"],
        default="ev",
        help="Units of E in output NPZ (default: ev; same = no conversion)",
    )
    units_group.add_argument(
        "--force-in",
        choices=["hartree-bohr", "ev-angstrom"],
        default="hartree-bohr",
        dest="force_in",
        help="Units of F in the input NPZ (default: hartree-bohr)",
    )
    units_group.add_argument(
        "--force-out",
        choices=["ev-angstrom", "hartree-bohr", "same"],
        default="ev-angstrom",
        dest="force_out",
        help="Units of F in output NPZ (default: ev-angstrom; same = no conversion)",
    )
    units_group.add_argument(
        "--dipole-in",
        choices=["debye", "e-angstrom"],
        default="debye",
        dest="dipole_in",
        help="Units of Dxyz in the input NPZ (default: debye)",
    )
    units_group.add_argument(
        "--dipole-out",
        choices=["e-angstrom", "debye", "same"],
        default="e-angstrom",
        dest="dipole_out",
        help="Units of Dxyz in output NPZ (default: e-angstrom; same = no conversion)",
    )
    units_group.add_argument(
        "--grid-coords-in",
        choices=["auto", "bohr", "angstrom", "index"],
        default="auto",
        help="Units of vdw_surface/vdw_grid/esp_grid (default: auto)",
    )
    units_group.add_argument(
        "--grid-coords-out",
        choices=["angstrom", "bohr", "same"],
        default="angstrom",
        help="Grid coordinate units in output NPZ (default: angstrom; same = no conversion)",
    )
    units_group.add_argument(
        "--preserve-units",
        action="store_true",
        help="Do not convert R, E, F, Dxyz, or grid coordinates (sets all *-out to same)",
    )
    
    parser.add_argument(
        '--efd', '--energies-forces-dipoles',
        type=Path,
        nargs='+',
        required=True,
        metavar='FILE',
        help='Path(s) to energies_forces_dipoles.npz file(s). Multiple files are concatenated.'
    )
    
    parser.add_argument(
        '--grid', '--grids-esp',
        type=Path,
        default=None,
        help='Path to grids_esp.npz file (optional)'
    )
    
    parser.add_argument(
        '--output-dir', '-o',
        type=Path,
        required=True,
        help='Directory to save output files'
    )
    
    parser.add_argument(
        '--train-frac',
        type=float,
        default=0.8,
        help='Fraction of data for training (default: 0.8)'
    )
    
    parser.add_argument(
        '--valid-frac',
        type=float,
        default=0.1,
        help='Fraction of data for validation (default: 0.1)'
    )
    
    parser.add_argument(
        '--test-frac',
        type=float,
        default=0.1,
        help='Fraction of data for testing (default: 0.1)'
    )
    
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for reproducible splits (default: 42)'
    )
    
    parser.add_argument(
        '--cube-spacing',
        type=float,
        default=0.25,
        help='Grid spacing in Bohr from original cube files (default: 0.25)'
    )
    
    parser.add_argument(
        '--skip-validation',
        action='store_true',
        help='Skip validation checks'
    )
    
    parser.add_argument(
        '--atomic-ref',
        type=str,
        default=None,
        metavar='SCHEME',
        help='Subtract per-atom reference energies (e.g. pbe0/sz, pbe0/def2-tzvp)'
    )
    parser.add_argument(
        '--atomic-ref-units',
        type=str,
        choices=['hartree', 'ev'],
        default='hartree',
        help='Units of refs in JSON: hartree (pbe0/def2-tzvp) or ev (pbe0/sz may use eV; default: hartree)'
    )
    
    parser.add_argument(
        '--n-grid-points',
        type=int,
        default=3000,
        metavar='N',
        help='Target number of ESP grid points per sample (default 3000). Excludes tails (±SD) and points near atoms.'
    )
    parser.add_argument(
        '--esp-sd-sigma',
        type=float,
        default=3.0,
        metavar='N',
        help='Exclude grid points beyond ±N SD from mean (default 3.0, ignores distribution tails)'
    )
    parser.add_argument(
        '--esp-max-abs-kcal-mol',
        type=float,
        default=100.0,
        metavar='X',
        help='Exclude grid points with |esp| > X kcal/mol/e (default 100.0)'
    )
    parser.add_argument(
        '--min-dist-to-atoms',
        type=float,
        default=1.0,
        metavar='Å',
        help='Exclude grid points closer than this to any atom in Å (default 1.0)'
    )
    
    parser.add_argument(
        '--quiet', '-q',
        action='store_true',
        help='Suppress detailed output'
    )

    parser.add_argument(
        '--flip-forces',
        action='store_true',
        help=(
            'Negate F before unit conversion (Ha/Bohr → eV/Å). Use when F stores the PySCF energy '
            'gradient ∂E/∂R instead of forces F = −∇E. mmml pyscf-evaluate NPZ already uses −gradient.'
        ),
    )

    parser.add_argument(
        '--energy-scale',
        type=float,
        default=1.0,
        metavar='X',
        help='Multiply E by X after Hartree→eV. Also efield_energy, efield_scf_energy if present (default 1.0).',
    )
    parser.add_argument(
        '--zscale-energies', '--z-scale-energies',
        action='store_true',
        help=(
            'Z-scale E after creating splits using training-set mean/std: '
            'E = (E_eV - mean_train) / std_train. Saves energy_zscale_stats.json.'
        ),
    )
    parser.add_argument(
        '--force-scale',
        type=float,
        default=1.0,
        metavar='X',
        help='Multiply F by X after Ha/Bohr→eV/Å. Also efield_scf_F if present (default 1.0).',
    )
    parser.add_argument(
        '--dipole-scale',
        type=float,
        default=1.0,
        metavar='X',
        help=(
            'Multiply Dxyz by X after Debye→e·Å if present. Also scales D, efield_scf_D, efield_D (Debye) if present.'
        ),
    )
    parser.add_argument(
        '--efield-scale',
        type=float,
        default=1.0,
        metavar='X',
        help='Multiply Ef, efield_Ef, efield_scf_Ef by X if present (default 1.0).',
    )
    parser.add_argument(
        '--esp-scale',
        type=float,
        default=1.0,
        metavar='X',
        help='Multiply esp by X [Hartree/e] on grid splits and on EFD if esp is stored there (default 1.0).',
    )
    parser.add_argument(
        '--charge-scale',
        type=float,
        default=1.0,
        metavar='X',
        help=(
            'Multiply NPZ key Q by X if present. For total molecular charge when Q stores charge. '
            'PySCF ESP export may use Q for quadrupole—verify your file (default 1.0).'
        ),
    )
    parser.add_argument(
        '--quadrupole-scale',
        type=float,
        default=None,
        metavar='X',
        help=argparse.SUPPRESS,
    )

    args = parser.parse_args()

    if args.quadrupole_scale is not None:
        if args.charge_scale != 1.0:
            parser.error("--quadrupole-scale is deprecated (was misnamed); use --charge-scale only, not both.")
        args.charge_scale = args.quadrupole_scale
    
    # Validate inputs
    efd_files = args.efd if isinstance(args.efd, list) else [args.efd]
    for f in efd_files:
        if not Path(f).exists():
            print(f"❌ Error: EFD file not found: {f}")
            sys.exit(1)
    
    if args.grid is not None and not args.grid.exists():
        print(f"❌ Error: Grid file not found: {args.grid}")
        sys.exit(1)
    
    if abs(args.train_frac + args.valid_frac + args.test_frac - 1.0) > 1e-6:
        print("❌ Error: Split fractions must sum to 1.0")
        print(f"   Got: {args.train_frac} + {args.valid_frac} + {args.test_frac} = "
              f"{args.train_frac + args.valid_frac + args.test_frac}")
        sys.exit(1)
    
    def _norm_force_unit(s: str) -> ForceIn:
        return s.replace("-", "_")  # type: ignore[return-value]

    def _norm_dipole_unit(s: str) -> DipoleIn:
        return s.replace("-", "_")  # type: ignore[return-value]

    # Run the conversion
    success = fix_and_split_data(
        efd_file=efd_files if len(efd_files) > 1 else efd_files[0],
        grid_file=args.grid,
        output_dir=args.output_dir,
        train_frac=args.train_frac,
        valid_frac=args.valid_frac,
        test_frac=args.test_frac,
        seed=args.seed,
        cube_spacing_bohr=args.cube_spacing,
        skip_validation=args.skip_validation,
        atomic_ref=getattr(args, 'atomic_ref', None),
        atomic_ref_units=getattr(args, 'atomic_ref_units', 'hartree'),
        n_grid_points=getattr(args, 'n_grid_points', 3000),
        esp_sd_sigma=getattr(args, 'esp_sd_sigma', 3.0),
        esp_max_abs_kcal_mol=getattr(args, 'esp_max_abs_kcal_mol', 100.0),
        min_dist_to_atoms=getattr(args, 'min_dist_to_atoms', 1.0),
        flip_forces=getattr(args, 'flip_forces', False),
        energy_scale=getattr(args, 'energy_scale', 1.0),
        force_scale=getattr(args, 'force_scale', 1.0),
        dipole_scale=getattr(args, 'dipole_scale', 1.0),
        efield_scale=getattr(args, 'efield_scale', 1.0),
        esp_scale=getattr(args, 'esp_scale', 1.0),
        charge_scale=getattr(args, 'charge_scale', 1.0),
        zscale_energies=getattr(args, 'zscale_energies', False),
        coords_in=args.coords_in,
        coords_out=args.coords_out,
        energy_in=args.energy_in,
        energy_out=args.energy_out,
        force_in=_norm_force_unit(args.force_in),
        force_out=_norm_force_unit(args.force_out),
        dipole_in=_norm_dipole_unit(args.dipole_in),
        dipole_out=_norm_dipole_unit(args.dipole_out),
        grid_coords_in=args.grid_coords_in,
        grid_coords_out=args.grid_coords_out,
        preserve_units=args.preserve_units,
        verbose=not args.quiet,
    )
    
    if not success:
        sys.exit(1)


if __name__ == "__main__":
    main()

