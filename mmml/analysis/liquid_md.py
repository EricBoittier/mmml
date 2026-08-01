"""Neat-liquid MD post-processing: density, element RDFs, COM MSD, timeseries."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

AVOGADRO = 6.02214076e23

# Experimental bulk liquids ~298 K (g/cm³, g/mol).
BULK_LIQUIDS: dict[str, dict[str, float]] = {
    "DCM": {"rho_g_cm3": 1.326, "mw_g_mol": 84.93, "atoms_per_monomer": 5},
    "ACO": {"rho_g_cm3": 0.784, "mw_g_mol": 58.08, "atoms_per_monomer": 10},
    "TIP3": {"rho_g_cm3": 0.9970, "mw_g_mol": 18.01528, "atoms_per_monomer": 3},
    "MEOH": {"rho_g_cm3": 0.7866, "mw_g_mol": 32.04186, "atoms_per_monomer": 6},
}

_ELEMENT_SYMBOLS = {1: "H", 6: "C", 7: "N", 8: "O", 9: "F", 16: "S", 17: "Cl"}


@dataclass(frozen=True)
class DensityReport:
    n_molecules: int
    solvent: str | None
    box_side_A: float
    density_g_cm3: float
    reference_g_cm3: float | None
    relative_error: float | None
    note: str


@dataclass(frozen=True)
class RdfPeak:
    pair: str
    peak_r_A: float
    peak_g: float


@dataclass(frozen=True)
class MsdReport:
    time_ps: np.ndarray
    msd_A2: np.ndarray
    diffusion_A2_per_ps: float
    diffusion_cm2_per_s: float
    fit_start_frame: int
    n_monomers: int


def packing_density_g_cm3(
    *,
    n_molecules: int,
    mw_g_mol: float,
    box_side_A: float,
) -> float:
    """Mass density for ``n`` molecules of molar mass ``mw`` in a cubic box."""
    if n_molecules <= 0:
        raise ValueError("n_molecules must be positive")
    if mw_g_mol <= 0:
        raise ValueError("mw_g_mol must be positive")
    if box_side_A <= 0:
        raise ValueError("box_side_A must be positive")
    volume_cm3 = (float(box_side_A) * 1e-8) ** 3
    mass_g = float(n_molecules) * float(mw_g_mol) / AVOGADRO
    return mass_g / volume_cm3


def infer_solvent_from_composition(composition: dict[str, Any] | str | None) -> str | None:
    """Return the sole solvent residue name when composition is neat."""
    if composition is None:
        return None
    if isinstance(composition, str):
        parts = [p.strip() for p in composition.split(",") if p.strip()]
        if len(parts) != 1 or ":" not in parts[0]:
            return None
        resid, _count = parts[0].split(":", 1)
        return resid.strip().upper() or None
    if len(composition) != 1:
        return None
    return str(next(iter(composition.keys()))).upper()


def density_report(
    *,
    n_molecules: int,
    box_side_A: float,
    solvent: str | None = None,
    mw_g_mol: float | None = None,
    note: str = "packing density from cubic box (NVT is not a ⟨ρ⟩ validation)",
) -> DensityReport:
    key = solvent.upper() if solvent else None
    props = BULK_LIQUIDS.get(key) if key else None
    mw = float(mw_g_mol) if mw_g_mol is not None else (props["mw_g_mol"] if props else None)
    if mw is None:
        raise ValueError("mw_g_mol required when solvent is unknown")
    rho = packing_density_g_cm3(
        n_molecules=n_molecules, mw_g_mol=mw, box_side_A=box_side_A
    )
    ref = float(props["rho_g_cm3"]) if props else None
    rel = (rho - ref) / ref if ref else None
    return DensityReport(
        n_molecules=int(n_molecules),
        solvent=key,
        box_side_A=float(box_side_A),
        density_g_cm3=float(rho),
        reference_g_cm3=ref,
        relative_error=float(rel) if rel is not None else None,
        note=note,
    )


def element_pair_rdfs_from_arrays(
    positions: np.ndarray,
    atomic_numbers: np.ndarray,
    *,
    box_side_A: float | np.ndarray,
    r_max: float = 12.0,
    n_bins: int = 120,
    atoms_per_monomer: int | None = None,
    exclude_intramolecular: bool = True,
) -> dict[str, Any]:
    """Partial g(r) for element pairs from (n_frames, n_atoms, 3) positions.

    When ``atoms_per_monomer`` is set (and ``exclude_intramolecular``), pairs
    that share a contiguous monomer block are dropped so bonded C–H etc. do not
    dominate the first peak used for liquid validation.
    """
    pos_all = np.asarray(positions, dtype=np.float64)
    z = np.asarray(atomic_numbers, dtype=int)
    if pos_all.ndim != 3 or pos_all.shape[0] == 0:
        return {"n_frames": 0, "pairs": {}, "bins_A": [], "r_max_A": r_max}
    if z.shape[0] != pos_all.shape[1]:
        raise ValueError(
            f"atomic_numbers length ({z.shape[0]}) != n_atoms ({pos_all.shape[1]})"
        )

    box = np.asarray(box_side_A, dtype=np.float64)
    if box.ndim == 0:
        box_vec = np.array([float(box), float(box), float(box)], dtype=np.float64)
    elif box.shape == (3,):
        box_vec = box
    else:
        raise ValueError(f"box_side_A must be scalar or length-3, got shape {box.shape}")

    symbols = np.asarray([_ELEMENT_SYMBOLS.get(int(zi), f"Z{int(zi)}") for zi in z])
    elements = sorted(set(symbols.tolist()))
    edges = np.linspace(0.0, r_max, n_bins + 1, dtype=np.float64)
    centers = 0.5 * (edges[:-1] + edges[1:])
    shell_vol = (4.0 / 3.0) * np.pi * (edges[1:] ** 3 - edges[:-1] ** 3)
    volume = float(np.prod(box_vec))
    n_frames = int(pos_all.shape[0])
    n_atoms = int(pos_all.shape[1])

    mol_id: np.ndarray | None = None
    if (
        exclude_intramolecular
        and atoms_per_monomer is not None
        and atoms_per_monomer > 0
        and n_atoms % int(atoms_per_monomer) == 0
    ):
        mol_id = np.arange(n_atoms, dtype=int) // int(atoms_per_monomer)

    pairs_out: dict[str, Any] = {}
    for i, ea in enumerate(elements):
        for eb in elements[i:]:
            idx_a = np.flatnonzero(symbols == ea)
            idx_b = np.flatnonzero(symbols == eb)
            same = ea == eb
            hist = np.zeros(n_bins, dtype=np.float64)
            n_pair_samples = 0
            for frame in range(n_frames):
                pos = pos_all[frame]
                pa = pos[idx_a]
                pb = pos[idx_b]
                if same:
                    if len(idx_a) < 2:
                        continue
                    d = pa[:, None, :] - pa[None, :, :]
                    d -= box_vec.reshape(1, 1, 3) * np.round(d / box_vec.reshape(1, 1, 3))
                    iu = np.triu_indices(len(idx_a), k=1)
                    dists = np.linalg.norm(d[iu], axis=-1)
                    if mol_id is not None:
                        keep = mol_id[idx_a][iu[0]] != mol_id[idx_a][iu[1]]
                        dists = dists[keep]
                else:
                    d = pa[:, None, :] - pb[None, :, :]
                    d -= box_vec.reshape(1, 1, 3) * np.round(d / box_vec.reshape(1, 1, 3))
                    dists = np.linalg.norm(d, axis=-1)
                    if mol_id is not None:
                        keep = mol_id[idx_a][:, None] != mol_id[idx_b][None, :]
                        dists = dists[keep]
                    else:
                        dists = dists.ravel()
                n_pair_samples += int(dists.size)
                if dists.size:
                    hist += np.histogram(dists, bins=edges)[0]
            if n_pair_samples == 0:
                continue
            if same:
                n_ideal = len(idx_a) * (len(idx_a) - 1) / 2.0
                if mol_id is not None:
                    # Drop same-monomer pairs from the ideal-gas count too.
                    n_mol = n_atoms // int(atoms_per_monomer)
                    n_per = int(np.sum(symbols[: int(atoms_per_monomer)] == ea))
                    n_ideal -= n_mol * n_per * (n_per - 1) / 2.0
            else:
                n_ideal = float(len(idx_a) * len(idx_b))
                if mol_id is not None:
                    n_mol = n_atoms // int(atoms_per_monomer)
                    n_a = int(np.sum(symbols[: int(atoms_per_monomer)] == ea))
                    n_b = int(np.sum(symbols[: int(atoms_per_monomer)] == eb))
                    n_ideal -= n_mol * n_a * n_b
            if n_ideal <= 0:
                continue
            # Ideal-gas shell expectation for this pair class.
            norm = n_frames * n_ideal * shell_vol / volume
            g_r = np.divide(hist, norm, out=np.zeros_like(hist), where=norm > 0)
            peak_i = int(np.argmax(g_r)) if g_r.size else 0
            label = f"{ea}-{eb}"
            pairs_out[label] = {
                "element_a": ea,
                "element_b": eb,
                "n_frames": n_frames,
                "bins_A": centers.tolist(),
                "g_r": g_r.tolist(),
                "peak_r_A": float(centers[peak_i]) if g_r.size else None,
                "peak_g": float(g_r[peak_i]) if g_r.size else None,
                "exclude_intramolecular": bool(mol_id is not None),
            }

    return {
        "n_frames": n_frames,
        "n_atoms": n_atoms,
        "elements": elements,
        "pairs": pairs_out,
        "r_max_A": float(r_max),
        "bins_A": centers.tolist(),
        "box_side_A": box_vec.tolist(),
        "exclude_intramolecular": bool(mol_id is not None),
        "atoms_per_monomer": int(atoms_per_monomer) if atoms_per_monomer else None,
    }


def monomer_com_msd(
    positions: np.ndarray,
    *,
    atoms_per_monomer: int,
    masses: np.ndarray | None = None,
    box_side_A: float,
    timestep_ps: float,
    fit_start_fraction: float = 0.5,
) -> MsdReport:
    """Einstein MSD of neat-liquid monomer centers of mass (MIC-unwrapped)."""
    pos = np.asarray(positions, dtype=np.float64)
    if pos.ndim != 3:
        raise ValueError("positions must have shape (n_frames, n_atoms, 3)")
    n_frames, n_atoms, _ = pos.shape
    if atoms_per_monomer <= 0 or n_atoms % atoms_per_monomer != 0:
        raise ValueError(
            f"n_atoms={n_atoms} not divisible by atoms_per_monomer={atoms_per_monomer}"
        )
    if timestep_ps <= 0:
        raise ValueError("timestep_ps must be positive")
    if not 0.0 <= fit_start_fraction < 1.0:
        raise ValueError("fit_start_fraction must be in [0, 1)")

    n_monomers = n_atoms // atoms_per_monomer
    box = float(box_side_A)
    if masses is None:
        w = np.ones(atoms_per_monomer, dtype=np.float64)
    else:
        w = np.asarray(masses, dtype=np.float64).reshape(n_monomers, atoms_per_monomer)[0]

    shaped = pos.reshape(n_frames, n_monomers, atoms_per_monomer, 3)
    # Unwrap monomer atoms then COM (fractional MIC steps).
    frac = shaped / box
    unwrapped = np.empty_like(frac)
    unwrapped[0] = frac[0]
    for i in range(1, n_frames):
        step = frac[i] - frac[i - 1]
        step -= np.rint(step)
        unwrapped[i] = unwrapped[i - 1] + step
    cart = unwrapped * box
    com = np.sum(cart * w[None, None, :, None], axis=2) / float(np.sum(w))

    msd = np.zeros(n_frames, dtype=np.float64)
    for lag in range(1, n_frames):
        disp = com[lag:] - com[:-lag]
        msd[lag] = float(np.mean(np.sum(disp**2, axis=-1)))

    time = np.arange(n_frames, dtype=np.float64) * float(timestep_ps)
    fit_start = max(1, int(n_frames * fit_start_fraction))
    if n_frames - fit_start < 2:
        slope = 0.0
    else:
        slope = float(np.polyfit(time[fit_start:], msd[fit_start:], 1)[0])
    diffusion = max(0.0, slope / 6.0)
    return MsdReport(
        time_ps=time,
        msd_A2=msd,
        diffusion_A2_per_ps=diffusion,
        diffusion_cm2_per_s=diffusion * 1e-4,
        fit_start_frame=fit_start,
        n_monomers=n_monomers,
    )


def resolve_box_side_A(
    *,
    h5_path: Path | None = None,
    run_dir: Path | None = None,
    fallback: float | None = None,
) -> float | None:
    """Find cubic box side from suite summary, handoff, or explicit fallback."""
    candidates: list[Path] = []
    if run_dir is not None:
        candidates.extend(
            [
                run_dir / "suite_summary_jaxmd.json",
                run_dir / "suite_summary.json",
                run_dir / "handoff" / "state.npz",
            ]
        )
    if h5_path is not None:
        candidates.extend(
            [
                h5_path.parent / "suite_summary_jaxmd.json",
                h5_path.parent / "suite_summary.json",
                h5_path.parent / "handoff" / "state.npz",
            ]
        )
    for path in candidates:
        if not path.is_file():
            continue
        if path.suffix == ".json":
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            box = data.get("box_A")
            if box is not None and float(box) > 0:
                return float(box)
        elif path.suffix == ".npz":
            try:
                data = np.load(path, allow_pickle=False)
                if "cell" in data:
                    cell = np.asarray(data["cell"], dtype=np.float64)
                    if cell.shape == (3, 3):
                        return float(cell[0, 0])
                    if cell.shape == (3,):
                        return float(cell[0])
            except (OSError, ValueError):
                continue
    return float(fallback) if fallback is not None and float(fallback) > 0 else None


def find_campaign_h5_files(campaign_dir: Path) -> list[Path]:
    """Newest-first jaxmd HDF5 trajectories under a campaign output root."""
    if not campaign_dir.is_dir():
        return []
    files = [
        p
        for p in campaign_dir.rglob("*.h5")
        if p.is_file() and p.stat().st_size > 64 and "jaxmd" in p.name.lower()
    ]
    if not files:
        files = [p for p in campaign_dir.rglob("*.h5") if p.is_file() and p.stat().st_size > 64]
    files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return files


def load_h5_for_analysis(
    h5_path: Path,
    *,
    stride: int = 1,
    max_frames: int | None = None,
) -> dict[str, Any]:
    """Load positions / energies / temperature from an mmml jaxmd HDF5."""
    import h5py

    stride = max(1, int(stride))
    with h5py.File(h5_path, "r") as handle:
        n_total = int(handle["positions"].shape[0])
        sl = slice(None, None, stride)
        positions = np.asarray(handle["positions"][sl], dtype=np.float64)
        time_ps = np.asarray(handle["time_ps"][sl], dtype=np.float64) if "time_ps" in handle else None
        temperature = (
            np.asarray(handle["temperature"][sl], dtype=np.float64)
            if "temperature" in handle
            else None
        )
        potential = (
            np.asarray(handle["potential_energy"][sl], dtype=np.float64)
            if "potential_energy" in handle
            else None
        )
        total = (
            np.asarray(handle["total_energy"][sl], dtype=np.float64)
            if "total_energy" in handle
            else None
        )
        z = np.asarray(handle.attrs["atomic_numbers"], dtype=int)
        dt_ps = float(handle.attrs["dt_ps"]) if "dt_ps" in handle.attrs else None
        steps_per_recording = (
            int(handle.attrs["steps_per_recording"])
            if "steps_per_recording" in handle.attrs
            else None
        )
        ensemble = str(handle.attrs["ensemble"]) if "ensemble" in handle.attrs else None

    if max_frames is not None and positions.shape[0] > int(max_frames):
        positions = positions[-int(max_frames) :]
        if time_ps is not None:
            time_ps = time_ps[-int(max_frames) :]
        if temperature is not None:
            temperature = temperature[-int(max_frames) :]
        if potential is not None:
            potential = potential[-int(max_frames) :]
        if total is not None:
            total = total[-int(max_frames) :]

    frame_dt = None
    if time_ps is not None and time_ps.size >= 2:
        frame_dt = float(np.median(np.diff(time_ps)))
    elif dt_ps is not None and steps_per_recording is not None:
        frame_dt = float(dt_ps) * float(steps_per_recording) * float(stride)

    return {
        "path": str(h5_path),
        "n_frames_file": n_total,
        "n_frames": int(positions.shape[0]),
        "positions": positions,
        "atomic_numbers": z,
        "time_ps": time_ps,
        "temperature": temperature,
        "potential_energy": potential,
        "total_energy": total,
        "dt_ps": dt_ps,
        "frame_dt_ps": frame_dt,
        "ensemble": ensemble,
    }


def analyze_h5(
    h5_path: Path,
    *,
    box_side_A: float | None = None,
    solvent: str | None = None,
    n_molecules: int | None = None,
    atoms_per_monomer: int | None = None,
    stride: int = 1,
    max_frames: int | None = 400,
    r_max: float = 12.0,
    n_bins: int = 120,
    do_msd: bool = True,
) -> dict[str, Any]:
    """Full neat-liquid analysis bundle for one jaxmd HDF5 trajectory."""
    box = resolve_box_side_A(h5_path=h5_path, fallback=box_side_A)
    if box is None:
        raise ValueError(
            f"could not resolve box_side_A for {h5_path}; pass --box-size"
        )
    data = load_h5_for_analysis(h5_path, stride=stride, max_frames=max_frames)
    z = data["atomic_numbers"]
    key = solvent.upper() if solvent else None
    props = BULK_LIQUIDS.get(key) if key else None
    apm = atoms_per_monomer or (int(props["atoms_per_monomer"]) if props else None)
    if n_molecules is None and apm is not None and len(z) % apm == 0:
        n_molecules = len(z) // apm
    if n_molecules is None:
        n_molecules = 1

    dens = density_report(
        n_molecules=int(n_molecules),
        box_side_A=float(box),
        solvent=key,
    )
    rdf = element_pair_rdfs_from_arrays(
        data["positions"],
        z,
        box_side_A=float(box),
        r_max=r_max,
        n_bins=n_bins,
        atoms_per_monomer=apm,
        exclude_intramolecular=True,
    )
    peaks = [
        RdfPeak(pair=label, peak_r_A=float(rec["peak_r_A"]), peak_g=float(rec["peak_g"]))
        for label, rec in rdf.get("pairs", {}).items()
        if rec.get("peak_r_A") is not None
    ]
    peaks.sort(key=lambda p: p.peak_g, reverse=True)

    msd_payload: dict[str, Any] | None = None
    if do_msd and apm is not None and data["frame_dt_ps"] and data["n_frames"] >= 3:
        msd = monomer_com_msd(
            data["positions"],
            atoms_per_monomer=int(apm),
            box_side_A=float(box),
            timestep_ps=float(data["frame_dt_ps"]),
        )
        msd_payload = {
            "diffusion_A2_per_ps": msd.diffusion_A2_per_ps,
            "diffusion_cm2_per_s": msd.diffusion_cm2_per_s,
            "fit_start_frame": msd.fit_start_frame,
            "n_monomers": msd.n_monomers,
            "time_ps": msd.time_ps.tolist(),
            "msd_A2": msd.msd_A2.tolist(),
            "note": "short trajectories yield noisy D; treat as qualitative",
        }

    temp = data["temperature"]
    pot = data["potential_energy"]
    tot = data["total_energy"]
    timeseries = {
        "time_ps": data["time_ps"].tolist() if data["time_ps"] is not None else None,
        "temperature_mean_K": float(np.mean(temp)) if temp is not None else None,
        "temperature_std_K": float(np.std(temp)) if temp is not None else None,
        "potential_energy_mean_eV": float(np.mean(pot)) if pot is not None else None,
        "total_energy_mean_eV": float(np.mean(tot)) if tot is not None else None,
        "total_energy_drift_eV": (
            float(tot[-1] - tot[0]) if tot is not None and tot.size >= 2 else None
        ),
    }

    return {
        "h5": str(h5_path),
        "ensemble": data["ensemble"],
        "n_frames_analyzed": data["n_frames"],
        "n_frames_file": data["n_frames_file"],
        "frame_dt_ps": data["frame_dt_ps"],
        "box_side_A": float(box),
        "density": asdict(dens),
        "rdf": {
            "r_max_A": rdf["r_max_A"],
            "n_frames": rdf["n_frames"],
            "pairs": rdf["pairs"],
            "top_peaks": [asdict(p) for p in peaks[:8]],
        },
        "msd": msd_payload,
        "timeseries": timeseries,
    }


def analyze_campaign_dir(
    campaign_dir: Path,
    *,
    box_side_A: float | None = None,
    solvent: str | None = None,
    prefer_run: str = "jaxmd_nvt",
    stride: int = 1,
    max_frames: int | None = 400,
    r_max: float = 12.0,
) -> dict[str, Any]:
    """Analyze the best available jaxmd HDF5 under a campaign output directory."""
    campaign_dir = Path(campaign_dir)
    h5_files = find_campaign_h5_files(campaign_dir)
    chosen: Path | None = None
    for path in h5_files:
        if prefer_run in path.parts:
            chosen = path
            break
    if chosen is None and h5_files:
        chosen = h5_files[0]
    if chosen is None:
        return {
            "campaign_dir": str(campaign_dir),
            "error": "no HDF5 trajectories found",
            "h5_files": [],
        }

    # Prefer composition / box from campaign_plan or suite summary.
    plan = campaign_dir / "campaign_plan.json"
    composition = None
    if plan.is_file():
        try:
            plan_data = json.loads(plan.read_text(encoding="utf-8"))
            defaults = plan_data.get("defaults") or {}
            composition = defaults.get("composition")
            if box_side_A is None and defaults.get("box_size") is not None:
                box_side_A = float(defaults["box_size"])
        except (OSError, json.JSONDecodeError, TypeError, ValueError):
            pass
    if solvent is None:
        solvent = infer_solvent_from_composition(composition)

    report = analyze_h5(
        chosen,
        box_side_A=box_side_A,
        solvent=solvent,
        stride=stride,
        max_frames=max_frames,
        r_max=r_max,
    )
    report["campaign_dir"] = str(campaign_dir)
    report["h5_files"] = [str(p.relative_to(campaign_dir)) for p in h5_files[:12]]
    report["composition"] = composition
    return report


def plot_timeseries_png(report: dict[str, Any], path: Path) -> bool:
    """Plot T(t) and E_pot(t) from an analyze_h5 report."""
    ts = report.get("timeseries") or {}
    time_ps = ts.get("time_ps")
    # Reload energies from the source h5 for plotting (kept out of summary JSON size).
    h5 = report.get("h5")
    if not h5 or not time_ps:
        return False
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import h5py
    except ImportError:
        return False

    with h5py.File(h5, "r") as handle:
        t = np.asarray(handle["time_ps"], dtype=np.float64)
        temp = np.asarray(handle["temperature"], dtype=np.float64) if "temperature" in handle else None
        pot = (
            np.asarray(handle["potential_energy"], dtype=np.float64)
            if "potential_energy" in handle
            else None
        )
        tot = (
            np.asarray(handle["total_energy"], dtype=np.float64)
            if "total_energy" in handle
            else None
        )

    panels: list[str] = []
    if temp is not None:
        panels.append("temp")
    if pot is not None or tot is not None:
        panels.append("energy")
    if not panels:
        return False
    fig, axes = plt.subplots(len(panels), 1, figsize=(7.5, 2.4 * len(panels)), sharex=True)
    if len(panels) == 1:
        axes = [axes]
    for ax, kind in zip(axes, panels, strict=True):
        if kind == "temp":
            ax.plot(t, temp, color="#1f4e79", lw=1.0)
            ax.set_ylabel("T (K)")
        else:
            if pot is not None:
                ax.plot(t, pot, color="#8b2942", lw=1.0, label="E_pot")
            if tot is not None:
                ax.plot(t, tot, color="#2f6f4e", lw=1.0, label="E_tot")
            ax.set_ylabel("E (eV)")
            ax.legend(fontsize=8)
        ax.grid(True, alpha=0.25)
    axes[-1].set_xlabel("time (ps)")
    dens = report.get("density") or {}
    title = Path(h5).name
    if dens.get("density_g_cm3") is not None:
        title += f"  ρ={dens['density_g_cm3']:.3f} g/cm³"
    fig.suptitle(title, fontsize=11)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return True


def plot_rdf_png(report: dict[str, Any], path: Path, *, max_panels: int = 6) -> bool:
    """Plot top element-pair RDFs from an analyze_h5 report."""
    pairs = ((report.get("rdf") or {}).get("pairs")) or {}
    if not pairs:
        return False
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return False

    ranked = sorted(
        pairs.items(),
        key=lambda kv: float(kv[1].get("peak_g") or 0.0),
        reverse=True,
    )[:max_panels]
    n = len(ranked)
    ncols = min(3, n)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.2 * ncols, 3.2 * nrows), squeeze=False)
    fig.suptitle("Element-pair g(r)", fontsize=11)
    for ax, (label, rec) in zip(axes.ravel(), ranked, strict=False):
        ax.plot(rec.get("bins_A") or [], rec.get("g_r") or [], lw=1.0, color="#005384")
        ax.set_title(label, fontsize=9)
        ax.set_xlabel("r (Å)")
        ax.set_ylabel("g(r)")
        ax.grid(True, alpha=0.25)
    for ax in axes.ravel()[n:]:
        ax.axis("off")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return True


def plot_msd_png(report: dict[str, Any], path: Path) -> bool:
    """Plot monomer COM MSD from an analyze_h5 report."""
    msd = report.get("msd") or {}
    time_ps = msd.get("time_ps")
    values = msd.get("msd_A2")
    if not time_ps or not values:
        return False
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return False
    fig, ax = plt.subplots(figsize=(6.2, 4.0))
    ax.plot(time_ps, values, color="#5c3d1e", lw=1.1)
    fit0 = int(msd.get("fit_start_frame") or 0)
    if 0 < fit0 < len(time_ps):
        ax.axvline(time_ps[fit0], color="#888888", ls="--", lw=0.8, label="fit start")
    d = msd.get("diffusion_cm2_per_s")
    title = "Monomer COM MSD"
    if d is not None:
        title += f"  D≈{d:.3e} cm²/s"
    ax.set_title(title)
    ax.set_xlabel("time (ps)")
    ax.set_ylabel("MSD (Å²)")
    ax.grid(True, alpha=0.25)
    if fit0:
        ax.legend(fontsize=8)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return True


def write_analysis_outputs(report: dict[str, Any], output_dir: Path) -> dict[str, str]:
    """Write metrics.json + standard PNGs; return artifact path map."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / "metrics.json"
    # Compact JSON: drop full RDF histograms from the summary file copy.
    summary = dict(report)
    rdf = dict(summary.get("rdf") or {})
    pairs = rdf.get("pairs") or {}
    rdf["pairs"] = {
        label: {
            "peak_r_A": rec.get("peak_r_A"),
            "peak_g": rec.get("peak_g"),
            "element_a": rec.get("element_a"),
            "element_b": rec.get("element_b"),
        }
        for label, rec in pairs.items()
    }
    summary["rdf"] = rdf
    msd = summary.get("msd")
    if isinstance(msd, dict):
        msd = {
            k: v
            for k, v in msd.items()
            if k not in {"time_ps", "msd_A2"}
        }
        summary["msd"] = msd
    metrics_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    artifacts = {"metrics": str(metrics_path)}
    ts_path = output_dir / "timeseries.png"
    if plot_timeseries_png(report, ts_path):
        artifacts["timeseries_png"] = str(ts_path)
    rdf_path = output_dir / "rdf.png"
    if plot_rdf_png(report, rdf_path):
        artifacts["rdf_png"] = str(rdf_path)
    msd_path = output_dir / "msd.png"
    if plot_msd_png(report, msd_path):
        artifacts["msd_png"] = str(msd_path)
    # Full RDF histograms for replotting.
    full_rdf_path = output_dir / "rdf_full.json"
    full_rdf_path.write_text(
        json.dumps(report.get("rdf") or {}, indent=2) + "\n", encoding="utf-8"
    )
    artifacts["rdf_full"] = str(full_rdf_path)
    return artifacts
