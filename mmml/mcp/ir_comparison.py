"""Publication-quality IR comparison: ML dipoles, MM charges, harmonic sticks, NIST."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

# NIST WebBook JCAMP C75092 — dichloromethane gas-phase IR (relative intensities).
NIST_DCM_IR: list[tuple[float, float, str]] = [
    (3019.0, 1.00, r"C–H stretch"),
    (2996.0, 0.88, r"C–H stretch"),
    (1575.0, 0.32, r"CH$_2$ bend"),
    (1470.0, 0.22, r"CH$_2$ bend"),
    (1150.0, 0.18, r"CH$_2$ rock"),
    (948.0, 0.12, r"C–C stretch"),
    (748.0, 0.92, r"C–Cl stretch"),
    (707.0, 0.78, r"C–Cl stretch"),
]


@dataclass(frozen=True)
class IRSpectrumLine:
    """Broadened line spectrum on a uniform frequency grid."""

    freq_cm: np.ndarray
    intensity: np.ndarray
    label: str
    kind: str  # "broadened" | "sticks"


def _normalize(y: np.ndarray, *, mode: str = "max") -> np.ndarray:
    y = np.asarray(y, dtype=np.float64)
    y = np.maximum(y, 0.0)
    if mode == "max":
        peak = float(y.max()) if y.size else 1.0
        return y / peak if peak > 0 else y
    if mode == "area":
        area = float(np.trapezoid(y))
        return y / area if area > 0 else y
    return y


def predict_dipoles_physnet_jit(
    frames,
    dipole_ckpt: Path,
    *,
    progress_every: int = 5000,
) -> np.ndarray:
    """Fast JIT PhysNet dipole inference over trajectory frames."""
    import e3x
    import jax
    import jax.numpy as jnp

    from mmml.cli.misc.physnet_evaluate import _load_physnet_checkpoint

    n_atoms = len(frames[0])
    _, params, model = _load_physnet_checkpoint(dipole_ckpt, n_atoms)
    if not getattr(model, "charges", False):
        raise ValueError(f"checkpoint does not predict dipoles: {dipole_ckpt}")

    dst_idx, src_idx = e3x.ops.sparse_pairwise_indices(n_atoms)

    @jax.jit
    def dipole_one(z, r):
        atom_mask = (z > 0).astype(jnp.float32)
        batch_segments = jnp.zeros((n_atoms,), dtype=jnp.int32)
        valid = (atom_mask[dst_idx] > 0) & (atom_mask[src_idx] > 0)
        batch_mask = valid.astype(jnp.float32)
        out = model.apply(
            params,
            atomic_numbers=z,
            positions=r,
            dst_idx=dst_idx,
            src_idx=src_idx,
            batch_segments=batch_segments,
            batch_size=1,
            batch_mask=batch_mask,
            atom_mask=atom_mask,
        )
        return out["dipoles"]

    dipoles = np.zeros((len(frames), 3), dtype=np.float64)
    for i, atoms in enumerate(frames):
        z = jnp.asarray(atoms.get_atomic_numbers(), dtype=jnp.int32)
        r = jnp.asarray(atoms.get_positions(), dtype=jnp.float32)
        dipoles[i] = np.asarray(dipole_one(z, r), dtype=np.float64).reshape(3)
        if progress_every and (i == 0 or (i + 1) % progress_every == 0):
            print(f"      ML dipole frame {i + 1:,}/{len(frames):,}")
    return dipoles


def sticks_to_spectrum(
    frequencies: np.ndarray,
    intensities: np.ndarray,
    freq_grid: np.ndarray,
    *,
    fwhm: float = 10.0,
) -> np.ndarray:
    """Lorentzian-broadened stick spectrum on ``freq_grid``."""
    sigma = max(fwhm, 1.0) / (2.0 * np.sqrt(2.0 * np.log(2.0)))
    spec = np.zeros_like(freq_grid, dtype=np.float64)
    for nu, intensity in zip(frequencies, intensities, strict=False):
        if intensity <= 0 or not np.isfinite(nu):
            continue
        spec += intensity * np.exp(-0.5 * ((freq_grid - nu) / sigma) ** 2)
    return spec


def build_comparison_spectra(
    *,
    mm_freq: np.ndarray,
    mm_ir: np.ndarray,
    ml_freq: np.ndarray,
    ml_ir: np.ndarray,
    harmonic_freqs: np.ndarray,
    harmonic_int: np.ndarray,
    freq_min: float = 600.0,
    freq_max: float = 3200.0,
    n_grid: int = 4000,
    stick_fwhm: float = 12.0,
) -> dict[str, IRSpectrumLine]:
    """Assemble normalized comparison spectra on a common grid."""
    grid = np.linspace(freq_min, freq_max, n_grid)
    mm_interp = np.interp(grid, mm_freq, mm_ir, left=0.0, right=0.0)
    ml_interp = np.interp(grid, ml_freq, ml_ir, left=0.0, right=0.0)

    nist_freqs = np.array([p[0] for p in NIST_DCM_IR], dtype=np.float64)
    nist_int = np.array([p[1] for p in NIST_DCM_IR], dtype=np.float64)
    nist_spec = sticks_to_spectrum(nist_freqs, nist_int, grid, fwhm=stick_fwhm)

    harm_mask = (np.abs(harmonic_freqs) >= freq_min) & (np.abs(harmonic_freqs) <= freq_max)
    harm_freqs = np.abs(harmonic_freqs[harm_mask])
    harm_int = harmonic_int[harm_mask]
    if harm_int.size:
        harm_int = harm_int / harm_int.max()
    harm_spec = sticks_to_spectrum(harm_freqs, harm_int, grid, fwhm=stick_fwhm)

    return {
        "mm": IRSpectrumLine(grid, _normalize(mm_interp), "MM (CGENFF $\\mu$)", "broadened"),
        "ml": IRSpectrumLine(grid, _normalize(ml_interp), "ML (PhysNet $\\mu$)", "broadened"),
        "nist": IRSpectrumLine(grid, _normalize(nist_spec), "NIST gas phase", "sticks"),
        "harmonic": IRSpectrumLine(
            grid, _normalize(harm_spec), "Harmonic (ML Hessian)", "sticks"
        ),
    }


def plot_ir_comparison_publication(
    spectra: dict[str, IRSpectrumLine],
    out_path: Path,
    *,
    title: str = "DCM dimer — infrared spectra",
    harmonic_freqs: np.ndarray | None = None,
    harmonic_int: np.ndarray | None = None,
    zoom_ch: tuple[float, float] = (2850.0, 3050.0),
    zoom_cl: tuple[float, float] = (650.0, 900.0),
) -> Path:
    """Write a publication-ready three-panel IR comparison figure."""
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    from matplotlib import gridspec
    from matplotlib.lines import Line2D

    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
            "mathtext.fontset": "dejavusans",
            "axes.labelsize": 11,
            "axes.titlesize": 11,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 9,
            "axes.linewidth": 1.0,
            "xtick.major.width": 1.0,
            "ytick.major.width": 1.0,
            "xtick.direction": "in",
            "ytick.direction": "in",
            "lines.linewidth": 1.6,
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.05,
        }
    )

    c_md_mm = "#0072B2"
    c_md_ml = "#D55E00"
    c_nist = "#000000"
    c_harm = "#009E73"

    fig = plt.figure(figsize=(7.5, 7.0))
    gs = gridspec.GridSpec(3, 1, height_ratios=[1.5, 1.0, 1.0], hspace=0.32)
    ax_full = fig.add_subplot(gs[0])
    ax_cl = fig.add_subplot(gs[1])
    ax_ch = fig.add_subplot(gs[2])

    grid = spectra["mm"].freq_cm
    fmin, fmax = float(grid[0]), float(grid[-1])

    def _md_curves(ax, lo: float, hi: float) -> None:
        for key, color, lbl in (
            ("mm", c_md_mm, "MD, MM charges (CGENFF)"),
            ("ml", c_md_ml, "MD, ML dipole (PhysNet)"),
        ):
            line = spectra[key]
            mask = (line.freq_cm >= lo) & (line.freq_cm <= hi)
            ax.plot(
                line.freq_cm[mask],
                line.intensity[mask],
                color=color,
                lw=1.7,
                label=lbl,
                alpha=0.95,
            )

    def _stick_overlay(
        ax,
        lo: float,
        hi: float,
        *,
        ymax: float = 1.05,
        show_harmonic: bool = True,
    ) -> None:
        """Draw NIST + harmonic sticks in the upper fraction of the panel."""
        stick_scale = 0.92 * ymax
        for nu, inten, _ in NIST_DCM_IR:
            if lo <= nu <= hi:
                ax.vlines(
                    nu,
                    0,
                    inten * stick_scale,
                    colors=c_nist,
                    linewidth=1.1,
                    alpha=0.75,
                    zorder=3,
                )
        if show_harmonic and harmonic_freqs is not None and harmonic_int is not None:
            hmask = (np.abs(harmonic_freqs) >= lo) & (np.abs(harmonic_freqs) <= hi)
            hf = np.abs(harmonic_freqs[hmask])
            hi = np.asarray(harmonic_int[hmask], dtype=np.float64)
            if hi.size and hi.max() > 0:
                hi = hi / hi.max()
            for nu, inten in zip(hf, hi, strict=False):
                ax.vlines(
                    nu,
                    0,
                    inten * stick_scale * 0.85,
                    colors=c_harm,
                    linewidth=1.0,
                    alpha=0.8,
                    zorder=2,
                    linestyles=(0, (1.2, 1.8)),
                )

    # (a) Full range — MD curves + stick references
    _md_curves(ax_full, fmin, fmax)
    _stick_overlay(ax_full, fmin, fmax)
    ax_full.set_xlim(fmin, fmax)
    ax_full.set_ylim(0, 1.08)
    ax_full.set_ylabel("Normalized intensity")
    ax_full.set_title(title, fontweight="bold", pad=8)
    ax_full.text(
        0.015,
        0.97,
        "20 ps NVT · 2× DCM · 28 Å PBC · 0.1 fs",
        transform=ax_full.transAxes,
        ha="left",
        va="top",
        fontsize=9,
        color="#333333",
    )
    ax_full.text(-0.09, 1.02, "a", transform=ax_full.transAxes, fontsize=13, fontweight="bold")
    ax_full.legend(
        handles=[
            Line2D([0], [0], color=c_md_mm, lw=1.8, label="MD, MM charges (CGENFF)"),
            Line2D([0], [0], color=c_md_ml, lw=1.8, label="MD, ML dipole (PhysNet)"),
            Line2D([0], [0], color=c_nist, lw=1.2, label="NIST gas phase (sticks)"),
            Line2D(
                [0],
                [0],
                color=c_harm,
                lw=1.2,
                ls=(0, (1.2, 1.8)),
                label="Harmonic, monomer (sticks)",
            ),
        ],
        loc="upper right",
        frameon=True,
        framealpha=0.95,
        edgecolor="#bbbbbb",
        fancybox=False,
    )
    ax_full.grid(True, which="major", alpha=0.18, linewidth=0.6)
    ax_full.tick_params(top=True, right=True)

    # (b) C–Cl region
    cl0, cl1 = zoom_cl
    _md_curves(ax_cl, cl0, cl1)
    _stick_overlay(ax_cl, cl0, cl1)
    ax_cl.set_xlim(cl0, cl1)
    ax_cl.set_ylim(0, 1.08)
    ax_cl.set_ylabel("Normalized intensity")
    ax_cl.set_title(r"C–Cl stretch region", fontsize=10.5, pad=5)
    ax_cl.text(-0.09, 1.02, "b", transform=ax_cl.transAxes, fontsize=13, fontweight="bold")
    ax_cl.grid(True, which="major", alpha=0.18, linewidth=0.6)
    ax_cl.tick_params(top=True, right=True)

    # (c) C–H region
    ch0, ch1 = zoom_ch
    _md_curves(ax_ch, ch0, ch1)
    _stick_overlay(ax_ch, ch0, ch1, show_harmonic=True)
    ax_ch.set_xlim(ch0, ch1)
    ax_ch.set_ylim(0, 1.08)
    ax_ch.set_xlabel(r"Wavenumber / cm$^{-1}$")
    ax_ch.set_ylabel("Normalized intensity")
    ax_ch.set_title(r"C–H stretch region", fontsize=10.5, pad=5)
    ax_ch.text(-0.09, 1.02, "c", transform=ax_ch.transAxes, fontsize=13, fontweight="bold")
    ax_ch.grid(True, which="major", alpha=0.18, linewidth=0.6)
    ax_ch.tick_params(top=True, right=True)

    fig.align_ylabels([ax_full, ax_cl, ax_ch])

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300)
    fig.savefig(out_path.with_suffix(".pdf"))
    plt.close(fig)
    return out_path


def plot_stick_comparison(
    harmonic_freqs: np.ndarray,
    harmonic_int: np.ndarray,
    out_path: Path,
    *,
    freq_min: float = 600.0,
    freq_max: float = 3200.0,
) -> Path:
    """Stick-only panel: harmonic vs NIST (publication style)."""
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
            "axes.labelsize": 11,
            "figure.dpi": 150,
            "savefig.dpi": 300,
        }
    )

    fig, ax = plt.subplots(figsize=(7.5, 2.8))
    hmask = (np.abs(harmonic_freqs) >= freq_min) & (np.abs(harmonic_freqs) <= freq_max)
    hf = np.abs(harmonic_freqs[hmask])
    hi = np.asarray(harmonic_int[hmask], dtype=np.float64)
    if hi.size and hi.max() > 0:
        hi = hi / hi.max()

    for nu, inten in zip(hf, hi, strict=False):
        ax.vlines(nu, 0, inten, color="#009E73", lw=1.4, alpha=0.9)

    for nu, inten, _ in NIST_DCM_IR:
        if freq_min <= nu <= freq_max:
            ax.vlines(nu, 0, inten, color="#000000", lw=1.4, alpha=0.8)

    ax.set_xlim(freq_min, freq_max)
    ax.set_ylim(0, 1.12)
    ax.set_xlabel(r"Wavenumber / cm$^{-1}$")
    ax.set_ylabel("Relative intensity")
    ax.set_title("Gas-phase harmonic sticks vs NIST (DCM monomer)", fontweight="bold")
    ax.grid(True, alpha=0.18, linewidth=0.6)
    ax.tick_params(top=True, right=True, direction="in")
    ax.legend(
        handles=[
            Line2D([0], [0], color="#009E73", lw=1.6, label="Harmonic (ML Hessian)"),
            Line2D([0], [0], color="#000000", lw=1.6, label="NIST gas phase"),
        ],
        loc="upper right",
        frameon=True,
        edgecolor="#bbbbbb",
        fancybox=False,
    )
    out_path = Path(out_path)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    return out_path


def generate_ir_comparison_figure(
    run_dir: Path,
    traj: Path,
    dipole_ckpt: Path,
    *,
    dt_fs: float = 0.1,
    steps_per_recording: int = 1,
    stride: int = 1,
    max_frames: int | None = None,
    out_dir: Path | None = None,
) -> dict[str, Any]:
    """Compute ML spectrum on trajectory and write publication comparison plots."""
    from ase.io.trajectory import Trajectory

    from mmml.spectra.spectra_md import dipole_fluctuation_ir_spectrum

    out_dir = out_dir or (run_dir / "spectra")
    out_dir.mkdir(parents=True, exist_ok=True)

    # MM spectrum from prior classical run if available
    mm_npz = out_dir / "correlation_spectra.npz"
    if not mm_npz.is_file():
        raise FileNotFoundError(f"MM spectrum not found: {mm_npz}")
    mm_data = np.load(mm_npz)
    mm_freq = np.asarray(mm_data["freq_cm"], dtype=np.float64)
    mm_ir = np.asarray(mm_data["ir"], dtype=np.float64)

    harm_npz = out_dir / "harmonic_ir.npz"
    if not harm_npz.is_file():
        raise FileNotFoundError(f"harmonic spectrum not found: {harm_npz}")
    harm_data = np.load(harm_npz)
    harm_freqs = np.asarray(harm_data["stick_frequencies"], dtype=np.float64)
    harm_int = np.asarray(harm_data["stick_intensities"], dtype=np.float64)

    print(f"Loading trajectory metadata {traj} ...")
    dip_cache = out_dir / f"ml_dipoles_n{200000 if max_frames is None else max_frames}_s{stride}.npz"
    # Try common cache names
    if not dip_cache.is_file():
        for cand in sorted(out_dir.glob("ml_dipoles_n*_s*.npz")):
            dip_cache = cand
            break

    if dip_cache.is_file():
        cached = np.load(dip_cache)
        dipoles = np.asarray(cached["dipoles"], dtype=np.float64)
        frame_dt_fs = float(cached.get("frame_dt_fs", dt_fs * steps_per_recording * stride))
        print(f"  Using cached ML dipoles: {dip_cache.name} ({len(dipoles):,} frames)")
    else:
        from ase.io.trajectory import Trajectory

        frames = list(Trajectory(str(traj)))
        if stride > 1:
            frames = frames[::stride]
        if max_frames is not None:
            frames = frames[:max_frames]
        frame_dt_fs = float(dt_fs) * max(1, int(steps_per_recording)) * stride
        print(f"  {len(frames):,} frames, frame_dt = {frame_dt_fs} fs")
        print("Computing ML dipoles (JIT PhysNet) ...")
        dipoles = predict_dipoles_physnet_jit(frames, dipole_ckpt)
        dip_cache = out_dir / f"ml_dipoles_n{len(dipoles)}_s{stride}.npz"
        np.savez(dip_cache, dipoles=dipoles, stride=stride, frame_dt_fs=frame_dt_fs)

    ml_freq, ml_ir = dipole_fluctuation_ir_spectrum(dipoles, frame_dt_fs)

    spectra = build_comparison_spectra(
        mm_freq=mm_freq,
        mm_ir=mm_ir,
        ml_freq=ml_freq,
        ml_ir=ml_ir,
        harmonic_freqs=harm_freqs,
        harmonic_int=harm_int,
    )

    main_png = out_dir / "ir_comparison.png"
    stick_png = out_dir / "ir_sticks_comparison.png"
    plot_ir_comparison_publication(
        spectra,
        main_png,
        harmonic_freqs=harm_freqs,
        harmonic_int=harm_int,
    )
    plot_stick_comparison(harm_freqs, harm_int, stick_png)

    meta = {
        "n_frames": len(dipoles),
        "stride": stride,
        "frame_dt_fs": frame_dt_fs,
        "dipole_checkpoint": str(dipole_ckpt),
        "outputs": {
            "comparison_png": str(main_png),
            "comparison_pdf": str(main_png.with_suffix(".pdf")),
            "sticks_png": str(stick_png),
            "ml_dipoles_cache": str(dip_cache),
        },
        "nist_peaks": [{"freq_cm": p[0], "intensity": p[1], "assignment": p[2]} for p in NIST_DCM_IR],
    }
    (out_dir / "ir_comparison_meta.json").write_text(
        json.dumps(meta, indent=2), encoding="utf-8"
    )
    np.savez(
        out_dir / "ir_comparison_spectra.npz",
        freq_cm=spectra["mm"].freq_cm,
        mm=spectra["mm"].intensity,
        ml=spectra["ml"].intensity,
        nist=spectra["nist"].intensity,
        harmonic=spectra["harmonic"].intensity,
        ml_freq_raw=ml_freq,
        ml_ir_raw=ml_ir,
    )
    print(f"Wrote {main_png} and {stick_png}")
    return meta
