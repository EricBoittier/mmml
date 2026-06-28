"""Publication-quality IR comparison: ML dipoles, MM charges, harmonic sticks, NIST."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

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

# Sans-serif stack (Helvetica Neue first when available on macOS; DejaVu is a solid Linux fallback).
_SANS_FONTS = [
    "Helvetica Neue",
    "Helvetica",
    "Arial",
    "Liberation Sans",
    "DejaVu Sans",
]


@dataclass(frozen=True)
class IRSpectrumLine:
    """Broadened line spectrum on a uniform frequency grid."""

    freq_cm: np.ndarray
    intensity: np.ndarray
    label: str
    kind: str  # "broadened" | "sticks"


@dataclass(frozen=True)
class IRProcessingMethod:
    """One IR derivation / post-processing pipeline."""

    key: str
    title: str
    description: str
    compute: Callable[[np.ndarray, float], tuple[np.ndarray, np.ndarray]]


def normalize01(y: np.ndarray) -> np.ndarray:
    """Scale non-negative values to [0, 1] (peak = 1)."""
    y = np.asarray(y, dtype=np.float64)
    y = np.maximum(y, 0.0)
    peak = float(y.max()) if y.size else 0.0
    if peak <= 0.0:
        return y
    return np.clip(y / peak, 0.0, 1.0)


def apply_mpl_style() -> None:
    """Configure matplotlib for clean sans-serif publication figures."""
    import matplotlib as mpl

    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": _SANS_FONTS,
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
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def _interp_to_grid(
    freq: np.ndarray, spec: np.ndarray, grid: np.ndarray
) -> np.ndarray:
    return np.interp(grid, freq, spec, left=0.0, right=0.0)


def _gaussian_smooth_1d(y: np.ndarray, sigma_pts: float) -> np.ndarray:
    """Gaussian kernel smoothing along a uniform grid."""
    from scipy.ndimage import gaussian_filter1d

    return gaussian_filter1d(np.asarray(y, dtype=np.float64), sigma=sigma_pts)


def _moving_average(y: np.ndarray, width: int) -> np.ndarray:
    width = max(3, int(width) | 1)
    kernel = np.ones(width, dtype=np.float64) / width
    return np.convolve(np.asarray(y, dtype=np.float64), kernel, mode="same")


def _savgol_smooth(y: np.ndarray, window: int, poly: int = 3) -> np.ndarray:
    from scipy.signal import savgol_filter

    window = max(poly + 2, int(window))
    if window % 2 == 0:
        window += 1
    if window >= len(y):
        window = max(poly + 2, (len(y) // 2) * 2 - 1)
    if window < poly + 2:
        return np.asarray(y, dtype=np.float64)
    return savgol_filter(np.asarray(y, dtype=np.float64), window, poly)


def compute_dipole_periodogram(
    dipoles: np.ndarray, frame_dt_fs: float
) -> tuple[np.ndarray, np.ndarray]:
    from mmml.spectra.spectra_md import dipole_fluctuation_ir_spectrum

    return dipole_fluctuation_ir_spectrum(dipoles, frame_dt_fs)


def compute_acf_spectrum(
    dipoles: np.ndarray,
    frame_dt_fs: float,
    *,
    qcf: str,
    window: str = "hann",
) -> tuple[np.ndarray, np.ndarray]:
    from mmml.spectra.spectra_md import autocorrelation, correlation_to_spectrum

    mu = np.asarray(dipoles, dtype=np.float64)
    mu = mu - mu.mean(axis=0, keepdims=True)
    acf = autocorrelation(mu)
    return correlation_to_spectrum(
        acf, frame_dt_fs, window=window, zero_pad=4, qcf=qcf
    )


def build_ir_processing_methods() -> list[IRProcessingMethod]:
    """Registered IR pipelines (each gets its own PNG)."""

    def periodogram(d, dt):
        return compute_dipole_periodogram(d, dt)

    def acf_harmonic(d, dt):
        return compute_acf_spectrum(d, dt, qcf="harmonic", window="hann")

    def acf_classical(d, dt):
        return compute_acf_spectrum(d, dt, qcf="classical", window="hann")

    def acf_blackman(d, dt):
        return compute_acf_spectrum(d, dt, qcf="harmonic", window="blackman")

    def acf_gaussian_window(d, dt):
        return compute_acf_spectrum(d, dt, qcf="harmonic", window="gaussian")

    return [
        IRProcessingMethod(
            "01_dipole_periodogram",
            "Dipole periodogram",
            r"$|\tilde{\mu}|^2 \times \omega$ (Hann window, zero-padded FFT)",
            periodogram,
        ),
        IRProcessingMethod(
            "02_acf_harmonic",
            "Autocorrelation · harmonic QCF",
            r"$\mathrm{FT}[\langle \mu(0)\!\cdot\!\mu(\tau)\rangle] \times \omega$",
            acf_harmonic,
        ),
        IRProcessingMethod(
            "03_acf_classical",
            "Autocorrelation · classical QCF",
            r"$\mathrm{FT}[\langle \mu(0)\!\cdot\!\mu(\tau)\rangle] \times \omega^2$",
            acf_classical,
        ),
        IRProcessingMethod(
            "04_acf_blackman",
            "ACF + Blackman window",
            "Autocorrelation spectrum with Blackman apodization",
            acf_blackman,
        ),
        IRProcessingMethod(
            "05_acf_gaussian_window",
            "ACF + Gaussian window",
            "Autocorrelation spectrum with Gaussian apodization",
            acf_gaussian_window,
        ),
    ]


def build_post_smooth_variants(
    grid: np.ndarray,
    mm_y: np.ndarray,
    ml_y: np.ndarray,
) -> list[tuple[str, str, str, np.ndarray, np.ndarray]]:
    """Return (key, title, description, mm_smooth, ml_smooth) tuples."""
    out: list[tuple[str, str, str, np.ndarray, np.ndarray]] = []
    specs = [
        (
            "06_gaussian_smooth",
            "Gaussian smoothing",
            r"$\sigma = 8$ cm$^{-1}$ on interpolated grid",
            lambda y: _gaussian_smooth_1d(y, sigma_pts=8.0),
        ),
        (
            "07_moving_average",
            "Moving average",
            "Boxcar filter, 15 cm$^{-1}$ equivalent width",
            lambda y: _moving_average(y, width=15),
        ),
        (
            "08_savgol",
            "Savitzky–Golay",
            "Polynomial order 3, 31-point window",
            lambda y: _savgol_smooth(y, window=31, poly=3),
        ),
    ]
    for key, title, desc, fn in specs:
        out.append((key, title, desc, normalize01(fn(mm_y)), normalize01(fn(ml_y))))
    return out


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


def compute_all_method_spectra(
    mm_dipoles: np.ndarray,
    ml_dipoles: np.ndarray,
    frame_dt_fs: float,
    *,
    freq_min: float = 600.0,
    freq_max: float = 3200.0,
    n_grid: int = 4000,
) -> dict[str, dict[str, np.ndarray]]:
    """Compute MM/ML spectra for every processing method (values in [0, 1])."""
    grid = np.linspace(freq_min, freq_max, n_grid)
    methods = build_ir_processing_methods()
    all_specs: dict[str, dict[str, np.ndarray]] = {}

    base_mm: np.ndarray | None = None
    base_ml: np.ndarray | None = None

    for method in methods:
        mm_f, mm_ir = method.compute(mm_dipoles, frame_dt_fs)
        ml_f, ml_ir = method.compute(ml_dipoles, frame_dt_fs)
        mm_y = normalize01(_interp_to_grid(mm_f, mm_ir, grid))
        ml_y = normalize01(_interp_to_grid(ml_f, ml_ir, grid))
        all_specs[method.key] = {
            "title": method.title,
            "description": method.description,
            "mm": mm_y,
            "ml": ml_y,
        }
        if method.key == "02_acf_harmonic":
            base_mm, base_ml = mm_y, ml_y

    if base_mm is not None and base_ml is not None:
        for key, title, desc, mm_s, ml_s in build_post_smooth_variants(
            grid, base_mm, base_ml
        ):
            all_specs[key] = {
                "title": title,
                "description": desc,
                "mm": mm_s,
                "ml": ml_s,
            }

    all_specs["_grid"] = {"freq_cm": grid}
    return all_specs


def _stick_overlay(
    ax,
    lo: float,
    hi: float,
    harmonic_freqs: np.ndarray | None,
    harmonic_int: np.ndarray | None,
    *,
    ymax: float = 1.0,
) -> None:
    """Draw NIST + harmonic sticks normalized to [0, ymax]."""
    for nu, inten, _ in NIST_DCM_IR:
        if lo <= nu <= hi:
            ax.vlines(nu, 0, float(inten) * ymax, colors="#111111", lw=1.1, alpha=0.8, zorder=3)
    if harmonic_freqs is not None and harmonic_int is not None:
        hmask = (np.abs(harmonic_freqs) >= lo) & (np.abs(harmonic_freqs) <= hi)
        hf = np.abs(harmonic_freqs[hmask])
        hi_arr = np.asarray(harmonic_int[hmask], dtype=np.float64)
        if hi_arr.size and hi_arr.max() > 0:
            hi_arr = normalize01(hi_arr)
        for nu, inten in zip(hf, hi_arr, strict=False):
            ax.vlines(
                nu,
                0,
                float(inten) * ymax * 0.9,
                colors="#009E73",
                lw=1.0,
                alpha=0.85,
                zorder=2,
                linestyles=(0, (1.2, 1.8)),
            )


def plot_method_figure(
    *,
    method_key: str,
    title: str,
    description: str,
    grid: np.ndarray,
    mm_y: np.ndarray,
    ml_y: np.ndarray,
    out_path: Path,
    harmonic_freqs: np.ndarray | None = None,
    harmonic_int: np.ndarray | None = None,
    zoom_ch: tuple[float, float] = (2850.0, 3050.0),
    zoom_cl: tuple[float, float] = (650.0, 900.0),
) -> Path:
    """One three-panel PNG for a single IR processing method."""
    import matplotlib.pyplot as plt
    from matplotlib import gridspec
    from matplotlib.lines import Line2D

    apply_mpl_style()
    c_mm, c_ml = "#0072B2", "#D55E00"

    fig = plt.figure(figsize=(7.5, 7.0))
    gs = gridspec.GridSpec(3, 1, height_ratios=[1.5, 1.0, 1.0], hspace=0.34)
    axes = [fig.add_subplot(gs[i]) for i in range(3)]
    fmin, fmax = float(grid[0]), float(grid[-1])
    panels = [
        (axes[0], fmin, fmax, "a", title),
        (axes[1], zoom_cl[0], zoom_cl[1], "b", r"C–Cl stretch"),
        (axes[2], zoom_ch[0], zoom_ch[1], "c", r"C–H stretch"),
    ]

    for ax, lo, hi, panel_lbl, panel_title in panels:
        mask = (grid >= lo) & (grid <= hi)
        ax.plot(grid[mask], mm_y[mask], color=c_mm, lw=1.7, label="MD, MM charges")
        ax.plot(grid[mask], ml_y[mask], color=c_ml, lw=1.7, label="MD, ML dipole")
        _stick_overlay(ax, lo, hi, harmonic_freqs, harmonic_int, ymax=1.0)
        ax.set_xlim(lo, hi)
        ax.set_ylim(0.0, 1.0)
        ax.set_ylabel("Intensity (0–1)")
        ax.set_title(panel_title, fontsize=10.5, pad=4)
        ax.text(-0.09, 1.02, panel_lbl, transform=ax.transAxes, fontsize=13, fontweight="bold")
        ax.grid(True, alpha=0.18, linewidth=0.6)
        ax.tick_params(top=True, right=True)

    axes[0].text(
        0.015,
        0.97,
        description,
        transform=axes[0].transAxes,
        ha="left",
        va="top",
        fontsize=8.5,
        color="#444444",
        wrap=True,
    )
    axes[0].text(
        0.015,
        0.88,
        "20 ps NVT · 2× DCM · 28 Å PBC",
        transform=axes[0].transAxes,
        ha="left",
        va="top",
        fontsize=8.5,
        color="#666666",
    )
    axes[0].legend(
        handles=[
            Line2D([0], [0], color=c_mm, lw=1.8, label="MD, MM charges (CGENFF)"),
            Line2D([0], [0], color=c_ml, lw=1.8, label="MD, ML dipole (PhysNet)"),
            Line2D([0], [0], color="#111111", lw=1.2, label="NIST gas phase"),
            Line2D(
                [0], [0], color="#009E73", lw=1.2, ls=(0, (1.2, 1.8)), label="Harmonic sticks"
            ),
        ],
        loc="upper right",
        frameon=True,
        framealpha=0.95,
        edgecolor="#cccccc",
        fancybox=False,
    )
    axes[2].set_xlabel(r"Wavenumber / cm$^{-1}$")
    fig.align_ylabels(axes)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    return out_path


def plot_ir_comparison_publication(
    grid: np.ndarray,
    mm_y: np.ndarray,
    ml_y: np.ndarray,
    out_path: Path,
    *,
    title: str = "DCM dimer — infrared spectra",
    subtitle: str = "ACF harmonic QCF (reference panel)",
    harmonic_freqs: np.ndarray | None = None,
    harmonic_int: np.ndarray | None = None,
) -> Path:
    """Summary three-panel figure (default method: ACF harmonic)."""
    return plot_method_figure(
        method_key="summary",
        title=title,
        description=subtitle,
        grid=grid,
        mm_y=mm_y,
        ml_y=ml_y,
        out_path=out_path,
        harmonic_freqs=harmonic_freqs,
        harmonic_int=harmonic_int,
    )


def plot_stick_comparison(
    harmonic_freqs: np.ndarray,
    harmonic_int: np.ndarray,
    out_path: Path,
    *,
    freq_min: float = 600.0,
    freq_max: float = 3200.0,
) -> Path:
    """Stick-only panel: harmonic vs NIST, intensities in [0, 1]."""
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    apply_mpl_style()
    fig, ax = plt.subplots(figsize=(7.5, 2.8))
    hmask = (np.abs(harmonic_freqs) >= freq_min) & (np.abs(harmonic_freqs) <= freq_max)
    hf = np.abs(harmonic_freqs[hmask])
    hi = normalize01(np.asarray(harmonic_int[hmask], dtype=np.float64))

    for nu, inten in zip(hf, hi, strict=False):
        ax.vlines(nu, 0, inten, color="#009E73", lw=1.4, alpha=0.9)

    for nu, inten, _ in NIST_DCM_IR:
        if freq_min <= nu <= freq_max:
            ax.vlines(nu, 0, inten, color="#111111", lw=1.4, alpha=0.85)

    ax.set_xlim(freq_min, freq_max)
    ax.set_ylim(0.0, 1.0)
    ax.set_xlabel(r"Wavenumber / cm$^{-1}$")
    ax.set_ylabel("Intensity (0–1)")
    ax.set_title("Gas-phase harmonic sticks vs NIST (DCM monomer)", fontweight="bold")
    ax.grid(True, alpha=0.18, linewidth=0.6)
    ax.tick_params(top=True, right=True, direction="in")
    ax.legend(
        handles=[
            Line2D([0], [0], color="#009E73", lw=1.6, label="Harmonic (ML Hessian)"),
            Line2D([0], [0], color="#111111", lw=1.6, label="NIST gas phase"),
        ],
        loc="upper right",
        frameon=True,
        edgecolor="#cccccc",
        fancybox=False,
    )
    out_path = Path(out_path)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
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
    """Compute ML/MM spectra, write one PNG per IR method (all normalized 0–1)."""
    from ase.io.trajectory import Trajectory

    out_dir = out_dir or (run_dir / "spectra")
    methods_dir = out_dir / "ir_methods"
    methods_dir.mkdir(parents=True, exist_ok=True)

    mm_npz = out_dir / "correlation_spectra.npz"
    if not mm_npz.is_file():
        raise FileNotFoundError(f"MM spectrum not found: {mm_npz}")
    mm_data = np.load(mm_npz)
    mm_dipoles = np.asarray(mm_data["dipoles"], dtype=np.float64)

    harm_npz = out_dir / "harmonic_ir.npz"
    if not harm_npz.is_file():
        raise FileNotFoundError(f"harmonic spectrum not found: {harm_npz}")
    harm_data = np.load(harm_npz)
    harm_freqs = np.asarray(harm_data["stick_frequencies"], dtype=np.float64)
    harm_int = np.asarray(harm_data["stick_intensities"], dtype=np.float64)

    dip_cache: Path | None = None
    for cand in sorted(out_dir.glob("ml_dipoles_n*_s*.npz")):
        dip_cache = cand
        break

    if dip_cache is not None and dip_cache.is_file():
        cached = np.load(dip_cache)
        ml_dipoles = np.asarray(cached["dipoles"], dtype=np.float64)
        frame_dt_fs = float(cached.get("frame_dt_fs", dt_fs * steps_per_recording * stride))
        print(f"  Using cached ML dipoles: {dip_cache.name} ({len(ml_dipoles):,} frames)")
    else:
        frames = list(Trajectory(str(traj)))
        if stride > 1:
            frames = frames[::stride]
        if max_frames is not None:
            frames = frames[:max_frames]
        frame_dt_fs = float(dt_fs) * max(1, int(steps_per_recording)) * stride
        print(f"  Computing ML dipoles for {len(frames):,} frames ...")
        ml_dipoles = predict_dipoles_physnet_jit(frames, dipole_ckpt)
        dip_cache = out_dir / f"ml_dipoles_n{len(ml_dipoles)}_s{stride}.npz"
        np.savez(dip_cache, dipoles=ml_dipoles, stride=stride, frame_dt_fs=frame_dt_fs)

    n = min(len(mm_dipoles), len(ml_dipoles))
    mm_dipoles = mm_dipoles[:n]
    ml_dipoles = ml_dipoles[:n]

    all_specs = compute_all_method_spectra(mm_dipoles, ml_dipoles, frame_dt_fs)
    grid = np.asarray(all_specs.pop("_grid")["freq_cm"], dtype=np.float64)

    method_pngs: dict[str, str] = {}
    for key, payload in sorted(all_specs.items()):
        if key.startswith("_"):
            continue
        png_path = methods_dir / f"ir_{key}.png"
        plot_method_figure(
            method_key=key,
            title=str(payload["title"]),
            description=str(payload["description"]),
            grid=grid,
            mm_y=np.asarray(payload["mm"]),
            ml_y=np.asarray(payload["ml"]),
            out_path=png_path,
            harmonic_freqs=harm_freqs,
            harmonic_int=harm_int,
        )
        method_pngs[key] = str(png_path)
        print(f"  Wrote {png_path.name}")

    ref = all_specs.get("02_acf_harmonic", all_specs[sorted(all_specs)[0]])
    main_png = out_dir / "ir_comparison.png"
    plot_ir_comparison_publication(
        grid,
        np.asarray(ref["mm"]),
        np.asarray(ref["ml"]),
        main_png,
        harmonic_freqs=harm_freqs,
        harmonic_int=harm_int,
    )
    stick_png = out_dir / "ir_sticks_comparison.png"
    plot_stick_comparison(harm_freqs, harm_int, stick_png)

    meta = {
        "n_frames": n,
        "stride": stride,
        "frame_dt_fs": frame_dt_fs,
        "dipole_checkpoint": str(dipole_ckpt),
        "normalization": "peak=1, clipped to [0, 1]",
        "outputs": {
            "comparison_png": str(main_png),
            "sticks_png": str(stick_png),
            "methods_dir": str(methods_dir),
            "method_pngs": method_pngs,
            "ml_dipoles_cache": str(dip_cache) if dip_cache else None,
        },
        "nist_peaks": [
            {"freq_cm": p[0], "intensity": p[1], "assignment": p[2]} for p in NIST_DCM_IR
        ],
    }
    (out_dir / "ir_comparison_meta.json").write_text(
        json.dumps(meta, indent=2), encoding="utf-8"
    )
    np.savez(
        out_dir / "ir_comparison_spectra.npz",
        freq_cm=grid,
        **{f"mm_{k}": v["mm"] for k, v in all_specs.items() if not k.startswith("_")},
        **{f"ml_{k}": v["ml"] for k, v in all_specs.items() if not k.startswith("_")},
    )
    print(f"Wrote {len(method_pngs)} method PNGs under {methods_dir}")
    return meta
