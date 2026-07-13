"""Shared plotting utilities for the dimer scan campaign.

Provides a single preprocessing function that enriches a raw scan DataFrame
with a combined ``multipoles_mbd`` backend (learned_multipole + learned_mbd)
so every plot script shows the total QM/ML interaction energy alongside the
individual components.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from ase.data import covalent_radii
from ase.data.colors import jmol_colors
from ase.io.utils import rotate as _ase_rotate
from matplotlib.patches import Circle

# ── Backend metadata ─────────────────────────────────────────────────────────

BACKEND_LABELS: dict[str, str] = {
    "learned_multipole": "Multipoles",
    "learned_mbd":       "MBD",
    "multipoles_mbd":    "Multipoles + MBD",
    "xtb_gfn2":          "GFN2-xTB",
    "dftb3_d4":         "DFTB3-D4/3ob-3-1",
    "charmm":            "CGenFF",
    "ccsd_def2svp_gpu4pyscf_cp": "CCSD/def2-SVP (GPU4PySCF)",
    "ccsd_def2svpd_gpu4pyscf_cp": "CCSD/def2-SVPD (GPU4PySCF)",
    "spookynet":         "SpookyNet",
    "spookynet_hybrid":  "SpookyNet Hybrid ML",
    "spookynet_muon_ep7": "SpookyNet (Muon e7)",
    "spookynet_hybrid_muon_ep7": "Hybrid (Muon e7)",
    "spookynet_mbdzbl_ep2": "MBD+ZBL (e2)",
    "spookynet_hybrid_mbdzbl_ep2": "Hybrid MBD+ZBL (e2)",
    "spookynet_hybrid_step3000": "SpookyNet (hybrid training, step 3000)",
    "spookynet_hybrid_hybrid_step3000": "Hybrid decomposition (step 3000)",
    "spookynet_hybrid_step3000_mbd": "Hybrid step 3000 + MBD",
    "spookynet_hybrid_muon_epoch1_mbd": "Muon epoch 1 + MBD",
    "spookynet_hybrid_step19800": "Hybrid step 19800",
    "spookynet_hybrid_step19800_mbd": "Hybrid step 19800 + MBD",
    "spookynet_hybrid_step36800": "Hybrid step 36800",
    "spookynet_hybrid_step36800_mbd": "Hybrid step 36800 + MBD",
    "spookynet_hybrid_step42600": "Hybrid step 42600",
    "spookynet_hybrid_step44600": "Hybrid step 44600",
    "spookynet_hybrid_step44600_mbd": "Hybrid step 44600 + MBD",
    "spookynet_hybrid_step45600": "Hybrid step 45600",
    "spookynet_hybrid_corrected_v2_step4600": "Spooky residual v2 step 4600",
    "hf_def2svp_cp":     "HF/def2-SVP (CP)",
    "mp2_def2svp_cp":    "MP2/def2-SVP (CP)",
    "hf_def2svp":        "HF/def2-SVP",
    "mp2_def2svp":       "MP2/def2-SVP",
    "b2plyp_def2svp_d3bj_cp":   "B2PLYP-D3BJ/def2-SVP (CP)",
    "dsdblyp_def2svp_d3bj_cp":  "DSD-BLYP-D3BJ/def2-SVP (CP)",
    "pwpb95_def2svp_d3bj_cp":   "PWPB95-D3BJ/def2-SVP (CP)",
    "hf_def2svp_gpu4pyscf_cp": "HF/def2-SVP (GPU4PySCF)",
    "mp2_def2svp_gpu4pyscf_cp": "MP2/def2-SVP (GPU4PySCF)",
    "pbe0_def2svp_gpu4pyscf_cp": "PBE0/def2-SVP (GPU4PySCF)",
    "pbe0_def2svp_gpu4pyscf_d3bj_cp": "PBE0-D3BJ/def2-SVP (GPU4PySCF)",
}

BACKEND_COLORS: dict[str, str] = {
    "learned_multipole": "#4e79a7",
    "learned_mbd":       "#f28e2b",
    "multipoles_mbd":    "#2ca02c",   # green — combined model
    "xtb_gfn2":          "#59a14f",
    "dftb3_d4":         "#edc948",
    "charmm":            "#e15759",
    "ccsd_def2svp_gpu4pyscf_cp": "#ff9da7",
    "ccsd_def2svpd_gpu4pyscf_cp": "#b07aa1",
    "spookynet":         "#b07aa1",
    "spookynet_hybrid":  "#9c755f",
    "hf_def2svp_cp":     "#76b7b2",
    "mp2_def2svp_cp":    "#edc948",
    "hf_def2svp":        "#76b7b2",
    "mp2_def2svp":       "#edc948",
    "b2plyp_def2svp_d3bj_cp":   "#af7aa1",
    "dsdblyp_def2svp_d3bj_cp":  "#ff9da7",
    "pwpb95_def2svp_d3bj_cp":   "#9c755f",
    "hf_def2svp_gpu4pyscf_cp": "#76b7b2",
    "mp2_def2svp_gpu4pyscf_cp": "#edc948",
    "pbe0_def2svp_gpu4pyscf_cp": "#59a14f",
    "pbe0_def2svp_gpu4pyscf_d3bj_cp": "#af7aa1",
}

BACKEND_CMAPS: dict[str, str] = {
    "learned_multipole": "RdBu_r",
    "learned_mbd":       "PuOr_r",
    "multipoles_mbd":    "PRGn_r",
    "xtb_gfn2":          "RdYlGn_r",
    "dftb3_d4":         "YlOrBr_r",
    "charmm":            "seismic",
    "ccsd_def2svp_gpu4pyscf_cp": "RdPu_r",
    "ccsd_def2svpd_gpu4pyscf_cp": "PuRd_r",
    "spookynet":         "BrBG_r",
    "spookynet_hybrid":  "PuOr",
    "spookynet_hybrid_step3000": "viridis",
    "spookynet_hybrid_hybrid_step3000": "viridis",
    "spookynet_hybrid_step3000_mbd": "viridis",
    "spookynet_hybrid_muon_epoch1_mbd": "viridis",
    "spookynet_hybrid_step19800": "viridis",
    "spookynet_hybrid_step19800_mbd": "viridis",
    "spookynet_hybrid_step36800": "viridis",
    "spookynet_hybrid_step36800_mbd": "viridis",
    "spookynet_hybrid_step42600": "viridis",
    "spookynet_hybrid_step44600": "viridis",
    "spookynet_hybrid_step44600_mbd": "viridis",
    "spookynet_hybrid_step45600": "viridis",
    "spookynet_hybrid_corrected_v2_step4600": "viridis",
    "hf_def2svp_cp":     "PuBuGn_r",
    "mp2_def2svp_cp":    "YlOrBr_r",
    "hf_def2svp":        "PuBuGn_r",
    "mp2_def2svp":       "YlOrBr_r",
    "b2plyp_def2svp_d3bj_cp":   "PuRd_r",
    "dsdblyp_def2svp_d3bj_cp":  "RdPu_r",
    "pwpb95_def2svp_d3bj_cp":   "copper_r",
    "hf_def2svp_gpu4pyscf_cp": "PuBuGn_r",
    "mp2_def2svp_gpu4pyscf_cp": "YlOrBr_r",
    "pbe0_def2svp_gpu4pyscf_cp": "GnBu_r",
    "pbe0_def2svp_gpu4pyscf_d3bj_cp": "PuRd_r",
}

# ── Backend grouping for harmonized colour scales ────────────────────────────
#
# Individual per-backend colormaps (BACKEND_CMAPS above) made cross-model
# comparison hard: every panel used a different diverging (two-hue,
# white-centered) colormap, so nothing about the *colour* was comparable
# between e.g. two SpookyNet checkpoints, let alone between an ML model and
# an ab initio reference. Group backends into families and give each family
# one shared, linear (sequential, single-hue) colormap instead: same colour
# = same physical meaning across every panel in that family, and equal energy
# differences map to equal colour differences everywhere (no special
# treatment of E_int=0 the way a diverging norm would give it).

GROUP_ML = "ml"
GROUP_QM = "qm"
GROUP_REFERENCE = "reference"

_ML_PREFIXES = ("spookynet", "learned_multipole", "learned_mbd", "multipoles_mbd")
_REFERENCE_BACKENDS = {"xtb_gfn2", "dftb3_d4", "charmm"}

GROUP_CMAPS: dict[str, str] = {
    GROUP_ML: "viridis",
    GROUP_QM: "magma",
    GROUP_REFERENCE: "cividis",
}

GROUP_LABELS: dict[str, str] = {
    GROUP_ML: "Learned / ML models",
    GROUP_QM: "Ab initio (HF/MP2/DFT)",
    GROUP_REFERENCE: "Empirical references (xTB/DFTB/CGenFF)",
}


def backend_group(backend: str) -> str:
    """Classify a backend name into a colour-scale family (ml/qm/reference)."""
    if backend in _REFERENCE_BACKENDS:
        return GROUP_REFERENCE
    if any(backend == p or backend.startswith(p) for p in _ML_PREFIXES):
        return GROUP_ML
    return GROUP_QM


def backend_cmap(backend: str) -> str:
    """Shared, linear (sequential) colormap for *backend*'s family."""
    return GROUP_CMAPS[backend_group(backend)]


# Canonical ordering for legend / subplot layout
BACKEND_ORDER: list[str] = [
    "multipoles_mbd",
    "learned_multipole",
    "learned_mbd",
    "spookynet",
    "spookynet_hybrid",
    "spookynet_hybrid_step3000",
    "spookynet_hybrid_hybrid_step3000",
    "spookynet_hybrid_step3000_mbd",
    "spookynet_hybrid_muon_epoch1_mbd",
    "spookynet_hybrid_step19800",
    "spookynet_hybrid_step19800_mbd",
    "spookynet_hybrid_step36800",
    "spookynet_hybrid_step36800_mbd",
    "spookynet_hybrid_step42600",
    "spookynet_hybrid_step44600",
    "spookynet_hybrid_step44600_mbd",
    "spookynet_hybrid_step45600",
    "spookynet_hybrid_corrected_v2_step4600",
    "xtb_gfn2",
    "dftb3_d4",
    "charmm",
    "ccsd_def2svp_gpu4pyscf_cp",
    "ccsd_def2svpd_gpu4pyscf_cp",
    "mp2_def2svp_gpu4pyscf_cp",
    "hf_def2svp_gpu4pyscf_cp",
    "pbe0_def2svp_gpu4pyscf_cp",
    "pbe0_def2svp_gpu4pyscf_d3bj_cp",
    "hf_def2svp_cp",
    "mp2_def2svp_cp",
    "hf_def2svp",
    "mp2_def2svp",
    "dsdblyp_def2svp_d3bj_cp",
    "b2plyp_def2svp_d3bj_cp",
    "pwpb95_def2svp_d3bj_cp",
]


# ── Key columns used to join multipole and MBD rows ─────────────────────────

_JOIN_KEYS = ["molecule_a", "molecule_b", "distance_angstrom", "offset_angstrom"]


def add_combined_backend(df: pd.DataFrame) -> pd.DataFrame:
    """Return *df* extended with a ``multipoles_mbd`` backend.

    The combined energy is the **sum** of ``learned_multipole`` and
    ``learned_mbd`` energies at matching (molecule_a, molecule_b,
    distance_angstrom, offset_angstrom) coordinates.

    Component columns (``comp_*``) are preserved from the multipole rows so
    the decomposition plots still work on the combined backend.

    If either backend is missing from *df*, the function returns *df* unchanged.
    """
    if "offset_angstrom" not in df.columns:
        df = df.copy()
        df["offset_angstrom"] = 0.0

    df_mp  = df[df["backend"] == "learned_multipole"].copy()
    df_mbd = df[df["backend"] == "learned_mbd"].copy()

    if df_mp.empty or df_mbd.empty:
        return df

    # Merge on geometry keys
    energy_cols = ["energy_ev", "energy_kcal_mol"]
    df_mp_e  = df_mp [_JOIN_KEYS + energy_cols].rename(
        columns={c: f"mp_{c}" for c in energy_cols}
    )
    df_mbd_e = df_mbd[_JOIN_KEYS + energy_cols].rename(
        columns={c: f"mbd_{c}" for c in energy_cols}
    )
    merged = df_mp_e.merge(df_mbd_e, on=_JOIN_KEYS, how="inner")
    if merged.empty:
        return df

    # Sum energies
    for c in energy_cols:
        merged[c] = merged[f"mp_{c}"] + merged[f"mbd_{c}"]
        merged.drop(columns=[f"mp_{c}", f"mbd_{c}"], inplace=True)

    merged["backend"] = "multipoles_mbd"

    # Carry component + contact-distance columns from multipole rows (same geometry)
    carry_cols = [c for c in df_mp.columns if c.startswith("comp_")]
    if "min_contact_angstrom" in df_mp.columns:
        carry_cols.append("min_contact_angstrom")
    if carry_cols:
        df_mp_comp = df_mp[_JOIN_KEYS + carry_cols]
        merged = merged.merge(df_mp_comp, on=_JOIN_KEYS, how="left")

    # Carry charmm-specific columns as NaN so concat works
    for extra in ["charmm_ELEC_kcal", "charmm_VDW_kcal"]:
        if extra in df.columns and extra not in merged.columns:
            merged[extra] = np.nan

    return pd.concat([df, merged], ignore_index=True)


def load_and_enrich(csv_path) -> pd.DataFrame:
    """Load a scan CSV and add the ``multipoles_mbd`` combined backend."""
    df = pd.read_csv(csv_path)
    if "offset_angstrom" not in df.columns:
        print("No 'offset_angstrom' column — adding 0.0 (treating as 1D scan).")
        df["offset_angstrom"] = 0.0
    return add_combined_backend(df)


# A geometric floor of 1.2 Å (roughly a bare covalent bond length) turns out
# to be too permissive: pairs like ACE+ACE stay strongly repulsive well past
# that (contact needs to clear ~1.4-1.5 Å before energies stop being
# dominated by the wall), so those in-between points were still slipping
# through and dominating colour scales. 1.5 Å is a safer default.
MIN_SAFE_CONTACT_ANGSTROM = 1.5


def flag_clashing_geometries(
    df: pd.DataFrame, min_contact: float = MIN_SAFE_CONTACT_ANGSTROM
) -> pd.DataFrame:
    """Mark scan rows whose fragments are at an unphysically close contact.

    ``distance_angstrom`` is measured between each monomer's chemically
    motivated anchor point, not its centroid or van der Waals surface — so a
    nominal "close" scan distance can put atoms from opposite fragments on
    top of each other (e.g. ACE+ACE needs ~5.25 Å center-to-center before
    fragment atoms stop overlapping). Those points produce huge, backend-
    dependent repulsive energies that dominate colour scales and make
    cross-backend comparisons meaningless. Adds a boolean ``is_clash``
    column; does not drop rows (caller decides whether to filter/mask).
    """
    df = df.copy()
    if "min_contact_angstrom" in df.columns:
        df["is_clash"] = df["min_contact_angstrom"] < min_contact
    else:
        df["is_clash"] = False
    return df


def flag_energy_outliers(
    df: pd.DataFrame, value_col: str, mad_thresh: float = 8.0
) -> pd.DataFrame:
    """Mark rows whose *value_col* is a robust outlier within *df*.

    Geometric clash filtering (``flag_clashing_geometries``) uses a single
    fixed contact-distance cutoff, which can still miss backend-specific
    energetic blow-ups (numerical instabilities right at the repulsive wall,
    or a pair-specific geometry the fixed cutoff didn't anticipate). This
    catches those directly from the energies themselves via a median absolute
    deviation (MAD) based z-score — robust to the very outliers it's
    detecting, unlike a mean/std z-score. Adds a boolean ``is_energy_outlier``
    column; does not drop rows.
    """
    df = df.copy()
    values = df[value_col].to_numpy(dtype=float)
    if len(values) < 4:
        df["is_energy_outlier"] = False
        return df
    med = np.median(values)
    mad = np.median(np.abs(values - med))
    if mad < 1e-9:
        df["is_energy_outlier"] = False
        return df
    robust_z = 0.6745 * (values - med) / mad
    df["is_energy_outlier"] = np.abs(robust_z) > mad_thresh
    return df


def robust_color_vmax(
    values: np.ndarray,
    *,
    percentile: float = 85.0,
    floor: float = 0.5,
    pad: float = 1.2,
    ceiling: float | None = None,
) -> float:
    """Pick a colour-scale ``vmax`` that isn't dominated by a repulsive wall.

    Uses a percentile (not max) of the *clean* (already clash/outlier
    filtered) energies so a handful of remaining steep points can't blow out
    the whole colour range, with a floor so near-flat surfaces still get
    visible contrast and an optional hard ceiling.
    """
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return floor
    vmax = float(np.percentile(np.abs(values), percentile)) * pad
    vmax = max(vmax, floor)
    if ceiling is not None:
        vmax = min(vmax, ceiling)
    return vmax


# ── Molecule rendering (bonds, depth transparency, optional force field) ────

def render_dimer_atoms(
    ax,
    atoms,
    fragments: tuple[np.ndarray, np.ndarray] | None = None,
    *,
    rotation: str = "15x,-20y,0z",
    radii_scale: float = 0.42,
    bond_cutoff_scale: float = 1.15,
    depth_alpha_range: tuple[float, float] = (0.4, 1.0),
    forces=None,
    force_scale: float = 1.5,
    force_color: str = "crimson",
    coord_axes: list[tuple[np.ndarray, str, str]] | None = None,
    title: str | None = None,
    title_fontsize: float = 7,
) -> None:
    """Ball-and-stick render of an ASE ``Atoms`` object onto *ax*.

    Draws covalent bonds (restricted to within-fragment pairs when
    *fragments* is given, so close intermolecular contacts don't render as
    spurious bonds) and depth-cues atoms with per-atom alpha so overlapping
    monomers in a dimer stay legible. If *forces* (N, 3) is given, overlays a
    2D-projected force-arrow field using the same camera rotation. If
    *coord_axes* is given (list of ``(vector3, color, label)`` in the same
    unrotated frame as the atoms), draws small arrows near the structure
    showing which direction each scan coordinate (e.g. approach distance,
    lateral offset) points, projected through the same camera rotation as
    everything else — so the arrows genuinely reflect the 3D geometry rather
    than being a flat, angle-independent annotation.
    """
    ax.set_axis_off()
    ax.set_aspect("equal")
    if title:
        ax.set_title(title, fontsize=title_fontsize, pad=1)
    if atoms is None or len(atoms) == 0:
        return

    pos = atoms.get_positions()
    Z = atoms.get_atomic_numbers()
    n = len(atoms)

    R = _ase_rotate(rotation)
    proj = pos @ R
    x, y, z = proj[:, 0], proj[:, 1], proj[:, 2]

    zmin, zmax = z.min(), z.max()
    depth_t = np.zeros(n) if zmax - zmin < 1e-9 else (z - zmin) / (zmax - zmin)
    alphas = depth_alpha_range[0] + depth_t * (depth_alpha_range[1] - depth_alpha_range[0])

    radii = covalent_radii[Z] * radii_scale
    colors = jmol_colors[Z]

    frag_id = np.zeros(n, dtype=int)
    if fragments is not None:
        for fi, idx in enumerate(fragments):
            frag_id[np.asarray(idx)] = fi

    # Bonds (behind atoms): within-fragment atom pairs closer than the
    # covalent-radius-sum cutoff. Restricting to same-fragment pairs avoids
    # drawing a "bond" for two monomers simply pushed close together.
    dmat = np.linalg.norm(pos[:, None, :] - pos[None, :, :], axis=-1)
    cutoff = bond_cutoff_scale * (covalent_radii[Z][:, None] + covalent_radii[Z][None, :])
    bonded = (dmat > 1e-6) & (dmat <= cutoff)
    for i in range(n):
        for j in range(i + 1, n):
            if not bonded[i, j]:
                continue
            if fragments is not None and frag_id[i] != frag_id[j]:
                continue
            ax.plot(
                [x[i], x[j]], [y[i], y[j]],
                color="#3a3a3a", lw=1.6, alpha=float(min(alphas[i], alphas[j])),
                zorder=1, solid_capstyle="round",
            )

    # Atoms, back-to-front so nearer atoms occlude farther ones correctly.
    for rank, i in enumerate(np.argsort(z)):
        ax.add_patch(
            Circle(
                (x[i], y[i]), radii[i],
                facecolor=colors[i], edgecolor="k", linewidth=0.4,
                alpha=float(alphas[i]), zorder=2 + rank * 0.001,
            )
        )

    if forces is not None:
        f_proj = np.asarray(forces) @ R
        ax.quiver(
            x, y, f_proj[:, 0], f_proj[:, 1],
            color=force_color, scale_units="xy", angles="xy",
            scale=1.0 / force_scale, width=0.01, zorder=10, alpha=0.9,
        )

    pad = (radii.max() if len(radii) else 1.0) * 2.2
    xlo, xhi = x.min() - pad, x.max() + pad
    ylo, yhi = y.min() - pad, y.max() + pad

    if coord_axes:
        # Anchor at the lower-left corner of the structure's bounding box
        # (offset slightly further out) so the arrows read as a small
        # coordinate-frame glyph rather than overlapping the molecule.
        span = max(xhi - xlo, yhi - ylo)
        arrow_len = span * 0.28
        origin = np.array([xlo + pad * 0.4, ylo + pad * 0.4])
        for vec3, color, label in coord_axes:
            v_proj = (np.asarray(vec3, dtype=float) @ R)[:2]
            norm = np.linalg.norm(v_proj)
            if norm < 1e-9:
                continue
            v_hat = v_proj / norm
            tip = origin + v_hat * arrow_len
            ax.annotate(
                "", xy=tuple(tip), xytext=tuple(origin),
                arrowprops=dict(arrowstyle="-|>", color=color, lw=1.8, shrinkA=0, shrinkB=0),
                zorder=11,
            )
            if label:
                label_pos = origin + v_hat * arrow_len * 1.25
                ax.text(
                    label_pos[0], label_pos[1], label, color=color,
                    fontsize=6, fontweight="bold", ha="center", va="center", zorder=11,
                )
                xlo, xhi = min(xlo, label_pos[0]), max(xhi, label_pos[0])
                ylo, yhi = min(ylo, label_pos[1]), max(yhi, label_pos[1])

    ax.set_xlim(xlo, xhi)
    ax.set_ylim(ylo, yhi)


def ordered_backends(df: pd.DataFrame, requested: list[str] | None = None) -> list[str]:
    """Return backends present in *df* sorted by ``BACKEND_ORDER``."""
    present = set(df["backend"].unique())
    pool = requested if requested else list(present)
    ordered = [b for b in BACKEND_ORDER if b in pool and b in present]
    # append any remaining backends not in BACKEND_ORDER
    for b in pool:
        if b in present and b not in ordered:
            ordered.append(b)
    return ordered
