"""Shared plotting utilities for the dimer scan campaign.

Provides a single preprocessing function that enriches a raw scan DataFrame
with a combined ``multipoles_mbd`` backend (learned_multipole + learned_mbd)
so every plot script shows the total QM/ML interaction energy alongside the
individual components.
"""

from __future__ import annotations

import pandas as pd
import numpy as np

# ── Backend metadata ─────────────────────────────────────────────────────────

BACKEND_LABELS: dict[str, str] = {
    "learned_multipole": "Multipoles",
    "learned_mbd":       "MBD",
    "multipoles_mbd":    "Multipoles + MBD",
    "xtb_gfn2":          "GFN2-xTB",
    "charmm":            "CGenFF",
    "spookynet":         "SpookyNet",
    "spookynet_hybrid":  "SpookyNet Hybrid ML",
}

BACKEND_COLORS: dict[str, str] = {
    "learned_multipole": "#4e79a7",
    "learned_mbd":       "#f28e2b",
    "multipoles_mbd":    "#2ca02c",   # green — combined model
    "xtb_gfn2":          "#59a14f",
    "charmm":            "#e15759",
    "spookynet":         "#b07aa1",
    "spookynet_hybrid":  "#9c755f",
}

BACKEND_CMAPS: dict[str, str] = {
    "learned_multipole": "RdBu_r",
    "learned_mbd":       "PuOr_r",
    "multipoles_mbd":    "PRGn_r",
    "xtb_gfn2":          "RdYlGn_r",
    "charmm":            "seismic",
    "spookynet":         "BrBG_r",
    "spookynet_hybrid":  "PuOr",
}

# Canonical ordering for legend / subplot layout
BACKEND_ORDER: list[str] = [
    "multipoles_mbd",
    "learned_multipole",
    "learned_mbd",
    "spookynet_hybrid",
    "spookynet",
    "xtb_gfn2",
    "charmm",
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


MIN_SAFE_CONTACT_ANGSTROM = 1.2


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
