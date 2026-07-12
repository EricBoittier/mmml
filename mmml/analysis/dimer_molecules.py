"""Reference molecular geometries and scan configurations for the 5-molecule dimer campaign.

Molecules
---------
DCM  – dichloromethane (CH₂Cl₂), NIST/literature geometry
ACE  – acetone (CH₃COCH₃), ASE experimental geometry
BENZ – benzene (C₆H₆), ASE (D₆ₕ, ring in XY plane)
TIP3 – water (H₂O), ASE
MEOH – methanol (CH₃OH), ASE

Scan design
-----------
All scans use Z as the approach axis (A centred at −d/2, B at +d/2).
Each pair has a chemically motivated pre-orientation:
  • A is rotated so its interaction vector points toward +Z (toward B).
  • B is rotated so its interaction vector points toward −Z (toward A).
The 2D scan sweeps a lateral offset along ``transverse_axis`` in addition
to the radial distance, capturing the physically relevant second coordinate
(H-bond lateral slip, π-stack slip, ring-offset, …).
"""

from __future__ import annotations

import numpy as np
from ase import Atoms
from ase.build import molecule as ase_molecule

from mmml.analysis.dimer_scans import centered_atoms


# ---------------------------------------------------------------------------
# Reference Geometries
# ---------------------------------------------------------------------------

def _make_tip3() -> Atoms:
    """H₂O TIP3P geometry from ASE: O at [0,0,0.119] Å, H at [0,±0.763,−0.477] Å."""
    return ase_molecule("H2O")


def _make_meoh() -> Atoms:
    """Methanol from ASE experimental geometry (C–O = 1.43 Å, O–H = 0.96 Å)."""
    return ase_molecule("CH3OH")


def _make_benz() -> Atoms:
    """Benzene from ASE (D₆ₕ, r_CC = 1.395 Å, ring in XY plane)."""
    return ase_molecule("C6H6")


def _make_ace() -> Atoms:
    """Acetone (CH₃COCH₃) from ASE experimental geometry (C=O along +Z)."""
    return ase_molecule("CH3COCH3")


def _make_dcm() -> Atoms:
    """Dichloromethane (CH₂Cl₂) with NIST literature geometry.

    r(C–Cl) = 1.765 Å, r(C–H) = 1.087 Å
    ∠ClCCl = 111.8 °,  ∠HCH = 112.0 °
    C₂ᵥ symmetry: Cl in XZ plane; H along ±Y/+Z.
    """
    cl_half = np.radians(111.8 / 2.0)
    h_half = np.radians(112.0 / 2.0)
    r_cl, r_h = 1.765, 1.087
    return Atoms(
        "CCl2H2",
        positions=[
            [0.0, 0.0, 0.0],                                              # C
            [ r_cl * np.sin(cl_half), 0.0, -r_cl * np.cos(cl_half)],     # Cl1
            [-r_cl * np.sin(cl_half), 0.0, -r_cl * np.cos(cl_half)],     # Cl2
            [0.0,  r_h * np.sin(h_half),  r_h * np.cos(h_half)],         # H1
            [0.0, -r_h * np.sin(h_half),  r_h * np.cos(h_half)],         # H2
        ],
    )


MOLECULES: dict[str, Atoms] = {
    "TIP3": _make_tip3(),
    "MEOH": _make_meoh(),
    "BENZ": _make_benz(),
    "ACE":  _make_ace(),
    "DCM":  _make_dcm(),
}


# ---------------------------------------------------------------------------
# Rotation Utilities
# ---------------------------------------------------------------------------

def rotation_matrix_align_to_z(vector: np.ndarray) -> np.ndarray:
    """Return a 3×3 rotation matrix *R* such that R @ vector/|vector| = [0, 0, 1].

    Uses Rodrigues' rotation formula for the general case; handles the special
    cases vector ∥ ±z analytically.
    """
    v = np.asarray(vector, dtype=float)
    v = v / np.linalg.norm(v)
    z = np.array([0.0, 0.0, 1.0])

    if np.allclose(v, z, atol=1e-8):
        return np.eye(3)
    if np.allclose(v, -z, atol=1e-8):
        # 180° around X axis
        return np.diag([1.0, -1.0, -1.0])

    axis = np.cross(v, z)
    axis /= np.linalg.norm(axis)
    angle = float(np.arccos(np.clip(np.dot(v, z), -1.0, 1.0)))
    c, s = np.cos(angle), np.sin(angle)
    K = np.array(
        [
            [0.0,     -axis[2],  axis[1]],
            [axis[2],  0.0,     -axis[0]],
            [-axis[1], axis[0],  0.0    ],
        ]
    )
    return c * np.eye(3) + s * K + (1.0 - c) * np.outer(axis, axis)


def orient_molecule(
    atoms: Atoms,
    interaction_vec: np.ndarray,
    *,
    point_toward_plus_z: bool,
) -> Atoms:
    """Return a *centred* copy of *atoms* rotated so *interaction_vec* points to ±Z.

    Parameters
    ----------
    atoms:
        Input molecule (any centering, will be centred first).
    interaction_vec:
        Direction from the molecular centroid toward the interaction site.
    point_toward_plus_z:
        If ``True``, the interaction vector is aligned to **+Z** (use for
        monomer A, which faces B at +Z).  If ``False``, aligned to **−Z**
        (use for monomer B, which faces A at −Z).
    """
    oriented = centered_atoms(atoms)
    v = np.asarray(interaction_vec, dtype=float)
    v = v / np.linalg.norm(v)

    # Aligning v to +Z  ↔  aligning −v to +Z means v ends at −Z
    R = rotation_matrix_align_to_z(v if point_toward_plus_z else -v)

    new_pos = oriented.get_positions() @ R.T
    oriented.set_positions(new_pos)
    return oriented


# ---------------------------------------------------------------------------
# Interaction Vectors
# ---------------------------------------------------------------------------

def _mean_unit(atoms: Atoms, indices: list[int]) -> np.ndarray:
    """Unit vector from geometric centroid to mean position of *indices*."""
    c = centered_atoms(atoms)
    pos = c.get_positions()
    vec = pos[indices].mean(axis=0)
    norm = float(np.linalg.norm(vec))
    if norm < 1e-8:
        raise ValueError(f"Degenerate interaction vector at indices {indices}.")
    return vec / norm


def _oh_h_index(atoms: Atoms) -> int:
    """Index of the hydroxyl H (closest H to the unique O) in methanol."""
    syms = atoms.get_chemical_symbols()
    pos = atoms.get_positions()
    o_idx = next(i for i, s in enumerate(syms) if s == "O")
    h_idxs = [i for i, s in enumerate(syms) if s == "H"]
    return h_idxs[int(np.argmin([np.linalg.norm(pos[i] - pos[o_idx]) for i in h_idxs]))]


def _o_index(atoms: Atoms) -> int:
    return next(i for i, s in enumerate(atoms.get_chemical_symbols()) if s == "O")


# ── TIP3 (H₂O): O=0, H=1,2 in ASE geometry ────────────────────────────────
# After centering: O at ~+0.40 Å in Z; H's at ~−0.20 Å in Z.
_TIP3_H_DONOR   = _mean_unit(MOLECULES["TIP3"], [1, 2])   # centroid → H bisector
_TIP3_O_ACCEPT  = _mean_unit(MOLECULES["TIP3"], [0])       # centroid → O (lone-pair side)

# ── MEOH (CH₃OH): ASE order C=0, O=1, H_OH=2(nearest to O), H×3 ──────────
_MEOH_H_DONOR  = _mean_unit(MOLECULES["MEOH"], [_oh_h_index(MOLECULES["MEOH"])])
_MEOH_O_ACCEPT = _mean_unit(MOLECULES["MEOH"], [_o_index(MOLECULES["MEOH"])])

# ── BENZ (C₆H₆): ring in XY plane → π-normal along Z ──────────────────────
_BENZ_PI = np.array([0.0, 0.0, 1.0])

# ── ACE (CH₃COCH₃): ASE order O=0, C(carbonyl)=1, C_Me=2,3, H×6 ──────────
_ACE_CO_O = _mean_unit(MOLECULES["ACE"], [0])    # centroid → carbonyl O

# ── DCM (CH₂Cl₂): C=0, Cl=1,2, H=3,4 ─────────────────────────────────────
_DCM_H_DONOR  = _mean_unit(MOLECULES["DCM"], [3, 4])   # centroid → H atoms
_DCM_CL_DIPOLE = _mean_unit(MOLECULES["DCM"], [1, 2])  # centroid → Cl atoms


# ---------------------------------------------------------------------------
# Pair Scan Configurations
# ---------------------------------------------------------------------------

# Each entry is keyed by the upper-triangular (A, B) pair label.
# Fields:
#   a_vec              – interaction vector of A (will be pointed toward +Z)
#   b_vec              – interaction vector of B (will be pointed toward −Z)
#   offsets_angstrom   – list of transverse displacements for the 2D scan
#   transverse_axis    – direction of lateral offset (orthogonal to Z)
#   description        – short physical description

PAIR_SCAN_CONFIG: dict[tuple[str, str], dict] = {
    # ── DCM pairs ──────────────────────────────────────────────────────────
    ("DCM", "DCM"): {
        # Head-to-tail: H(A)···Cl(B) along the molecular dipole direction
        "a_vec": _DCM_H_DONOR,
        "b_vec": _DCM_CL_DIPOLE,
        "offsets_angstrom": [0.0, 0.5, 1.0, 1.5, 2.0],
        "transverse_axis": [1, 0, 0],
        "description": "DCM head-to-tail H···Cl (dipole alignment)",
    },
    ("DCM", "ACE"): {
        # Weak H-bond: DCM C–H → ACE C=O oxygen
        "a_vec": _DCM_H_DONOR,
        "b_vec": _ACE_CO_O,
        "offsets_angstrom": [0.0, 0.5, 1.0, 1.5, 2.0],
        "transverse_axis": [1, 0, 0],
        "description": "DCM C–H···O=C (weak H-bond)",
    },
    ("DCM", "BENZ"): {
        # C–H···π: DCM H pointing into BENZ ring face
        "a_vec": _DCM_H_DONOR,
        "b_vec": _BENZ_PI,
        "offsets_angstrom": [0.0, 0.7, 1.4, 2.1, 2.8],
        "transverse_axis": [1, 0, 0],
        "description": "DCM C–H···π (ring-centre to off-axis)",
    },
    ("DCM", "TIP3"): {
        # Weak H-bond: DCM C–H → TIP3 O lone pairs
        "a_vec": _DCM_H_DONOR,
        "b_vec": _TIP3_O_ACCEPT,
        "offsets_angstrom": [0.0, 0.5, 1.0, 1.5, 2.0],
        "transverse_axis": [1, 0, 0],
        "description": "DCM C–H···O(H₂O) weak H-bond",
    },
    ("DCM", "MEOH"): {
        # Weak H-bond: DCM C–H → MEOH O lone pairs
        "a_vec": _DCM_H_DONOR,
        "b_vec": _MEOH_O_ACCEPT,
        "offsets_angstrom": [0.0, 0.5, 1.0, 1.5, 2.0],
        "transverse_axis": [1, 0, 0],
        "description": "DCM C–H···O(MeOH) weak H-bond",
    },
    # ── ACE pairs ──────────────────────────────────────────────────────────
    ("ACE", "ACE"): {
        # Antiparallel C=O — both O face the interface; lateral offset reveals
        # the true minimum of the antiparallel dipole-dipole interaction
        "a_vec": _ACE_CO_O,
        "b_vec": _ACE_CO_O,   # B's O also faces toward A → antiparallel C=O
        "offsets_angstrom": [0.0, 0.5, 1.0, 1.5, 2.5],
        "transverse_axis": [1, 0, 0],
        "description": "ACE antiparallel C=O···O=C (dipole–dipole)",
    },
    ("ACE", "BENZ"): {
        # n→π* / C=O···π: ACE C=O oxygen points toward BENZ π cloud
        "a_vec": _ACE_CO_O,
        "b_vec": _BENZ_PI,
        "offsets_angstrom": [0.0, 0.7, 1.4, 2.1, 2.8],
        "transverse_axis": [1, 0, 0],
        "description": "ACE C=O···π(BENZ) (n→π*-like)",
    },
    ("ACE", "TIP3"): {
        # H-bond: TIP3 O–H donates to ACE C=O
        "a_vec": _ACE_CO_O,
        "b_vec": _TIP3_H_DONOR,
        "offsets_angstrom": [0.0, 0.5, 1.0, 1.5, 2.0],
        "transverse_axis": [1, 0, 0],
        "description": "TIP3 O–H···O=C(ACE) H-bond",
    },
    ("ACE", "MEOH"): {
        # H-bond: MEOH O–H donates to ACE C=O
        "a_vec": _ACE_CO_O,
        "b_vec": _MEOH_H_DONOR,
        "offsets_angstrom": [0.0, 0.5, 1.0, 1.5, 2.0],
        "transverse_axis": [1, 0, 0],
        "description": "MeOH O–H···O=C(ACE) H-bond",
    },
    # ── BENZ pairs ─────────────────────────────────────────────────────────
    ("BENZ", "BENZ"): {
        # π–π stacking: face-to-face sandwich
        # 2D slip: 0 Å = perfect sandwich; ~1.4 Å = graphite stacking; ~2.8 Å = T-shape approach
        "a_vec": _BENZ_PI,
        "b_vec": _BENZ_PI,
        "offsets_angstrom": [0.0, 0.7, 1.4, 2.1, 2.8],
        "transverse_axis": [1, 0, 0],
        "description": "BENZ π-stacking sandwich → slip-stacked",
    },
    ("BENZ", "TIP3"): {
        # O–H···π: TIP3 H donor points toward BENZ ring face
        "a_vec": _BENZ_PI,
        "b_vec": _TIP3_H_DONOR,
        "offsets_angstrom": [0.0, 0.7, 1.4, 2.1],
        "transverse_axis": [1, 0, 0],
        "description": "TIP3 O–H···π(BENZ) (ring-centre to edge)",
    },
    ("BENZ", "MEOH"): {
        # O–H···π: MEOH O–H donor points toward BENZ ring face
        "a_vec": _BENZ_PI,
        "b_vec": _MEOH_H_DONOR,
        "offsets_angstrom": [0.0, 0.7, 1.4, 2.1],
        "transverse_axis": [1, 0, 0],
        "description": "MeOH O–H···π(BENZ) (ring-centre to edge)",
    },
    # ── TIP3 pairs ─────────────────────────────────────────────────────────
    ("TIP3", "TIP3"): {
        # Classic water dimer: A accepts, B donates (O–H···O linear)
        "a_vec": _TIP3_O_ACCEPT,
        "b_vec": _TIP3_H_DONOR,
        "offsets_angstrom": [0.0, 0.5, 1.0, 1.5, 2.0],
        "transverse_axis": [1, 0, 0],
        "description": "TIP3 O–H···O H-bond (water dimer)",
    },
    ("TIP3", "MEOH"): {
        # H-bond: MEOH O–H donates to TIP3 O lone pairs
        "a_vec": _TIP3_O_ACCEPT,
        "b_vec": _MEOH_H_DONOR,
        "offsets_angstrom": [0.0, 0.5, 1.0, 1.5, 2.0],
        "transverse_axis": [1, 0, 0],
        "description": "MeOH O–H···O(TIP3) H-bond",
    },
    # ── MEOH pairs ─────────────────────────────────────────────────────────
    ("MEOH", "MEOH"): {
        # MEOH H-bond chain: A accepts, B donates
        "a_vec": _MEOH_O_ACCEPT,
        "b_vec": _MEOH_H_DONOR,
        "offsets_angstrom": [0.0, 0.5, 1.0, 1.5, 2.0],
        "transverse_axis": [1, 0, 0],
        "description": "MeOH O–H···O H-bond (methanol dimer)",
    },
}


# ---------------------------------------------------------------------------
# Pre-oriented Monomers (computed at import time, cached)
# ---------------------------------------------------------------------------

ORIENTED_MONOMERS: dict[tuple[str, str], dict[str, Atoms]] = {}

for _pair, _cfg in PAIR_SCAN_CONFIG.items():
    _label_a, _label_b = _pair
    ORIENTED_MONOMERS[_pair] = {
        "a": orient_molecule(MOLECULES[_label_a], _cfg["a_vec"], point_toward_plus_z=True),
        "b": orient_molecule(MOLECULES[_label_b], _cfg["b_vec"], point_toward_plus_z=False),
    }


# ---------------------------------------------------------------------------
# Convenience Factory
# ---------------------------------------------------------------------------

def make_oriented_scan_geometries(
    label_a: str,
    label_b: str,
    distances_angstrom,
    offsets_angstrom=None,
):
    """Yield DimerGeometry objects for a pre-oriented (A, B) pair.

    Uses the chemically motivated orientations from ``PAIR_SCAN_CONFIG``.
    If *offsets_angstrom* is ``None``, uses the config defaults (2D scan).
    Pass ``[0.0]`` for a 1D on-axis scan only.

    Parameters
    ----------
    label_a, label_b:
        Molecule labels (must be keys of ``MOLECULES``).
    distances_angstrom:
        Iterable of centre-to-centre distances (Å) along Z.
    offsets_angstrom:
        Iterable of lateral displacements (Å); ``None`` → config defaults.
    """
    from mmml.analysis.dimer_scans import distance_scan_geometries_2d

    pair = (label_a, label_b)
    if pair not in PAIR_SCAN_CONFIG:
        raise KeyError(f"Pair {pair!r} not in PAIR_SCAN_CONFIG.")

    cfg = PAIR_SCAN_CONFIG[pair]
    monomers = ORIENTED_MONOMERS[pair]

    if offsets_angstrom is None:
        offsets_angstrom = cfg["offsets_angstrom"]

    yield from distance_scan_geometries_2d(
        monomers["a"],
        monomers["b"],
        distances_angstrom,
        offsets_angstrom,
        pair=pair,
        axis=(0.0, 0.0, 1.0),           # approach along Z
        transverse_axis=cfg["transverse_axis"],
        center="none",                   # monomers already centred and oriented
        mol_id_array="mol_id",
    )
