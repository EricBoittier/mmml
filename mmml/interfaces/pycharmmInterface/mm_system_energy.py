"""Full-system JAX MM energy: bonded terms + CHARMM-style switched nonbonded (MIC PBC).

Used to cross-check solvated peptide/water boxes against PyCHARMM ``ENER FORCE``
without the monomer-decomposed COM switching in :mod:`mm_energy_forces`.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array

from mmml.interfaces.pycharmmInterface.cgenff_bonded import bonded_energy_and_forces
from mmml.interfaces.pycharmmInterface.cgenff_topology import (
    CgenffBondedSystem,
    load_cgenff_bonded_from_psf,
)
from mmml.interfaces.pycharmmInterface.long_range_backend import (
    box_length_from_cell,
    compute_jax_pme_coulomb,
    pick_lr_solver,
    resolve_jax_pme_dispersion,
    resolve_jax_pme_method,
)
from mmml.interfaces.pycharmmInterface.pbc_utils_jax import mic_displacement

COULOMB_KCAL = 332.063711


@dataclass(frozen=True, slots=True)
class CharmmNbondSettings:
    """CHARMM ``nbonds`` switched cutoffs (Å) for cdie + VDW switching.

    ``elec_switch`` / ``vdw_switch`` select CHARMM ``enbfast`` / ``enbaexp`` modes:

    - ``fshift`` — cdie force shift (``CSHIFT``; CGENFF PRM ``fshift`` keyword)
    - ``fswitch`` — cdie force switch (``CFSWIT``; PyCHARMM ``fswitch=True``)
    - ``pswitch`` — potential switch (``CSWIT`` / ``LVSW``; legacy JAX default)
    - ``vfswitch`` — VDW force switch (``LVFSW``; PyCHARMM ``vfswitch=True``)
    """

    cutnb: float
    ctonnb: float
    ctofnb: float
    eps: float = 1.0
    e14fac: float = 1.0
    vdw14fac: float = 0.0
    elec_switch: str = "fswitch"
    vdw_switch: str = "vfswitch"

    @property
    def c2onnb(self) -> float:
        return float(self.ctonnb) ** 2

    @property
    def c2ofnb(self) -> float:
        return float(self.ctofnb) ** 2

    @property
    def ctrof2(self) -> float:
        return -1.0 / self.c2ofnb

    @property
    def min2of(self) -> float:
        return 2.0 / float(self.ctofnb)

    @property
    def rul3(self) -> float:
        denom = self.c2ofnb - self.c2onnb
        if abs(denom) < 1e-12:
            return 0.0
        return 1.0 / (denom**3)

    @property
    def rul12(self) -> float:
        denom = self.c2ofnb - self.c2onnb
        if abs(denom) < 1e-12:
            return 0.0
        return 12.0 * self.rul3


@dataclass(frozen=True, slots=True)
class CharmmVfswitchCoeffs:
    """Precomputed VDW force-switch coefficients (``LVFSW`` init in ``enbfast``)."""

    recof6: float
    recof3: float
    onoff6: float
    onoff3: float
    ofdif6: float
    ofdif3: float


@dataclass(frozen=True, slots=True)
class CharmmFswitchCoeffs:
    """Precomputed cdie force-switch coefficients (``CFSWIT`` init in ``enbfast``)."""

    acoef: float
    bcoef: float
    cover3: float
    dover5: float
    const: float
    eadd: float


def charmm_vfswitch_coeffs(settings: CharmmNbondSettings) -> CharmmVfswitchCoeffs:
    """Coefficients for CHARMM VDW force switch (``LVFSW``)."""
    c2of = settings.c2ofnb
    b = float(settings.ctofnb)
    off3 = c2of * b
    off6 = off3 * off3
    recof6 = 1.0 / off6
    if float(settings.ctonnb) < b:
        c2on = settings.c2onnb
        on3 = c2on * float(settings.ctonnb)
        on6 = on3 * on3
        recof3 = 1.0 / off3
        ofdif6 = off6 / (off6 - on6)
        ofdif3 = off3 / (off3 - on3)
        onoff6 = recof6 / on6
        onoff3 = recof3 / on3
    else:
        onoff6 = recof6 * recof6
        onoff3 = recof6
        recof3 = 1.0 / off3
        ofdif6 = 1.0
        ofdif3 = 1.0
    return CharmmVfswitchCoeffs(
        recof6=recof6,
        recof3=recof3,
        onoff6=onoff6,
        onoff3=onoff3,
        ofdif6=ofdif6,
        ofdif3=ofdif3,
    )


def charmm_fswitch_coeffs(settings: CharmmNbondSettings) -> CharmmFswitchCoeffs:
    """Coefficients for CHARMM cdie force switch (``CFSWIT``)."""
    c2on = settings.c2onnb
    c2of = settings.c2ofnb
    b = float(settings.ctofnb)
    cton = float(settings.ctonnb)
    if cton < b:
        onoff2 = c2on * c2of
        on3 = c2on * cton
        off3 = c2of * b
        off4 = c2of * c2of
        off5 = off3 * c2of
        denom = 1.0 / (c2of - c2on) ** 3
        eadd = (onoff2 * (b - cton) - (off5 - on3 * c2on) / 5.0) * 8.0 * denom
        acoef = off4 * (c2of - 3.0 * c2on) * denom
        bcoef = 6.0 * onoff2 * denom
        cover3 = -(c2on + c2of) * denom
        dover5 = 2.0 * denom / 5.0
        const = bcoef * b - acoef / b + cover3 * off3 + dover5 * off5
    else:
        eadd = -1.0 / b
        acoef = bcoef = cover3 = dover5 = const = 0.0
    return CharmmFswitchCoeffs(
        acoef=acoef,
        bcoef=bcoef,
        cover3=cover3,
        dover5=dover5,
        const=const,
        eadd=eadd,
    )


@dataclass(frozen=True, slots=True)
class NonbondedSystemData:
    """Per-atom nonbonded parameters and PSF exclusion list."""

    charges: np.ndarray
    at_codes: np.ndarray
    epsilon: np.ndarray
    rmin: np.ndarray
    excluded_pairs: frozenset[tuple[int, int]]
    e14_pairs: frozenset[tuple[int, int]]
    psf_path: Path | None = None
    psf_bonds: np.ndarray | None = None


@dataclass(frozen=True, slots=True)
class MmSystemEnergyResult:
    bonded: dict[str, float]
    nonbonded: dict[str, float]
    total_energy: float
    forces: np.ndarray


def charmm_switch_factor(r_sq: Array, settings: CharmmNbondSettings) -> Array:
    """CHARMM CSWIT/DSWIT potential switch (0 outside ``ctofnb``)."""
    c2on = settings.c2onnb
    c2of = settings.c2ofnb
    rijl = c2on - r_sq
    riju = c2of - r_sq
    funct = riju * riju * (riju - 3.0 * rijl) * settings.rul3
    inside = (r_sq <= c2of) & (r_sq > c2on)
    below = r_sq <= c2on
    return jnp.where(below, 1.0, jnp.where(inside, funct, 0.0))


def charmm_fshift_elec(r: Array, qq: Array, settings: CharmmNbondSettings) -> Array:
    """CHARMM cdie force shift (``CSHIFT`` / PRM ``fshift``).

    ``qq`` is ``q_i * q_j / eps`` (no Coulomb constant); multiply by ``COULOMB_KCAL``.
    """
    r = jnp.asarray(r, dtype=jnp.float64)
    qq = jnp.asarray(qq, dtype=jnp.float64)
    r_safe = jnp.maximum(r, 1e-10)
    r_sq = r_safe * r_safe
    r1 = 1.0 / r_safe
    ch = qq * r1
    ctrof2 = jnp.asarray(settings.ctrof2, dtype=jnp.float64)
    min2of = jnp.asarray(settings.min2of, dtype=jnp.float64)
    return ch * (1.0 + r_sq * (min2of * r1 - ctrof2))


def charmm_fswitch_elec(
    r: Array,
    qq: Array,
    settings: CharmmNbondSettings,
    coeffs: CharmmFswitchCoeffs,
) -> Array:
    """CHARMM cdie force switch (``CFSWIT`` / PyCHARMM ``fswitch``)."""
    r = jnp.asarray(r, dtype=jnp.float64)
    qq = jnp.asarray(qq, dtype=jnp.float64)
    r_safe = jnp.maximum(r, 1e-10)
    r_sq = r_safe * r_safe
    r1 = 1.0 / r_safe
    outer = r_sq > settings.c2onnb
    acoef = jnp.asarray(coeffs.acoef, dtype=jnp.float64)
    bcoef = jnp.asarray(coeffs.bcoef, dtype=jnp.float64)
    cover3 = jnp.asarray(coeffs.cover3, dtype=jnp.float64)
    dover5 = jnp.asarray(coeffs.dover5, dtype=jnp.float64)
    const = jnp.asarray(coeffs.const, dtype=jnp.float64)
    eadd = jnp.asarray(coeffs.eadd, dtype=jnp.float64)
    inner = r1 + eadd
    switched = r1 * (acoef - r_sq * (bcoef + r_sq * (cover3 + dover5 * r_sq))) + const
    return qq * jnp.where(outer, switched, inner)


def charmm_vfswitch_vdw(
    r: Array,
    a_coef: Array,
    b_coef: Array,
    settings: CharmmNbondSettings,
    coeffs: CharmmVfswitchCoeffs,
) -> Array:
    """CHARMM VDW force switch (``LVFSW`` / PyCHARMM ``vfswitch``).

    ``a_coef`` = ``epsilon * sigma^12``, ``b_coef`` = ``2 * epsilon * sigma^6``.
    """
    r = jnp.asarray(r, dtype=jnp.float64)
    a_coef = jnp.asarray(a_coef, dtype=jnp.float64)
    b_coef = jnp.asarray(b_coef, dtype=jnp.float64)
    r_safe = jnp.maximum(r, 1e-10)
    r_sq = r_safe * r_safe
    r1 = 1.0 / r_safe
    tr2 = r1 * r1
    tr6 = tr2 * tr2 * tr2
    outer = r_sq > settings.c2onnb
    recof6 = jnp.asarray(coeffs.recof6, dtype=jnp.float64)
    recof3 = jnp.asarray(coeffs.recof3, dtype=jnp.float64)
    onoff6 = jnp.asarray(coeffs.onoff6, dtype=jnp.float64)
    onoff3 = jnp.asarray(coeffs.onoff3, dtype=jnp.float64)
    ofdif6 = jnp.asarray(coeffs.ofdif6, dtype=jnp.float64)
    ofdif3 = jnp.asarray(coeffs.ofdif3, dtype=jnp.float64)
    r3 = r1 * tr2
    rjunk6 = tr6 - recof6
    rjunk3 = r3 - recof3
    cr12 = a_coef * ofdif6 * rjunk6
    cr6 = b_coef * ofdif3 * rjunk3
    switched = cr12 * rjunk6 - cr6 * rjunk3
    ca = a_coef * tr6 * tr6
    enevdw = ca - b_coef * tr6
    inner = enevdw + b_coef * onoff3 - a_coef * onoff6
    return jnp.where(outer, switched, inner)


def _pair_elec_energy(
    r: Array,
    qq: Array,
    settings: CharmmNbondSettings,
    fswitch_coeffs: CharmmFswitchCoeffs,
) -> Array:
    mode = settings.elec_switch
    if mode == "fshift":
        raw = charmm_fshift_elec(r, qq, settings)
    elif mode == "fswitch":
        raw = charmm_fswitch_elec(r, qq, settings, fswitch_coeffs)
    elif mode == "pswitch":
        r_sq = r * r
        coul = qq / jnp.maximum(r, 1e-10)
        return coul * charmm_switch_factor(r_sq, settings) * COULOMB_KCAL
    else:
        raise ValueError(f"unknown elec_switch {mode!r}")
    return COULOMB_KCAL * raw


def _pair_vdw_energy(
    r: Array,
    ep: Array,
    sig: Array,
    settings: CharmmNbondSettings,
    vfswitch_coeffs: CharmmVfswitchCoeffs,
    *,
    use_jax_pme_dispersion: bool,
) -> Array:
    a_coef = ep * sig**12
    b_coef = 2.0 * ep * sig**6
    mode = settings.vdw_switch
    if mode == "vfswitch":
        return charmm_vfswitch_vdw(r, a_coef, b_coef, settings, vfswitch_coeffs)
    if mode == "pswitch":
        r_sq = r * r
        r_safe = jnp.maximum(r, 1e-10)
        sig_r6 = (sig / r_safe) ** 6
        vdw_r12 = ep * (sig_r6 * sig_r6)
        vdw_full = ep * (sig_r6 * sig_r6 - 2.0 * sig_r6)
        vdw = vdw_r12 if use_jax_pme_dispersion else vdw_full
        return vdw * charmm_switch_factor(r_sq, settings)
    raise ValueError(f"unknown vdw_switch {mode!r}")


def fully_excluded_pairs(iblo: Iterable[int], inb: Iterable[int], natom: int) -> frozenset[tuple[int, int]]:
    """Return 0-based atom pairs fully excluded by CHARMM ``IBLO``/``INB``."""
    iblo_list = list(iblo)
    inb_list = list(inb)
    if natom <= 0:
        return frozenset()
    if not inb_list:
        return frozenset()
    excluded: set[tuple[int, int]] = set()
    for i in range(natom):
        if i >= len(iblo_list):
            break
        start = int(iblo_list[i]) - 1
        if start < 0:
            continue
        end = int(iblo_list[i + 1]) - 1 if i + 1 < len(iblo_list) else len(inb_list)
        end = min(end, len(inb_list))
        for idx in range(start, end):
            j = int(inb_list[idx]) - 1
            if j < 0 or j >= natom:
                continue
            a, b = (i, j) if i < j else (j, i)
            excluded.add((a, b))
    return frozenset(excluded)


def excluded_pairs_from_psf_inb_iblo(
    inb: np.ndarray,
    iblo: np.ndarray,
    natom: int,
) -> frozenset[tuple[int, int]]:
    """Build exclusion pairs from PSF ``INB`` + ``IBLO`` (CHARMM formatted PSF)."""
    if int(natom) <= 0 or inb.size == 0 or iblo.size == 0:
        return frozenset()
    return fully_excluded_pairs(iblo, inb, int(natom))


def excluded_pairs_from_psf_nnb(
    nnb_indices: np.ndarray,
    natom: int,
    *,
    iblo_indices: np.ndarray | None = None,
) -> frozenset[tuple[int, int]]:
    """Build fully excluded pairs from PSF ``!NNB`` data.

    CHARMM PSF EXT files store a flat ``INB`` partner list plus per-atom ``IBLO``
    pointers (see ``psfres.F90``).  When ``iblo_indices`` is supplied, use the
    Fortran layout via :func:`fully_excluded_pairs`.  Otherwise fall back to the
    legacy packed ``count + partners`` encoding (rare in MMML PSF fixtures).
    """
    iblo = np.asarray(iblo_indices, dtype=np.int32) if iblo_indices is not None else None
    inb = np.asarray(nnb_indices, dtype=np.int32)
    if iblo is not None and iblo.size > 0 and inb.size > 0:
        return excluded_pairs_from_psf_inb_iblo(inb, iblo, natom)
    if inb.size == 0 or natom <= 0:
        return frozenset()
    excluded: set[tuple[int, int]] = set()
    cursor = 0
    for i in range(natom):
        if cursor >= inb.size:
            break
        n_excl = int(inb[cursor])
        cursor += 1
        for _ in range(n_excl):
            if cursor >= inb.size:
                break
            j = int(inb[cursor]) - 1
            cursor += 1
            if j < 0 or j >= natom:
                continue
            a, b = (i, j) if i < j else (j, i)
            excluded.add((a, b))
    return frozenset(excluded)


def _excluded_pairs_from_psf_file(
    psf_path: Path | str,
    bonds: np.ndarray,
    *,
    natom: int,
) -> frozenset[tuple[int, int]]:
    """Load exclusions from a PSF on disk (``INB``/``IBLO`` or bond fallback)."""
    from mmml.interfaces.pycharmmInterface.cgenff_topology import parse_psf_ext

    psf_data = parse_psf_ext(psf_path)
    if psf_data.nnb_indices.size > 0 and psf_data.iblo_indices.size > 0:
        from_psf = excluded_pairs_from_psf_inb_iblo(
            psf_data.nnb_indices,
            psf_data.iblo_indices,
            natom,
        )
        if from_psf:
            return from_psf
    if psf_data.nnb_indices.size > 0:
        from_packed = excluded_pairs_from_psf_nnb(psf_data.nnb_indices, natom)
        if from_packed:
            return from_packed
    return excluded_pairs_from_psf_bonds(bonds)


def resolve_nonbonded_excluded_pairs(
    psf_path: Path | str,
    bonds: np.ndarray,
    *,
    natom: int,
) -> frozenset[tuple[int, int]]:
    """Return CHARMM-style excluded pairs for JAX MIC pair lists.

    Prefer PSF ``INB``/``IBLO`` from ``psf_path`` (written after ``upinb`` at box
    build).  Live PyCHARMM ``get_iblo_inb()`` often returns ``nnb=0`` after param
    reads even though the saved PSF carries the full exclusion table.  Fall back to
    bond-derived 1–2/1–3 pairs only when neither PSF nor live lists are available.
    """
    from_psf = _excluded_pairs_from_psf_file(psf_path, bonds, natom=natom)
    if from_psf:
        return from_psf

    try:
        import pycharmm.nbonds as nbonds
        import pycharmm.psf as psf

        try:
            nbonds.update_bnbnd()
        except Exception:
            pass
        iblo, inb = psf.get_iblo_inb()
        excluded = fully_excluded_pairs(iblo, inb, natom)
    except Exception:
        excluded = frozenset()

    if excluded:
        return excluded

    return excluded_pairs_from_psf_bonds(bonds)


def excluded_pairs_from_psf_bonds(bonds: np.ndarray) -> frozenset[tuple[int, int]]:
    """Build CHARMM-style 1–2 and 1–3 exclusion pairs from PSF bonds (0-based)."""
    from mmml.utils.geometry_checks import build_bond_exclusion_pairs

    bonds = np.asarray(bonds, dtype=np.int32)
    if bonds.size == 0:
        return frozenset()
    ib = bonds[:, 0] + 1
    jb = bonds[:, 1] + 1
    return build_bond_exclusion_pairs(ib, jb, exclude_1_3=True)


def one_four_pairs_from_bonds(bonds: np.ndarray, natom: int) -> frozenset[tuple[int, int]]:
    """Infer 1–4 pairs (0-based) from PSF bond list for ``e14fac`` electrostatic scaling."""
    neighbors: dict[int, set[int]] = {i: set() for i in range(natom)}
    for i_raw, j_raw in bonds:
        i, j = int(i_raw), int(j_raw)
        neighbors[i].add(j)
        neighbors[j].add(i)

    pairs: set[tuple[int, int]] = set()
    for a, b in bonds:
        a_i, b_i = int(a), int(b)
        for c in neighbors[a_i]:
            if c == b_i:
                continue
            for d in neighbors[c]:
                if d in (a_i, b_i):
                    continue
                pairs.add((min(a_i, d), max(a_i, d)))
                pairs.add((min(b_i, d), max(b_i, d)))
    return frozenset(pairs)


def _pair_lj_epsilon(ep_i: Array, ep_j: Array) -> Array:
    """CHARMM geometric ε_ij = sqrt(|ε_i ε_j|) (kcal/mol, positive)."""
    return jnp.sqrt(jnp.abs(ep_i * ep_j))


def _live_charmm_nonbonded_arrays(
    natom: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    """Per-atom charges, ε, and Rmin/2 from the active PyCHARMM session."""
    try:
        import pycharmm.param as param
        import pycharmm.psf as psf

        charges = np.asarray(psf.get_charges(), dtype=np.float64)
        epsilon = np.asarray(param.get_epsilon(), dtype=np.float64)
        rmin = np.asarray(param.get_vdwr(), dtype=np.float64)
        if charges.shape[0] != natom or epsilon.shape[0] != natom or rmin.shape[0] != natom:
            return None
        return charges, epsilon, rmin
    except Exception:
        return None


def parse_lj_tables_from_prm(prm_path: Path | str) -> dict[str, tuple[float, float]]:
    """Parse CHARMM NONBONDED epsilon (kcal/mol) and Rmin/2 (Å) by atom type."""
    tables: dict[str, tuple[float, float]] = {}
    for line in Path(prm_path).read_text(encoding="utf-8", errors="replace").splitlines():
        if len(line) <= 5 or line[0] == "!":
            continue
        parts = line.split()
        if len(parts) < 4 or parts[1] != "0.0":
            continue
        atype, ep, sig = parts[0], parts[2], parts[3]
        try:
            tables[atype] = (float(ep), float(sig))
        except ValueError:
            continue
    return tables


def load_nonbonded_system_from_charmm(
    psf_path: Path | str,
    *prm_paths: Path | str,
) -> NonbondedSystemData:
    """Load charges, LJ tables, and exclusions from the active PyCHARMM PSF."""
    from mmml.interfaces.pycharmmInterface.cgenff_topology import parse_psf_ext

    psf_data = parse_psf_ext(psf_path)
    natom = psf_data.n_atoms

    live = _live_charmm_nonbonded_arrays(natom)
    if live is not None:
        charges, atom_eps, atom_rmin = live
    else:
        lj: dict[str, tuple[float, float]] = {}
        for prm_path in prm_paths:
            lj.update(parse_lj_tables_from_prm(prm_path))
        atom_eps = np.array(
            [lj.get(str(t), (0.0, 0.0))[0] for t in psf_data.atom_types],
            dtype=np.float64,
        )
        atom_rmin = np.array(
            [lj.get(str(t), (0.0, 0.0))[1] for t in psf_data.atom_types],
            dtype=np.float64,
        )
        charges = np.asarray(psf_data.charges, dtype=np.float64)
    try:
        import pycharmm.psf as psf

        iac = np.asarray(psf.get_iac(), dtype=np.int32)
    except Exception:
        iac = np.ones(natom, dtype=np.int32)
    # TIP3 ``HT`` and other types with ``iac==0`` carry no CHARMM VDW term.
    zero_lj = iac <= 0
    atom_eps = np.asarray(atom_eps, dtype=np.float64)
    atom_rmin = np.asarray(atom_rmin, dtype=np.float64)
    atom_eps[zero_lj] = 0.0
    atom_rmin[zero_lj] = 0.0
    at_codes = np.zeros(natom, dtype=np.int32)

    excluded = resolve_nonbonded_excluded_pairs(
        psf_path,
        psf_data.bonds,
        natom=natom,
    )
    e14 = one_four_pairs_from_bonds(psf_data.bonds, natom) - excluded

    return NonbondedSystemData(
        charges=charges,
        at_codes=at_codes,
        epsilon=atom_eps,
        rmin=atom_rmin,
        excluded_pairs=excluded,
        e14_pairs=e14,
        psf_path=Path(psf_path),
        psf_bonds=np.asarray(psf_data.bonds, dtype=np.int32),
    )


def _build_pair_indices(
    positions: np.ndarray,
    cell: np.ndarray,
    excluded: frozenset[tuple[int, int]],
    cutoff: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Host-side O(N²) MIC pair list (``i < j``, ``r < cutoff``)."""
    pos = np.asarray(positions, dtype=np.float64)
    cell_mat = np.asarray(cell, dtype=np.float64)
    if cell_mat.shape == (3,):
        cell_mat = np.diag(cell_mat)
    inv = np.linalg.inv(cell_mat)
    cutoff_sq = float(cutoff) ** 2
    n = pos.shape[0]
    pairs_i: list[int] = []
    pairs_j: list[int] = []
    for i in range(n):
        for j in range(i + 1, n):
            if (i, j) in excluded:
                continue
            dr = pos[j] - pos[i]
            frac = dr @ inv.T
            frac = frac - np.round(frac)
            dr_mic = frac @ cell_mat
            r_sq = float(np.dot(dr_mic, dr_mic))
            if r_sq < cutoff_sq:
                pairs_i.append(i)
                pairs_j.append(j)
    return (
        np.asarray(pairs_i, dtype=np.int32),
        np.asarray(pairs_j, dtype=np.int32),
    )


def nonbonded_energy_and_forces(
    positions: Array | np.ndarray,
    nbond_data: NonbondedSystemData,
    cell: Array | np.ndarray,
    settings: CharmmNbondSettings,
    *,
    pair_i: np.ndarray | None = None,
    pair_j: np.ndarray | None = None,
    lr_solver: str | None = None,
    jax_pme_method: str | None = None,
    jax_pme_sr_cutoff_A: float = 6.0,
    jax_pme_dispersion: bool | None = None,
) -> tuple[dict[str, Array], Array]:
    """Switched VDW + Coulomb for all atom pairs within ``cutnb``.

    When ``lr_solver`` resolves to ``jax_pme``, the pair list supplies switched
    r⁻¹² repulsion (LB σ_ij, ε_ij) and Coulomb plus r⁻⁶ dispersion are evaluated
    with jax-pme (Ewald/PME/P3M).
    """
    use_jax_pme = pick_lr_solver(lr_solver) == "jax_pme"
    use_jax_pme_dispersion = use_jax_pme and resolve_jax_pme_dispersion(jax_pme_dispersion)
    pme_method = resolve_jax_pme_method(jax_pme_method)
    pos = jnp.asarray(positions, dtype=jnp.float64)
    cell_j = jnp.asarray(cell, dtype=jnp.float64)
    if cell_j.ndim == 1 and cell_j.shape[0] == 3:
        cell_j = jnp.diag(cell_j)

    excluded_pairs = nbond_data.excluded_pairs
    if nbond_data.psf_path is not None and nbond_data.psf_bonds is not None:
        excluded_pairs = resolve_nonbonded_excluded_pairs(
            nbond_data.psf_path,
            nbond_data.psf_bonds,
            natom=int(np.asarray(nbond_data.charges).shape[0]),
        )

    if pair_i is None or pair_j is None:
        host_i, host_j = _build_pair_indices(
            np.asarray(positions),
            np.asarray(cell),
            excluded_pairs,
            settings.cutnb,
        )
        pair_i = host_i
        pair_j = host_j

    pi = jnp.asarray(pair_i, dtype=jnp.int32)
    pj = jnp.asarray(pair_j, dtype=jnp.int32)

    e14_scale_np = np.ones(len(pair_i), dtype=np.float64)
    vdw14_scale_np = np.ones(len(pair_i), dtype=np.float64)
    for k, (i, j) in enumerate(zip(pair_i, pair_j, strict=True)):
        if (int(i), int(j)) in nbond_data.e14_pairs:
            e14_scale_np[k] = settings.e14fac
            vdw14_scale_np[k] = settings.vdw14fac

    q = jnp.asarray(nbond_data.charges, dtype=jnp.float64)
    eps_tbl = jnp.asarray(nbond_data.epsilon, dtype=jnp.float64)
    rm_tbl = jnp.asarray(nbond_data.rmin, dtype=jnp.float64)
    e14_scale = jnp.asarray(e14_scale_np, dtype=jnp.float64)
    vdw14_scale = jnp.asarray(vdw14_scale_np, dtype=jnp.float64)
    vfswitch_coeffs = charmm_vfswitch_coeffs(settings)
    fswitch_coeffs = charmm_fswitch_coeffs(settings)
    c2of = settings.c2ofnb

    def _pair_terms(positions_arg: Array) -> tuple[Array, Array, Array]:
        ri = positions_arg[pi]
        rj = positions_arg[pj]
        disp = jax.vmap(lambda a, b: mic_displacement(a, b, cell_j))(ri, rj)
        r = jnp.linalg.norm(disp, axis=-1)
        r_sq = r * r
        within_ctof = r_sq < c2of

        ep_i = eps_tbl[pi]
        ep_j = eps_tbl[pj]
        rm_i = rm_tbl[pi]
        rm_j = rm_tbl[pj]
        sig = rm_i + rm_j
        ep = _pair_lj_epsilon(ep_i, ep_j)

        qq = q[pi] * q[pj] * e14_scale / settings.eps
        vdw = _pair_vdw_energy(
            r,
            ep,
            sig,
            settings,
            vfswitch_coeffs,
            use_jax_pme_dispersion=use_jax_pme_dispersion,
        )
        vdw = vdw * vdw14_scale
        elec = _pair_elec_energy(r, qq, settings, fswitch_coeffs)

        vdw = jnp.where(within_ctof, vdw, 0.0)
        elec = jnp.where(within_ctof, elec, 0.0)

        vdw_sw = jnp.sum(vdw)
        if use_jax_pme:
            elec_sw = jnp.array(0.0, dtype=pos.dtype)
        else:
            elec_sw = jnp.sum(elec)
        return vdw_sw, elec_sw, vdw_sw + elec_sw

    def _energy(positions_arg: Array) -> Array:
        return _pair_terms(positions_arg)[2]

    energy = _energy(pos)
    vdw_energy, elec_energy, _ = _pair_terms(pos)
    forces = -jax.grad(_energy)(pos)
    if use_jax_pme:
        pos_np = np.asarray(positions, dtype=np.float64)
        pme = compute_jax_pme_coulomb(
            pos_np,
            nbond_data.charges,
            box_length_A=box_length_from_cell(np.asarray(cell)),
            method=pme_method,
            sr_cutoff_A=float(jax_pme_sr_cutoff_A),
        )
        elec_energy = jnp.asarray(pme.energy_kcalmol, dtype=pos.dtype)
        forces = forces + jnp.asarray(pme.forces_kcalmol_A, dtype=pos.dtype)
        if use_jax_pme_dispersion:
            from mmml.interfaces.pycharmmInterface.long_range_backend import (
                compute_jax_pme_lj_dispersion,
                per_atom_jax_pme_c6_sqrt,
            )

            c6_sqrt = per_atom_jax_pme_c6_sqrt(
                np.abs(nbond_data.epsilon),
                nbond_data.rmin,
            )
            disp = compute_jax_pme_lj_dispersion(
                pos_np,
                c6_sqrt,
                box_length_A=box_length_from_cell(np.asarray(cell)),
                method=pme_method,
                sr_cutoff_A=float(jax_pme_sr_cutoff_A),
            )
            vdw_disp = jnp.asarray(disp.energy_kcalmol, dtype=pos.dtype)
            vdw_energy = vdw_energy + vdw_disp
            forces = forces + jnp.asarray(disp.forces_kcalmol_A, dtype=pos.dtype)
        energy = vdw_energy + elec_energy
    components = {
        "vdw": vdw_energy,
        "elec": elec_energy,
        "total": energy,
    }
    return components, forces


@dataclass(frozen=True, slots=True)
class NonbondedPairDecomposition:
    """Per-pair MIC nonbonded contributions (kcal/mol) aligned with ``pair_i``/``pair_j``."""

    pair_i: np.ndarray
    pair_j: np.ndarray
    r_A: np.ndarray
    vdw_kcal: np.ndarray
    elec_kcal: np.ndarray

    @property
    def n_pairs(self) -> int:
        return int(self.pair_i.shape[0])

    def totals(self) -> dict[str, float]:
        return {
            "vdw": float(np.sum(self.vdw_kcal)),
            "elec": float(np.sum(self.elec_kcal)),
            "total": float(np.sum(self.vdw_kcal) + np.sum(self.elec_kcal)),
        }


def decompose_nonbonded_pair_energies(
    positions: Array | np.ndarray,
    nbond_data: NonbondedSystemData,
    cell: Array | np.ndarray,
    settings: CharmmNbondSettings,
    *,
    pair_i: np.ndarray | None = None,
    pair_j: np.ndarray | None = None,
    lr_solver: str | None = None,
) -> NonbondedPairDecomposition:
    """Return per-pair switched VDW and Coulomb energies (MIC, ``lr_solver=mic`` only)."""
    if pick_lr_solver(lr_solver) == "jax_pme":
        raise ValueError("decompose_nonbonded_pair_energies supports lr_solver=mic only")
    pos = jnp.asarray(positions, dtype=jnp.float64)
    cell_j = jnp.asarray(cell, dtype=jnp.float64)
    if cell_j.ndim == 1 and cell_j.shape[0] == 3:
        cell_j = jnp.diag(cell_j)

    excluded_pairs = nbond_data.excluded_pairs
    if nbond_data.psf_path is not None and nbond_data.psf_bonds is not None:
        excluded_pairs = resolve_nonbonded_excluded_pairs(
            nbond_data.psf_path,
            nbond_data.psf_bonds,
            natom=int(np.asarray(nbond_data.charges).shape[0]),
        )

    if pair_i is None or pair_j is None:
        host_i, host_j = _build_pair_indices(
            np.asarray(positions),
            np.asarray(cell),
            excluded_pairs,
            settings.cutnb,
        )
        pair_i = host_i
        pair_j = host_j

    pi = jnp.asarray(pair_i, dtype=jnp.int32)
    pj = jnp.asarray(pair_j, dtype=jnp.int32)

    e14_scale_np = np.ones(len(pair_i), dtype=np.float64)
    vdw14_scale_np = np.ones(len(pair_i), dtype=np.float64)
    for k, (i, j) in enumerate(zip(pair_i, pair_j, strict=True)):
        if (int(i), int(j)) in nbond_data.e14_pairs:
            e14_scale_np[k] = settings.e14fac
            vdw14_scale_np[k] = settings.vdw14fac

    q = jnp.asarray(nbond_data.charges, dtype=jnp.float64)
    eps_tbl = jnp.asarray(nbond_data.epsilon, dtype=jnp.float64)
    rm_tbl = jnp.asarray(nbond_data.rmin, dtype=jnp.float64)
    e14_scale = jnp.asarray(e14_scale_np, dtype=jnp.float64)
    vdw14_scale = jnp.asarray(vdw14_scale_np, dtype=jnp.float64)
    vfswitch_coeffs = charmm_vfswitch_coeffs(settings)
    fswitch_coeffs = charmm_fswitch_coeffs(settings)
    c2of = settings.c2ofnb

    ri = pos[pi]
    rj = pos[pj]
    disp = jax.vmap(lambda a, b: mic_displacement(a, b, cell_j))(ri, rj)
    r = jnp.linalg.norm(disp, axis=-1)
    r_sq = r * r
    within_ctof = r_sq < c2of

    ep_i = eps_tbl[pi]
    ep_j = eps_tbl[pj]
    rm_i = rm_tbl[pi]
    rm_j = rm_tbl[pj]
    sig = rm_i + rm_j
    ep = _pair_lj_epsilon(ep_i, ep_j)
    qq = q[pi] * q[pj] * e14_scale / settings.eps

    vdw = _pair_vdw_energy(
        r,
        ep,
        sig,
        settings,
        vfswitch_coeffs,
        use_jax_pme_dispersion=False,
    )
    vdw = vdw * vdw14_scale
    elec = _pair_elec_energy(r, qq, settings, fswitch_coeffs)
    vdw = jnp.where(within_ctof, vdw, 0.0)
    elec = jnp.where(within_ctof, elec, 0.0)

    return NonbondedPairDecomposition(
        pair_i=np.asarray(pair_i, dtype=np.int32),
        pair_j=np.asarray(pair_j, dtype=np.int32),
        r_A=np.asarray(r, dtype=np.float64),
        vdw_kcal=np.asarray(vdw, dtype=np.float64),
        elec_kcal=np.asarray(elec, dtype=np.float64),
    )


def mm_system_energy_and_forces(
    positions: Array | np.ndarray,
    bonded_system: CgenffBondedSystem,
    nbond_data: NonbondedSystemData,
    cell: Array | np.ndarray,
    settings: CharmmNbondSettings,
    *,
    prm_file: Path | str | None = None,
    include_cmap: bool = True,
    lr_solver: str | None = None,
    jax_pme_method: str | None = None,
    jax_pme_sr_cutoff_A: float = 6.0,
    jax_pme_dispersion: bool | None = None,
) -> MmSystemEnergyResult:
    """Bonded + switched nonbonded MM energy and forces (kcal/mol, kcal/mol/Å)."""
    _ = prm_file
    bonded_comp, bonded_forces = bonded_energy_and_forces(
        jnp.asarray(positions),
        bonded_system.topology,
        bonded_system.bonded,
        urey_k=bonded_system.urey_k,
        urey_r0=bonded_system.urey_r0,
        energy_unit="kcal/mol",
        include_cmap=include_cmap,
    )
    nb_comp, nb_forces = nonbonded_energy_and_forces(
        positions,
        nbond_data,
        cell,
        settings,
        lr_solver=lr_solver,
        jax_pme_method=jax_pme_method,
        jax_pme_sr_cutoff_A=jax_pme_sr_cutoff_A,
        jax_pme_dispersion=jax_pme_dispersion,
    )
    forces = np.asarray(bonded_forces + nb_forces, dtype=np.float64)
    bonded = {k: float(v) for k, v in bonded_comp.items()}
    nonbonded = {
        "vdw": float(nb_comp["vdw"]),
        "elec": float(nb_comp["elec"]),
        "total": float(nb_comp["total"]),
    }
    total = bonded["total"] + nonbonded["total"]
    return MmSystemEnergyResult(
        bonded=bonded,
        nonbonded=nonbonded,
        total_energy=total,
        forces=forces,
    )


def load_bonded_system_from_psf(
    psf_path: Path | str,
    positions: Array | np.ndarray,
    *,
    prm_file: Path | str,
    extra_prm_files: Sequence[Path | str] = (),
) -> CgenffBondedSystem:
    """Load bonded topology/parameters using explicit PRM path(s)."""
    return load_cgenff_bonded_from_psf(
        psf_path,
        positions,
        prm_file=prm_file,
        extra_prm_files=extra_prm_files,
    )
