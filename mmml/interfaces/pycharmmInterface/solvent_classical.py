"""Classical CHARMM intramolecular energy for a subset of molecules (``MMML_ML_SOLUTE`` solvent).

With ``MMML_ML_SOLUTE`` set, the molecule-pair MM (:mod:`mm_molpair`) treats every solvent-solvent pair with plain
CGenFF/TIP3P LJ + Coulomb. This module supplies the other half of a classical solvent: each solvent molecule's
intramolecular energy from the PSF + CHARMM parameter files,

    bonds        K_b (r - r0)^2
    angles       K_t (theta - theta0)^2 + K_ub (S - S0)^2          (Urey-Bradley on the 1-3 distance)
    dihedrals    sum_n K_n (1 + cos(n phi - delta_n))               (exact type match, else X-b-c-X wildcards)
    impropers    K (psi - psi0)^2 (n = 0) / K (1 + cos(n psi - psi0))
    nonbonded    NBXMOD 5: 1-2 and 1-3 pairs excluded; 1-4 pairs with the 1-4 LJ columns (eps14, Rmin14/2) and
                 E14FAC = 1 Coulomb; 1-5 and further intramolecular pairs full LJ + Coulomb, no cutoff (the
                 molecules are made whole first; they are far smaller than any MM cutoff)

so a TIP3P water is flexible TIP3P (HT-OT 450 kcal/mol/A^2 at 0.9572 A, HT-OT-HT 55 kcal/mol/rad^2 at 104.52 deg,
CHARMM toppar) and a CGenFF octanol is the full CGenFF molecule. LJ is the CHARMM Lorentz-Berthelot combination
(eps_ij = sqrt(eps_i eps_j), Rmin_ij = Rmin_i/2 + Rmin_j/2); the Coulomb constant is the molecule-pair MM's
(:data:`mm_molpair.COULOMB_K`, = OpenMM's). NBFIX pairs inside a solvent molecule are refused (none in CGenFF
water/methanol/octanol).

Self-contained on purpose: no pycharmm / libcharmm import (the runner may import this under Slurm PMI).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

_SECTIONS = {
    "BOND": "bonds", "BONDS": "bonds",
    "ANGL": "angles", "ANGLE": "angles", "ANGLES": "angles", "THETA": "angles", "THETAS": "angles",
    "DIHE": "dihedrals", "DIHEDRAL": "dihedrals", "DIHEDRALS": "dihedrals", "PHI": "dihedrals",
    "IMPR": "impropers", "IMPROPER": "impropers", "IMPROPERS": "impropers", "IMPH": "impropers", "IMPHI": "impropers",
    "CMAP": "cmap", "NONB": "nonbonded", "NONBONDED": "nonbonded", "NBON": "nonbonded", "NBONDED": "nonbonded",
    "NBFIX": "nbfix", "HBOND": "hbond", "HBON": "hbond", "END": "end",
}


def default_prm_files() -> List[Path]:
    return [Path(__file__).resolve().parents[2] / "data" / "charmm" / "par_all36_cgenff.prm"]


@dataclass
class CharmmParams:
    bonds: Dict[Tuple[str, str], Tuple[float, float]]
    angles: Dict[Tuple[str, str, str], Tuple[float, float, float, float]]   # K, theta0 (deg), K_ub, S0
    dihedrals: Dict[Tuple[str, str, str, str], List[Tuple[float, int, float]]]   # [(K, n, delta deg)]
    impropers: Dict[Tuple[str, str, str, str], List[Tuple[float, int, float]]]
    nonbonded: Dict[str, Tuple[float, float, float, float]]   # eps, Rmin/2, eps14, Rmin14/2 (eps <= 0 as in the prm)
    nbfix: Dict[Tuple[str, str], Tuple[float, float]]


def _float(s: str) -> Optional[float]:
    try:
        return float(s)
    except ValueError:
        return None


def _param_lines(path: Path) -> List[str]:
    """Parameter lines of a .prm, or of the ``read para`` blocks of a toppar .str."""
    raw = path.read_text(encoding="utf-8", errors="replace").splitlines()
    if path.suffix.lower() != ".str":
        return raw
    out, inside = [], False
    for line in raw:
        s = line.strip().lower()
        if s.startswith("read para"):
            inside = True
            continue
        if inside:
            out.append(line)
            if s.split("!")[0].strip() == "end":
                inside = False
    return out


def read_charmm_params(paths: Sequence[Path | str]) -> CharmmParams:
    """Bond/angle/UB/dihedral/improper/LJ/NBFIX parameters; a later file overrides an earlier one (flex append)."""
    p = CharmmParams({}, {}, {}, {}, {}, {})
    for path in paths:
        section = None
        cont = False
        seen_dihe: set = set()   # a new file's dihedral key replaces (not extends) an earlier file's
        seen_impr: set = set()
        for line in _param_lines(Path(path)):
            body = line.split("!")[0].rstrip()
            if not body.strip():
                continue
            if cont:   # continuation of a NONBONDED/HBOND header line
                cont = body.endswith("-")
                continue
            tok = body.split()
            key = tok[0].upper()
            if key in _SECTIONS and (len(tok) == 1 or _float(tok[1]) is None):
                section = _SECTIONS[key]
                cont = body.endswith("-")
                if section == "end":
                    section = None
                continue
            if key.startswith("ATOM") or key.startswith("MASS") or key.startswith("*"):
                continue
            if section == "bonds" and len(tok) >= 4:
                k, r0 = float(tok[2]), float(tok[3])
                p.bonds[(tok[0], tok[1])] = p.bonds[(tok[1], tok[0])] = (k, r0)
            elif section == "angles" and len(tok) >= 5:
                k, t0 = float(tok[3]), float(tok[4])
                kub, s0 = (float(tok[5]), float(tok[6])) if len(tok) >= 7 else (0.0, 0.0)
                p.angles[(tok[0], tok[1], tok[2])] = p.angles[(tok[2], tok[1], tok[0])] = (k, t0, kub, s0)
            elif section in ("dihedrals", "impropers") and len(tok) >= 7:
                d = p.dihedrals if section == "dihedrals" else p.impropers
                seen = seen_dihe if section == "dihedrals" else seen_impr
                kk = tuple(tok[:4])
                term = (float(tok[4]), int(float(tok[5])), float(tok[6]))
                if kk not in seen:
                    d[kk] = []
                    seen.add(kk)
                d[kk].append(term)
            elif section == "nonbonded" and len(tok) >= 4 and _float(tok[1]) is not None:
                eps, r2 = float(tok[2]), float(tok[3])
                e14, r14 = (float(tok[5]), float(tok[6])) if len(tok) >= 7 else (eps, r2)
                p.nonbonded[tok[0]] = (eps, r2, e14, r14)
            elif section == "nbfix" and len(tok) >= 4:
                v = (float(tok[2]), float(tok[3]))
                p.nbfix[(tok[0], tok[1])] = p.nbfix[(tok[1], tok[0])] = v
    return p


def _lookup_dihedral(d, a, b, c, e):
    for k in ((a, b, c, e), (e, c, b, a)):
        if k in d:
            return d[k]
    for k in (("X", b, c, "X"), ("X", c, b, "X")):
        if k in d:
            return d[k]
    return None


def _lookup_improper(d, a, b, c, e):
    for k in ((a, b, c, e), (e, c, b, a), (a, "X", "X", e), (e, "X", "X", a),
              ("X", b, c, e), ("X", c, b, a), ("X", "X", c, e), ("X", "X", b, a)):
        if k in d:
            return d[k]
    return None


@dataclass
class PsfData:
    types: List[str]
    charges: np.ndarray
    bonds: np.ndarray
    angles: np.ndarray
    dihedrals: np.ndarray
    impropers: np.ndarray


def read_psf(path: Path | str) -> PsfData:
    """Atom types/charges and bonded lists of a CHARMM PSF (standard, EXT or XPLOR)."""
    lines = Path(path).read_text(encoding="utf-8", errors="replace").splitlines()
    types: List[str] = []
    charges: List[float] = []
    lists = {}
    i = 0
    while i < len(lines):
        line = lines[i]
        if "!" not in line:
            i += 1
            continue
        head, tag = line.split("!", 1)
        tag = tag.split(":")[0].strip().upper()
        if tag == "NATOM":
            n = int(head.split()[0])
            for j in range(i + 1, i + 1 + n):
                parts = lines[j].split()
                types.append(parts[5])
                charges.append(float(parts[6]))
            i += 1 + n
            continue
        width = {"NBOND": 2, "NTHETA": 3, "NPHI": 4, "NIMPHI": 4}.get(tag)
        if width is not None:
            n = int(head.split()[0])
            vals: List[int] = []
            i += 1
            while len(vals) < n * width:
                vals.extend(int(x) for x in lines[i].split())
                i += 1
            lists[tag] = np.asarray(vals, dtype=np.int64).reshape(-1, width) - 1
            continue
        i += 1
    z = lambda w: np.zeros((0, w), dtype=np.int64)
    return PsfData(types, np.asarray(charges, dtype=np.float64), lists.get("NBOND", z(2)), lists.get("NTHETA", z(3)),
                   lists.get("NPHI", z(4)), lists.get("NIMPHI", z(4)))


@dataclass
class IntraTerms:
    """Index/parameter arrays of the classical intramolecular energy of the selected molecules (kcal/mol, A, rad)."""

    bond_idx: np.ndarray
    bond_k: np.ndarray
    bond_r0: np.ndarray
    angle_idx: np.ndarray
    angle_k: np.ndarray
    angle_t0: np.ndarray
    ub_idx: np.ndarray
    ub_k: np.ndarray
    ub_s0: np.ndarray
    dihe_idx: np.ndarray
    dihe_k: np.ndarray
    dihe_n: np.ndarray
    dihe_d: np.ndarray
    impr_idx: np.ndarray
    impr_k: np.ndarray
    impr_n: np.ndarray
    impr_d: np.ndarray
    pair_idx: np.ndarray      # (P, 2) 1-4 and 1-5+ pairs
    pair_qq: np.ndarray       # q_i q_j (E14FAC = 1)
    pair_eps: np.ndarray      # sqrt(eps_i eps_j) > 0
    pair_rmin: np.ndarray     # Rmin_ij
    n_molecules: int
    n_atoms: int

    def summary(self) -> str:
        return (f"{self.n_molecules} molecules / {self.n_atoms} atoms: bonds {len(self.bond_idx)}, angles "
                f"{len(self.angle_idx)} (UB {len(self.ub_idx)}), dihedral terms {len(self.dihe_idx)}, improper terms "
                f"{len(self.impr_idx)}, 1-4/1-5+ pairs {len(self.pair_idx)}")


def build_intra_terms(psf: PsfData, params: CharmmParams, atom_mol: np.ndarray, select: np.ndarray) -> IntraTerms:
    """Terms of the molecules with ``select[mol]`` (``atom_mol``: molecule id per atom)."""
    atom_mol = np.asarray(atom_mol, dtype=np.int64)
    if atom_mol.shape[0] != len(psf.types):
        raise ValueError(f"PSF has {len(psf.types)} atoms, the system {atom_mol.shape[0]}")
    t = psf.types
    sel_atom = np.asarray(select, bool)[atom_mol]

    def keep(rows):
        if len(rows) == 0:
            return rows
        m = atom_mol[rows]
        if np.any(m != m[:, :1]):
            bad = rows[np.nonzero(np.any(m != m[:, :1], axis=1))[0][0]]
            raise ValueError(f"PSF term {bad.tolist()} spans molecules: monomer offsets do not follow the PSF")
        return rows[sel_atom[rows[:, 0]]]

    bonds, angles, dihes, imprs = keep(psf.bonds), keep(psf.angles), keep(psf.dihedrals), keep(psf.impropers)
    missing = []
    bk, br = [], []
    for a, b in bonds:
        v = params.bonds.get((t[a], t[b]))
        if v is None:
            missing.append(("bond", t[a], t[b]))
            v = (0.0, 0.0)
        bk.append(v[0]); br.append(v[1])
    ak, at, ub_rows, uk, us = [], [], [], [], []
    for a, b, c in angles:
        v = params.angles.get((t[a], t[b], t[c]))
        if v is None:
            missing.append(("angle", t[a], t[b], t[c]))
            v = (0.0, 0.0, 0.0, 0.0)
        ak.append(v[0]); at.append(np.radians(v[1]))
        if v[2] != 0.0:
            ub_rows.append((a, c)); uk.append(v[2]); us.append(v[3])
    d_rows, dk, dn, dd = [], [], [], []
    for row in dihes:
        terms = _lookup_dihedral(params.dihedrals, *(t[x] for x in row))
        if terms is None:
            missing.append(("dihedral",) + tuple(t[x] for x in row))
            continue
        for k, n, d in terms:
            d_rows.append(row); dk.append(k); dn.append(n); dd.append(np.radians(d))
    i_rows, ik, i_n, idl = [], [], [], []
    for row in imprs:
        terms = _lookup_improper(params.impropers, *(t[x] for x in row))
        if terms is None:
            missing.append(("improper",) + tuple(t[x] for x in row))
            continue
        for k, n, d in terms:
            i_rows.append(row); ik.append(k); i_n.append(n); idl.append(np.radians(d))
    if missing:
        uniq = sorted(set(missing))
        raise ValueError(f"missing CHARMM parameters for {len(uniq)} term types, e.g. {uniq[:5]}")

    # NBXMOD 5 from the PSF lists, as CHARMM's MAKINB: 1-2 (bonds) and 1-3 (angle ends) excluded, 1-4 from the
    # dihedral ends unless already excluded; every other intramolecular pair is a full nonbonded pair.
    excl = {tuple(sorted(map(int, b))) for b in bonds} | {tuple(sorted((int(a[0]), int(a[2])))) for a in angles}
    one4 = {tuple(sorted((int(d[0]), int(d[3])))) for d in dihes} - excl
    pairs, q, eps, rmin = [], [], [], []
    sel_atoms = np.nonzero(sel_atom)[0]
    mol_of = atom_mol[sel_atoms]
    order = np.argsort(mol_of, kind="stable")
    sel_atoms, mol_of = sel_atoms[order], mol_of[order]
    bounds = np.nonzero(np.diff(mol_of))[0] + 1
    for grp in np.split(sel_atoms, bounds):
        for ii in range(len(grp)):
            for jj in range(ii + 1, len(grp)):
                a, b = int(grp[ii]), int(grp[jj])
                key = (min(a, b), max(a, b))
                if key in excl:
                    continue
                is14 = key in one4
                if (t[a], t[b]) in params.nbfix:
                    raise NotImplementedError(f"NBFIX {t[a]}-{t[b]} inside a solvent molecule is not supported")
                pa, pb = params.nonbonded[t[a]], params.nonbonded[t[b]]
                ea, ra = (pa[2], pa[3]) if is14 else (pa[0], pa[1])
                eb, rb = (pb[2], pb[3]) if is14 else (pb[0], pb[1])
                pairs.append(key); q.append(psf.charges[a] * psf.charges[b])
                eps.append(np.sqrt(abs(ea) * abs(eb))); rmin.append(ra + rb)
    f = lambda x: np.asarray(x, dtype=np.float64)
    ii_ = lambda x, w: np.asarray(x, dtype=np.int32).reshape(-1, w)
    return IntraTerms(
        ii_(bonds, 2), f(bk), f(br), ii_(angles, 3), f(ak), f(at), ii_(ub_rows, 2), f(uk), f(us),
        ii_(d_rows, 4), f(dk), np.asarray(dn, np.int32), f(dd), ii_(i_rows, 4), f(ik), np.asarray(i_n, np.int32),
        f(idl), ii_(pairs, 2), f(q), f(eps), f(rmin), int(len(np.unique(mol_of))), int(len(sel_atoms)),
    )


def make_intra_energy_fn(terms: IntraTerms, *, coulomb_k: float, energy_scale: float = 1.0) -> Callable:
    """``E(positions) -> scalar`` (kcal/mol x ``energy_scale``) on WHOLE molecules (no minimum image)."""
    import jax.numpy as jnp

    T = {k: jnp.asarray(v) for k, v in terms.__dict__.items() if isinstance(v, np.ndarray)}

    def _norm(v):
        return jnp.sqrt(jnp.maximum(jnp.sum(v * v, axis=-1), 1e-30))

    def _dihedral(x, idx):
        p0, p1, p2, p3 = (x[idx[:, k]] for k in range(4))
        b1, b2, b3 = p1 - p0, p2 - p1, p3 - p2
        n1, n2 = jnp.cross(b1, b2), jnp.cross(b2, b3)
        m1 = jnp.cross(n1, b2 / _norm(b2)[:, None])
        return jnp.arctan2(jnp.sum(m1 * n2, -1), jnp.sum(n1 * n2, -1))

    def energy(x):
        dt = x.dtype
        P = {k: v.astype(dt) if v.dtype.kind == "f" else v for k, v in T.items()}
        e = jnp.zeros((), dt)
        if terms.bond_idx.shape[0]:
            r = _norm(x[P["bond_idx"][:, 0]] - x[P["bond_idx"][:, 1]])
            e = e + jnp.sum(P["bond_k"] * (r - P["bond_r0"]) ** 2)
        if terms.angle_idx.shape[0]:
            u = x[P["angle_idx"][:, 0]] - x[P["angle_idx"][:, 1]]
            v = x[P["angle_idx"][:, 2]] - x[P["angle_idx"][:, 1]]
            th = jnp.arctan2(_norm(jnp.cross(u, v)), jnp.sum(u * v, -1))
            e = e + jnp.sum(P["angle_k"] * (th - P["angle_t0"]) ** 2)
        if terms.ub_idx.shape[0]:
            s = _norm(x[P["ub_idx"][:, 0]] - x[P["ub_idx"][:, 1]])
            e = e + jnp.sum(P["ub_k"] * (s - P["ub_s0"]) ** 2)
        if terms.dihe_idx.shape[0]:
            phi = _dihedral(x, P["dihe_idx"])
            e = e + jnp.sum(P["dihe_k"] * (1.0 + jnp.cos(P["dihe_n"].astype(dt) * phi - P["dihe_d"])))
        if terms.impr_idx.shape[0]:
            psi = _dihedral(x, P["impr_idx"])
            dlt = jnp.mod(psi - P["impr_d"] + jnp.pi, 2.0 * jnp.pi) - jnp.pi
            e_h = P["impr_k"] * dlt * dlt
            e_p = P["impr_k"] * (1.0 + jnp.cos(P["impr_n"].astype(dt) * psi - P["impr_d"]))
            e = e + jnp.sum(jnp.where(P["impr_n"] == 0, e_h, e_p))
        if terms.pair_idx.shape[0]:
            r = _norm(x[P["pair_idx"][:, 0]] - x[P["pair_idx"][:, 1]])
            sr6 = (P["pair_rmin"] / r) ** 6
            e = e + jnp.sum(P["pair_eps"] * (sr6 * sr6 - 2.0 * sr6) + coulomb_k * P["pair_qq"] / r)
        return e * energy_scale

    return energy


def intra_components(terms: IntraTerms, x: np.ndarray, *, coulomb_k: float) -> Dict[str, float]:
    """Per-term energies in kcal/mol (numpy, float64): bond, angle, urey, dihedral, improper, lj, elec."""
    x = np.asarray(x, np.float64)
    nrm = lambda v: np.linalg.norm(v, axis=-1)
    out = {}
    out["bond"] = float(np.sum(terms.bond_k * (nrm(x[terms.bond_idx[:, 0]] - x[terms.bond_idx[:, 1]]) - terms.bond_r0) ** 2)) \
        if len(terms.bond_idx) else 0.0
    if len(terms.angle_idx):
        u = x[terms.angle_idx[:, 0]] - x[terms.angle_idx[:, 1]]
        v = x[terms.angle_idx[:, 2]] - x[terms.angle_idx[:, 1]]
        th = np.arctan2(nrm(np.cross(u, v)), np.sum(u * v, -1))
        out["angle"] = float(np.sum(terms.angle_k * (th - terms.angle_t0) ** 2))
    else:
        out["angle"] = 0.0
    out["urey"] = float(np.sum(terms.ub_k * (nrm(x[terms.ub_idx[:, 0]] - x[terms.ub_idx[:, 1]]) - terms.ub_s0) ** 2)) \
        if len(terms.ub_idx) else 0.0

    def dih(idx):
        p0, p1, p2, p3 = (x[idx[:, k]] for k in range(4))
        b1, b2, b3 = p1 - p0, p2 - p1, p3 - p2
        n1, n2 = np.cross(b1, b2), np.cross(b2, b3)
        m1 = np.cross(n1, b2 / nrm(b2)[:, None])
        return np.arctan2(np.sum(m1 * n2, -1), np.sum(n1 * n2, -1))

    out["dihedral"] = float(np.sum(terms.dihe_k * (1 + np.cos(terms.dihe_n * dih(terms.dihe_idx) - terms.dihe_d)))) \
        if len(terms.dihe_idx) else 0.0
    if len(terms.impr_idx):
        psi = dih(terms.impr_idx)
        dl = np.mod(psi - terms.impr_d + np.pi, 2 * np.pi) - np.pi
        out["improper"] = float(np.sum(np.where(terms.impr_n == 0, terms.impr_k * dl * dl,
                                                terms.impr_k * (1 + np.cos(terms.impr_n * psi - terms.impr_d)))))
    else:
        out["improper"] = 0.0
    if len(terms.pair_idx):
        r = nrm(x[terms.pair_idx[:, 0]] - x[terms.pair_idx[:, 1]])
        sr6 = (terms.pair_rmin / r) ** 6
        out["lj"] = float(np.sum(terms.pair_eps * (sr6 * sr6 - 2 * sr6)))
        out["elec"] = float(np.sum(coulomb_k * terms.pair_qq / r))
    else:
        out["lj"] = out["elec"] = 0.0
    out["total"] = sum(out.values())
    return out


def solvent_mask_from_env(n_molecules: int) -> Optional[np.ndarray]:
    """``MMML_ML_SOLUTE`` -> bool (n_molecules,) True for solute molecules, or None when unset."""
    import os

    s = os.environ.get("MMML_ML_SOLUTE", "").strip()
    if not s:
        return None
    ids = sorted({int(v) for v in s.replace(" ", "").split(",") if v})
    if not ids or min(ids) < 0 or max(ids) >= n_molecules:
        raise ValueError(f"MMML_ML_SOLUTE molecule ids must lie in [0, {n_molecules}), got {s!r}")
    m = np.zeros(n_molecules, bool)
    m[ids] = True
    return m


def build_solvent_intra_from_psf(psf_path, monomer_offsets, solute_mask, prm_files=None) -> IntraTerms:
    """Classical intramolecular terms of every molecule NOT in ``solute_mask`` (PSF order = system order)."""
    import os

    if prm_files is None:
        env = os.environ.get("MMML_ML_SOLUTE_PRM", "").strip()
        prm_files = [Path(x) for x in env.split(",") if x] if env else default_prm_files()
    offs = np.asarray(monomer_offsets, dtype=np.int64)
    atom_mol = np.repeat(np.arange(len(offs) - 1), np.diff(offs))
    psf = read_psf(psf_path)
    return build_intra_terms(psf, read_charmm_params(prm_files), atom_mol, ~np.asarray(solute_mask, bool))
