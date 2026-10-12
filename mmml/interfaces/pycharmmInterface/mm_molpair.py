"""Molecule-pair (dimer-block) MM tail, on-device Verlet lists and ML dimer buckets.

Same physics as the ``jax_mic`` switched MM tail of
``mm_energy_forces.build_mm_energy_forces_fn`` (dynamic pair-list path)::

    E_MM = sum_{dimers a<b} s(R_ab) * sum_{i in a, j in b} [LJ(r~_ij) + C(r~_ij)]

with ``R_ab`` the minimum-image distance of the unweighted monomer centroids,
``s`` the complementary COM handoff (``handoff * taper``), ``r_ij`` the
minimum-image atom-atom distance, ``r~ = softplus(20 (r - 0.6)) / 20 + 0.6`` the
soft core, ``LJ = eps_ij ((Rmin_ij / r~)^12 - 2 (Rmin_ij / r~)^6)`` and
``C = 332.0637 q_i q_j / r~`` (kcal/mol), all scaled by ``lambda_a lambda_b``.
The old path enumerates atom pairs within ``R_off + 2 * extent + skin``; every
atom pair of a dimer with ``R_ab < R_off`` is in it, and ``s = 0`` beyond
``R_off``, so summing whole dimer blocks over ``R_ab < R_off + skin`` is the
same sum (float32 summation order aside).

The calculator's short-range inter-monomer wall (``pair_wall_energy``, eV,
nonzero only for ``r_ij < 1 A``) is fused into the same block pass: such pairs
have ``R_ab < 1 + 2 * extent << R_off``, so the list holds all of them.

Lists are molecule pairs, bucketed by molecule *class* pair (class = atom count
+ element signature, e.g. water / methanol), so each bucket is a dense
``(P, n_a, n_b)`` block with no per-atom padding.  They live on the device and
are rebuilt inside the jitted MD loop only when some centroid moved more than
``skin / 2 - guard`` since the last build (no host round trip, no recompiles);
static capacities come from the measured counts plus a margin and overflow sets
a sticky flag the host checks once per block.

The same build also emits the sparse-ML candidate list (``ml_dimer_candidates``
of the hybrid calculator: sorted global dimer ids padded with ``n_dimers``) and,
through :func:`ml_dimer_buckets`, per-class-pair index lists for a bucketed ML
batch (see README in perf/overheads for the interface).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, NamedTuple, Optional, Sequence, Tuple

import numpy as np

import jax
import jax.numpy as jnp
from jax import Array

COULOMB_K = 3.32063711e2  # kcal/mol * A / e^2 (same constant as mm_energy_forces)
try:  # the exact conversion factor mmml_calculator uses (ASE kcal/mol)
    import ase.units as _au

    KCALMOL2EV = float(_au.kcal / _au.mol)
except Exception:  # pragma: no cover
    KCALMOL2EV = 1.0 / 23.060548867

#: Rebuild guard (A): positions move between the displacement test and the force
#: evaluation of the same MD step; 0.1 A covers |v| dt for any sane dt.
DEFAULT_REBUILD_GUARD_A = 0.1


# ---------------------------------------------------------------------------
# Topology
# ---------------------------------------------------------------------------
@dataclass
class MolPairTopology:
    """Host description of the molecules (static for a run)."""

    n_atoms: int
    n_mol: int
    atom_idx: np.ndarray  # (M, A_max) int32, padded with the molecule's first atom
    atom_mask: np.ndarray  # (M, A_max) bool
    n_atoms_mol: np.ndarray  # (M,)
    mol_class: np.ndarray  # (M,) int32
    class_n_atoms: list  # atoms per class
    class_signature: list  # tuple of Z per class
    buckets: list = field(default_factory=list)  # [(ca, cb)] with ca <= cb

    @property
    def n_dimers(self) -> int:
        return self.n_mol * (self.n_mol - 1) // 2


def build_topology(
    monomer_offsets: Sequence[int], atomic_numbers: Optional[Sequence[int]] = None
) -> MolPairTopology:
    off = np.asarray(monomer_offsets, dtype=np.int64)
    M = len(off) - 1
    n_at = np.diff(off).astype(np.int32)
    A = int(n_at.max())
    idx = np.zeros((M, A), dtype=np.int32)
    mask = np.zeros((M, A), dtype=bool)
    for m in range(M):
        k = int(n_at[m])
        idx[m, :k] = np.arange(off[m], off[m + 1])
        idx[m, k:] = off[m]
        mask[m, :k] = True
    Z = None if atomic_numbers is None else np.asarray(atomic_numbers, dtype=np.int64)
    sigs: list = []
    cls = np.zeros(M, dtype=np.int32)
    for m in range(M):
        sig = (int(n_at[m]),) + (tuple(Z[off[m] : off[m + 1]].tolist()) if Z is not None else ())
        if sig not in sigs:
            sigs.append(sig)
        cls[m] = sigs.index(sig)
    # order classes by size (small first) so bucket (W, M) is (0, 1)
    order = sorted(range(len(sigs)), key=lambda c: (sigs[c][0], sigs[c]))
    remap = np.empty(len(sigs), dtype=np.int32)
    for new, old in enumerate(order):
        remap[old] = new
    cls = remap[cls]
    sigs = [sigs[o] for o in order]
    K = len(sigs)
    buckets = [(a, b) for a in range(K) for b in range(a, K)]
    return MolPairTopology(
        n_atoms=int(off[-1]),
        n_mol=M,
        atom_idx=idx,
        atom_mask=mask,
        n_atoms_mol=n_at,
        mol_class=cls,
        class_n_atoms=[int(s[0]) for s in sigs],
        class_signature=[s[1:] for s in sigs],
        buckets=buckets,
    )


# ---------------------------------------------------------------------------
# Small helpers (same MIC arithmetic as pbc_utils_jax.mic_displacements_batched)
# ---------------------------------------------------------------------------
def _cell3(box: Array) -> Array:
    """Box (scalar, (3,) or (3, 3)) -> 3x3 cell (static-shape version of
    mm_energy_forces._box_to_cell_3x3)."""
    b = jnp.asarray(box)
    if b.ndim == 2:
        return b
    if b.ndim == 1 and b.shape[0] == 3:
        return jnp.diag(b)
    return jnp.diag(jnp.broadcast_to(b.reshape(-1)[0], (3,)))


def _mic(d: Array, cell: Array, inv: Array) -> Array:
    s = d @ inv
    return (s - jnp.round(s)) @ cell


def centroids(pos: Array, atom_idx: Array, atom_mask: Array) -> Array:
    """Unweighted molecule centroids (M, 3) (= monomer_coms_segment without masses)."""
    x = pos[atom_idx]  # (M, A, 3)
    w = atom_mask.astype(pos.dtype)
    return jnp.sum(x * w[..., None], axis=1) / jnp.sum(w, axis=1, keepdims=True)


def _sharpstep(r, x0, x1, gamma):
    from mmml.interfaces.pycharmmInterface.calculator_utils import _sharpstep as _ss

    return _ss(r, x0, x1, gamma=gamma)


# ---------------------------------------------------------------------------
# Neighbour list state (a pytree; carried through lax loops)
# ---------------------------------------------------------------------------
class MolPairNL(NamedTuple):
    pairs: Tuple[Array, ...]  # per bucket (cap_b, 2) int32 mol ids, class-ordered; pad = M
    counts: Array  # (n_buckets,) int32 true counts at last build
    ml_cand: Array  # (cap_ml,) int32 sorted global dimer ids; pad = n_dimers
    ml_count: Array  # () int32
    ref_com: Array  # (M, 3) centroids at last build
    overflow: Array  # () bool, sticky
    n_builds: Array  # () int32


@dataclass
class MolPairCaps:
    mm: Tuple[int, ...]
    ml: int


class MolPairMM:
    """Builds lists and evaluates the fused LJ + Coulomb + wall dimer-block tail.

    ``spec`` is the dict exported by ``build_mm_energy_forces_fn`` as
    ``update_fn.molpair_spec`` (per-atom q / Rmin/2 / eps, lambda, switching).
    """

    def __init__(
        self,
        spec: dict,
        topo: MolPairTopology,
        *,
        skin_A: float = 1.0,
        ml_active_radius_A: Optional[float] = None,
        include_wall: bool = True,
        wall_r_on: Optional[float] = None,
        wall_k: Optional[float] = None,
        guard_A: float = DEFAULT_REBUILD_GUARD_A,
        cap_margin: float = 1.15,
        cap_pad: int = 64,
        dtype=jnp.float32,
    ):
        if spec.get("hybrid_hamiltonian", "handoff") != "handoff":
            raise ValueError("molpair MM supports hybrid_hamiltonian=handoff only")
        if spec.get("use_jax_pme"):
            raise ValueError("molpair MM supports lr_solver=mic (jax_mic) only")
        self.spec = spec
        self.topo = topo
        self.dtype = dtype
        self.skin = float(skin_A)
        self.guard = float(min(guard_A, 0.25 * self.skin))
        self.r_off = float(spec["mm_switch_on"]) + float(spec["mm_switch_width"])
        self.r_mm_list = self.r_off + self.skin
        self.ml_active_radius = None if ml_active_radius_A is None else float(ml_active_radius_A)
        self.r_ml_list = (
            None if self.ml_active_radius is None else self.ml_active_radius + self.skin + 1e-3
        )
        self.include_wall = bool(include_wall)
        from mmml.models.short_range_wall import DEFAULT_WALL_K_EV_A2, DEFAULT_WALL_R_ON_A

        self.wall_r_on = float(DEFAULT_WALL_R_ON_A if wall_r_on is None else wall_r_on)
        self.wall_k = float(DEFAULT_WALL_K_EV_A2 if wall_k is None else wall_k)
        self.cap_margin = float(cap_margin)
        self.cap_pad = int(cap_pad)
        M = topo.n_mol
        self.M = M
        self.n_dimers = topo.n_dimers
        self._atom_idx = jnp.asarray(topo.atom_idx)
        self._atom_mask = jnp.asarray(topo.atom_mask)
        self._cls = jnp.asarray(topo.mol_class)
        self._q = jnp.asarray(spec["q"], dtype=dtype)
        self._rm = jnp.asarray(spec["rmin_half"], dtype=dtype)
        self._ep = jnp.asarray(spec["eps"], dtype=dtype)  # signed (<= 0)
        self._lam = jnp.asarray(spec["lambda_monomer"], dtype=dtype)
        self.caps: Optional[MolPairCaps] = None
        self._jit_cache: dict = {}

    # ------------------------------------------------------------------ lists
    def _build_impl(self, pos: Array, box: Array, caps: MolPairCaps, prev_overflow, prev_builds) -> MolPairNL:
        M = self.M
        cell = _cell3(box).astype(pos.dtype)
        from mmml.interfaces.pycharmmInterface.pbc_utils_jax import cell_inverse

        inv = cell_inverse(cell)
        com = centroids(pos, self._atom_idx, self._atom_mask)
        d = com[None, :, :] - com[:, None, :]  # (M, M, 3)
        d = _mic(d, cell, inv)
        r2 = jnp.sum(d * d, axis=-1)
        ar = jnp.arange(M)
        upper = ar[None, :] > ar[:, None]
        within = upper & (r2 < self.r_mm_list**2)
        ci = self._cls[:, None]
        cj = self._cls[None, :]
        pairs, counts = [], []
        overflow = jnp.asarray(False)
        for b, (ca, cb) in enumerate(self.topo.buckets):
            if ca == cb:
                sel = within & (ci == ca) & (cj == ca)
            else:
                sel = within & (((ci == ca) & (cj == cb)) | ((ci == cb) & (cj == ca)))
            cap = int(caps.mm[b])
            flat = jnp.nonzero(sel.reshape(-1), size=cap, fill_value=M * M)[0]
            valid = flat < M * M
            i = jnp.where(valid, flat // M, M)
            j = jnp.where(valid, flat % M, M)
            if ca != cb:  # orient so the first molecule has class ca
                swap = valid & (self._cls[jnp.minimum(i, M - 1)] != ca)
                i, j = jnp.where(swap, j, i), jnp.where(swap, i, j)
            n = jnp.sum(sel, dtype=jnp.int32)
            overflow = overflow | (n > cap)
            pairs.append(jnp.stack([i, j], axis=1).astype(jnp.int32))
            counts.append(n)
        if self.r_ml_list is not None:
            sel_ml = upper & (r2 < self.r_ml_list**2)
            flat = jnp.nonzero(sel_ml.reshape(-1), size=caps.ml, fill_value=M * M)[0]
            valid = flat < M * M
            i = flat // M
            j = flat % M
            did = i * (2 * M - i - 1) // 2 + (j - i - 1)
            ml_cand = jnp.where(valid, did, self.n_dimers).astype(jnp.int32)
            ml_count = jnp.sum(sel_ml, dtype=jnp.int32)
            overflow = overflow | (ml_count > caps.ml)
        else:
            ml_cand = jnp.full((max(1, caps.ml),), self.n_dimers, dtype=jnp.int32)
            ml_count = jnp.asarray(0, dtype=jnp.int32)
        return MolPairNL(
            pairs=tuple(pairs),
            counts=jnp.stack(counts).astype(jnp.int32),
            ml_cand=ml_cand,
            ml_count=ml_count,
            ref_com=com,
            overflow=jnp.logical_or(overflow, prev_overflow),
            n_builds=prev_builds + 1,
        )

    def _count_impl(self, pos: Array, box: Array):
        """Exact counts for capacity sizing (host helper)."""
        cell = _cell3(box).astype(pos.dtype)
        from mmml.interfaces.pycharmmInterface.pbc_utils_jax import cell_inverse

        inv = cell_inverse(cell)
        com = centroids(pos, self._atom_idx, self._atom_mask)
        d = _mic(com[None] - com[:, None], cell, inv)
        r2 = jnp.sum(d * d, axis=-1)
        ar = jnp.arange(self.M)
        upper = ar[None, :] > ar[:, None]
        within = upper & (r2 < self.r_mm_list**2)
        ci = self._cls[:, None]
        cj = self._cls[None, :]
        out = []
        for ca, cb in self.topo.buckets:
            if ca == cb:
                sel = within & (ci == ca) & (cj == ca)
            else:
                sel = within & (((ci == ca) & (cj == cb)) | ((ci == cb) & (cj == ca)))
            out.append(jnp.sum(sel))
        n_ml = (
            jnp.sum(upper & (r2 < self.r_ml_list**2)) if self.r_ml_list is not None else jnp.asarray(0)
        )
        return jnp.stack(out), n_ml

    def size_caps(self, pos, box, *, margin: Optional[float] = None, floor: Optional[MolPairCaps] = None) -> MolPairCaps:
        """Capacities = measured counts * margin + pad (never below ``floor``)."""
        mg = self.cap_margin if margin is None else float(margin)
        n_mm, n_ml = jax.device_get(jax.jit(self._count_impl)(jnp.asarray(pos, self.dtype), jnp.asarray(box, self.dtype)))
        mm = [int(np.ceil(int(n) * mg)) + self.cap_pad for n in np.asarray(n_mm)]
        ml = int(np.ceil(int(n_ml) * mg)) + self.cap_pad
        if floor is not None:
            mm = [max(a, b) for a, b in zip(mm, floor.mm)]
            ml = max(ml, floor.ml)
        self.caps = MolPairCaps(mm=tuple(mm), ml=ml)
        return self.caps

    def build(self, pos, box, caps: Optional[MolPairCaps] = None) -> MolPairNL:
        caps = caps or self.caps
        key = ("build", caps.mm, caps.ml)
        if key not in self._jit_cache:
            self._jit_cache[key] = jax.jit(
                lambda p, b: self._build_impl(p, b, caps, jnp.asarray(False), jnp.asarray(0, jnp.int32))
            )
        return self._jit_cache[key](jnp.asarray(pos, self.dtype), jnp.asarray(box, self.dtype))

    def maybe_update(self, pos: Array, box: Array, nl: MolPairNL, guard: Optional[float] = None) -> MolPairNL:
        """Traceable: rebuild iff max centroid displacement > skin/2 - guard.

        ``guard`` defaults to ``self.guard`` (the integrator carry: positions
        still drift by |v| dt before the force call). Inside the force call the
        positions are exact, so it passes ``guard=0``.
        """
        caps = MolPairCaps(mm=tuple(int(p.shape[0]) for p in nl.pairs), ml=int(nl.ml_cand.shape[0]))
        cell = _cell3(box).astype(pos.dtype)
        from mmml.interfaces.pycharmmInterface.pbc_utils_jax import cell_inverse

        inv = cell_inverse(cell)
        com = centroids(pos, self._atom_idx, self._atom_mask)
        dd = _mic(com - nl.ref_com, cell, inv)
        dmax2 = jnp.max(jnp.sum(dd * dd, axis=-1))
        thr = 0.5 * self.skin - (self.guard if guard is None else float(guard))
        need = dmax2 > thr * thr
        return jax.lax.cond(
            need,
            lambda: self._build_impl(pos, box, caps, nl.overflow, nl.n_builds),
            lambda: nl,
        )

    # ----------------------------------------------------------------- energy
    def energy_terms(
        self, pos: Array, box: Array, nl: MolPairNL, charges: Optional[Array] = None
    ) -> Tuple[Array, Array, Array, Array]:
        """(E_mm_eV, E_wall_eV, E_vdw_eV, E_elec_eV); positions must be whole molecules."""
        sp = self.spec
        cell = _cell3(box).astype(pos.dtype)
        from mmml.interfaces.pycharmmInterface.pbc_utils_jax import cell_inverse
        from mmml.interfaces.pycharmmInterface.cutoffs import GAMMA_OFF, GAMMA_ON

        inv = cell_inverse(cell)
        com = centroids(pos, self._atom_idx, self._atom_mask)
        q = self._q if charges is None else jnp.asarray(charges, pos.dtype)
        on, w, mlw = float(sp["mm_switch_on"]), float(sp["mm_switch_width"]), float(sp["ml_switch_width"])
        r_min = sp.get("mm_r_min")
        e_vdw = jnp.zeros((), pos.dtype)
        e_el = jnp.zeros((), pos.dtype)
        e_wall = jnp.zeros((), pos.dtype)
        M = self.M
        for b, (ca, cb) in enumerate(self.topo.buckets):
            pr = nl.pairs[b]
            if pr.shape[0] == 0:
                continue
            na, nb = self.topo.class_n_atoms[ca], self.topo.class_n_atoms[cb]
            valid = pr[:, 0] < M
            ma = jnp.where(valid, pr[:, 0], 0)
            mb = jnp.where(valid, pr[:, 1], 0)
            ia = self._atom_idx[ma, :na]  # (P, na)
            ib = self._atom_idx[mb, :nb]
            xa = pos[ia]
            xb = pos[ib]
            d = xb[:, None, :, :] - xa[:, :, None, :]  # (P, na, nb, 3)  r_j - r_i
            d = _mic(d, cell, inv)
            r = jnp.sqrt(jnp.maximum(jnp.sum(d * d, axis=-1), 1e-22))
            lam = self._lam[ma] * self._lam[mb]  # (P,)
            if self.include_wall:
                rw = jnp.maximum(r, 1e-6)
                inside = rw < self.wall_r_on
                dw = jnp.where(inside, self.wall_r_on - rw, 0.0)
                ew = jnp.where(inside, self.wall_k * dw**3 / rw, 0.0)
                # the wall belongs to the pair interaction: it scales with lambda like ML and MM (alchemy)
                e_wall = e_wall + jnp.sum(jnp.where(valid[:, None, None], ew * lam[:, None, None], 0.0))
            rs = jax.nn.softplus(20.0 * (r - 0.6)) / 20.0 + 0.6
            qq = q[ia][:, :, None] * q[ib][:, None, :]
            rm = self._rm[ia][:, :, None] + self._rm[ib][:, None, :]
            ep = (self._ep[ia][:, :, None] * self._ep[ib][:, None, :]) ** 0.5
            r6 = (rm / jnp.maximum(rs, 1e-10)) ** 6
            lj = jnp.sum(ep * (r6 * r6 - 2.0 * r6), axis=(1, 2))
            el = COULOMB_K * jnp.sum(qq / jnp.maximum(rs, 1e-10), axis=(1, 2))
            dc = _mic(com[mb] - com[ma], cell, inv)
            # padded slots have ma == mb (R = 0): keep the norm's gradient finite
            R = jnp.sqrt(jnp.maximum(jnp.sum(dc * dc, axis=-1), 1e-22))
            if sp.get("complementary_handoff", True):
                s = _sharpstep(R, on - mlw, on, GAMMA_ON) * (1.0 - _sharpstep(R, on, on + w, GAMMA_OFF))
            else:
                s = _sharpstep(R, on, on + w, GAMMA_ON) * (1.0 - _sharpstep(R, on + w, on + 2.0 * w, GAMMA_OFF))
            if r_min is not None:
                s = s * (R >= float(r_min)).astype(s.dtype)
            s = jnp.where(valid, s * lam, 0.0)
            e_vdw = e_vdw + jnp.sum(s * lj)
            e_el = e_el + jnp.sum(s * el)
        e_vdw = e_vdw * KCALMOL2EV
        e_el = e_el * KCALMOL2EV
        return e_vdw + e_el, e_wall, e_vdw, e_el

    def energy(self, pos, box, nl, charges=None) -> Array:
        e_mm, e_wall, _, _ = self.energy_terms(pos, box, nl, charges)
        return e_mm + e_wall

    def energy_and_forces(self, pos, box, nl, charges=None):
        e, g = jax.value_and_grad(self.energy)(pos, box, nl, charges)
        return e, -g


# ---------------------------------------------------------------------------
# ML dimer / monomer buckets (interface for a bucketed ML batch)
# ---------------------------------------------------------------------------
def monomer_buckets(topo: MolPairTopology) -> dict:
    """Static per-class monomer lists: ``{class: (mol_ids int32 (n_c,), n_atoms_c)}``."""
    out = {}
    for c, n in enumerate(topo.class_n_atoms):
        out[c] = (np.nonzero(topo.mol_class == c)[0].astype(np.int32), int(n))
    return out


def ml_dimer_buckets(
    mp: MolPairMM,
    nl: MolPairNL,
    pos: Array,
    box: Array,
    caps: Sequence[int],
    *,
    active_radius: Optional[float] = None,
) -> Tuple[dict, Array]:
    """Per class-pair ML dimer lists, compacted to the *active* dimers.

    Pure, traceable, static shapes. For bucket ``b = (ca, cb)`` (``ca <= cb``,
    classes from :func:`build_topology`, smallest molecule first) returns

    * ``mol_a``, ``mol_b``: int32 ``(caps[b],)`` molecule ids, ``class(mol_a) ==
      ca``; padding = ``M`` (out of range).
    * ``dimer_id``: int32 global dimer id (combinations order, ``min < max``),
      padding = ``n_dimers`` -- the id the scatter-back of the existing
      calculator uses.
    * ``valid``: bool, ``n``: int32 active count, ``r_com``: centroid distance.

    Active = centroid MIC distance ``< active_radius`` (default: the radius the
    candidate list was built for minus the skin), the same test as the sparse
    ML path. Candidates come from ``nl.ml_cand`` (exact while the list is
    valid). Returns ``(buckets, overflow)``; ``overflow`` is True if any bucket
    had more active dimers than its capacity (fail closed: caller must grow).
    """
    M = mp.M
    rad = (mp.r_ml_list - mp.skin - 1e-3) if active_radius is None else float(active_radius)
    cand = nl.ml_cand
    valid_c = cand < mp.n_dimers
    did = jnp.where(valid_c, cand, 0)
    # invert combinations-order id -> (i, j)
    n = M
    i = (
        (2 * n - 1 - jnp.sqrt((2.0 * n - 1.0) ** 2 - 8.0 * did.astype(jnp.float32))) / 2.0
    ).astype(jnp.int32)
    i = jnp.clip(i, 0, n - 2)
    start = i * (2 * n - i - 1) // 2
    i = jnp.where(did < start, i - 1, i)
    start = i * (2 * n - i - 1) // 2
    nxt = (i + 1) * (2 * n - i - 2) // 2
    i = jnp.where(did >= nxt, i + 1, i)
    start = i * (2 * n - i - 1) // 2
    j = did - start + i + 1
    cell = _cell3(box).astype(pos.dtype)
    from mmml.interfaces.pycharmmInterface.pbc_utils_jax import cell_inverse

    inv = cell_inverse(cell)
    com = centroids(pos, mp._atom_idx, mp._atom_mask)
    dc = _mic(com[j] - com[i], cell, inv)
    r = jnp.linalg.norm(dc, axis=-1)
    active = valid_c & (r < rad)
    ci = mp._cls[i]
    cj = mp._cls[j]
    out = {}
    overflow = jnp.asarray(False)
    for b, (ca, cb) in enumerate(mp.topo.buckets):
        if ca == cb:
            sel = active & (ci == ca) & (cj == ca)
        else:
            sel = active & (((ci == ca) & (cj == cb)) | ((ci == cb) & (cj == ca)))
        cap = int(caps[b])
        k = jnp.nonzero(sel, size=cap, fill_value=cand.shape[0])[0]
        kv = k < cand.shape[0]
        ks = jnp.where(kv, k, 0)
        a_, b_ = i[ks], j[ks]
        if ca != cb:
            sw = mp._cls[a_] != ca
            a_, b_ = jnp.where(sw, b_, a_), jnp.where(sw, a_, b_)
        cnt = jnp.sum(sel, dtype=jnp.int32)
        overflow = overflow | (cnt > cap)
        out[(ca, cb)] = dict(
            mol_a=jnp.where(kv, a_, M).astype(jnp.int32),
            mol_b=jnp.where(kv, b_, M).astype(jnp.int32),
            dimer_id=jnp.where(kv, did[ks], mp.n_dimers).astype(jnp.int32),
            valid=kv,
            n=cnt,
            r_com=jnp.where(kv, r[ks], jnp.inf),
        )
    return out, overflow


def count_dimers_within(positions, monomer_offsets, box_L: float, radius_A: float) -> int:
    """Host count of dimers with centroid MIC distance < ``radius_A`` (cubic box).

    Molecules are made whole around their first atom first (as the calculator
    does), so the centroids match the in-graph ones.
    """
    pos = np.asarray(positions, dtype=np.float64)
    off = np.asarray(monomer_offsets, dtype=np.int64)
    L = float(box_L)
    com = np.empty((len(off) - 1, 3))
    for m in range(len(off) - 1):
        x = pos[off[m] : off[m + 1]]
        d = x - x[0]
        x = x - L * np.round(d / L)
        com[m] = x.mean(axis=0)
    d = com[None, :, :] - com[:, None, :]
    d -= L * np.round(d / L)
    r2 = np.sum(d * d, axis=-1)
    iu = np.triu_indices(len(com), k=1)
    return int(np.sum(r2[iu] < radius_A**2))


def auto_ml_dimer_cap(n_active: int, margin: float, n_dimers: int, pad: int = 64) -> int:
    """Sparse-ML slot cap from a measured active count: ``ceil(n * margin) + pad``."""
    return int(min(n_dimers, int(np.ceil(n_active * float(margin))) + int(pad)))
