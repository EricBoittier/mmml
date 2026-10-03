"""Packed variable-size batches for full SPICE-α efield training.

Instead of padding every frame to the largest system, each batch concatenates
whole frames until an atom / edge / molecule budget is reached, then pads to
the fixed budget so XLA compiles one shape. Edges are an intra-frame
neighbour list within ``cutoff`` (directed, i != j), built on the CPU.

Layout of a packed batch (M molecule slots, A atoms, E edges):
  atomic_numbers (A,) int32, positions (A, 3), forces (A, 3), batch_segments (A,)
  dst_idx_flat / src_idx_flat (E,), electric_field (M, 3), energies (M,),
  dipoles (M, 3), polar (M, 3, 3), mol_mask (M,), subset (M,)
The last molecule slot and the last atom are always padding; padding atoms
have Z = 0 and padding edges are (A-1, A-1), which the model masks out.
"""

from __future__ import annotations

import queue
import threading
from dataclasses import dataclass
from typing import Iterator, Mapping

import jax
import jax.numpy as jnp
import numpy as np
import optax

from mmml.models.efield.model_functions import predicted_polarizability_bohr3


@dataclass(frozen=True)
class PackSpec:
    max_molecules: int = 256   # M, including one padding slot
    max_atoms: int = 4096      # A, including at least one padding atom
    max_edges: int = 131072    # E
    cutoff: float = 10.0       # Å, neighbour-list radius (match the model cutoff)
    # Separate Coulomb pair list: None reuses the message-passing edges (Coulomb
    # truncated at ``cutoff``); a float (math.inf = all intra-frame pairs) builds
    # a second list with its own budget.
    coulomb_cutoff: float | None = None
    max_coulomb_edges: int = 0


def _pair_d2(R: np.ndarray) -> np.ndarray:
    d2 = np.sum((R[:, None, :] - R[None, :, :]) ** 2, axis=-1)
    np.fill_diagonal(d2, np.inf)
    return d2


def frame_edges(R: np.ndarray, cutoff: float, d2: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Directed intra-frame pairs (dst, src) with i != j and |r_ij| < cutoff."""
    d2 = _pair_d2(R) if d2 is None else d2
    mask = np.isfinite(d2) if np.isinf(cutoff) else d2 < cutoff * cutoff
    dst, src = np.nonzero(mask)
    return dst.astype(np.int32), src.astype(np.int32)


def _empty_batch(spec: PackSpec) -> dict[str, np.ndarray]:
    M, A, E = spec.max_molecules, spec.max_atoms, spec.max_edges
    return {
        "atomic_numbers": np.zeros((A,), np.int32),
        "positions": np.zeros((A, 3), np.float32),
        "forces": np.zeros((A, 3), np.float32),
        "batch_segments": np.full((A,), M - 1, np.int32),
        "dst_idx_flat": np.full((E,), A - 1, np.int32),
        "src_idx_flat": np.full((E,), A - 1, np.int32),
        **({"coulomb_dst_idx_flat": np.full((spec.max_coulomb_edges,), A - 1, np.int32),
            "coulomb_src_idx_flat": np.full((spec.max_coulomb_edges,), A - 1, np.int32)}
           if spec.coulomb_cutoff is not None else {}),
        "electric_field": np.zeros((M, 3), np.float32),
        "energies": np.zeros((M,), np.float32),
        "dipoles": np.zeros((M, 3), np.float32),
        "polar": np.zeros((M, 3, 3), np.float32),
        "mol_mask": np.zeros((M,), np.float32),
        "subset": np.full((M,), -1, np.int32),
    }


def iter_packed_batches(
    data: Mapping[str, np.ndarray],
    order: np.ndarray,
    spec: PackSpec,
    *,
    e_ref: np.ndarray | None = None,
) -> Iterator[dict[str, np.ndarray]]:
    """Greedily pack frames (in ``order``) into fixed-budget batches.

    ``e_ref`` (indexed by Z, eV) is subtracted from frame energies, so the
    model learns residual energies instead of absolute ones.
    """
    M, A, E = spec.max_molecules, spec.max_atoms, spec.max_edges
    EC = spec.max_coulomb_edges
    split_coulomb = spec.coulomb_cutoff is not None
    off = data["offsets"]
    batch, m, a, e, ec = _empty_batch(spec), 0, 0, 0, 0
    for f in order:
        lo, hi = int(off[f]), int(off[f + 1])
        n = hi - lo
        R = data["R"][lo:hi]
        d2 = _pair_d2(R)
        dst, src = frame_edges(R, spec.cutoff, d2)
        ne = len(dst)
        nec = 0
        if split_coulomb:
            cdst, csrc = frame_edges(R, spec.coulomb_cutoff, d2)
            nec = len(cdst)
        if n > A - 1 or ne > E or nec > EC and split_coulomb:
            raise ValueError(f"frame {f} ({n} atoms, {ne} edges, {nec} coulomb edges) exceeds batch budget {spec}")
        if m + 1 > M - 1 or a + n > A - 1 or e + ne > E or (split_coulomb and ec + nec > EC):
            yield batch
            batch, m, a, e, ec = _empty_batch(spec), 0, 0, 0, 0
        Z = data["Z"][lo:hi].astype(np.int32)
        batch["atomic_numbers"][a:a + n] = Z
        batch["positions"][a:a + n] = R
        batch["forces"][a:a + n] = data["F"][lo:hi]
        batch["batch_segments"][a:a + n] = m
        batch["dst_idx_flat"][e:e + ne] = dst + a
        batch["src_idx_flat"][e:e + ne] = src + a
        if split_coulomb:
            batch["coulomb_dst_idx_flat"][ec:ec + nec] = cdst + a
            batch["coulomb_src_idx_flat"][ec:ec + nec] = csrc + a
        energy = float(data["E"][f])
        if e_ref is not None:
            energy -= float(e_ref[Z].sum())
        batch["energies"][m] = energy
        batch["dipoles"][m] = data["D"][f]
        batch["polar"][m] = data["polar"][f]
        batch["mol_mask"][m] = 1.0
        batch["subset"][m] = int(data["subset"][f])
        m, a, e, ec = m + 1, a + n, e + ne, ec + nec
    if m:
        yield batch


def prefetch(it: Iterator, size: int = 8) -> Iterator:
    """Run a batch iterator in a background thread."""
    q: queue.Queue = queue.Queue(maxsize=size)
    sentinel = object()

    def worker():
        try:
            for item in it:
                q.put(item)
        except BaseException as exc:  # surface errors in the consumer
            q.put(exc)
        q.put(sentinel)

    threading.Thread(target=worker, daemon=True).start()
    while True:
        item = q.get()
        if item is sentinel:
            return
        if isinstance(item, BaseException):
            raise item
        yield item


def _forward(model_apply, params, batch, M):
    (energy, dipole), state = model_apply(
        params,
        atomic_numbers=batch["atomic_numbers"],
        positions=batch["positions"],
        Ef=batch["electric_field"],
        dst_idx_flat=batch["dst_idx_flat"],
        src_idx_flat=batch["src_idx_flat"],
        batch_segments=batch["batch_segments"],
        batch_size=M,
        coulomb_dst_idx_flat=batch.get("coulomb_dst_idx_flat"),
        coulomb_src_idx_flat=batch.get("coulomb_src_idx_flat"),
        mutable=["intermediates"],
    )
    return energy, dipole, state


def _polar(model_apply, params, batch, M, field_scale):
    return predicted_polarizability_bohr3(
        model_apply, params, batch["atomic_numbers"], batch["positions"],
        batch["dst_idx_flat"], batch["src_idx_flat"], batch["batch_segments"], M,
        field_scale=field_scale,
    )


def packed_losses(model_apply, params, batch, M, weights, field_scale, gradient_checkpoint):
    """Masked losses (0.5·squared error, as optax.l2_loss elsewhere) and predictions."""
    mol = batch["mol_mask"]
    n_mol = jnp.maximum(mol.sum(), 1.0)
    real = (batch["atomic_numbers"] > 0).astype(jnp.float32)
    n_atom = jnp.maximum(real.sum(), 1.0)

    def energy_fn(pos):
        energy, dipole, state = _forward(model_apply, params, {**batch, "positions": pos}, M)
        return -jnp.sum(energy * mol), (energy, dipole, state)

    if gradient_checkpoint:
        energy_fn = jax.checkpoint(energy_fn, policy=jax.checkpoint_policies.nothing_saveable)
    (_, (energy, dipole, state)), forces = jax.value_and_grad(energy_fn, has_aux=True)(batch["positions"])
    charges = state["intermediates"]["atomic_charges"][-1]  # (A,)
    q_mol = jax.ops.segment_sum(charges, batch["batch_segments"], num_segments=M)

    terms = {
        "energy": jnp.sum(0.5 * (energy - batch["energies"]) ** 2 * mol) / n_mol,
        "forces": jnp.sum(0.5 * (forces - batch["forces"]) ** 2 * real[:, None]) / (3.0 * n_atom),
        "dipole": jnp.sum(0.5 * (dipole - batch["dipoles"]) ** 2 * mol[:, None]) / (3.0 * n_mol),
        "charge": jnp.sum(q_mol ** 2 * mol) / n_mol,
    }
    polar = jnp.zeros_like(batch["polar"])
    if weights["polar"] != 0.0:
        polar = _polar(model_apply, params, batch, M, field_scale)
        terms["polar"] = jnp.sum(0.5 * (polar - batch["polar"]) ** 2 * mol[:, None, None]) / (9.0 * n_mol)
    else:
        terms["polar"] = jnp.asarray(0.0)
    total = sum(weights[k] * terms[k] for k in terms)
    preds = {"energy": energy, "forces": forces, "dipole": dipole, "polar": polar}
    return total, (terms, preds)


def make_steps(model_apply, optimizer, M, weights, field_scale, gradient_checkpoint, ema_decay):
    """jit-compiled (train_step, eval_step) for packed batches."""

    def loss_fn(params, batch):
        return packed_losses(model_apply, params, batch, M, weights, field_scale, gradient_checkpoint)

    @jax.jit
    def train_step(params, ema_params, opt_state, batch):
        (loss, (terms, _)), grads = jax.value_and_grad(loss_fn, has_aux=True)(params, batch)
        finite = jnp.isfinite(loss)
        grads = jax.tree_util.tree_map(lambda g: jnp.where(jnp.isfinite(g), g, 0.0), grads)
        updates, new_opt_state = optimizer.update(grads, opt_state, params)
        new_params = optax.apply_updates(params, updates)
        params = jax.tree_util.tree_map(lambda n, o: jnp.where(finite, n, o), new_params, params)
        opt_state = jax.tree_util.tree_map(lambda n, o: jnp.where(finite, n, o), new_opt_state, opt_state)
        ema_params = jax.tree_util.tree_map(
            lambda e, p: jnp.where(finite, ema_decay * e + (1.0 - ema_decay) * p, e), ema_params, params)
        return params, ema_params, opt_state, loss, terms, finite

    @jax.jit
    def eval_step(params, batch):
        loss, (terms, preds) = loss_fn(params, batch)
        return loss, terms, preds

    return train_step, eval_step


class SubsetMetrics:
    """Accumulates MACE-MDP-style RMSEs per subset (dipole e·Å, polar e·Å²/V)."""

    BOHR3_PER_EA2V = 97.17

    def __init__(self, subset_names: Mapping[int, str]):
        self.names = dict(subset_names)
        self.acc: dict[str, dict[str, float]] = {}
        self._diag = np.eye(3, dtype=bool)

    def _add(self, key, name, sq, count):
        d = self.acc.setdefault(key, {})
        d[name + "_sq"] = d.get(name + "_sq", 0.0) + float(sq)
        d[name + "_n"] = d.get(name + "_n", 0.0) + float(count)

    def update(self, batch, preds):
        mol = np.asarray(batch["mol_mask"]) > 0
        sub = np.asarray(batch["subset"])
        seg = np.asarray(batch["batch_segments"])
        real = np.asarray(batch["atomic_numbers"]) > 0
        derr = (np.asarray(preds["dipole"]) - np.asarray(batch["dipoles"]))
        perr = (np.asarray(preds["polar"]) - np.asarray(batch["polar"])) / self.BOHR3_PER_EA2V
        eerr = np.asarray(preds["energy"]) - np.asarray(batch["energies"])
        ferr = np.asarray(preds["forces"]) - np.asarray(batch["forces"])
        for s in np.unique(sub[mol]):
            m = mol & (sub == s)
            for key in ("all", self.names.get(int(s), str(s))):
                self._add(key, "dipole", np.sum(derr[m] ** 2), 3 * m.sum())
                self._add(key, "polar_diag", np.sum(perr[m][:, self._diag] ** 2), 3 * m.sum())
                self._add(key, "polar_offdiag", np.sum(perr[m][:, ~self._diag] ** 2), 6 * m.sum())
                self._add(key, "energy", np.sum(eerr[m] ** 2), m.sum())
                atoms = real & m[seg]
                self._add(key, "forces", np.sum(ferr[atoms] ** 2), 3 * atoms.sum())

    def result(self) -> dict[str, dict[str, float]]:
        out = {}
        for key, d in self.acc.items():
            out[key] = {name: float(np.sqrt(d[name + "_sq"] / max(d[name + "_n"], 1.0)))
                        for name in ("dipole", "polar_diag", "polar_offdiag", "energy", "forces")}
            out[key]["n_frames"] = int(d["energy_n"])
        return out
