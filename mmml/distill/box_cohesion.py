"""Teacher / student interaction labels of whole liquid frames (for MM tuning).

For a periodic frame of ``M`` identical molecules this computes

* teacher: ``E_int = E(box, PBC) - sum_A E(A)`` and
  ``F_int = F(box) - F(A)`` per atom (gas-phase monomers),
* student (PhysNet ML/MM monomer+dimer model): every molecule pair with
  centroid separation below ``r_pair_max`` -> ``e = P(AB) - P(A) - P(B)`` and
  the matching interaction forces.

The ML taper is *not* applied here, so a single label file can be refitted
for any handoff window with ``mm_switch_on <= r_pair_max``
(:func:`frame_features_from_labels`). Energies are stored in kcal/mol.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

from mmml.models.mm_nonbonded_tune import (
    EV_TO_KCAL,
    FrameFeatures,
    MonomerNonbonded,
    SwitchConfig,
    frame_features,
    min_image_pairs,
    force_blocks,
    ml_pair_energy_forces,
)

__all__ = [
    "BoxFrame",
    "PhysNetPairEvaluator",
    "iter_box_frames",
    "label_frame",
    "save_labels",
    "load_labels",
    "frame_features_from_labels",
]


@dataclass
class BoxFrame:
    group: int
    index: int
    numbers: np.ndarray  # (N,)
    positions: np.ndarray  # (N, 3) as stored (wrapped)
    cell: np.ndarray  # (3, 3)
    mols: np.ndarray  # (M, a, 3) unwrapped around each molecule's first atom
    phase: str = ""
    energy_eV: float | None = None  # stored box energy (extxyz), if any
    forces: np.ndarray | None = None  # stored box forces eV/A, if any


def iter_box_frames(
    paths: Sequence[Path | str],
    *,
    atoms_per_molecule: int,
    phase: str | None = "md",
    stride: int = 1,
    max_per_file: int | None = None,
    group_every: int | None = None,
) -> Iterable[BoxFrame]:
    """Frames from extxyz trajectories (``seed``/``phase`` info keys optional).

    Without a ``seed`` key the group (bootstrap / hold-out unit) is the file
    index, or with ``group_every`` a time block of that many kept frames:
    ``1000 * file_index + kept // group_every``.
    """
    from ase.io import iread

    from mmml.distill.box_clusters import whole_molecules

    for fi, path in enumerate(paths):
        kept = 0
        n_match = 0
        for k, atoms in enumerate(iread(str(path), index=":")):
            ph = str(atoms.info.get("phase", ""))
            if phase and ph != phase:
                continue
            n_match += 1
            if (n_match - 1) % max(1, int(stride)):
                continue
            if max_per_file is not None and kept >= int(max_per_file):
                break
            if "seed" in atoms.info:
                group = int(atoms.info["seed"])
            elif group_every:
                group = 1000 * fi + kept // int(group_every)
            else:
                group = fi
            e_file = f_file = None
            if atoms.calc is not None:
                try:
                    e_file = float(atoms.get_potential_energy())
                    f_file = np.asarray(atoms.get_forces(), dtype=np.float64)
                except Exception:  # pragma: no cover - frames without results
                    e_file = f_file = None
            yield BoxFrame(
                group=group,
                index=k,
                numbers=np.asarray(atoms.numbers),
                positions=np.asarray(atoms.positions, dtype=np.float64),
                cell=np.asarray(atoms.cell[:], dtype=np.float64),
                mols=whole_molecules(atoms, atoms_per_molecule),
                phase=ph,
                energy_eV=e_file,
                forces=f_file,
            )
            kept += 1


class PhysNetPairEvaluator:
    """Batched PhysNet (hybrid-MLpot checkpoint) energies/forces, kcal/mol.

    Structures are padded to ``max_atoms``; a fixed batch shape keeps one jit
    compilation. ``energy_to_kcal`` converts the checkpoint's energy unit
    (eV for the PET-distilled students).
    """

    def __init__(
        self,
        checkpoint: Path | str,
        *,
        max_atoms: int,
        batch_size: int = 256,
        energy_to_kcal: float = EV_TO_KCAL,
    ) -> None:
        import e3x
        import jax
        import jax.numpy as jnp

        from mmml.interfaces.calculators.checkpoint_loading import (
            load_physnet_for_hybrid_mlpot,
        )

        self.model, self.params, _ = load_physnet_for_hybrid_mlpot(
            checkpoint, max_padded_atoms=int(max_atoms)
        )
        self.n = int(max_atoms)
        self.B = int(batch_size)
        self.scale = float(energy_to_kcal)
        dst, src = e3x.ops.sparse_pairwise_indices(self.n)
        offs = np.arange(self.B) * self.n
        self._dst = jnp.asarray((np.asarray(dst)[None] + offs[:, None]).reshape(-1))
        self._src = jnp.asarray((np.asarray(src)[None] + offs[:, None]).reshape(-1))
        self._dst_local = np.asarray(dst)
        self._src_local = np.asarray(src)
        self._seg = jnp.asarray(np.repeat(np.arange(self.B), self.n))
        model = self.model

        @jax.jit
        def _apply(params, Z, R, batch_mask, atom_mask):
            out = model.apply(
                params,
                atomic_numbers=Z,
                positions=R,
                dst_idx=self._dst,
                src_idx=self._src,
                batch_segments=self._seg,
                batch_size=self.B,
                batch_mask=batch_mask,
                atom_mask=atom_mask,
            )
            return out["energy"].reshape(-1), out["forces"]

        self._apply = _apply

    def evaluate(
        self, structures: Sequence[tuple[np.ndarray, np.ndarray]]
    ) -> tuple[np.ndarray, list[np.ndarray]]:
        n, B = self.n, self.B
        energies = np.zeros(len(structures))
        forces: list[np.ndarray] = [None] * len(structures)  # type: ignore[list-item]
        for start in range(0, len(structures), B):
            chunk = structures[start : start + B]
            Z = np.zeros((B, n), dtype=np.int32)
            R = np.zeros((B, n, 3), dtype=np.float32)
            nat = np.zeros(B, dtype=np.int32)
            for k, (z, r) in enumerate(chunk):
                m = len(z)
                if m > n:
                    raise ValueError(f"structure with {m} atoms > max_atoms={n}")
                Z[k, :m] = z
                R[k, :m] = r
                # park padding atoms far apart so no spurious short distances
                R[k, m:] = 100.0 + 10.0 * np.arange(n - m)[:, None]
                nat[k] = m
            bm = (
                (self._dst_local[None] < nat[:, None]) & (self._src_local[None] < nat[:, None])
            ).reshape(-1)
            am = (Z > 0).reshape(-1)
            e, f = self._apply(
                self.params,
                Z.reshape(-1),
                R.reshape(-1, 3),
                bm.astype(np.float32),
                am.astype(np.float32),
            )
            e = np.asarray(e, dtype=np.float64) * self.scale
            f = np.asarray(f, dtype=np.float64).reshape(B, n, 3) * self.scale
            for k, (z, _) in enumerate(chunk):
                energies[start + k] = e[k]
                forces[start + k] = f[k, : len(z)]
        return energies, forces


def teacher_energy_only(teacher, structures: Sequence[tuple]) -> np.ndarray:
    """Teacher energies (eV) without building an autograd graph.

    A large PET on a ~3000-atom periodic box does not fit a 32 GB GPU once
    forces are requested; energy-only evaluation does. ``teacher`` is a
    :class:`mmml.distill.batched_teacher.BatchedMetatomicTeacher`.
    """
    import torch
    from metatomic.torch import ModelEvaluationOptions, ModelOutput

    out = np.zeros(len(structures))
    for k, item in enumerate(structures):
        with torch.no_grad():
            systems = teacher._systems([item])
            for s in systems:
                s.positions.requires_grad_(False)
            teacher._attach_neighbors(systems)
            options = ModelEvaluationOptions(
                length_unit="angstrom",
                outputs={"energy": ModelOutput(quantity="energy", unit="eV", per_atom=False)},
            )
            res = teacher.model(systems, options, check_consistency=False)
            out[k] = float(res["energy"].block().values.reshape(-1)[0])
        del systems, res
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
    return out


def label_frame(
    frame: BoxFrame,
    *,
    teacher=None,
    student: PhysNetPairEvaluator | None = None,
    r_pair_max: float = 8.0,
    box_forces: bool = True,
    box_from_file: bool = False,
) -> dict[str, Any]:
    """Teacher and/or student interaction labels of one frame (kcal/mol).

    ``box_forces=False`` evaluates the teacher box energy without forces
    (memory); ``teacher_f_int`` is then omitted and fits are energy-only.
    """
    mols = frame.mols
    M, a, _ = mols.shape
    z_mol = frame.numbers[:a]
    out: dict[str, Any] = {
        "group": frame.group,
        "index": frame.index,
        "cell": frame.cell,
        "mols": mols,
        "z_mol": z_mol,
    }
    if teacher is not None:
        box = (frame.numbers, frame.positions, frame.cell)
        res = teacher.evaluate([(z_mol, mols[m]) for m in range(M)])
        e_mono = np.array([r[0] for r in res])
        f_mono = np.stack([r[1] for r in res])
        if box_from_file:
            # frames sampled *by the teacher itself*: reuse its stored E/F
            if frame.energy_eV is None or frame.forces is None:
                raise ValueError("box_from_file needs energy/forces in the trajectory")
            e_box, f_box = frame.energy_eV, frame.forces
            out["teacher_f_int"] = (f_box.reshape(M, a, 3) - f_mono) * EV_TO_KCAL
        elif box_forces:
            e_box, f_box = teacher.evaluate([box])[0]
            out["teacher_f_int"] = (f_box.reshape(M, a, 3) - f_mono) * EV_TO_KCAL
        else:
            e_box = float(teacher_energy_only(teacher, [box])[0])
        out["teacher_e_int"] = float(e_box - e_mono.sum()) * EV_TO_KCAL
        out["teacher_e_box"] = float(e_box) * EV_TO_KCAL
        out["teacher_e_mono"] = e_mono * EV_TO_KCAL
    if student is not None:
        pairs, shifts, r_com = min_image_pairs(mols, frame.cell, r_pair_max)
        structs = [(z_mol, mols[m]) for m in range(M)]
        structs += [
            (np.concatenate([z_mol, z_mol]), np.concatenate([mols[i], mols[j] + s]))
            for (i, j), s in zip(pairs, shifts)
        ]
        e, f = student.evaluate(structs)
        e_mono, f_mono = e[:M], f[:M]
        e_ab = e[M:]
        f_ab = np.stack(f[M:]) if len(pairs) else np.zeros((0, 2 * a, 3))
        e_pair = e_ab - e_mono[pairs[:, 0]] - e_mono[pairs[:, 1]]
        f_int = f_ab.copy()
        f_mono_arr = np.stack(f_mono)
        f_int[:, :a] -= f_mono_arr[pairs[:, 0]]
        f_int[:, a:] -= f_mono_arr[pairs[:, 1]]
        out.update(
            pairs=pairs, shifts=shifts, r_com=r_com, student_e_pair=e_pair,
            student_f_pair=f_int, student_e_mono=e_mono,
        )
    return out


_RAGGED = ("pairs", "shifts", "r_com", "student_e_pair", "student_f_pair")


def save_labels(path: Path | str, frames: Sequence[dict[str, Any]], meta: dict | None = None) -> None:
    """Stack per-frame label dicts into one NPZ (pair arrays ragged via offsets)."""
    import json

    arrays: dict[str, Any] = {}
    keys = set().union(*[f.keys() for f in frames])
    for k in sorted(keys):
        vals = [f[k] for f in frames]
        if k in _RAGGED:
            arrays[k] = np.concatenate(vals, axis=0)
            if k == "pairs":
                arrays["pair_offsets"] = np.cumsum([0] + [len(v) for v in vals])
        else:
            arrays[k] = np.stack([np.asarray(v) for v in vals])
    arrays["meta"] = json.dumps(meta or {})
    np.savez_compressed(path, **arrays)


def load_labels(path: Path | str) -> list[dict[str, Any]]:
    d = np.load(path, allow_pickle=False)
    n = d["group"].shape[0]
    frames = []
    off = d["pair_offsets"] if "pair_offsets" in d.files else None
    for i in range(n):
        f: dict[str, Any] = {}
        for k in d.files:
            if k in ("meta", "pair_offsets"):
                continue
            if k in _RAGGED:
                f[k] = d[k][off[i] : off[i + 1]]
            else:
                f[k] = d[k][i]
        frames.append(f)
    return frames


def frame_features_from_labels(
    frames: Sequence[dict[str, Any]],
    ff: MonomerNonbonded,
    switch: SwitchConfig,
    *,
    with_forces: bool = True,
    progress: bool = False,
) -> FrameFeatures:
    """Reduce labelled frames to :class:`FrameFeatures` for one switch config."""
    G, e_res, n_mol, group, A, b, f2, n_at, e_t, e_ml = ([] for _ in range(10))
    for k, fr in enumerate(frames):
        mols = np.asarray(fr["mols"])
        M, a, _ = mols.shape
        e_ml_k, f_ml = ml_pair_energy_forces(
            switch, M, a, fr["pairs"], fr["r_com"], fr["shifts"], mols,
            fr["student_e_pair"], fr["student_f_pair"] if with_forces else None,
        )
        g, J = frame_features(ff, switch, mols, fr["cell"], with_jacobian=with_forces)
        G.append(g)
        e_t.append(float(fr["teacher_e_int"]))
        e_ml.append(e_ml_k)
        e_res.append(float(fr["teacher_e_int"]) - e_ml_k)
        n_mol.append(M)
        n_at.append(M * a)
        group.append(int(fr["group"]))
        if with_forces:
            f_res = (np.asarray(fr["teacher_f_int"]) - f_ml).reshape(-1)
            Lk, yk, fp = force_blocks(J, f_res)
            A.append(Lk)
            b.append(yk)
            f2.append(fp)
        if progress and (k + 1) % 20 == 0:
            print(f"  features {k + 1}/{len(frames)}", flush=True)
    arr = lambda x: np.asarray(x, dtype=np.float64)  # noqa: E731
    return FrameFeatures(
        G=arr(G), e_res=arr(e_res), n_mol=arr(n_mol), group=np.asarray(group),
        L=arr(A) if with_forces else None, y=arr(b) if with_forces else None,
        f_perp_sq=arr(f2) if with_forces else None, n_atoms=arr(n_at),
        e_teacher=arr(e_t), e_ml=arr(e_ml), meta={"switch": switch.to_dict()},
    )
