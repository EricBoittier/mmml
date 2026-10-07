"""ASE/matplotlib diagnostics for MM cross-checks (structure, masks, neighbor lists).

Example::

    from mmml.interfaces.pycharmmInterface.mm_system_plot import plot_mm_system_diagnostics

    fig = plot_mm_system_diagnostics(
        positions,
        symbols=symbols,
        charges=charges,
        subsystems={"peptide": peptide_mask, "water": water_mask},
        excluded_pairs=nbond_data.excluded_pairs,
        pair_i=pair_i,
        pair_j=pair_j,
        save_path="diag.png",
        show=False,
    )
"""

from __future__ import annotations

from pathlib import Path
from typing import Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np
from ase import Atoms
from ase.data import chemical_symbols
from ase.visualize.plot import plot_atoms


def _as_symbols(symbols: Sequence[str] | None, n_atoms: int) -> list[str]:
    if symbols is None:
        return ["X"] * n_atoms
    out = [str(s) for s in symbols]
    if len(out) != n_atoms:
        raise ValueError(f"symbols length {len(out)} != n_atoms {n_atoms}")
    return out


def _charge_cmap(charges: np.ndarray) -> np.ndarray:
    """Map charges to RGBA (blue = negative, red = positive)."""
    q = np.asarray(charges, dtype=float)
    if q.size == 0:
        return np.zeros((0, 4))
    scale = max(float(np.max(np.abs(q))), 1e-6)
    norm = np.clip(q / scale, -1.0, 1.0)
    rgba = np.zeros((len(norm), 4))
    rgba[:, 0] = np.clip(norm, 0.0, 1.0)
    rgba[:, 2] = np.clip(-norm, 0.0, 1.0)
    rgba[:, 3] = 1.0
    return rgba


def _pair_matrix(
    n_atoms: int,
    pairs: Sequence[tuple[int, int]] | None,
    *,
    symmetric: bool = True,
) -> np.ndarray:
    mat = np.zeros((n_atoms, n_atoms), dtype=np.float32)
    if not pairs:
        return mat
    for a_raw, b_raw in pairs:
        a, b = int(a_raw), int(b_raw)
        if a < 0 or b < 0 or a >= n_atoms or b >= n_atoms:
            continue
        mat[a, b] = 1.0
        if symmetric:
            mat[b, a] = 1.0
    return mat


def _masked_atoms(
    positions: np.ndarray,
    symbols: list[str],
    mask: np.ndarray,
) -> Atoms:
    idx = np.flatnonzero(np.asarray(mask, dtype=bool))
    if idx.size == 0:
        return Atoms()
    pos = np.asarray(positions, dtype=float)[idx]
    sym = [symbols[int(i)] for i in idx]
    return Atoms(sym, positions=pos)


def plot_mm_system_diagnostics(
    positions: np.ndarray,
    *,
    symbols: Sequence[str] | None = None,
    charges: Sequence[float] | np.ndarray | None = None,
    subsystems: Mapping[str, np.ndarray] | None = None,
    excluded_pairs: frozenset[tuple[int, int]] | set[tuple[int, int]] | None = None,
    e14_pairs: frozenset[tuple[int, int]] | set[tuple[int, int]] | None = None,
    pair_i: np.ndarray | None = None,
    pair_j: np.ndarray | None = None,
    cell: np.ndarray | None = None,
    rotation: str = "0x,0y,0z",
    save_path: Path | str | None = None,
    show: bool = True,
    dpi: int = 120,
) -> plt.Figure:
    """Plot structure, subsystem breakdowns, charge coloring, and pair matrices.

    Layout:
      - Row 0: full system (ASE ball-and-stick)
      - Row 1: one panel per named subsystem mask
      - Row 2: charge-colored structure (if ``charges`` given)
      - Row 3+: binary matrices (exclusions, 1–4 pairs, neighbor list)
    """
    pos = np.asarray(positions, dtype=float)
    n_atoms = int(pos.shape[0])
    sym = _as_symbols(symbols, n_atoms)
    atoms = Atoms(sym, positions=pos)
    if cell is not None:
        cell_mat = np.asarray(cell, dtype=float)
        if cell_mat.shape == (3,):
            atoms.set_cell(np.diag(cell_mat))
        else:
            atoms.set_cell(cell_mat)
        atoms.set_pbc(True)

    subsystem_items = list(subsystems.items()) if subsystems else []
    n_sub = max(len(subsystem_items), 1)

    has_charges = charges is not None
    matrices: list[tuple[str, np.ndarray]] = []
    if excluded_pairs:
        matrices.append(("excluded (1–2/1–3)", _pair_matrix(n_atoms, excluded_pairs)))
    if e14_pairs:
        matrices.append(("1–4 pairs", _pair_matrix(n_atoms, e14_pairs)))
    if pair_i is not None and pair_j is not None:
        nl_pairs = [
            (int(i), int(j))
            for i, j in zip(np.asarray(pair_i).ravel(), np.asarray(pair_j).ravel())
        ]
        matrices.append(("neighbor list", _pair_matrix(n_atoms, nl_pairs)))

    n_matrix_rows = max(len(matrices), 0)
    n_rows = 2 + (1 if has_charges else 0) + (1 if n_matrix_rows else 0)
    fig_h = 3.2 * n_rows
    fig_w = max(4.0 * n_sub, 4.0 * min(n_matrix_rows, 3), 8.0)
    fig = plt.figure(figsize=(fig_w, fig_h), constrained_layout=True)

    row = 0
    ax_full = fig.add_subplot(n_rows, 1, row + 1)
    plot_atoms(atoms, ax_full, rotation=rotation, radii=0.35)
    ax_full.set_title(f"Full system ({n_atoms} atoms)")
    row += 1

    if subsystem_items:
        gs = fig.add_gridspec(n_rows, n_sub, figure=fig)
        for col, (name, mask) in enumerate(subsystem_items):
            ax = fig.add_subplot(gs[row, col])
            sub_atoms = _masked_atoms(pos, sym, np.asarray(mask))
            if len(sub_atoms) == 0:
                ax.set_title(f"{name} (empty)")
                ax.axis("off")
                continue
            plot_atoms(sub_atoms, ax, rotation=rotation, radii=0.35)
            ax.set_title(f"{name} ({int(np.sum(mask))} atoms)")
        row += 1
    else:
        ax_sub = fig.add_subplot(n_rows, 1, row + 1)
        plot_atoms(atoms, ax_sub, rotation=rotation, radii=0.35)
        ax_sub.set_title("Subsystem (none specified — full system repeated)")
        row += 1

    if has_charges:
        ax_q = fig.add_subplot(n_rows, 1, row + 1)
        colors = _charge_cmap(np.asarray(charges, dtype=float))
        plot_atoms(atoms, ax_q, rotation=rotation, radii=0.35, colors=colors)
        ax_q.set_title("Atoms colored by partial charge")
        row += 1

    if matrices:
        n_cols = min(len(matrices), 3)
        gs_m = fig.add_gridspec(n_rows, n_cols, figure=fig)
        for col, (title, mat) in enumerate(matrices[:n_cols]):
            ax_m = fig.add_subplot(gs_m[row, col])
            im = ax_m.imshow(mat, origin="lower", cmap="Blues", vmin=0.0, vmax=1.0)
            ax_m.set_title(title)
            ax_m.set_xlabel("atom j")
            ax_m.set_ylabel("atom i")
            fig.colorbar(im, ax=ax_m, fraction=0.046, pad=0.04)
        row += 1

    if save_path is not None:
        out = Path(save_path).expanduser().resolve()
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=dpi, bbox_inches="tight")

    if show:
        plt.show()
    return fig


def symbols_from_atom_types(atom_types: Sequence[str]) -> list[str]:
    """Best-effort element symbols from CHARMM atom type names (for ASE plots)."""
    out: list[str] = []
    for atype in atom_types:
        name = str(atype)
        if name.startswith(("H", "h")) or "H" in name[:3]:
            out.append("H")
        elif name.startswith(("O", "o")) or "O" in name[:3]:
            out.append("O")
        elif name.startswith(("N", "n")) or "N" in name[:3]:
            out.append("N")
        elif name.startswith(("C", "c")) or "C" in name[:3]:
            out.append("C")
        elif name.startswith(("S", "s")):
            out.append("S")
        elif name.startswith(("P", "p")):
            out.append("P")
        else:
            out.append("X")
    return out


def symbols_from_atomic_numbers(atomic_numbers: Sequence[int]) -> list[str]:
    return [chemical_symbols[int(z)] for z in atomic_numbers]
