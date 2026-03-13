"""
Braille rasterizer for molecular structures in the terminal.

Orthographic projection, zoom-to-fill, Jmol colors. Renders atoms as spheres
and force vectors as arrows. No bonds. Retro 64-bit vibe.
"""
from __future__ import annotations

from typing import Optional

import numpy as np

# Braille dot bit positions: (col, row) in 2x4 cell -> bit
# Standard 8-dot braille: left col = dots 1,2,3,7; right col = 4,5,6,8
_BRAILLE_BITS = [
    (0, 0, 0x01),
    (1, 0, 0x08),
    (0, 1, 0x02),
    (1, 1, 0x10),
    (0, 2, 0x04),
    (1, 2, 0x20),
    (0, 3, 0x40),
    (1, 3, 0x80),
]

# Force arrow color (cyan for visibility)
_FORCE_COLOR = np.array([0.0, 1.0, 1.0], dtype=np.float64)


def _rgb_to_ansi256(r: int, g: int, b: int) -> str:
    """Map RGB (0-255) to ANSI 256-color escape sequence."""
    if r == g == b and r < 8:
        return "\033[38;5;0m"  # Black
    if r == g == b and r > 248:
        return f"\033[38;5;{231}\033[m"  # White
    # 6x6x6 color cube
    r6 = min(5, r * 6 // 256)
    g6 = min(5, g * 6 // 256)
    b6 = min(5, b * 6 // 256)
    idx = 16 + 36 * r6 + 6 * g6 + b6
    return f"\033[38;5;{idx}m"


def _rasterize_circle(
    buf_r: np.ndarray,
    buf_g: np.ndarray,
    buf_b: np.ndarray,
    cx: float,
    cy: float,
    radius: float,
    r: float,
    g: float,
    b: float,
) -> None:
    """Draw filled circle in pixel buffers. Coords in pixel space."""
    h, w = buf_r.shape
    x0 = int(max(0, cx - radius - 1))
    x1 = int(min(w, cx + radius + 2))
    y0 = int(max(0, cy - radius - 1))
    y1 = int(min(h, cy + radius + 2))
    r2 = radius * radius
    for py in range(y0, y1):
        for px in range(x0, x1):
            dx = px - cx
            dy = py - cy
            if dx * dx + dy * dy <= r2:
                buf_r[py, px] = r
                buf_g[py, px] = g
                buf_b[py, px] = b


def _rasterize_line(
    buf_r: np.ndarray,
    buf_g: np.ndarray,
    buf_b: np.ndarray,
    x0: float,
    y0: float,
    x1: float,
    y1: float,
    r: float,
    g: float,
    b: float,
    thickness: float = 0.5,
) -> None:
    """Bresenham-style line rasterization with thickness."""
    h, w = buf_r.shape
    steps = int(max(abs(x1 - x0), abs(y1 - y0), 1) * 2)
    for i in range(steps + 1):
        t = i / steps
        x = x0 + t * (x1 - x0)
        y = y0 + t * (y1 - y0)
        ix, iy = int(x), int(y)
        for dy in range(-1, 2):
            for dx in range(-1, 2):
                px, py = ix + dx, iy + dy
                if 0 <= px < w and 0 <= py < h:
                    buf_r[py, px] = r
                    buf_g[py, px] = g
                    buf_b[py, px] = b


def render_braille(
    positions: np.ndarray,
    atomic_numbers: np.ndarray,
    forces: Optional[np.ndarray] = None,
    *,
    width: int = 80,
    height: int = 24,
    projection: str = "xy",
    atom_scale: float = 1.2,
    force_scale: Optional[float] = None,
) -> str:
    """
    Render molecule to braille string with ANSI colors.

    Args:
        positions: (N, 3) atom positions in Angstroms
        atomic_numbers: (N,) atomic numbers
        forces: (N, 3) force vectors, optional. If None, forces are not drawn.
        width: Braille cells wide (2 pixels per cell)
        height: Braille cells tall (4 pixels per cell)
        projection: 'xy' (top-down), 'xz' (front), or 'yz' (side)
        atom_scale: Scale factor for atom radii (covalent * scale)
        force_scale: Scale for force arrows. Auto if None (fit to view).

    Returns:
        ANSI-colored string with newlines, ready to print.
    """
    from ase.data.colors import jmol_colors
    from ase.data import covalent_radii

    # Project to 2D
    if projection == "xy":
        proj = positions[:, :2]  # x, y
    elif projection == "xz":
        proj = positions[:, [0, 2]]  # x, z
    elif projection == "yz":
        proj = positions[:, 1:]  # y, z
    else:
        proj = positions[:, :2]

    # Pixel dimensions (2 cols x 4 rows per braille cell)
    pw = width * 2
    ph = height * 4

    # Zoom to fit with padding (orthographic, aspect-preserving)
    x_min, x_max = proj[:, 0].min(), proj[:, 0].max()
    y_min, y_max = proj[:, 1].min(), proj[:, 1].max()
    pad = 0.1
    range_x = (x_max - x_min) or 1.0
    range_y = (y_max - y_min) or 1.0
    scale_x = (1 - 2 * pad) * pw / range_x
    scale_y = (1 - 2 * pad) * ph / range_y
    scale = min(scale_x, scale_y)
    pad_px = pad * pw
    pad_py = pad * ph

    def to_pixel(x: float, y: float) -> tuple[float, float]:
        px = (x - x_min) * scale + pad_px
        py = (y_max - y) * scale + pad_py  # flip y for screen coords
        return px, py

    # Allocate buffers (0 = background/dark)
    buf_r = np.zeros((ph, pw), dtype=np.float64)
    buf_g = np.zeros((ph, pw), dtype=np.float64)
    buf_b = np.zeros((ph, pw), dtype=np.float64)

    # Get radii and colors
    radii = np.array([covalent_radii[z] if z > 0 else 0.3 for z in atomic_numbers])
    radii = radii * atom_scale * scale

    colors = np.array([jmol_colors[z] for z in atomic_numbers], dtype=np.float64)

    # Draw atoms
    for i in range(len(positions)):
        px, py = to_pixel(proj[i, 0], proj[i, 1])
        r, g, b = colors[i]
        _rasterize_circle(buf_r, buf_g, buf_b, px, py, radii[i], r, g, b)

    # Draw force vectors
    if forces is not None and np.any(np.abs(forces) > 1e-10):
        fmag = np.linalg.norm(forces, axis=1)
        fmax = float(np.max(fmag)) or 1.0
        if force_scale is None:
            # Auto-scale: arrow length ~ 1/4 of view
            force_scale = min(range_x, range_y) / 4.0 / fmax
        for i in range(len(positions)):
            f = forces[i]
            if np.linalg.norm(f) < 1e-10:
                continue
            x0, y0 = to_pixel(proj[i, 0], proj[i, 1])
            dx = f[0] * force_scale * scale
            dy = -f[1] * force_scale * scale  # flip y
            if projection == "xz":
                dx = f[0] * force_scale * scale
                dy = -f[2] * force_scale * scale
            elif projection == "yz":
                dx = f[1] * force_scale * scale
                dy = -f[2] * force_scale * scale
            x1, y1 = x0 + dx, y0 + dy
            _rasterize_line(
                buf_r, buf_g, buf_b,
                x0, y0, x1, y1,
                _FORCE_COLOR[0], _FORCE_COLOR[1], _FORCE_COLOR[2],
            )

    # Convert to braille
    reset = "\033[m"
    lines = []
    for cy in range(height):
        row = []
        for cx in range(width):
            bits = 0
            r_sum, g_sum, b_sum = 0.0, 0.0, 0.0
            n = 0
            for dx, dy, bit in _BRAILLE_BITS:
                px = cx * 2 + dx
                py = cy * 4 + dy
                if buf_r[py, px] > 0 or buf_g[py, px] > 0 or buf_b[py, px] > 0:
                    bits |= bit
                    r_sum += buf_r[py, px]
                    g_sum += buf_g[py, px]
                    b_sum += buf_b[py, px]
                    n += 1
            if n > 0:
                r = int(255 * r_sum / n)
                g = int(255 * g_sum / n)
                b = int(255 * b_sum / n)
                row.append(_rgb_to_ansi256(r, g, b) + chr(0x2800 + bits))
            else:
                row.append(reset + " ")
        lines.append("".join(row) + reset)
    return "\n".join(lines)


def render_atoms_braille(
    atoms: object,
    *,
    forces: Optional[np.ndarray] = None,
    **kwargs,
) -> str:
    """
    Convenience wrapper for ASE Atoms.

    Args:
        atoms: ASE Atoms object
        forces: Optional (N,3) array. If None, uses atoms.get_forces() if calc exists.
        **kwargs: Passed to render_braille

    Returns:
        Braille string
    """
    positions = np.asarray(atoms.get_positions())
    atomic_numbers = np.asarray(atoms.get_atomic_numbers())
    if forces is None:
        calc = getattr(atoms, "calc", None)
        if calc is not None:
            try:
                forces = atoms.get_forces()
            except Exception:
                forces = None
    return render_braille(
        positions,
        atomic_numbers,
        forces=forces,
        **kwargs,
    )
