"""

OpenGL renderer for molecular structures: spheres (atoms) and cylinders (bonds).
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import ctypes
from OpenGL.GL import (
    GL_AMBIENT,
    GL_AMBIENT_AND_DIFFUSE,
    GL_LIGHT_MODEL_AMBIENT,
    GL_BLEND,
    GL_CLAMP_TO_EDGE,
    GL_COLOR_BUFFER_BIT,
    GL_DEPTH_BUFFER_BIT,
    GL_DEPTH_TEST,
    GL_DIFFUSE,
    GL_FRONT,
    GL_LIGHT0,
    GL_LIGHTING,
    GL_LINES,
    GL_LINE_STRIP,
    GL_LINEAR,
    GL_ONE_MINUS_SRC_ALPHA,
    GL_POSITION,
    GL_QUADS,
    GL_RGBA,
    GL_SHININESS,
    GL_SMOOTH,
    GL_SPECULAR,
    GL_SRC_ALPHA,
    GL_TEXTURE_2D,
    GL_TRIANGLES,
    GL_TEXTURE_MAG_FILTER,
    GL_TEXTURE_MIN_FILTER,
    GL_TEXTURE_WRAP_S,
    GL_TEXTURE_WRAP_T,
    GL_UNSIGNED_BYTE,
    glBegin,
    glBindTexture,
    glBlendFunc,
    glClear,
    glClearColor,
    glColor3f,
    glColor4f,
    glDisable,
    glEnable,
    glEnd,
    glGenTextures,
    glLightModelfv,
    glLightfv,
    glLineWidth,
    glLoadIdentity,
    glMaterialfv,
    glOrtho,
    glMatrixMode,
    GL_MODELVIEW,
    GL_PROJECTION,
    glNormal3f,
    glPopMatrix,
    glShadeModel,
    glPushMatrix,
    glRotatef,
    glTexCoord2f,
    glTexImage2D,
    glTexParameteri,
    glTranslatef,
    glVertex3f,
    glViewport,
    glGetIntegerv,
    GL_VIEWPORT,
)
from OpenGL.GLU import (
    GLU_SMOOTH,
    gluCylinder,
    gluLookAt,
    gluNewQuadric,
    gluPerspective,
    gluQuadricNormals,
    gluSphere,
)

from .molecule import (
    Atom,
    CPK_COLORS,
    VDW_RADII,
    compute_angles,
    compute_bonds,
    compute_dihedrals,
)

if TYPE_CHECKING:
    from typing import Sequence


_SPHERE_QUADRIC = None
_CYLINDER_QUADRIC = None


def _get_sphere_quadric():
    global _SPHERE_QUADRIC
    if _SPHERE_QUADRIC is None:
        _SPHERE_QUADRIC = gluNewQuadric()
        gluQuadricNormals(_SPHERE_QUADRIC, GLU_SMOOTH)
    return _SPHERE_QUADRIC


def _get_cylinder_quadric():
    global _CYLINDER_QUADRIC
    if _CYLINDER_QUADRIC is None:
        _CYLINDER_QUADRIC = gluNewQuadric()
        gluQuadricNormals(_CYLINDER_QUADRIC, GLU_SMOOTH)
    return _CYLINDER_QUADRIC


def _sphere_vertices(radius: float, slices: int = 12, stacks: int = 8) -> list[tuple[float, float, float]]:
    """Generate vertices for a sphere (simple wireframe or low-poly)."""
    verts: list[tuple[float, float, float]] = []
    for i in range(stacks + 1):
        phi = math.pi * i / stacks
        for j in range(slices + 1):
            theta = 2 * math.pi * j / slices
            x = radius * math.sin(phi) * math.cos(theta)
            y = radius * math.cos(phi)
            z = radius * math.sin(phi) * math.sin(theta)
            verts.append((x, y, z))
    return verts


def _draw_sphere_solid(x: float, y: float, z: float, radius: float, color: tuple[float, float, float], slices: int = 24, stacks: int = 16) -> None:
    """Draw a solid lit sphere using GLU quadric."""
    glMaterialfv(GL_FRONT, GL_AMBIENT_AND_DIFFUSE, (*color, 1.0))
    glMaterialfv(GL_FRONT, GL_SPECULAR, (0.4, 0.4, 0.4, 1.0))
    glMaterialfv(GL_FRONT, GL_SHININESS, (32.0,))
    glPushMatrix()
    glTranslatef(x, y, z)
    gluSphere(_get_sphere_quadric(), radius, slices, stacks)
    glPopMatrix()


def _draw_sphere_wire(x: float, y: float, z: float, radius: float, color: tuple[float, float, float], slices: int = 8, stacks: int = 6) -> None:
    """Draw a wireframe sphere (faster)."""
    glColor3f(*color)
    glTranslatef(x, y, z)
    # Latitude rings
    for i in range(1, stacks):
        phi = math.pi * i / stacks
        r = radius * math.sin(phi)
        yc = radius * math.cos(phi)
        glBegin(GL_LINE_STRIP)
        for j in range(slices + 1):
            theta = 2 * math.pi * j / slices
            glVertex3f(r * math.cos(theta), yc, r * math.sin(theta))
        glEnd()
    # Longitude lines
    for j in range(slices):
        theta = 2 * math.pi * j / slices
        glBegin(GL_LINE_STRIP)
        for i in range(stacks + 1):
            phi = math.pi * i / stacks
            glVertex3f(
                radius * math.sin(phi) * math.cos(theta),
                radius * math.cos(phi),
                radius * math.sin(phi) * math.sin(theta),
            )
        glEnd()
    glTranslatef(-x, -y, -z)


def draw_atoms(atoms: Sequence[Atom], atom_scale: float = 0.3, wireframe: bool = False) -> None:
    """Draw atoms as spheres."""
    for a in atoms:
        r = VDW_RADII.get(a.element, 1.5) * atom_scale
        color = CPK_COLORS.get(a.element, (0.5, 0.5, 0.5))
        if wireframe:
            glDisable(GL_LIGHTING)
            _draw_sphere_wire(a.x, a.y, a.z, r, color)
            glEnable(GL_LIGHTING)
        else:
            _draw_sphere_solid(a.x, a.y, a.z, r, color)


def _draw_cylinder_between(
    x1: float, y1: float, z1: float,
    x2: float, y2: float, z2: float,
    radius: float, slices: int = 8,
) -> None:
    """Draw a cylinder from (x1,y1,z1) to (x2,y2,z2)."""
    dx = x2 - x1
    dy = y2 - y1
    dz = z2 - z1
    length = math.sqrt(dx * dx + dy * dy + dz * dz)
    if length < 1e-6:
        return
    vx, vy, vz = dx / length, dy / length, dz / length
    glPushMatrix()
    glTranslatef(x1, y1, z1)
    if vz > 0.999:
        pass  # already along +Z
    elif vz < -0.999:
        glRotatef(180.0, 1.0, 0.0, 0.0)
    else:
        ax = -vy
        ay = vx
        az = 0.0
        angle = math.degrees(math.acos(vz))
        glRotatef(angle, ax, ay, az)
    gluCylinder(_get_cylinder_quadric(), radius, radius, length, slices, 1)
    glPopMatrix()


def draw_bonds(
    atoms: Sequence[Atom],
    bonds: list[tuple[int, int]],
    color: tuple[float, float, float] = (0.5, 0.5, 0.5),
    radius: float = 0.12,
) -> None:
    """Draw bonds as thick cylinders."""
    glMaterialfv(GL_FRONT, GL_AMBIENT_AND_DIFFUSE, (*color, 1.0))
    glMaterialfv(GL_FRONT, GL_SPECULAR, (0.2, 0.2, 0.2, 1.0))
    glMaterialfv(GL_FRONT, GL_SHININESS, (16.0,))
    for i, j in bonds:
        if i < len(atoms) and j < len(atoms):
            a, b = atoms[i], atoms[j]
            _draw_cylinder_between(a.x, a.y, a.z, b.x, b.y, b.z, radius)


def draw_forces(atoms: Sequence[Atom], scale: float = 0.15, color: tuple[float, float, float] = (1.0, 0.0, 0.0), arrow_radius: float = 0.06) -> None:
    """Draw force vectors as arrows from each atom when forces are available."""
    glMaterialfv(GL_FRONT, GL_AMBIENT_AND_DIFFUSE, (*color, 1.0))
    glMaterialfv(GL_FRONT, GL_SPECULAR, (0.2, 0.2, 0.2, 1.0))
    glMaterialfv(GL_FRONT, GL_SHININESS, (16.0,))
    for a in atoms:
        f = a.force
        if f is None or (f[0] == 0 and f[1] == 0 and f[2] == 0):
            continue
        mag = math.sqrt(f[0] ** 2 + f[1] ** 2 + f[2] ** 2)
        if mag < 1e-9:
            continue
        # Arrow from atom to atom + scale * force
        ex = a.x + scale * f[0]
        ey = a.y + scale * f[1]
        ez = a.z + scale * f[2]
        _draw_cylinder_between(a.x, a.y, a.z, ex, ey, ez, arrow_radius)
        # Small cone at tip (simplified: extra cylinder segment)
        tip_frac = 0.85
        tx = a.x + tip_frac * scale * f[0]
        ty = a.y + tip_frac * scale * f[1]
        tz = a.z + tip_frac * scale * f[2]
        _draw_cylinder_between(tx, ty, tz, ex, ey, ez, arrow_radius * 1.8)


def draw_molecule(atoms: Sequence[Atom], bonds: list[tuple[int, int]] | None = None, atom_scale: float = 0.3, wireframe: bool = False, bond_radius: float | None = None, force_scale: float = 6.0) -> None:
    """Draw full molecule: atoms, bonds, and forces (when available)."""
    if bonds is None:
        bonds = compute_bonds(list(atoms))
    br = bond_radius if bond_radius is not None else atom_scale * 0.4
    draw_bonds(atoms, bonds, radius=br)
    draw_atoms(atoms, atom_scale=atom_scale, wireframe=wireframe)
    if any(a.force is not None for a in atoms):
        # Auto-scale: normalize by max force magnitude for visibility
        mags = [math.sqrt(f[0]**2 + f[1]**2 + f[2]**2) for a in atoms if (f := a.force) is not None]
        max_mag = max(mags) if mags else 1.0
        effective_scale = force_scale / max(1e-9, max_mag)
        draw_forces(atoms, scale=effective_scale)


def draw_pointer_ray(
    start: tuple[float, float, float],
    direction: tuple[float, float, float],
    length: float = 2.0,
    color: tuple[float, float, float] = (0.2, 0.8, 1.0),
    line_width: float = 3.0,
) -> None:
    """Draw a laser pointer ray from start in direction. direction should be unit length."""
    dx, dy, dz = direction
    ex = start[0] + length * dx
    ey = start[1] + length * dy
    ez = start[2] + length * dz
    glDisable(GL_LIGHTING)
    glLineWidth(line_width)
    glColor3f(*color)
    glBegin(GL_LINES)
    glVertex3f(start[0], start[1], start[2])
    glVertex3f(ex, ey, ez)
    glEnd()
    glEnable(GL_LIGHTING)


# VMD-style cell colors: a=X=red, b=Y=green, c=Z=blue
_CELL_COLOR_A = (1.0, 0.2, 0.2)  # red (X)
_CELL_COLOR_B = (0.2, 1.0, 0.2)  # green (Y)
_CELL_COLOR_C = (0.2, 0.2, 1.0)  # blue (Z)


def draw_pbc_cell(
    cell: tuple[tuple[float, float, float], tuple[float, float, float], tuple[float, float, float]],
    center: tuple[float, float, float],
    scale: float = 1.0,
) -> None:
    """Draw PBC cell as wireframe box. VMD-style: a=red, b=green, c=blue."""
    a, b, c = cell
    corners = [
        (0, 0, 0),
        (a[0], a[1], a[2]),
        (b[0], b[1], b[2]),
        (c[0], c[1], c[2]),
        (a[0] + b[0], a[1] + b[1], a[2] + b[2]),
        (a[0] + c[0], a[1] + c[1], a[2] + c[2]),
        (b[0] + c[0], b[1] + c[1], b[2] + c[2]),
        (a[0] + b[0] + c[0], a[1] + b[1] + c[1], a[2] + b[2] + c[2]),
    ]
    transformed = [
        (
            scale * (p[0] - center[0]),
            scale * (p[1] - center[1]),
            scale * (p[2] - center[2]),
        )
        for p in corners
    ]
    # Edges: (i,j) -> color by which cell vector. a-edges, b-edges, c-edges
    edges_a = [(0, 1), (2, 4), (3, 5), (6, 7)]  # parallel to a
    edges_b = [(0, 2), (1, 4), (3, 6), (5, 7)]  # parallel to b
    edges_c = [(0, 3), (1, 5), (2, 6), (4, 7)]  # parallel to c
    glDisable(GL_LIGHTING)
    glLineWidth(2.0)
    for i, j in edges_a:
        glColor3f(*_CELL_COLOR_A)
        glBegin(GL_LINES)
        glVertex3f(transformed[i][0], transformed[i][1], transformed[i][2])
        glVertex3f(transformed[j][0], transformed[j][1], transformed[j][2])
        glEnd()
    for i, j in edges_b:
        glColor3f(*_CELL_COLOR_B)
        glBegin(GL_LINES)
        glVertex3f(transformed[i][0], transformed[i][1], transformed[i][2])
        glVertex3f(transformed[j][0], transformed[j][1], transformed[j][2])
        glEnd()
    for i, j in edges_c:
        glColor3f(*_CELL_COLOR_C)
        glBegin(GL_LINES)
        glVertex3f(transformed[i][0], transformed[i][1], transformed[i][2])
        glVertex3f(transformed[j][0], transformed[j][1], transformed[j][2])
        glEnd()
    glEnable(GL_LIGHTING)


# Simple 5x7 bitmap font for digits, '.', '°', overlay labels. Each char is 6x8 in texture (1px padding).
_FONT_5X7: dict[str, list[list[int]]] = {
    "0": [[1,1,1],[1,0,1],[1,0,1],[1,0,1],[1,1,1]],
    "1": [[0,1,0],[1,1,0],[0,1,0],[0,1,0],[1,1,1]],
    "2": [[1,1,1],[0,0,1],[1,1,1],[1,0,0],[1,1,1]],
    "3": [[1,1,1],[0,0,1],[0,1,1],[0,0,1],[1,1,1]],
    "4": [[1,0,1],[1,0,1],[1,1,1],[0,0,1],[0,0,1]],
    "5": [[1,1,1],[1,0,0],[1,1,1],[0,0,1],[1,1,1]],
    "6": [[1,1,1],[1,0,0],[1,1,1],[1,0,1],[1,1,1]],
    "7": [[1,1,1],[0,0,1],[0,0,1],[0,0,1],[0,0,1]],
    "8": [[1,1,1],[1,0,1],[1,1,1],[1,0,1],[1,1,1]],
    "9": [[1,1,1],[1,0,1],[1,1,1],[0,0,1],[1,1,1]],
    ".": [[0],[0],[0],[0],[1]],
    "°": [[1,1,1],[1,0,1],[1,1,1],[0,0,0],[0,0,0]],
    " ": [[0],[0],[0],[0],[0]],
    ":": [[0],[1],[0],[1],[0]],
    "/": [[0,0,1],[0,1,0],[0,1,0],[1,0,0],[0,0,0]],
    "-": [[0,0,0],[1,1,1],[0,0,0],[0,0,0],[0,0,0]],
    "E": [[1,1,1],[1,0,0],[1,1,1],[1,0,0],[1,1,1]],
    "K": [[1,0,1],[1,1,0],[1,0,0],[1,1,0],[1,0,1]],
    "T": [[1,1,1],[0,1,0],[0,1,0],[0,1,0],[0,1,0]],
    "F": [[1,1,1],[1,0,0],[1,1,1],[1,0,0],[1,0,0]],
    "r": [[0,0,0],[0,1,1],[1,0,0],[1,0,0],[1,0,0]],
    "a": [[0,0,0],[1,1,1],[0,0,1],[1,0,1],[1,1,1]],
    "m": [[0,0,0],[1,0,1],[1,1,1],[1,0,1],[1,0,1]],
    "e": [[0,0,0],[1,1,1],[1,0,1],[1,1,1],[1,0,0]],
    "i": [[0,1,0],[0,0,0],[0,1,0],[0,1,0],[0,1,0]],
    "n": [[0,0,0],[1,1,1],[1,0,1],[1,0,1],[1,0,1]],
    "t": [[0,1,0],[1,1,1],[0,1,0],[0,1,0],[0,1,0]],
    "c": [[0,0,0],[1,1,1],[1,0,0],[1,0,0],[1,1,1]],
    "o": [[0,0,0],[1,1,1],[1,0,1],[1,0,1],[1,1,1]],
    "l": [[0,1,0],[0,1,0],[0,1,0],[0,1,0],[0,1,0]],
    "p": [[0,0,0],[1,1,1],[1,0,1],[1,1,1],[1,0,0]],
    "s": [[0,0,0],[1,1,1],[1,0,0],[0,0,1],[1,1,1]],
    "u": [[0,0,0],[1,0,1],[1,0,1],[1,0,1],[1,1,1]],
}
_FONT_CHARS = "0123456789.° :/-EKTFramenitcolpsu"
_FONT_W, _FONT_H = 6, 8
_FONT_TEXTURE_ID: int | None = None


def _build_font_texture() -> int:
    """Build and upload font texture. Returns texture ID."""
    global _FONT_TEXTURE_ID
    if _FONT_TEXTURE_ID is not None:
        return _FONT_TEXTURE_ID
    import numpy as np
    n = len(_FONT_CHARS)
    tex_w, tex_h = n * _FONT_W, _FONT_H
    data = np.zeros((tex_h, tex_w, 4), dtype=np.uint8)
    for ci, ch in enumerate(_FONT_CHARS):
        glyph = _FONT_5X7.get(ch, _FONT_5X7["0"])
        gw = max(len(row) for row in glyph) if glyph else 3
        gh = len(glyph)
        ox = ci * _FONT_W + 1
        oy = 1
        for y, row in enumerate(glyph):
            for x, v in enumerate(row):
                if v and ox + x < tex_w and oy + y < tex_h:
                    data[oy + y, ox + x] = [255, 255, 255, 255]
    tex_id = glGenTextures(1)
    glBindTexture(GL_TEXTURE_2D, tex_id)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, tex_w, tex_h, 0, GL_RGBA, GL_UNSIGNED_BYTE, data)
    _FONT_TEXTURE_ID = tex_id
    return tex_id


def _draw_text_billboard(
    text: str,
    px: float, py: float, pz: float,
    cam_pos: tuple[float, float, float],
    scale: float = 0.05,
    color: tuple[float, float, float] = (1.0, 1.0, 1.0),
) -> None:
    """Draw text at 3D position, billboarded toward camera."""
    cx, cy, cz = cam_pos
    dx = px - cx
    dy = py - cy
    dz = pz - cz
    d = (dx * dx + dy * dy + dz * dz) ** 0.5
    if d < 1e-6:
        return
    dx, dy, dz = dx / d, dy / d, dz / d
    up = (0.0, 1.0, 0.0)
    rx = up[1] * dz - up[2] * dy
    ry = up[2] * dx - up[0] * dz
    rz = up[0] * dy - up[1] * dx
    rn = (rx * rx + ry * ry + rz * rz) ** 0.5
    if rn < 1e-6:
        up = (1.0, 0.0, 0.0)
        rx = up[1] * dz - up[2] * dy
        ry = up[2] * dx - up[0] * dz
        rz = up[0] * dy - up[1] * dx
        rn = (rx * rx + ry * ry + rz * rz) ** 0.5
    if rn >= 1e-6:
        rx, ry, rz = rx / rn, ry / rn, rz / rn
    ux = ry * dz - rz * dy
    uy = rz * dx - rx * dz
    uz = rx * dy - ry * dx
    tex_id = _build_font_texture()
    glDisable(GL_LIGHTING)
    glEnable(GL_TEXTURE_2D)
    glEnable(GL_BLEND)
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
    glBindTexture(GL_TEXTURE_2D, tex_id)
    glColor4f(color[0], color[1], color[2], 1.0)
    n = len(_FONT_CHARS)
    char_w = scale * 0.6
    char_h = scale
    total_w = len(text) * char_w
    start_x = -total_w / 2
    for i, ch in enumerate(text):
        if ch not in _FONT_CHARS:
            continue
        idx = _FONT_CHARS.index(ch)
        u0 = idx / n
        u1 = (idx + 1) / n
        x0 = start_x + i * char_w
        x1 = x0 + char_w
        glBegin(GL_QUADS)
        glTexCoord2f(u0, 0); glVertex3f(px + x0 * rx + 0 * ux, py + x0 * ry + 0 * uy, pz + x0 * rz + 0 * uz)
        glTexCoord2f(u1, 0); glVertex3f(px + x1 * rx + 0 * ux, py + x1 * ry + 0 * uy, pz + x1 * rz + 0 * uz)
        glTexCoord2f(u1, 1); glVertex3f(px + x1 * rx + char_h * ux, py + x1 * ry + char_h * uy, pz + x1 * rz + char_h * uz)
        glTexCoord2f(u0, 1); glVertex3f(px + x0 * rx + char_h * ux, py + x0 * ry + char_h * uy, pz + x0 * rz + char_h * uz)
        glEnd()
    glDisable(GL_BLEND)
    glDisable(GL_TEXTURE_2D)
    glEnable(GL_LIGHTING)


def draw_overlay(
    width: int,
    height: int,
    frame_idx: int,
    num_frames: int,
    metadata: dict | None = None,
    char_height: int | None = None,
    center: bool = True,
) -> None:
    """Draw 2D overlay text: frame, energy, kinetic_energy, total_energy (if available).
    Screen-space (angle-independent). If width or height is 0, uses current viewport (VR).
    char_height: None = auto from viewport (larger in VR). center: horizontal center."""
    meta = metadata or {}
    lines: list[str] = []
    lines.append(f"Frame {frame_idx + 1}/{num_frames}")
    if "energy" in meta:
        lines.append(f"E: {meta['energy']:.4f}")
    if "kinetic_energy" in meta:
        lines.append(f"K: {meta['kinetic_energy']:.4f}")
    if "total_energy" in meta:
        lines.append(f"T: {meta['total_energy']:.4f}")
    if len(lines) <= 1 and not meta and num_frames <= 1:
        return
    if width <= 0 or height <= 0:
        vp = (ctypes.c_int * 4)()
        glGetIntegerv(GL_VIEWPORT, vp)
        width, height = vp[2], vp[3]
    if char_height is None:
        char_height = max(24, height // 20)  # scale with viewport for VR
    glViewport(0, 0, width, height)
    glMatrixMode(GL_PROJECTION)
    glPushMatrix()
    glLoadIdentity()
    glOrtho(0, width, height, 0, -1, 1)
    glMatrixMode(GL_MODELVIEW)
    glPushMatrix()
    glLoadIdentity()
    glDisable(GL_LIGHTING)
    glDisable(GL_DEPTH_TEST)
    glEnable(GL_BLEND)
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
    char_w = char_height * 0.6
    line_h = char_height + 6
    block_h = len(lines) * line_h + 16
    block_w = max(len(t) * char_w for t in lines) + 24
    pad_top = int(height * 0.18)  # lower down, easier to see in VR
    # Semi-transparent background bar (upper-center)
    bar_left = (width - block_w) / 2 if center else 12
    bar_top = pad_top
    glColor4f(0.0, 0.0, 0.0, 0.6)
    glBegin(GL_QUADS)
    glVertex3f(bar_left, bar_top, 0)
    glVertex3f(bar_left + block_w, bar_top, 0)
    glVertex3f(bar_left + block_w, bar_top + block_h, 0)
    glVertex3f(bar_left, bar_top + block_h, 0)
    glEnd()
    glEnable(GL_TEXTURE_2D)
    tex_id = _build_font_texture()
    glBindTexture(GL_TEXTURE_2D, tex_id)
    n = len(_FONT_CHARS)
    for row, text in enumerate(lines):
        y0 = pad_top + 8 + row * line_h
        line_w = sum(1 for c in text if c in _FONT_CHARS) * char_w
        x_start = (width - line_w) / 2 if center else 12
        x_off = x_start
        for ch in text:
            if ch not in _FONT_CHARS:
                continue
            idx = _FONT_CHARS.index(ch)
            u0 = idx / n
            u1 = (idx + 1) / n
            x0, x1 = x_off, x_off + char_w
            x_off += char_w
            glColor4f(1.0, 1.0, 1.0, 0.95)
            glBegin(GL_QUADS)
            glTexCoord2f(u0, 0); glVertex3f(x0, y0, 0)
            glTexCoord2f(u1, 0); glVertex3f(x1, y0, 0)
            glTexCoord2f(u1, 1); glVertex3f(x1, y0 + char_height, 0)
            glTexCoord2f(u0, 1); glVertex3f(x0, y0 + char_height, 0)
            glEnd()
    glDisable(GL_TEXTURE_2D)
    glDisable(GL_BLEND)
    glEnable(GL_DEPTH_TEST)
    glEnable(GL_LIGHTING)
    glPopMatrix()
    glMatrixMode(GL_PROJECTION)
    glPopMatrix()
    glMatrixMode(GL_MODELVIEW)


def draw_annotations(
    atoms: Sequence[Atom],
    bonds: list[tuple[int, int]],
    cam_pos: tuple[float, float, float],
    scale: float = 0.05,
    show_bonds: bool = True,
    show_angles: bool = True,
    show_dihedrals: bool = True,
    molecule_world_offset: tuple[float, float, float] | None = None,
) -> None:
    """Draw bond lengths, angles, and dihedrals as 3D billboard text.
    molecule_world_offset: if set (VR), annotation positions are offset for correct billboard orientation.
    """
    if not atoms:
        return
    off = molecule_world_offset or (0.0, 0.0, 0.0)
    angles = compute_angles(atoms, bonds) if show_angles else []
    dihedrals = compute_dihedrals(atoms, bonds) if show_dihedrals else []
    for i, j in bonds:
        if i >= len(atoms) or j >= len(atoms):
            continue
        a, b = atoms[i], atoms[j]
        mx = (a.x + b.x) / 2
        my = (a.y + b.y) / 2
        mz = (a.z + b.z) / 2
        d = ((a.x - b.x) ** 2 + (a.y - b.y) ** 2 + (a.z - b.z) ** 2) ** 0.5
        if show_bonds:
            _draw_text_billboard(f"{d:.2f}", mx, my, mz, (cam_pos[0] - off[0], cam_pos[1] - off[1], cam_pos[2] - off[2]), scale=scale)
    for i, j, k, ang in angles:
        if j >= len(atoms):
            continue
        b = atoms[j]
        _draw_text_billboard(f"{ang:.1f}°", b.x, b.y, b.z, (cam_pos[0] - off[0], cam_pos[1] - off[1], cam_pos[2] - off[2]), scale=scale, color=(0.8, 0.9, 1.0))
    for i, j, k, l, dih in dihedrals:
        if j >= len(atoms) or k >= len(atoms):
            continue
        b, c = atoms[j], atoms[k]
        mx = (b.x + c.x) / 2
        my = (b.y + c.y) / 2
        mz = (b.z + c.z) / 2
        if show_dihedrals:
            _draw_text_billboard(f"{dih:.1f}°", mx, my, mz, (cam_pos[0] - off[0], cam_pos[1] - off[1], cam_pos[2] - off[2]), scale=scale, color=(1.0, 0.9, 0.8))


def setup_projection(width: int, height: int, fov: float = 50.0, near: float = 0.1, far: float = 1000.0, zoom: float = 1.0, orthographic: bool = True) -> None:
    """Set up projection. orthographic=True uses ortho (no perspective distortion)."""
    glViewport(0, 0, width, height)
    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()
    aspect = width / height if height else 1.0
    if orthographic:
        half = 50.0 / zoom
        glOrtho(-half * aspect, half * aspect, -half, half, -far, far)
    else:
        gluPerspective(fov, aspect, near, far)
    glMatrixMode(GL_MODELVIEW)


def setup_view(eye: tuple[float, float, float], center: tuple[float, float, float], up: tuple[float, float, float] = (0, 1, 0)) -> None:
    """Set up modelview (camera)."""
    glLoadIdentity()
    gluLookAt(eye[0], eye[1], eye[2], center[0], center[1], center[2], up[0], up[1], up[2])


def setup_projection_vr(fov, near: float = 0.1, far: float = 500.0, orthographic: bool = True, ortho_scale: float = 4.0, zoom: float = 1.0) -> None:
    """
    Set up projection from OpenXR Fovf.
    fov must have: angle_left, angle_right, angle_up, angle_down (radians).
    orthographic: use ortho projection (no perspective).
    zoom: scale ortho bounds (larger = zoomed out, smaller = zoomed in).
    """
    import math
    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()
    if orthographic:
        scale = ortho_scale * zoom
        left = math.tan(fov.angle_left) * scale
        right = math.tan(fov.angle_right) * scale
        bottom = math.tan(fov.angle_down) * scale
        top = math.tan(fov.angle_up) * scale
        glOrtho(left, right, bottom, top, -far, far)
    else:
        left = math.tan(fov.angle_left) * near
        right = math.tan(fov.angle_right) * near
        bottom = math.tan(fov.angle_down) * near
        top = math.tan(fov.angle_up) * near
        from OpenGL.GL import glFrustum
        glFrustum(left, right, bottom, top, near, far)
    glMatrixMode(GL_MODELVIEW)


def setup_view_vr(view_matrix_4x4, flip_x: bool = False, flip_y: bool = False) -> None:
    """
    Set up modelview from 4x4 view matrix (column-major, e.g. from xr.utils.view_matrix_from_posef).
    flip_x: fix inverted yaw (head left/right) common with some OpenXR/OpenGL combinations.
    flip_y: fix inverted pitch (head up/down) common with some OpenXR/OpenGL combinations.
    """
    from OpenGL.GL import glLoadMatrixf
    m = view_matrix_4x4.astype("float32").copy()
    if flip_x:
        m[0, :] = -m[0, :]  # negate X row to fix inverted head yaw
    if flip_y:
        m[1, :] = -m[1, :]  # negate Y row to fix inverted head pitch
    glLoadMatrixf(m.flatten(order="F"))


def clear_screen(bg: tuple[float, float, float] = (0.05, 0.05, 0.08)) -> None:
    """Clear color and depth buffers."""
    glClearColor(*bg, 1.0)
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)


def init_gl() -> None:
    """Initialize OpenGL state for rendering."""
    glEnable(GL_DEPTH_TEST)
    glShadeModel(GL_SMOOTH)
    glEnable(GL_LIGHTING)
    glEnable(GL_LIGHT0)
    glLightfv(GL_LIGHT0, GL_POSITION, (2.0, 4.0, 4.0, 0.0))  # directional (w=0)
    glLightfv(GL_LIGHT0, GL_AMBIENT, (0.25, 0.25, 0.25, 1.0))
    glLightfv(GL_LIGHT0, GL_DIFFUSE, (0.7, 0.7, 0.7, 1.0))
    glLightfv(GL_LIGHT0, GL_SPECULAR, (0.5, 0.5, 0.5, 1.0))
    glLightModelfv(GL_LIGHT_MODEL_AMBIENT, (0.35, 0.35, 0.35, 1.0))
