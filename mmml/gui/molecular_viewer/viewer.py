"""
OpenXR molecular viewer with desktop fallback.
Runs in VR when pyopenxr and a headset are available, otherwise uses a desktop window.
"""

from __future__ import annotations

# Python 3.14 removed ctypes.pointer; pyopenxr still needs it
import ctypes

def _ctypes_pointer(obj):
    return ctypes.POINTER(type(obj))(obj)

if not hasattr(ctypes, "pointer"):
    ctypes.pointer = _ctypes_pointer
try:
    from _ctypes import pointer  # noqa: F401
except ImportError:
    import _ctypes
    _ctypes.pointer = _ctypes_pointer

import argparse
import math
import sys
import time
from pathlib import Path
from typing import Sequence

import glfw

from .debug import debug, log, set_debug
from .molecule import Atom, center_and_scale, compute_bonds, load_structure
from .renderer import (
    clear_screen,
    draw_annotations,
    draw_molecule,
    draw_overlay,
    draw_pbc_cell,
    init_gl,
    setup_projection,
    setup_view,
)

# Optional OpenXR for VR mode
try:
    import xr  # noqa: F401
    from .vr_session import run_vr_session
    XR_AVAILABLE = True
except ImportError:
    XR_AVAILABLE = False
    run_vr_session = None


class MolecularViewer:
    """Desktop molecular viewer with mouse orbit and trajectory playback."""

    def __init__(
        self,
        atoms: list[Atom] | list[list[Atom]],
        window_width: int = 1280,
        window_height: int = 720,
        cells: list | None = None,
        metadata: list | None = None,
    ):
        self._frames: list[list[Atom]] = []
        if atoms and isinstance(atoms[0], list):
            self._frames = atoms
        else:
            self._frames = [list(atoms)] if atoms else [[]]
        self._cells: list = cells if cells is not None else [None] * len(self._frames)
        if len(self._cells) < len(self._frames):
            self._cells.extend([None] * (len(self._frames) - len(self._cells)))
        self._metadata: list = metadata if metadata is not None else [{}] * len(self._frames)
        if len(self._metadata) < len(self._frames):
            self._metadata.extend([{}] * (len(self._frames) - len(self._metadata)))
        self._frame_idx = 0
        self._playing = False
        self._last_time = time.monotonic()
        self._rot_x = 0.0
        self._rot_y = 0.0
        self._zoom = 1.0
        self._pan_x = 0.0
        self._pan_y = 0.0
        self._mouse_down = False
        self._pan_down = False
        self._last_mouse = (0.0, 0.0)
        self._window_w = window_width
        self._window_h = window_height
        self._atom_scale = 0.25
        self._scale = 1.0
        self._show_overlay = True
        self._show_annotations = True

    def _current_atoms(self) -> list[Atom]:
        return self._frames[self._frame_idx] if self._frames else []

    def _center_and_scale_atoms(self, atoms: list[Atom]) -> list[Atom]:
        if not atoms:
            return []
        centered = center_and_scale(atoms, scale=self._scale)
        return centered

    def _render_frame(self, width: int, height: int) -> None:
        """Render current frame to current OpenGL context."""
        clear_screen()
        setup_projection(width, height, fov=50.0, zoom=self._zoom)
        # Camera: orbit around origin, looking at (0,0,0)
        dist = 50.0 / self._zoom
        cx = math.cos(self._rot_y) * math.cos(self._rot_x)
        cy = math.sin(self._rot_x)
        cz = math.sin(self._rot_y) * math.cos(self._rot_x)
        eye = (dist * cx + self._pan_x, dist * cy + self._pan_y, dist * cz)
        center = (self._pan_x, self._pan_y, 0.0)
        setup_view(eye, center)
        raw_atoms = self._current_atoms()
        atoms = self._center_and_scale_atoms(raw_atoms)
        if atoms:
            n = len(raw_atoms)
            center = (
                sum(a.x for a in raw_atoms) / n if n else 0,
                sum(a.y for a in raw_atoms) / n if n else 0,
                sum(a.z for a in raw_atoms) / n if n else 0,
            )
            cell = self._cells[self._frame_idx] if self._frame_idx < len(self._cells) else None
            if cell is not None:
                draw_pbc_cell(cell, center, self._scale)
            bonds = compute_bonds(atoms)
            draw_molecule(atoms, bonds=bonds, atom_scale=self._atom_scale)
            if self._show_annotations:
                draw_annotations(atoms, bonds, eye, scale=5.0)
        if self._show_overlay:
            meta = self._metadata[self._frame_idx] if self._frame_idx < len(self._metadata) else {}
            draw_overlay(width, height, self._frame_idx, len(self._frames), meta)

    def _run_desktop(self, switch_to_vr: list | None = None) -> bool:
        """Run desktop GLFW window loop. If switch_to_vr provided, V key sets it and closes. Returns True if switched."""
        if not glfw.init():
            raise RuntimeError("glfw.init failed")
        glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 2)
        glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 1)
        win = glfw.create_window(self._window_w, self._window_h, "Molecular Viewer (Desktop)", None, None)
        if not win:
            glfw.terminate()
            raise RuntimeError("glfw.create_window failed")
        glfw.make_context_current(win)
        init_gl()

        def on_mouse_button(w, button, action, mods):
            if button == glfw.MOUSE_BUTTON_LEFT:
                self._mouse_down = action == glfw.PRESS
            elif button == glfw.MOUSE_BUTTON_RIGHT:
                self._pan_down = action == glfw.PRESS

        def on_cursor_pos(w, x, y):
            dx = x - self._last_mouse[0]
            dy = y - self._last_mouse[1]
            if self._mouse_down:
                self._rot_y += dx * 0.01
                self._rot_x += dy * 0.01
                self._rot_x = max(-1.5, min(1.5, self._rot_x))
            elif self._pan_down:
                self._pan_x -= dx * 0.5
                self._pan_y += dy * 0.5
            self._last_mouse = (x, y)

        def on_scroll(w, xoff, yoff):
            self._zoom *= 1.0 + yoff * 0.1
            self._zoom = max(0.01, min(100.0, self._zoom))

        # Orbit rotation deltas (keys: add rotation around axes)
        ROTATE_DELTA = math.radians(15)
        ROTATE_KEYS = {
            glfw.KEY_1: (0, ROTATE_DELTA),    # rotate around X
            glfw.KEY_2: (ROTATE_DELTA, 0),   # rotate around Y
            glfw.KEY_3: (0, -ROTATE_DELTA),
            glfw.KEY_4: (0, -ROTATE_DELTA),
            glfw.KEY_5: (-ROTATE_DELTA, 0),
            glfw.KEY_6: (0, ROTATE_DELTA),
            glfw.KEY_X: (0, ROTATE_DELTA),
            glfw.KEY_Y: (ROTATE_DELTA, 0),
            glfw.KEY_Z: (0, -ROTATE_DELTA),
        }

        def on_key(w, key, scancode, action, mods):
            if action != glfw.PRESS:
                return
            if key == glfw.KEY_ESCAPE:
                glfw.set_window_should_close(win, True)
            elif key == glfw.KEY_O:
                self._show_overlay = not self._show_overlay
            elif key == glfw.KEY_A:
                self._show_annotations = not self._show_annotations
            elif key == glfw.KEY_V and XR_AVAILABLE and run_vr_session is not None:
                if switch_to_vr is not None:
                    switch_to_vr[0] = True
                glfw.set_window_should_close(win, True)
            elif key == glfw.KEY_SPACE:
                self._playing = not self._playing
            elif key == glfw.KEY_R:
                # Reset view: center, default zoom
                self._rot_x = self._rot_y = 0.0
                self._zoom = 1.0
                self._pan_x = self._pan_y = 0.0
            elif key in ROTATE_KEYS:
                dx, dy = ROTATE_KEYS[key]
                self._rot_x += dx
                self._rot_y += dy
            elif key == glfw.KEY_LEFT and len(self._frames) > 1:
                self._frame_idx = (self._frame_idx - 1) % len(self._frames)
            elif key == glfw.KEY_RIGHT and len(self._frames) > 1:
                self._frame_idx = (self._frame_idx + 1) % len(self._frames)

        glfw.set_mouse_button_callback(win, on_mouse_button)
        glfw.set_cursor_pos_callback(win, on_cursor_pos)
        glfw.set_scroll_callback(win, on_scroll)
        glfw.set_key_callback(win, on_key)

        while not glfw.window_should_close(win):
            now = time.monotonic()
            if self._playing and len(self._frames) > 1:
                self._frame_idx = (self._frame_idx + 1) % len(self._frames)
                self._last_time = now
            w, h = glfw.get_framebuffer_size(win)
            self._render_frame(w, h)
            glfw.swap_buffers(win)
            glfw.poll_events()
        glfw.terminate()
        return bool(switch_to_vr and switch_to_vr[0])


def run_viewer(
    path: str | Path,
    vr: bool = True,
    window_width: int = 1280,
    window_height: int = 720,
    pointer: bool = True,
) -> None:
    """
    Load a molecular file and run the viewer.
    - path: PDB or XYZ file (single or multi-frame)
    - vr: if True and pyopenxr + headset available, run in VR; else desktop
    - window_width/height: desktop window size
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    debug("Loading %s", path)
    data, cell_or_cells, metadata = load_structure(path)
    if isinstance(data, list) and data and isinstance(data[0], Atom):
        frames = [data]
        cells = [cell_or_cells[0]] if cell_or_cells else [None]
        metadata = metadata if metadata else [{}]
    else:
        frames = data
        cells = cell_or_cells if cell_or_cells else [None] * len(frames)
        metadata = metadata if metadata else [{}] * len(frames)
    if not frames:
        raise ValueError(f"No structure loaded from {path}")
    debug("Loaded %d frame(s)", len(frames))
    mode = "vr" if vr else "desktop"
    while True:
        if mode == "vr" and XR_AVAILABLE and run_vr_session is not None:
            debug("VR mode: attempting OpenXR session")
            switch_to_desktop = [False]
            try:
                run_vr_session(
                    frames=frames,
                    cells=cells,
                    metadata=metadata,
                    atom_scale=0.25,
                    scale=1.0,
                    molecule_offset_z=-2.0,
                    playback_fps=10.0,
                    pointer=pointer,
                    switch_to_desktop=switch_to_desktop,
                )
                if switch_to_desktop[0]:
                    mode = "desktop"
                    continue
            except Exception as e:
                log("VR mode failed (%s), falling back to desktop.", e)
                mode = "desktop"
                continue
            break
        else:
            viewer = MolecularViewer(frames, window_width, window_height, cells=cells, metadata=metadata)
            switch_to_vr = [False] if (XR_AVAILABLE and run_vr_session is not None) else None
            switched = viewer._run_desktop(switch_to_vr=switch_to_vr)
            if switched and switch_to_vr and switch_to_vr[0]:
                mode = "vr"
                continue
            break


def main() -> None:
    parser = argparse.ArgumentParser(
        description="OpenXR Molecular Trajectory Viewer",
        epilog="Desktop keys: O=overlay, A=annotations, V=switch to VR. VR: B=toggle text, Menu=switch to desktop.",
    )
    parser.add_argument("file", type=Path, help="PDB or XYZ file (single or trajectory)")
    parser.add_argument("--no-vr", action="store_true", help="Force desktop mode")
    parser.add_argument("--no-pointer", action="store_true", help="Disable VR controller laser pointer")
    parser.add_argument("--debug", action="store_true", help="Enable debug logs (to vconsole/stderr)")
    parser.add_argument("--width", type=int, default=1280, help="Window width")
    parser.add_argument("--height", type=int, default=720, help="Window height")
    args = parser.parse_args()
    set_debug(args.debug)
    run_viewer(args.file, vr=not args.no_vr, window_width=args.width, window_height=args.height, pointer=not args.no_pointer)


if __name__ == "__main__":
    main()
