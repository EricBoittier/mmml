#!/usr/bin/env python3
"""POV-Ray stills for DCM–DCM hybrid dimer scan geometries.

House glossy style (``docs/plotting-style-guide.md``):
- shared world bounding box → consistent framing, no cropped arrows/atoms
- red force arrows from the hybrid model on the exact frame
- gold per-monomer dipoles from PhysNet ``q_ML``
- charge stills colored with ``crameri:vik`` (actual q values + colorbar)

Example::

    uv run python scripts/slurm/dense_dt_campaign/render_dimer_scan_povray.py
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

# Repo root on sys.path for ``scripts.render_povray_*`` imports.
_REPO = Path(__file__).resolve().parents[3]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from mmml.analysis.dimer_scans import (  # noqa: E402
    DEFAULT_ORIENT_MIN_CONTACT_A,
    intermolecular_min_distance,
)
from scripts.render_povray_multipoles import _arrow  # noqa: E402
from scripts.render_povray_style_catalog import STYLES, bonds, vec  # noqa: E402

Z_TO_SYM = {1: "H", 6: "C", 17: "Cl"}
ELEMENT_COLORS = {
    "H": (0.86, 0.87, 0.90),
    "C": (0.22, 0.24, 0.28),
    "Cl": (0.20, 0.72, 0.28),
}
FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
FORCE_COLOR = (0.78, 0.04, 0.12)
DIPOLE_COLOR = (0.95, 0.62, 0.06)
BOX_EDGE = (0.52, 0.55, 0.60)
CHARGE_CMAP_NAME = "crameri:vik"  # house red/blue for signed charge (style guide)
DIPOLE_ARROW_LEN = 2.0
BOX_PAD_A = 0.55  # padding beyond atoms/arrow tips
CAMERA_MARGIN = 1.08  # orthographic margin around the cube


def fibonacci_sphere(n: int) -> np.ndarray:
    i = np.arange(n) + 0.5
    phi = np.arccos(1.0 - 2.0 * i / n)
    theta = np.pi * (1.0 + 5.0**0.5) * i
    return np.stack(
        [np.cos(theta) * np.sin(phi), np.sin(theta) * np.sin(phi), np.cos(phi)],
        axis=1,
    )


def super_fibonacci(n: int) -> np.ndarray:
    phi = np.sqrt(2.0)
    psi = 1.533751168755204288118041
    i = np.arange(n) + 0.5
    s = i / n
    t = s * n / phi
    d = 2.0 * np.pi * (t - np.floor(t))
    r = np.sqrt(s)
    R = np.sqrt(1.0 - s)
    t2 = i / psi
    a = 2.0 * np.pi * (t2 - np.floor(t2))
    return np.stack(
        [r * np.sin(d), r * np.cos(d), R * np.sin(a), R * np.cos(a)], axis=1
    )


def quat_to_matrix(q: np.ndarray) -> np.ndarray:
    x, y, z, w = q
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )


def build_dimer(R1: np.ndarray, Z1: np.ndarray, dvec: np.ndarray, quat: np.ndarray, r: float):
    from ase import Atoms

    Rb0 = R1 @ quat_to_matrix(quat).T
    Ra = R1 - 0.5 * r * dvec
    Rb = Rb0 + 0.5 * r * dvec
    pos = np.vstack([Ra, Rb])
    syms = [Z_TO_SYM[int(z)] for z in Z1] * 2
    return Atoms(symbols=syms, positions=pos)


def _r_tag(r: float) -> str:
    """Filename-safe r label (avoid Path.with_suffix eating '.5')."""
    return f"{r:.2f}".replace(".", "p")


def _pov_include_dirs(povray: str) -> list[Path]:
    exe = Path(povray).resolve()
    candidates = [
        exe.parent.parent / "share" / "povray-3.7" / "include",
        exe.parent.parent / "share" / "povray" / "include",
        Path("/usr/share/povray/include"),
        Path("/usr/share/povray-3.7/include"),
    ]
    return [p for p in candidates if (p / "colors.inc").is_file()]


def _molecular_dipole(pos: np.ndarray, q: np.ndarray) -> np.ndarray:
    """Dipole in e·Å relative to the fragment COM."""
    com = pos.mean(axis=0)
    return np.sum(q[:, None] * (pos - com), axis=0)


def _centered_xyz(atoms) -> np.ndarray:
    return atoms.positions - atoms.positions.mean(0)


def _force_tip_points(
    xyz: np.ndarray, forces: np.ndarray, scale: float, max_len: float
) -> np.ndarray:
    tips = []
    for p, force in zip(xyz, forces):
        d = force * scale
        n = float(np.linalg.norm(d))
        if n < 0.04:
            continue
        if n > max_len:
            d = d * (max_len / n)
            n = max_len
        u = d / n
        tips.append(p + 0.42 * u + d)
    return np.asarray(tips).reshape(-1, 3)


def _dipole_tip_points(
    xyz: np.ndarray, charges: np.ndarray, n_mono: int, arrow_scale: float
) -> np.ndarray:
    tips = []
    for mol in (0, 1):
        sl = slice(mol * n_mono, (mol + 1) * n_mono)
        pos = xyz[sl]
        mu = _molecular_dipole(pos, charges[sl])
        n = float(np.linalg.norm(mu))
        if n < 1e-8:
            continue
        com = pos.mean(axis=0)
        tips.append(com + (mu / n) * arrow_scale)
    return np.asarray(tips).reshape(-1, 3)


def _frame_extent_points(
    xyz: np.ndarray,
    forces: np.ndarray,
    charges: np.ndarray,
    n_mono: int,
    *,
    force_scale: float,
    force_cap: float,
    dipole_len: float,
) -> np.ndarray:
    chunks = [xyz]
    ft = _force_tip_points(xyz, forces, force_scale, force_cap)
    if len(ft):
        chunks.append(ft)
    dt = _dipole_tip_points(xyz, charges, n_mono, dipole_len)
    if len(dt):
        chunks.append(dt)
    return np.vstack(chunks)


def compute_shared_box_half(
    frames: list[dict],
    pred: dict[str, np.ndarray],
    n_mono: int,
    *,
    force_scale: float,
    force_cap: float,
    dipole_len: float = DIPOLE_ARROW_LEN,
    pad: float = BOX_PAD_A,
) -> float:
    """Half-edge of the axis-aligned cube that contains every frame's glyphs."""
    lo = np.full(3, np.inf)
    hi = np.full(3, -np.inf)
    for i, fr in enumerate(frames):
        xyz = _centered_xyz(fr["atoms"])
        pts = _frame_extent_points(
            xyz,
            pred["forces"][i],
            pred["charges"][i],
            n_mono,
            force_scale=force_scale,
            force_cap=force_cap,
            dipole_len=dipole_len,
        )
        lo = np.minimum(lo, pts.min(axis=0))
        hi = np.maximum(hi, pts.max(axis=0))
    half = float(np.max(np.abs(np.vstack([lo, hi])))) + pad
    # Keep a minimum so tiny monomers don't fill the frame.
    return max(half, 4.0)


def _wire_box(half: float, *, radius: float = 0.028) -> list[str]:
    """Thin cube edges at ±half — shared crop/framing guide."""
    h = float(half)
    # 12 edges of the cube [-h,h]^3.
    edge_defs = [
        # bottom face (y=-h)
        ((-1, -1, -1), (1, -1, -1)),
        ((1, -1, -1), (1, -1, 1)),
        ((1, -1, 1), (-1, -1, 1)),
        ((-1, -1, 1), (-1, -1, -1)),
        # top face (y=+h)
        ((-1, 1, -1), (1, 1, -1)),
        ((1, 1, -1), (1, 1, 1)),
        ((1, 1, 1), (-1, 1, 1)),
        ((-1, 1, 1), (-1, 1, -1)),
        # verticals
        ((-1, -1, -1), (-1, 1, -1)),
        ((1, -1, -1), (1, 1, -1)),
        ((1, -1, 1), (1, 1, 1)),
        ((-1, -1, 1), (-1, 1, 1)),
    ]
    pigment = (
        f"pigment {{ color rgbt <{BOX_EDGE[0]},{BOX_EDGE[1]},{BOX_EDGE[2]},0.35> }} "
        f"finish {{ emission 0.08 }}"
    )
    lines = []
    for a_s, b_s in edge_defs:
        a = np.array(a_s, dtype=float) * h
        b = np.array(b_s, dtype=float) * h
        lines.append(
            f"cylinder {{ {vec(a)}, {vec(b)}, {radius} {pigment} no_shadow }}"
        )
    return lines


def _camera_lights(half: float, width: int, height: int) -> list[str]:
    span = 2.0 * half * CAMERA_MARGIN
    camera = np.array([half * 1.35, half * 0.95, -half * 3.1])
    shadow_y = -half - 0.05
    return [
        "#version 3.7;",
        "global_settings { assumed_gamma 1.0 }",
        "background { color rgbt <1,1,1,1> }",
        (
            f"camera {{ orthographic location {vec(camera)} look_at <0,0,0> "
            f"right x*{span:.6f} up y*{span * height / width:.6f} }}"
        ),
        (
            f"light_source {{ {vec((-half, half * 2.2, -half * 2))} color rgb <1,1,1> "
            f"area_light <2,0,0>,<0,2,0>,5,5 adaptive 1 jitter }}"
        ),
        f"light_source {{ {vec((half * 2, -half, -half))} color rgb <0.35,0.40,0.52> }}",
        (
            f"plane {{ y, {shadow_y:.4f} pigment {{ color rgbt <0.91,0.92,0.94,0.82> }} "
            f"finish {{ diffuse 0.75 }} }}"
        ),
    ]


def _atom_bond_lines(atoms, xyz: np.ndarray, colors: list[tuple[float, float, float]]) -> list[str]:
    from ase.data import covalent_radii

    finish = STYLES["glossy"]
    lines: list[str] = []
    for i, j in bonds(atoms):
        a, b = xyz[i], xyz[j]
        lines.append(
            f"cylinder {{ {vec(a)}, {vec(b)}, 0.105 "
            f"pigment {{ color rgb <0.55,0.57,0.61> }} "
            f"finish {{ diffuse 0.7 phong 0.25 }} }}"
        )
    for p, z, color in zip(xyz, atoms.numbers, colors):
        radius = max(0.24, covalent_radii[z] * 0.52)
        lines.append(
            f"sphere {{ {vec(p)}, {radius:.4f} texture {{ "
            f"pigment {{ color rgb {vec(color)} }} {finish} }} }}"
        )
    return lines


def glossy_scene(
    atoms,
    *,
    half: float,
    width: int,
    height: int,
    atom_colors: list[tuple[float, float, float]] | None = None,
    draw_box: bool = True,
) -> str:
    """Glossy molecule in a fixed world box (same camera for every frame)."""
    xyz = _centered_xyz(atoms)
    if atom_colors is None:
        atom_colors = [
            ELEMENT_COLORS.get(s, (0.55, 0.55, 0.58)) for s in atoms.get_chemical_symbols()
        ]
    lines = _camera_lights(half, width, height)
    if draw_box:
        lines += _wire_box(half)
    lines += _atom_bond_lines(atoms, xyz, atom_colors)
    return "\n".join(lines) + "\n"


def _force_overlay(
    xyz: np.ndarray,
    forces: np.ndarray,
    scale: float,
    *,
    max_len: float = 2.5,
) -> list[str]:
    lines: list[str] = []
    for p, force in zip(xyz, forces):
        d = force * scale
        n = float(np.linalg.norm(d))
        if n < 0.04:
            continue
        if n > max_len:
            d = d * (max_len / n)
            n = max_len
        u = d / n
        start = p + 0.42 * u
        tip = start + d
        neck = tip - min(0.22, 0.28 * n) * u
        pigment = (
            f"pigment {{ color rgb {vec(FORCE_COLOR)} }} finish {{ emission 0.18 }}"
        )
        lines += [
            f"cylinder {{ {vec(start)}, {vec(neck)}, 0.045 {pigment} }}",
            f"cone {{ {vec(neck)}, 0.12, {vec(tip)}, 0 {pigment} }}",
        ]
    return lines


def _dipole_overlay(
    xyz: np.ndarray,
    charges: np.ndarray,
    n_mono: int,
    *,
    arrow_scale: float = DIPOLE_ARROW_LEN,
) -> tuple[list[str], list[float]]:
    lines: list[str] = []
    norms: list[float] = []
    for mol in (0, 1):
        sl = slice(mol * n_mono, (mol + 1) * n_mono)
        pos = xyz[sl]
        q = charges[sl]
        mu = _molecular_dipole(pos, q)
        norms.append(float(np.linalg.norm(mu)))
        if norms[-1] < 1e-8:
            continue
        com = pos.mean(axis=0)
        u = mu / norms[-1]
        lines += _arrow(com, com + u * arrow_scale, DIPOLE_COLOR, radius=0.065)
    return lines, norms


def _load_charge_cmap():
    from cmap import Colormap

    return Colormap(CHARGE_CMAP_NAME)


def _charge_rgb(q: float, q_lim: float, cmap) -> tuple[float, float, float]:
    """Map signed charge onto house diverging scale (vik: blue− / red+)."""
    t = 0.5 * (float(np.clip(q / max(q_lim, 1e-12), -1.0, 1.0)) + 1.0)
    rgba = cmap(t)
    return float(rgba[0]), float(rgba[1]), float(rgba[2])


def _nice_q_lim(q_abs_max: float) -> float:
    """Round charge limit up to a readable tick (e)."""
    if q_abs_max <= 0:
        return 0.1
    exp = np.floor(np.log10(q_abs_max))
    base = 10.0**exp
    for m in (1.0, 1.5, 2.0, 2.5, 5.0, 10.0):
        if m * base >= q_abs_max:
            return float(m * base)
    return float(10.0 * base)


def _normalize_alpha(image):
    alpha = image.getchannel("A")
    baseline = alpha.getextrema()[0]
    if 0 < baseline < 255:
        image.putalpha(
            alpha.point(
                lambda value: max(0, round((value - baseline) * 255 / (255 - baseline)))
            )
        )
    return image


def _draw_charge_colorbar(
    draw,
    *,
    image_size: tuple[int, int],
    q_lim: float,
    cmap,
    font,
) -> None:
    """Vertical colorbar with actual charge ticks (e)."""
    w, h = image_size
    bar_w = max(14, w // 48)
    bar_h = int(h * 0.42)
    x0 = w - bar_w - 56
    y0 = (h - bar_h) // 2
    n = 128
    for i in range(n):
        # top = +q_lim (red in vik), bottom = −q_lim (blue)
        t = 1.0 - i / (n - 1)
        rgba = cmap(t)
        color = tuple(int(255 * c) for c in rgba[:3]) + (235,)
        y = y0 + int(i * bar_h / n)
        y2 = y0 + int((i + 1) * bar_h / n)
        draw.rectangle((x0, y, x0 + bar_w, y2), fill=color)
    draw.rectangle((x0 - 1, y0 - 1, x0 + bar_w + 1, y0 + bar_h + 1), outline=(40, 44, 52, 220))
    ticks = [q_lim, 0.0, -q_lim]
    for q in ticks:
        t = 0.5 * (1.0 - q / q_lim)  # 0 at top (+), 1 at bottom (−)
        y = y0 + int(t * bar_h)
        label = f"{q:+.3f}" if q != 0 else "0"
        draw.line((x0 + bar_w + 2, y, x0 + bar_w + 8, y), fill=(30, 34, 42, 255), width=2)
        draw.text((x0 + bar_w + 10, y - 8), f"{label} e", font=font, fill=(18, 22, 30, 255))
    draw.text((x0 - 4, y0 - 22), "q_ML", font=font, fill=(18, 22, 30, 255))


def _annotate_image(
    png: Path,
    *,
    title: str,
    lines: list[str],
    width: int,
    colorbar: dict | None = None,
) -> None:
    from PIL import Image, ImageDraw, ImageFont

    image = Image.open(png).convert("RGBA")
    image = _normalize_alpha(image)
    draw = ImageDraw.Draw(image, "RGBA")
    try:
        title_font = ImageFont.truetype(FONT, max(18, width // 28))
        body_font = ImageFont.truetype(FONT, max(14, width // 40))
        tick_font = ImageFont.truetype(FONT, max(12, width // 48))
    except OSError:
        title_font = ImageFont.load_default()
        body_font = title_font
        tick_font = title_font
    pad = 14
    text_block = [title, *lines]
    boxes = [
        draw.textbbox((0, 0), t, font=(title_font if i == 0 else body_font))
        for i, t in enumerate(text_block)
    ]
    tw = max(b[2] - b[0] for b in boxes) + 2 * pad + 16
    th = sum(b[3] - b[1] + 6 for b in boxes) + 2 * pad
    draw.rounded_rectangle((16, 14, 16 + tw, 14 + th), radius=12, fill=(250, 250, 252, 225))
    y = 14 + pad
    for i, t in enumerate(text_block):
        font = title_font if i == 0 else body_font
        draw.text((16 + pad, y), t, font=font, fill=(18, 22, 30, 255))
        y += (boxes[i][3] - boxes[i][1]) + 6
    if colorbar is not None:
        _draw_charge_colorbar(
            draw,
            image_size=image.size,
            q_lim=float(colorbar["q_lim"]),
            cmap=colorbar["cmap"],
            font=tick_font,
        )
    image.save(png)


def render_pov(
    pov_text: str,
    out_png: Path,
    *,
    povray: str,
    width: int,
    height: int,
    include_dirs: list[Path],
) -> bool:
    out_png.parent.mkdir(parents=True, exist_ok=True)
    stem = out_png.parent / out_png.name[: -len(out_png.suffix)]
    pov_path = Path(str(stem) + ".pov")
    ini = Path(str(stem) + ".ini")
    pov_path.write_text(pov_text)
    lib_lines = "".join(f'Library_Path="{lib}"\n' for lib in include_dirs)
    ini.write_text(
        lib_lines
        + f'Input_File_Name="{pov_path.name}"\n'
        + f'Output_File_Name="{out_png.name}"\n'
        + f"Width={width}\nHeight={height}\n"
        + "Antialias=On\nAntialias_Threshold=0.15\n"
        + "Output_Alpha=On\nDisplay=Off\n"
    )
    env = dict(os.environ)
    env.pop("POVINI", None)
    proc = subprocess.run(
        [povray, ini.name],
        capture_output=True,
        text=True,
        cwd=str(stem.parent),
        env=env,
    )
    ok = proc.returncode == 0 and out_png.is_file()
    if not ok:
        tail = (proc.stderr or proc.stdout or "").strip().splitlines()[-8:]
        print(f"  POV fail {out_png.name}: rc={proc.returncode}")
        for line in tail:
            print(f"    {line}")
    else:
        ini.unlink(missing_ok=True)
        pov_path.unlink(missing_ok=True)
    return ok


class HybridFrameEval:
    """Evaluate hybrid forces + ML charges for a small set of dimer frames."""

    def __init__(
        self,
        *,
        checkpoint: Path,
        sidecar: Path,
        data: Path,
        mm_switch_on: float,
        ml_switch_width: float,
        mm_switch_width: float,
        n_mono: int = 5,
    ):
        import jax
        import jax.numpy as jnp
        from mmml.cli.misc.physnet_evaluate import _load_physnet_checkpoint
        from mmml.models.hybrid_energy import HYBRID_MM_BATCH_KEYS, hybrid_forward
        from mmml.models.physnetjax.physnetjax.data.batches import prepare_batches_jit

        self.jax = jax
        self.jnp = jnp
        self.n_mono = n_mono
        self.pad = 2 * n_mono
        self.mm_switch_on = mm_switch_on
        self.ml_switch_width = ml_switch_width
        self.mm_switch_width = mm_switch_width

        raw = dict(np.load(data, allow_pickle=True))
        self.Z1 = np.asarray(raw["Z"][0])[:n_mono]
        self.R1 = np.asarray(raw["R"][0])[:n_mono]
        self.R1 = self.R1 - self.R1.mean(axis=0)
        self.t1 = np.asarray(raw["cgenff_type_idx"][0])[:n_mono]
        self.q1 = np.asarray(raw["cgenff_charge"][0])[:n_mono]
        side = json.loads(sidecar.read_text())
        self.sig_scale = jnp.asarray(side["mm_lj_sigma_scale"], dtype=jnp.float32)
        self.eps_scale = jnp.asarray(side["mm_lj_epsilon_scale"], dtype=jnp.float32)
        self.master_sig = jnp.asarray(raw["cgenff_master_sigmas"])
        self.master_eps = jnp.asarray(raw["cgenff_master_epsilons"])

        _, self.params, self.model = _load_physnet_checkpoint(
            checkpoint, self.pad, use_ema=True
        )
        self._prepare_batches_jit = prepare_batches_jit
        self._hybrid_forward = hybrid_forward
        self._HYBRID_MM_BATCH_KEYS = HYBRID_MM_BATCH_KEYS
        self._fwd = None

    def _ensure_fwd(self, batch_size: int):
        if self._fwd is not None:
            return
        jnp = self.jnp
        jax = self.jax
        hybrid_forward = self._hybrid_forward
        model = self.model
        params = self.params
        master_sig = self.master_sig
        master_eps = self.master_eps
        sig_scale = self.sig_scale
        eps_scale = self.eps_scale
        mm_switch_on = self.mm_switch_on
        mm_switch_width = self.mm_switch_width
        ml_switch_width = self.ml_switch_width

        self._fwd = jax.jit(
            lambda b: hybrid_forward(
                model.apply,
                params,
                b,
                batch_size,
                master_sig,
                master_eps,
                mm_switch_on=mm_switch_on,
                mm_switch_width=mm_switch_width,
                ml_switch_width=ml_switch_width,
                learn_mm_lj_scales=True,
                mm_lj_sigma_scale=sig_scale,
                mm_lj_epsilon_scale=eps_scale,
                lr_solver="mic",
                include_lj=True,
            )
        )

    def evaluate(self, positions: list[np.ndarray]) -> dict[str, np.ndarray]:
        """Return forces (eV/Å), ML charges (e), energies (eV) for each frame."""
        jnp = self.jnp
        jax = self.jax
        n = len(positions)
        pad = self.pad
        n_mono = self.n_mono
        batch_size = n
        R_all = np.zeros((n, pad, 3), dtype=np.float64)
        Z_all = np.zeros((n, pad), dtype=np.int32)
        T_all = np.full((n, pad), -1, dtype=np.int32)
        Q_all = np.zeros((n, pad), dtype=np.float64)
        M_all = np.full((n, pad), -1, dtype=np.int32)
        for i, pos in enumerate(positions):
            R_all[i] = pos
            Z_all[i, :n_mono] = self.Z1
            Z_all[i, n_mono:pad] = self.Z1
            T_all[i, :n_mono] = self.t1
            T_all[i, n_mono:pad] = self.t1
            Q_all[i, :n_mono] = self.q1
            Q_all[i, n_mono:pad] = self.q1
            M_all[i, :n_mono] = 0
            M_all[i, n_mono:pad] = 1

        d = {
            "R": jnp.asarray(R_all),
            "Z": jnp.asarray(Z_all),
            "F": jnp.zeros_like(jnp.asarray(R_all)),
            "E": jnp.zeros((n, 1)),
            "N": jnp.full((n,), pad),
            "D": jnp.zeros((n, 3)),
            "cgenff_type_idx": jnp.asarray(T_all),
            "cgenff_charge": jnp.asarray(Q_all),
            "mol_id": jnp.asarray(M_all),
            "id": jnp.arange(n),
        }
        keys = [
            "R",
            "Z",
            "F",
            "E",
            "N",
            "D",
            "dst_idx",
            "src_idx",
            "batch_segments",
            "id",
        ] + list(self._HYBRID_MM_BATCH_KEYS)
        batches = self._prepare_batches_jit(
            jax.random.PRNGKey(0),
            d,
            batch_size,
            num_atoms=pad,
            data_keys=keys,
            include_id=True,
        )
        self._ensure_fwd(batch_size)
        forces = np.zeros((n, pad, 3), dtype=np.float64)
        charges = np.zeros((n, pad), dtype=np.float64)
        energy = np.zeros(n, dtype=np.float64)
        charge_source = "cgenff_charge"
        for b in batches:
            out = self._fwd(b)
            ids = np.asarray(b["id"])
            f = np.asarray(out["forces"]).reshape(batch_size, pad, 3)
            e = np.asarray(out["energy"]).reshape(batch_size)
            q = out.get("charges")
            if q is None:
                q = np.asarray(b["cgenff_charge"]).reshape(batch_size, pad)
            else:
                q = np.asarray(q).reshape(batch_size, pad)
                charge_source = "PhysNet q_ML"
            forces[ids] = f
            charges[ids] = q
            energy[ids] = e
        return {
            "forces": forces,
            "charges": charges,
            "energy": energy,
            "charge_source": charge_source,
        }


def make_contact_sheet(
    paths: list[Path], labels: list[str], out: Path, cols: int = 4, title: str = ""
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.image as mpimg

    n = len(paths)
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(
        rows, cols, figsize=(3.2 * cols, 3.0 * rows), constrained_layout=True
    )
    axes = np.atleast_2d(axes)
    for i, ax in enumerate(axes.ravel()):
        ax.axis("off")
        if i >= n:
            continue
        img = mpimg.imread(paths[i])
        ax.imshow(img)
        ax.set_title(labels[i], fontsize=8)
    fig.suptitle(title or "DCM–DCM dimer scan (POV-Ray)", fontsize=12)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=140)
    plt.close(fig)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--data",
        type=Path,
        default=Path("artifacts/lj_scales/dataset_cgenff.npz"),
    )
    p.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("artifacts/lj_scales/ckpts/params_hybrid_mm_fixed_lj_scales_epoch222.json"),
    )
    p.add_argument(
        "--sidecar",
        type=Path,
        default=Path(
            "artifacts/lj_scales/ckpts/"
            "hybrid_mm_fixed_lj_scales-4d68132d-c686-4ded-9887-efc16d5b2638/hybrid_mm.json"
        ),
    )
    p.add_argument(
        "--components-csv",
        type=Path,
        default=Path(
            "artifacts/lj_scales/dense_dt_campaign/dimer_scans/orient_components.csv"
        ),
        help="Optional campaign CSV to pick deepest / softest rays",
    )
    p.add_argument(
        "--out",
        type=Path,
        default=Path("docs/images/dense-dt-campaign/dimer_scans/povray"),
    )
    p.add_argument("--n-directions", type=int, default=4)
    p.add_argument("--n-orientations", type=int, default=4)
    p.add_argument(
        "--r-grid",
        type=float,
        default=4.5,
        help="COM distance (Å) for the orientation grid (must clear min-contact)",
    )
    p.add_argument(
        "--r-values",
        default="4.0,4.5,5.5,8.0",
        help="COM distances (Å) for the approach series (clash-free by default)",
    )
    p.add_argument(
        "--min-contact",
        type=float,
        default=DEFAULT_ORIENT_MIN_CONTACT_A,
        help="Skip frames with intermolecular atom–atom dmin below this (Å)",
    )
    p.add_argument("--width", type=int, default=720)
    p.add_argument("--height", type=int, default=540)
    p.add_argument(
        "--mm-switch-on",
        type=float,
        default=8.0,
        help="Match epoch222 dimer-scan handoff (train taper)",
    )
    p.add_argument("--ml-switch-width", type=float, default=1.5)
    p.add_argument("--mm-switch-width", type=float, default=5.0)
    p.add_argument(
        "--povray",
        default="/mmhome/boittier/home/miniforge3/envs/jaxphyscharmm/bin/povray",
    )
    args = p.parse_args()

    povray = args.povray if Path(args.povray).is_file() else (shutil.which("povray") or "")
    if not povray:
        print("ERROR: povray not found")
        return 2

    n_mono = 5
    print("Loading hybrid evaluator…", flush=True)
    ev = HybridFrameEval(
        checkpoint=args.checkpoint,
        sidecar=args.sidecar,
        data=args.data,
        mm_switch_on=args.mm_switch_on,
        ml_switch_width=args.ml_switch_width,
        mm_switch_width=args.mm_switch_width,
        n_mono=n_mono,
    )
    R1, Z1 = ev.R1, ev.Z1
    dirs = fibonacci_sphere(args.n_directions)
    quats = super_fibonacci(args.n_orientations)
    rs = [float(x) for x in args.r_values.split(",") if x.strip()]

    args.out.mkdir(parents=True, exist_ok=True)
    for stale in args.out.glob("*.pov"):
        stale.unlink(missing_ok=True)
    for stale in args.out.glob("*.ini"):
        stale.unlink(missing_ok=True)

    include_dirs = _pov_include_dirs(povray)
    if not include_dirs:
        print(f"WARNING: colors.inc not found near {povray}")
    else:
        print(f"POV include: {include_dirs[0]}")

    frames: list[dict] = []
    min_contact = float(args.min_contact)
    skipped_clash = 0

    def _dmin(atoms) -> float:
        return intermolecular_min_distance(
            atoms.positions[:n_mono], atoms.positions[n_mono:]
        )

    def _add(atoms, *, stem, kind, r, direction, orientation, label, ray=None):
        nonlocal skipped_clash
        dmin = _dmin(atoms)
        if dmin < min_contact:
            skipped_clash += 1
            print(f"  skip clash {stem}: dmin={dmin:.2f} Å < {min_contact:g}")
            return
        frames.append(
            dict(
                stem=stem,
                kind=kind,
                atoms=atoms,
                dmin=dmin,
                r=float(r),
                direction=direction,
                orientation=orientation,
                ray=ray,
                label=label + f"\ndmin={dmin:.2f}",
            )
        )

    r_grid = float(args.r_grid)
    print(
        f"Orientation grid at r={r_grid} Å ({len(dirs)}×{len(quats)}), "
        f"min_contact={min_contact:g} Å"
    )
    for di, dvec in enumerate(dirs):
        for qi, q in enumerate(quats):
            atoms = build_dimer(R1, Z1, dvec, q, r_grid)
            _add(
                atoms,
                stem=f"ori_d{di:02d}_q{qi:02d}_r{_r_tag(r_grid)}",
                kind="orientation_grid",
                r=r_grid,
                direction=di,
                orientation=qi,
                label=f"d{di} q{qi}\nr={r_grid:g}",
            )

    # Prefer a campaign ray that is contact-ok along the approach series.
    approach_di, approach_qi = 0, 0
    approach_dvec, approach_quat = dirs[0], quats[0]
    if args.components_csv.is_file():
        import pandas as pd

        from scripts.slurm.dense_dt_campaign.dimer_scan_contacts import (
            annotate_dmin,
            contact_filtered_metrics,
        )

        df = pd.read_csv(args.components_csv)
        # Campaign is 8 dirs × 12 oris (not the POV 4×4 subsample).
        n_dir_csv = int(df["direction"].max()) + 1
        n_ori_csv = int(df["orientation"].max()) + 1
        dirs_c = fibonacci_sphere(n_dir_csv)
        quats_c = super_fibonacci(n_ori_csv)
        if "dmin_A" not in df.columns:
            df = annotate_dmin(df, R1=R1, n_directions=n_dir_csv, n_orientations=n_ori_csv)
            df.to_csv(args.components_csv, index=False)
        metrics = contact_filtered_metrics(df, min_contact=min_contact)
        soft_wells = sorted(
            metrics.get("soft_wells") or [], key=lambda x: x["E_int_kcal"]
        )
        print(
            f"CSV contact-ok soft wells: {len(soft_wells)} "
            f"(median {metrics.get('median_soft_well_kcal')})"
        )

        def pick(ray: int, r: float, tag: str):
            di = int(ray // n_ori_csv)
            qi = int(ray % n_ori_csv)
            if di >= len(dirs_c) or qi >= len(quats_c):
                return
            atoms = build_dimer(R1, Z1, dirs_c[di], quats_c[qi], float(r))
            _add(
                atoms,
                stem=f"{tag}_ray{ray:03d}_r{_r_tag(float(r))}",
                kind=tag,
                r=float(r),
                direction=di,
                orientation=qi,
                ray=int(ray),
                label=f"{tag}\nray {ray} r={float(r):.2f}",
            )

        if soft_wells:
            softest = soft_wells[0]
            shallow = soft_wells[-1]
            mid = soft_wells[len(soft_wells) // 2]
            pick(int(softest["ray"]), float(softest["r_A"]), "softest_well")
            pick(int(mid["ray"]), float(mid["r_A"]), "median_soft")
            pick(int(shallow["ray"]), float(shallow["r_A"]), "shallow_soft")
            # Approach series along the median soft-well orientation.
            approach_di = int(mid["direction"])
            approach_qi = int(mid["orientation"])
            approach_dvec = dirs_c[approach_di]
            approach_quat = quats_c[approach_qi]

    print(f"Approach series d{approach_di} q{approach_qi}")
    for r in rs:
        atoms = build_dimer(R1, Z1, approach_dvec, approach_quat, r)
        _add(
            atoms,
            stem=f"approach_d{approach_di:02d}_q{approach_qi:02d}_r{_r_tag(r)}",
            kind="approach",
            r=r,
            direction=approach_di,
            orientation=approach_qi,
            label=f"approach\nr={r:g}",
        )

    if not frames:
        print("ERROR: no contact-ok frames to render; relax --min-contact or raise --r-grid")
        return 2
    print(
        f"Evaluating hybrid on {len(frames)} contact-ok frames "
        f"({skipped_clash} clash frames skipped)…",
        flush=True,
    )
    pred = ev.evaluate([f["atoms"].positions.copy() for f in frames])
    charge_source = str(pred.get("charge_source", "PhysNet q_ML"))
    fmax_panel = float(np.linalg.norm(pred["forces"], axis=-1).max())
    soft_fmax = [
        float(np.linalg.norm(pred["forces"][i], axis=-1).max())
        for i, fr in enumerate(frames)
        if fr["dmin"] >= min_contact
    ]
    f_ref = float(np.percentile(soft_fmax, 90)) if soft_fmax else fmax_panel
    force_scale = 1.35 / max(f_ref, 1e-12)
    force_arrow_cap_A = 2.5

    q_all = pred["charges"]
    q_abs_max = float(np.max(np.abs(q_all))) if q_all.size else 0.1
    q_lim = _nice_q_lim(q_abs_max)
    charge_cmap = _load_charge_cmap()

    box_half = compute_shared_box_half(
        frames,
        pred,
        n_mono,
        force_scale=force_scale,
        force_cap=force_arrow_cap_A,
        dipole_len=DIPOLE_ARROW_LEN,
        pad=BOX_PAD_A,
    )
    print(
        f"  |F|_max={fmax_panel:.4f} eV/Å  soft |F|_ref={f_ref:.4f} eV/Å  "
        f"force_scale={force_scale:.4f} Å/(eV/Å)  cap={force_arrow_cap_A} Å",
        flush=True,
    )
    print(
        f"  shared box ±{box_half:.3f} Å  |  {charge_source}  "
        f"|q|_max={q_abs_max:.4f} e  colorbar ±{q_lim:.3f} e ({CHARGE_CMAP_NAME})",
        flush=True,
    )

    sheet_fd: list[Path] = []
    sheet_fd_labels: list[str] = []
    sheet_q: list[Path] = []
    sheet_q_labels: list[str] = []
    manifest = []

    for i, fr in enumerate(frames):
        atoms = fr["atoms"]
        forces = pred["forces"][i]
        charges = pred["charges"][i]
        xyz = _centered_xyz(atoms)

        # --- forces + dipoles ---
        base = glossy_scene(
            atoms, half=box_half, width=args.width, height=args.height, draw_box=True
        )
        overlay = _force_overlay(
            xyz, forces, force_scale, max_len=force_arrow_cap_A
        )
        dip_lines, mu_norms = _dipole_overlay(xyz, charges, n_mono)
        overlay += dip_lines
        png_fd = args.out / f"{fr['stem']}_forces_dipoles.png"
        ok = render_pov(
            base + "\n".join(overlay) + "\n",
            png_fd,
            povray=povray,
            width=args.width,
            height=args.height,
            include_dirs=include_dirs,
        )
        if ok:
            fmax = float(np.linalg.norm(forces, axis=-1).max())
            capped = fmax * force_scale > force_arrow_cap_A
            scale_note = (
                f"soft-ref scale {force_scale:.3f} Å per eV/Å"
                + (f"; arrows capped at {force_arrow_cap_A:g} Å" if capped else "")
            )
            _annotate_image(
                png_fd,
                title=f"{fr['stem']}  |  forces + dipoles",
                lines=[
                    f"r={fr['r']:.2f} Å   dmin={fr['dmin']:.2f} Å   box ±{box_half:.2f} Å",
                    f"red F  |F|_max={fmax:.3f} eV/Å  ({scale_note})",
                    f"gold μ  |μ|_A={mu_norms[0]:.3f}  |μ|_B={mu_norms[1]:.3f} e·Å",
                    f"hybrid on={args.mm_switch_on:g} / "
                    f"ml_w={args.ml_switch_width:g} / mm_w={args.mm_switch_width:g}",
                ],
                width=args.width,
            )
            sheet_fd.append(png_fd)
            sheet_fd_labels.append(fr["label"] + "\nF+μ")
            print(f"  {png_fd.name} ok")
        else:
            print(f"  {png_fd.name} FAIL")

        # --- charge-colored (continuous house scale) ---
        q_colors = [_charge_rgb(float(q), q_lim, charge_cmap) for q in charges]
        q_scene = glossy_scene(
            atoms,
            half=box_half,
            width=args.width,
            height=args.height,
            atom_colors=q_colors,
            draw_box=True,
        )
        q_overlay, mu_norms_q = _dipole_overlay(xyz, charges, n_mono)
        png_q = args.out / f"{fr['stem']}_by_charge.png"
        ok_q = render_pov(
            q_scene + "\n".join(q_overlay) + "\n",
            png_q,
            povray=povray,
            width=args.width,
            height=args.height,
            include_dirs=include_dirs,
        )
        if ok_q:
            qmin = float(charges.min())
            qmax = float(charges.max())
            _annotate_image(
                png_q,
                title=f"{fr['stem']}  |  atoms by charge",
                lines=[
                    f"r={fr['r']:.2f} Å   dmin={fr['dmin']:.2f} Å   box ±{box_half:.2f} Å",
                    f"{CHARGE_CMAP_NAME}  red=+q  blue=−q  "
                    f"(scale ±{q_lim:.3f} e)",
                    f"gold μ  |μ|_A={mu_norms_q[0]:.3f}  |μ|_B={mu_norms_q[1]:.3f} e·Å",
                    f"{charge_source}  q∈[{qmin:+.4f},{qmax:+.4f}]  Σq={float(charges.sum()):+.4f} e",
                ],
                width=args.width,
                colorbar={"q_lim": q_lim, "cmap": charge_cmap},
            )
            sheet_q.append(png_q)
            sheet_q_labels.append(fr["label"] + "\nq")
            print(f"  {png_q.name} ok")
        else:
            print(f"  {png_q.name} FAIL")

        # Element-colored overview (same box).
        png_plain = args.out / f"{fr['stem']}.png"
        ok_p = render_pov(
            glossy_scene(
                atoms, half=box_half, width=args.width, height=args.height, draw_box=True
            ),
            png_plain,
            povray=povray,
            width=args.width,
            height=args.height,
            include_dirs=include_dirs,
        )
        if ok_p:
            _annotate_image(
                png_plain,
                title=fr["stem"],
                lines=[
                    f"r={fr['r']:.2f} Å   dmin={fr['dmin']:.2f} Å   box ±{box_half:.2f} Å",
                    "element colors (Cl green / C dark / H light)",
                ],
                width=args.width,
            )

        manifest.append(
            dict(
                file=fr["stem"] + ".png",
                forces_dipoles=fr["stem"] + "_forces_dipoles.png",
                by_charge=fr["stem"] + "_by_charge.png",
                kind=fr["kind"],
                direction=fr.get("direction"),
                orientation=fr.get("orientation"),
                ray=fr.get("ray"),
                r_A=fr["r"],
                dmin_A=fr["dmin"],
                fmax_eV_A=float(np.linalg.norm(forces, axis=-1).max()),
                mu_A_eA=mu_norms[0] if ok else None,
                mu_B_eA=mu_norms[1] if ok else None,
                q_sum_e=float(charges.sum()),
                q_min_e=float(charges.min()),
                q_max_e=float(charges.max()),
            )
        )

    if sheet_fd:
        make_contact_sheet(
            sheet_fd,
            sheet_fd_labels,
            args.out / "dimer_scan_povray_sheet_forces_dipoles.png",
            cols=4,
            title="DCM–DCM dimer scan — hybrid forces (red) + monomer dipoles (gold)",
        )
        print(f"wrote {args.out / 'dimer_scan_povray_sheet_forces_dipoles.png'}")
    if sheet_q:
        make_contact_sheet(
            sheet_q,
            sheet_q_labels,
            args.out / "dimer_scan_povray_sheet_by_charge.png",
            cols=4,
            title=(
                f"DCM–DCM dimer scan — {CHARGE_CMAP_NAME} charge coloring "
                f"(±{q_lim:.3f} e)"
            ),
        )
        print(f"wrote {args.out / 'dimer_scan_povray_sheet_by_charge.png'}")
    plain = [args.out / m["file"] for m in manifest if (args.out / m["file"]).is_file()]
    plain_labels = [
        f"{m['kind']}\nr={m['r_A']:g}" for m in manifest if (args.out / m["file"]).is_file()
    ]
    if plain:
        make_contact_sheet(
            plain,
            plain_labels,
            args.out / "dimer_scan_povray_sheet.png",
            cols=4,
            title="DCM–DCM dimer scan geometries (POV-Ray, element colors)",
        )
        print(f"wrote {args.out / 'dimer_scan_povray_sheet.png'}")

    meta = {
        "checkpoint": str(args.checkpoint),
        "sidecar": str(args.sidecar),
        "mm_switch_on": args.mm_switch_on,
        "ml_switch_width": args.ml_switch_width,
        "mm_switch_width": args.mm_switch_width,
        "force_scale_A_per_eV_A": force_scale,
        "force_scale_ref": f"90th percentile |F|_max among contact-ok frames (dmin>={min_contact:g} Å)",
        "min_contact_A": min_contact,
        "n_frames_skipped_clash": skipped_clash,
        "force_ref_eV_A": f_ref,
        "force_arrow_cap_A": force_arrow_cap_A,
        "fmax_panel_eV_A": fmax_panel,
        "shared_box_half_A": box_half,
        "shared_box_note": (
            "Axis-aligned cube ±half about dimer COM; camera fixed across all "
            "frames. Box encloses atoms + force tips + dipole tips + pad."
        ),
        "charge_source": charge_source,
        "charge_cmap": CHARGE_CMAP_NAME,
        "charge_colorbar_lim_e": q_lim,
        "charge_abs_max_e": q_abs_max,
        "dipole_units": "e*Angstrom from q*(R-COM) per monomer",
        "style": "docs/plotting-style-guide.md glossy + shared box + crameri:vik charges",
        "frames": manifest,
    }
    (args.out / "manifest.json").write_text(json.dumps(meta, indent=2) + "\n")
    (args.out / "README.md").write_text(
        "\n".join(
            [
                "# DCM–DCM dimer scan POV-Ray stills",
                "",
                "House glossy POV style (`docs/plotting-style-guide.md`):",
                "",
                "- **shared bounding box** — one cube (±half Å about COM) and one",
                "  orthographic camera for every frame, so atoms / bonds / force",
                "  arrows / dipoles are never cropped and spacing stays consistent.",
                f"- **contact filter** — frames with intermolecular $d_\\mathrm{{min}}"
                f" < {args.min_contact:g}$ Å are skipped (COM–COM $r$ alone is not",
                "  steric for DCM; clash geometries invent huge forces / deep wells).",
                "- **forces** — red arrows from the hybrid model on the exact frame;",
                "  fixed soft-well panel normalization (see `manifest.json`).",
                "- **dipoles** — gold per-monomer μ from PhysNet `q_ML` (e·Å).",
                f"- **by charge** — continuous `{CHARGE_CMAP_NAME}` (red = +q,",
                "  blue = −q) with a colorbar in e.",
                "",
                "| Asset | Content |",
                "|---|---|",
                "| `*_forces_dipoles.png` | Glossy atoms + red F + gold μ + box |",
                "| `*_by_charge.png` | vik charge colors + colorbar + gold μ + box |",
                "| `ori_*/approach_*/…png` | Element-colored overview stills + box |",
                "| `dimer_scan_povray_sheet_forces_dipoles.png` | F+μ contact sheet |",
                "| `dimer_scan_povray_sheet_by_charge.png` | Charge contact sheet |",
                "| `dimer_scan_povray_sheet.png` | Element-color contact sheet |",
                "",
                f"Handoff used for forces: `mm_switch_on={args.mm_switch_on:g}`, "
                f"`ml_switch_width={args.ml_switch_width:g}`, "
                f"`mm_switch_width={args.mm_switch_width:g}` "
                "(epoch222 train taper by default).",
                "",
                "Regenerate:",
                "```bash",
                "uv run python scripts/slurm/dense_dt_campaign/render_dimer_scan_povray.py",
                "```",
                "",
            ]
        )
    )
    print(f"done → {args.out} ({len(manifest)} frames × 3 variants)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
