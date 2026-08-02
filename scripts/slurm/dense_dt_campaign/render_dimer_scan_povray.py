#!/usr/bin/env python3
"""POV-Ray stills for DCM–DCM hybrid dimer scan geometries.

House glossy style (``docs/plotting-style-guide.md``):
- red force arrows from the hybrid model on the exact frame
- gold per-monomer dipoles from PhysNet ``q_ML``
- separate charge-colored stills (blue +, red −)

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

from scripts.render_povray_multipoles import _arrow  # noqa: E402
from scripts.render_povray_style_catalog import scene, vec  # noqa: E402

Z_TO_SYM = {1: "H", 6: "C", 17: "Cl"}
EV_TO_KCAL = 23.0605
FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
FORCE_COLOR = (0.78, 0.04, 0.12)
DIPOLE_COLOR = (0.95, 0.62, 0.06)
Q_POS = (0.16, 0.38, 0.92)
Q_NEG = (0.88, 0.12, 0.20)


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


def _charge_rgb(q: float, q_ref: float) -> tuple[float, float, float]:
    """Signed charge → blue(+)/red(−), desaturated near zero."""
    t = float(np.clip(abs(q) / max(q_ref, 1e-8), 0.0, 1.0))
    base = Q_POS if q >= 0 else Q_NEG
    gray = (0.55, 0.56, 0.58)
    return tuple((1.0 - t) * g + t * c for g, c in zip(gray, base))


def _force_overlay(
    xyz: np.ndarray,
    forces: np.ndarray,
    scale: float,
    *,
    max_len: float = 2.5,
) -> list[str]:
    """Red force glyphs. ``scale`` is Å per (eV/Å); lengths capped at ``max_len``."""
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
    xyz: np.ndarray, charges: np.ndarray, n_mono: int, *, arrow_scale: float = 2.0
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


def _charge_atom_scene(
    atoms,
    charges: np.ndarray,
    *,
    width: int,
    height: int,
    q_ref: float,
) -> str:
    """Glossy scene with atom spheres recolored by signed charge."""
    from ase.data import covalent_radii

    xyz = atoms.positions - atoms.positions.mean(0)
    span = max(float(np.ptp(xyz, axis=0).max()), 3.0)
    camera = np.array([span * 1.15, span * 0.80, -span * 2.7])
    finish = 'finish { phong 0.85 phong_size 90 reflection 0.06 }'
    frame_scale = 1.72
    lines = [
        "#version 3.7;",
        "global_settings { assumed_gamma 1.0 }",
        "background { color rgbt <1,1,1,1> }",
        (
            f"camera {{ orthographic location {vec(camera)} look_at <0,0,0> "
            f"right x*{frame_scale*span} up y*{frame_scale*span*height/width} }}"
        ),
        (
            f"light_source {{ {vec((-span, span*2, -span*2))} color rgb <1,1,1> "
            f"area_light <2,0,0>,<0,2,0>,5,5 adaptive 1 jitter }}"
        ),
        f"light_source {{ {vec((span*2, -span, -span))} color rgb <0.35,0.40,0.52> }}",
        (
            "plane { y, -2.15 pigment { color rgbt <0.91,0.92,0.94,0.82> } "
            "finish { diffuse 0.75 } }"
        ),
    ]
    from scripts.render_povray_style_catalog import bonds

    for i, j in bonds(atoms):
        a, b = xyz[i], xyz[j]
        lines.append(
            f"cylinder {{ {vec(a)}, {vec(b)}, 0.105 "
            f"pigment {{ color rgb <0.55,0.57,0.61> }} "
            f"finish {{ diffuse 0.7 phong 0.25 }} }}"
        )
    for p, z, q in zip(xyz, atoms.numbers, charges):
        color = _charge_rgb(float(q), q_ref)
        radius = max(0.24, covalent_radii[z] * 0.52)
        lines.append(
            f"sphere {{ {vec(p)}, {radius:.4f} texture {{ "
            f"pigment {{ color rgb {vec(color)} }} {finish} }} }}"
        )
        # Soft signed halo (style-guide multipole convention).
        halo = radius + 0.07 + 0.40 * abs(float(q)) / max(q_ref, 1e-8)
        hc = Q_POS if q >= 0 else Q_NEG
        lines.append(
            f"sphere {{ {vec(p)}, {halo:.4f} pigment {{ "
            f"color rgbt <{hc[0]},{hc[1]},{hc[2]},0.82> }} "
            f"finish {{ emission 0.04 phong 0.3 }} no_shadow }}"
        )
    return "\n".join(lines) + "\n"


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


def _label_image(
    png: Path,
    *,
    title: str,
    lines: list[str],
    width: int,
) -> None:
    from PIL import Image, ImageDraw, ImageFont

    image = Image.open(png).convert("RGBA")
    image = _normalize_alpha(image)
    draw = ImageDraw.Draw(image, "RGBA")
    try:
        title_font = ImageFont.truetype(FONT, max(18, width // 28))
        body_font = ImageFont.truetype(FONT, max(14, width // 40))
    except OSError:
        title_font = ImageFont.load_default()
        body_font = title_font
    pad = 14
    text_block = [title, *lines]
    boxes = [draw.textbbox((0, 0), t, font=(title_font if i == 0 else body_font))
             for i, t in enumerate(text_block)]
    tw = max(b[2] - b[0] for b in boxes) + 2 * pad + 16
    th = sum(b[3] - b[1] + 6 for b in boxes) + 2 * pad
    draw.rounded_rectangle((16, 14, 16 + tw, 14 + th), radius=12, fill=(250, 250, 252, 225))
    y = 14 + pad
    for i, t in enumerate(text_block):
        font = title_font if i == 0 else body_font
        draw.text((16 + pad, y), t, font=font, fill=(18, 22, 30, 255))
        y += (boxes[i][3] - boxes[i][1]) + 6
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
            forces[ids] = f
            charges[ids] = q
            energy[ids] = e
        return {"forces": forces, "charges": charges, "energy": energy}


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
        "--r-values",
        default="2.8,3.5,5.0,8.0",
        help="COM distances (Å) for the approach series",
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

    # Collect frames to evaluate / render.
    frames: list[dict] = []

    r_grid = 3.5
    print(f"Orientation grid at r={r_grid} Å ({len(dirs)}×{len(quats)})")
    for di, dvec in enumerate(dirs):
        for qi, q in enumerate(quats):
            atoms = build_dimer(R1, Z1, dvec, q, r_grid)
            pa, pb = atoms.positions[:n_mono], atoms.positions[n_mono:]
            dmin = float(np.linalg.norm(pa[:, None, :] - pb[None, :, :], axis=-1).min())
            stem = f"ori_d{di:02d}_q{qi:02d}_r{_r_tag(r_grid)}"
            frames.append(
                dict(
                    stem=stem,
                    kind="orientation_grid",
                    atoms=atoms,
                    dmin=dmin,
                    r=r_grid,
                    direction=di,
                    orientation=qi,
                    label=f"d{di} q{qi}\nr={r_grid:g} dmin={dmin:.2f}",
                )
            )

    print("Approach series d0 q0")
    for r in rs:
        atoms = build_dimer(R1, Z1, dirs[0], quats[0], r)
        pa, pb = atoms.positions[:n_mono], atoms.positions[n_mono:]
        dmin = float(np.linalg.norm(pa[:, None, :] - pb[None, :, :], axis=-1).min())
        stem = f"approach_d00_q00_r{_r_tag(r)}"
        frames.append(
            dict(
                stem=stem,
                kind="approach",
                atoms=atoms,
                dmin=dmin,
                r=r,
                direction=0,
                orientation=0,
                label=f"approach\nr={r:g} dmin={dmin:.2f}",
            )
        )

    if args.components_csv.is_file():
        import pandas as pd

        df = pd.read_csv(args.components_csv)
        n_ori_csv = 8
        n_rays = int(df.ray.max()) + 1
        n_dir_csv = max(1, n_rays // n_ori_csv)
        dirs_c = fibonacci_sphere(n_dir_csv)
        quats_c = super_fibonacci(n_ori_csv)
        print(f"CSV extremes assuming {n_dir_csv}×{n_ori_csv} rays")

        def pick(ray: int, r: float, tag: str):
            di = int(ray // n_ori_csv)
            qi = int(ray % n_ori_csv)
            if di >= len(dirs_c) or qi >= len(quats_c):
                return
            atoms = build_dimer(R1, Z1, dirs_c[di], quats_c[qi], float(r))
            pa, pb = atoms.positions[:n_mono], atoms.positions[n_mono:]
            dmin = float(np.linalg.norm(pa[:, None, :] - pb[None, :, :], axis=-1).min())
            stem = f"{tag}_ray{ray:03d}_r{_r_tag(float(r))}"
            frames.append(
                dict(
                    stem=stem,
                    kind=tag,
                    atoms=atoms,
                    dmin=dmin,
                    r=float(r),
                    direction=di,
                    orientation=qi,
                    ray=int(ray),
                    label=f"{tag}\nray {ray} r={float(r):.2f}\ndmin={dmin:.2f}",
                )
            )

        idx = df.groupby("ray")["E_int_kcal"].idxmin()
        wells = df.loc[idx].sort_values("E_int_kcal")
        deep = wells.iloc[0]
        pick(int(deep.ray), float(deep.r_A), "deepest_well")
        soft = df[df.r_A >= 3.4]
        idx_s = soft.groupby("ray")["E_int_kcal"].idxmin()
        soft_wells = soft.loc[idx_s].sort_values("E_int_kcal")
        softest = soft_wells.iloc[0]
        pick(int(softest.ray), float(softest.r_A), "softest_well")
        shallow = soft_wells.iloc[-1]
        pick(int(shallow.ray), float(shallow.r_A), "shallow_soft")

    print(f"Evaluating hybrid on {len(frames)} frames…", flush=True)
    pred = ev.evaluate([f["atoms"].positions.copy() for f in frames])
    fmax_panel = float(np.linalg.norm(pred["forces"], axis=-1).max())
    # Soft-well reference for fixed panel scale (style guide). Exclude
    # contact/clash frames (short COM or short atom–atom contacts) so a single
    # Cl–Cl clash does not shrink every soft-well arrow.
    soft_fmax = [
        float(np.linalg.norm(pred["forces"][i], axis=-1).max())
        for i, fr in enumerate(frames)
        if fr["r"] >= 3.2 and fr["dmin"] >= 1.6
    ]
    f_ref = float(np.percentile(soft_fmax, 90)) if soft_fmax else fmax_panel
    force_scale = 1.35 / max(f_ref, 1e-12)
    force_arrow_cap_A = 2.5
    q_abs = np.abs(pred["charges"])
    q_ref = float(np.percentile(q_abs[q_abs > 0], 90)) if np.any(q_abs > 0) else 0.1
    print(
        f"  |F|_max={fmax_panel:.4f} eV/Å  soft |F|_ref={f_ref:.4f} eV/Å  "
        f"force_scale={force_scale:.4f} Å/(eV/Å)  cap={force_arrow_cap_A} Å  "
        f"q_ref={q_ref:.4f} e",
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
        xyz = atoms.positions - atoms.positions.mean(0)
        # Center forces/charges with the same COM shift used in the POV scene.
        # Forces are translationally invariant; positions for arrows use centered xyz.

        # --- forces + dipoles ---
        base = scene(
            atoms,
            "glossy",
            "",
            vectors=False,
            geometry=False,
            font="",
            width=args.width,
            height=args.height,
        )
        # scene() recenters; rebuild overlays in the same frame.
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
            _label_image(
                png_fd,
                title=f"{fr['stem']}  |  forces + dipoles",
                lines=[
                    f"r={fr['r']:.2f} Å   dmin={fr['dmin']:.2f} Å",
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

        # --- charge-colored ---
        q_scene = _charge_atom_scene(
            atoms, charges, width=args.width, height=args.height, q_ref=q_ref
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
            _label_image(
                png_q,
                title=f"{fr['stem']}  |  atoms by charge",
                lines=[
                    f"r={fr['r']:.2f} Å   dmin={fr['dmin']:.2f} Å",
                    "blue +q / red −q  (PhysNet q_ML; halo ∝ |q|)",
                    f"gold μ  |μ|_A={mu_norms_q[0]:.3f}  |μ|_B={mu_norms_q[1]:.3f} e·Å",
                    f"Σq={float(charges.sum()):.4f} e   q_ref={q_ref:.3f} e",
                ],
                width=args.width,
            )
            sheet_q.append(png_q)
            sheet_q_labels.append(fr["label"] + "\nq")
            print(f"  {png_q.name} ok")
        else:
            print(f"  {png_q.name} FAIL")

        # Keep plain element-colored still (no vectors) for quick overview.
        png_plain = args.out / f"{fr['stem']}.png"
        ok_p = render_pov(
            scene(
                atoms,
                "glossy",
                "",
                vectors=False,
                geometry=False,
                font="",
                width=args.width,
                height=args.height,
            ),
            png_plain,
            povray=povray,
            width=args.width,
            height=args.height,
            include_dirs=include_dirs,
        )
        if ok_p:
            _label_image(
                png_plain,
                title=fr["stem"],
                lines=[
                    f"r={fr['r']:.2f} Å   dmin={fr['dmin']:.2f} Å",
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
            title="DCM–DCM dimer scan — atoms colored by PhysNet charge (blue+/red−)",
        )
        print(f"wrote {args.out / 'dimer_scan_povray_sheet_by_charge.png'}")
    # Overview sheet of plain element-colored stills.
    plain = [args.out / (m["file"]) for m in manifest if (args.out / m["file"]).is_file()]
    plain_labels = [f"{m['kind']}\nr={m['r_A']:g}" for m in manifest if (args.out / m["file"]).is_file()]
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
        "force_scale_ref": "90th percentile |F|_max among frames with r>=3.2 Å and dmin>=1.6 Å",
        "force_ref_eV_A": f_ref,
        "force_arrow_cap_A": force_arrow_cap_A,
        "fmax_panel_eV_A": fmax_panel,
        "q_ref_e": q_ref,
        "charge_source": "PhysNet q_ML (model charges head)",
        "dipole_units": "e*Angstrom from q_ML*(R-COM) per monomer",
        "style": "docs/plotting-style-guide.md glossy + force/dipole/charge conventions",
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
                "- **forces** — red arrows from the hybrid model on the exact frame;",
                "  fixed panel normalization (see `manifest.json`).",
                "- **dipoles** — gold per-monomer μ from PhysNet `q_ML` (e·Å).",
                "- **by charge** — atom spheres + soft halos, blue = +, red = −.",
                "",
                "| Asset | Content |",
                "|---|---|",
                "| `*_forces_dipoles.png` | Glossy atoms + red F + gold μ |",
                "| `*_by_charge.png` | Charge-colored atoms + gold μ |",
                "| `ori_*/approach_*/…png` | Element-colored overview stills |",
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
