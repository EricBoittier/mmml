#!/usr/bin/env python3
"""POV-Ray stills for DCM–DCM hybrid dimer scan geometries.

Renders a small grid of orientations / COM distances so the 1D multi-ray
profiles can be eyeballed against real structures (Cl–Cl clashes, H-facing
contacts, etc.).

Example::

    uv run python scripts/slurm/dense_dt_campaign/render_dimer_scan_povray.py
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
from pathlib import Path

import numpy as np

COLORS = {
    "H": (0.85, 0.85, 0.88),
    "C": (0.35, 0.35, 0.38),
    "Cl": (0.30, 0.75, 0.30),
}
RADII = {"H": 0.28, "C": 0.55, "Cl": 0.80}
Z_TO_SYM = {1: "H", 6: "C", 17: "Cl"}


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
    """Locate POV-Ray ``include/`` next to the binary / share tree."""
    exe = Path(povray).resolve()
    candidates = [
        exe.parent.parent / "share" / "povray-3.7" / "include",
        exe.parent.parent / "share" / "povray" / "include",
        Path("/usr/share/povray/include"),
        Path("/usr/share/povray-3.7/include"),
    ]
    return [p for p in candidates if (p / "colors.inc").is_file()]


def render_atoms(
    atoms,
    out_png: Path,
    *,
    povray: str,
    width: int = 640,
    rotation: str = "15x,-20y,5z",
    include_dirs: list[Path] | None = None,
) -> bool:
    from ase.io import write
    from ase.io.pov import get_bondpairs

    out_png.parent.mkdir(parents=True, exist_ok=True)
    # Do not use Path.with_suffix on names containing dots (r3.5 → truncated).
    stem = out_png.parent / out_png.name[: -len(out_png.suffix)]
    pov_path = Path(str(stem) + ".pov")
    ini = Path(str(stem) + ".ini")
    syms = atoms.get_chemical_symbols()
    colors = np.array([COLORS.get(s, (0.6, 0.6, 0.6)) for s in syms])
    radii = np.array([RADII.get(s, 0.55) for s in syms])
    # Skip Cl–Cl bonds across the dimer (same as povray_tool).
    bondpairs = []
    for a, b, *rest in get_bondpairs(atoms, radius=0.95):
        sa, sb = atoms[a].symbol, atoms[b].symbol
        if {sa, sb} == {"Cl"}:
            continue
        # Drop spurious cross-monomer bonds except short contacts for viz.
        if (a < 5) != (b < 5):
            dist = float(np.linalg.norm(atoms.positions[a] - atoms.positions[b]))
            if dist > 1.6:
                continue
        bondpairs.append((a, b, *rest) if rest else (a, b))

    write(
        str(pov_path),
        atoms,
        format="pov",
        radii=radii,
        colors=colors,
        rotation=rotation,
        show_unit_cell=0,
        povray_settings=dict(
            canvas_width=width,
            background="White",
            transparent=False,
            display=False,
            camera_type="orthographic",
            bondlinewidth=0.08,
            bondatoms=bondpairs,
            textures=["jmol"] * len(atoms),
        ),
    )
    # Conda povray often can't find its system conf; inject Library_Path.
    libs = include_dirs if include_dirs is not None else _pov_include_dirs(povray)
    if libs and ini.is_file():
        extra = "".join(f'Library_Path="{lib}"\n' for lib in libs)
        ini.write_text(extra + ini.read_text())

    env = dict(os.environ)
    # Avoid hard-coded broken system conf paths from relocated conda builds.
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
    if ok:
        ini.unlink(missing_ok=True)
    return ok


def make_contact_sheet(paths: list[Path], labels: list[str], out: Path, cols: int = 4) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.image as mpimg

    n = len(paths)
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(3.2 * cols, 3.0 * rows), constrained_layout=True)
    axes = np.atleast_2d(axes)
    for i, ax in enumerate(axes.ravel()):
        ax.axis("off")
        if i >= n:
            continue
        img = mpimg.imread(paths[i])
        ax.imshow(img)
        ax.set_title(labels[i], fontsize=8)
    fig.suptitle("DCM–DCM dimer scan geometries (POV-Ray)", fontsize=12)
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
        "--components-csv",
        type=Path,
        default=Path("artifacts/lj_scales/dense_dt_campaign/dimer_scans/orient_components.csv"),
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
        help="COM distances (Å) for the orientation grid",
    )
    p.add_argument("--width", type=int, default=560)
    p.add_argument(
        "--povray",
        default="/mmhome/boittier/home/miniforge3/envs/jaxphyscharmm/bin/povray",
    )
    args = p.parse_args()

    povray = args.povray if Path(args.povray).is_file() else (shutil.which("povray") or "")
    if not povray:
        print("ERROR: povray not found")
        return 2

    raw = dict(np.load(args.data, allow_pickle=True))
    n_mono = 5
    Z1 = np.asarray(raw["Z"][0])[:n_mono]
    R1 = np.asarray(raw["R"][0])[:n_mono]
    R1 = R1 - R1.mean(axis=0)

    dirs = fibonacci_sphere(args.n_directions)
    quats = super_fibonacci(args.n_orientations)
    rs = [float(x) for x in args.r_values.split(",") if x.strip()]

    args.out.mkdir(parents=True, exist_ok=True)
    # Clear prior failed half-renders with truncated names
    for stale in args.out.glob("*.pov"):
        stale.unlink(missing_ok=True)
    for stale in args.out.glob("*.ini"):
        stale.unlink(missing_ok=True)

    sheet_paths: list[Path] = []
    sheet_labels: list[str] = []
    manifest = []
    include_dirs = _pov_include_dirs(povray)
    if not include_dirs:
        print(f"WARNING: colors.inc not found near {povray}")
    else:
        print(f"POV include: {include_dirs[0]}")

    # 1) Orientation grid at r = 3.5 Å (soft-well region)
    r_grid = 3.5
    print(f"Orientation grid at r={r_grid} Å ({len(dirs)}×{len(quats)})")
    for di, dvec in enumerate(dirs):
        for qi, q in enumerate(quats):
            atoms = build_dimer(R1, Z1, dvec, q, r_grid)
            # min interatomic distance across monomers (sanity)
            pa, pb = atoms.positions[:n_mono], atoms.positions[n_mono:]
            dmin = float(np.linalg.norm(pa[:, None, :] - pb[None, :, :], axis=-1).min())
            name = f"ori_d{di:02d}_q{qi:02d}_r{_r_tag(r_grid)}.png"
            png = args.out / name
            ok = render_atoms(
                atoms, png, povray=povray, width=args.width, include_dirs=include_dirs
            )
            print(f"  {name} dmin={dmin:.2f} Å ok={ok}")
            if ok:
                sheet_paths.append(png)
                sheet_labels.append(f"d{di} q{qi}\nr={r_grid:g} dmin={dmin:.2f}")
                manifest.append(
                    dict(file=name, kind="orientation_grid", direction=di, orientation=qi, r_A=r_grid, dmin_A=dmin)
                )

    # 2) Approach series for orientation (0,0)
    print("Approach series d0 q0")
    for r in rs:
        atoms = build_dimer(R1, Z1, dirs[0], quats[0], r)
        pa, pb = atoms.positions[:n_mono], atoms.positions[n_mono:]
        dmin = float(np.linalg.norm(pa[:, None, :] - pb[None, :, :], axis=-1).min())
        name = f"approach_d00_q00_r{_r_tag(r)}.png"
        png = args.out / name
        ok = render_atoms(
            atoms, png, povray=povray, width=args.width, include_dirs=include_dirs
        )
        print(f"  {name} dmin={dmin:.2f} Å ok={ok}")
        if ok:
            sheet_paths.append(png)
            sheet_labels.append(f"approach\nr={r:g} dmin={dmin:.2f}")
            manifest.append(
                dict(file=name, kind="approach", direction=0, orientation=0, r_A=r, dmin_A=dmin)
            )

    # 3) Deepest / softest rays from campaign CSV (if present)
    if args.components_csv.is_file():
        import pandas as pd

        df = pd.read_csv(args.components_csv)
        # Campaign used 96 rays = 12 dirs × 8 oris typically; map ray→(di,qi) if possible.
        # Our grid is smaller — rebuild using CSV ray's r at minimum with fibonacci matching
        # by regenerating the same n_dirs/n_oris as the CSV if encoded, else sample extremes.
        n_rays = int(df.ray.max()) + 1
        # Infer factorization from common campaign settings (12×8 or 8×8 etc.)
        n_ori_csv = 8
        n_dir_csv = max(1, n_rays // n_ori_csv)
        dirs_c = fibonacci_sphere(n_dir_csv)
        quats_c = super_fibonacci(n_ori_csv)
        print(f"CSV extremes assuming {n_dir_csv}×{n_ori_csv} = {n_dir_csv * n_ori_csv} rays")

        def pick(ray: int, r: float, tag: str):
            di = int(ray // n_ori_csv)
            qi = int(ray % n_ori_csv)
            if di >= len(dirs_c) or qi >= len(quats_c):
                return
            atoms = build_dimer(R1, Z1, dirs_c[di], quats_c[qi], float(r))
            pa, pb = atoms.positions[:n_mono], atoms.positions[n_mono:]
            dmin = float(np.linalg.norm(pa[:, None, :] - pb[None, :, :], axis=-1).min())
            name = f"{tag}_ray{ray:03d}_r{_r_tag(float(r))}.png"
            png = args.out / name
            ok = render_atoms(
                atoms, png, povray=povray, width=args.width, include_dirs=include_dirs
            )
            print(f"  {name} dmin={dmin:.2f} Å ok={ok}")
            if ok:
                sheet_paths.append(png)
                sheet_labels.append(f"{tag}\nray {ray} r={float(r):.2f}\ndmin={dmin:.2f}")
                manifest.append(
                    dict(
                        file=name,
                        kind=tag,
                        ray=int(ray),
                        direction=di,
                        orientation=qi,
                        r_A=float(r),
                        dmin_A=dmin,
                    )
                )

        # deepest contact well
        idx = df.groupby("ray")["E_int_kcal"].idxmin()
        wells = df.loc[idx].sort_values("E_int_kcal")
        deep = wells.iloc[0]
        pick(int(deep.ray), float(deep.r_A), "deepest_well")
        # softest soft well
        soft = df[df.r_A >= 3.4]
        idx_s = soft.groupby("ray")["E_int_kcal"].idxmin()
        soft_wells = soft.loc[idx_s].sort_values("E_int_kcal")
        softest = soft_wells.iloc[0]
        pick(int(softest.ray), float(softest.r_A), "softest_well")
        # shallowest soft well (least bound) for contrast
        shallow = soft_wells.iloc[-1]
        pick(int(shallow.ray), float(shallow.r_A), "shallow_soft")

    if sheet_paths:
        make_contact_sheet(sheet_paths, sheet_labels, args.out / "dimer_scan_povray_sheet.png", cols=4)
        print(f"wrote {args.out / 'dimer_scan_povray_sheet.png'}")

    (args.out / "manifest.json").write_text(json.dumps({"frames": manifest}, indent=2) + "\n")
    readme = args.out / "README.md"
    readme.write_text(
        "\n".join(
            [
                "# DCM–DCM dimer scan POV-Ray stills",
                "",
                "Visual check that multi-orientation COM scans are chemically sensible.",
                "",
                "| Asset | Content |",
                "|---|---|",
                "| `ori_d*_q*_r3.5.png` | Orientation grid at soft-well r≈3.5 Å |",
                "| `approach_d00_q00_r*.png` | One ray approaching from contact → MM region |",
                "| `deepest_well_*.png` / `softest_well_*.png` | Extremes from campaign CSV |",
                "| `dimer_scan_povray_sheet.png` | Contact sheet |",
                "",
                "Green = Cl, dark = C, light = H. `dmin` in titles is the shortest",
                "cross-monomer atom–atom distance (Å).",
                "",
                "Regenerate:",
                "```bash",
                "uv run python scripts/slurm/dense_dt_campaign/render_dimer_scan_povray.py",
                "```",
                "",
            ]
        )
    )
    print(f"done → {args.out} ({len(manifest)} frames)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
