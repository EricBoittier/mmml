"""Bake learned per-type LJ scales into a CGenFF ``.prm`` NONBONDED section.

Why this exists
---------------
``hybrid_mm.json`` carries per-CGenFF-type multiplicative LJ scales fitted by
hybrid ML/MM training. Deploying them in MD has so far required the JAX MM
path, because that is where the scales are applied::

    sigma_eff[t] = master_sigmas[t]  * sigma_scale[t]
    eps_eff[t]   = master_epsilons[t] * epsilon_scale[t]

That leaves ``mm_nonbond_mode=periodic_external`` -- the only full-box PME path
for large boxes -- unable to consume a trained sidecar, because there CHARMM's
IMAGE VDW is the LJ backend and never sees the scales (``hybrid_mlpot`` sets
``do_mm = include_mm and not periodic_mode``).

Rewriting the parameter file closes that gap without a JAX VDW kernel, and does
so *exactly* rather than approximately. The scales are per **type**, applied to
the master tables **before** any pair combining, and the CGenFF reader defines::

    master_sigmas[t] = rmin_half[t] * 2 / 2**(1/6)
    master_epsilons[t] = abs(epsilon[t])

Because both relations are linear in the stored column, scaling ``Rmin/2`` by
``sigma_scale`` and ``epsilon`` by ``epsilon_scale`` reproduces the JAX
effective tables identically -- and CHARMM then applies its usual
Lorentz-Berthelot combining to already-scaled per-type values, exactly as the
JAX path does. ``test_scaled_cgenff_prm.py`` asserts this round-trip through the
production parser rather than trusting the algebra.

1-4 parameters (the optional trailing three columns) are **not** scaled. They
were never part of the fit, and they only affect intramolecular 1-4 pairs, which
are inside the ML region and never contribute to the intermolecular MM energy
that ``periodic_external`` computes. Pass ``scale_14=True`` to override.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping

__all__ = [
    "ScaledPrmStats",
    "default_parameter_files",
    "deploy_scaled_lj_into_charmm",
    "scale_nonbonded_block",
    "write_scaled_cgenff_prm",
]

# A NONBONDED parameter line: type, ignored, -eps, Rmin/2 [, ignored, -eps14, Rmin14/2]
_NB_LINE = re.compile(
    r"^(?P<lead>\s*)(?P<type>[A-Za-z0-9_+\-*']+)(?P<gap>\s+)(?P<rest>[-0-9.eE\s]+?)"
    r"(?P<tail>\s*(?:!.*)?)$"
)


@dataclass
class ScaledPrmStats:
    scaled: list[str] = field(default_factory=list)
    unscaled: list[str] = field(default_factory=list)
    missing_from_prm: list[str] = field(default_factory=list)

    def summary(self) -> str:
        return (
            f"{len(self.scaled)} types scaled, {len(self.unscaled)} left unchanged, "
            f"{len(self.missing_from_prm)} sidecar types not present in the prm"
        )


def scale_nonbonded_block(
    text: str,
    sigma_scale: Mapping[str, float],
    epsilon_scale: Mapping[str, float],
    *,
    scale_14: bool = False,
) -> tuple[str, ScaledPrmStats]:
    """Return ``(new_text, stats)`` with NONBONDED entries scaled per type.

    Only the NONBONDED block is touched; BONDS/ANGLES/DIHEDRALS/NBFIX/HBOND and
    every comment are passed through byte-for-byte. Types absent from the scale
    maps are left exactly as they were, so a partial sidecar is safe.
    """
    lines = text.splitlines(keepends=True)
    out: list[str] = []
    stats = ScaledPrmStats()
    seen: set[str] = set()

    in_nb = False
    skip_continuation = False
    for raw in lines:
        stripped = raw.strip()
        upper = stripped.upper()

        if upper.startswith("NONBONDED"):
            in_nb = True
            # The NONBONDED header may continue onto following lines with `-`.
            skip_continuation = stripped.endswith("-")
            out.append(raw)
            continue

        if in_nb and skip_continuation:
            skip_continuation = stripped.endswith("-")
            out.append(raw)
            continue

        # Any other section keyword ends the NONBONDED block.
        if in_nb and re.match(
            r"^(NBFIX|HBOND|BONDS|ANGLES|DIHEDRALS|IMPROPER|CMAP|THOLE|END)\b",
            upper,
        ):
            in_nb = False

        if not in_nb or not stripped or stripped.startswith("!"):
            out.append(raw)
            continue

        m = _NB_LINE.match(raw.rstrip("\n"))
        if m is None:
            out.append(raw)
            continue

        atom_type = m.group("type")
        nums = m.group("rest").split()
        # 4 numeric columns total => [ignored, eps, rmin_half]; 7 => plus 1-4.
        if len(nums) not in (3, 6):
            out.append(raw)
            continue

        s = float(sigma_scale.get(atom_type, 1.0))
        e = float(epsilon_scale.get(atom_type, 1.0))
        seen.add(atom_type)
        if s == 1.0 and e == 1.0:
            stats.unscaled.append(atom_type)
            out.append(raw)
            continue

        new_nums = list(nums)
        # Column 1 is epsilon (negative by CHARMM convention); 2 is Rmin/2.
        # 12 decimals, not 6: at 6 the rounding alone put a ~6e-5 relative error
        # into the deployed LJ, which is a real (if small) physics change and
        # defeats the whole point of an *exact* rewrite. Fixed-point rather than
        # %g so no value can come out in scientific notation, which CHARMM's
        # parameter reader is not guaranteed to accept.
        new_nums[1] = f"{float(nums[1]) * e:.12f}"
        new_nums[2] = f"{float(nums[2]) * s:.12f}"
        if scale_14 and len(nums) == 6:
            new_nums[4] = f"{float(nums[4]) * e:.12f}"
            new_nums[5] = f"{float(nums[5]) * s:.12f}"

        newline = "\n" if raw.endswith("\n") else ""
        rebuilt = (
            f"{m.group('lead')}{atom_type}{m.group('gap')}"
            f"{'  '.join(new_nums)}{m.group('tail')}{newline}"
        )
        out.append(rebuilt)
        stats.scaled.append(atom_type)

    stats.missing_from_prm = sorted(
        {t for t in set(sigma_scale) | set(epsilon_scale) if t not in seen}
    )
    return "".join(out), stats


def _scale_maps(sidecar: str | Path) -> tuple[dict[str, float], dict[str, float]]:
    from mmml.models.mm_lj_scales import load_mm_lj_scales_sidecar

    payload = load_mm_lj_scales_sidecar(Path(sidecar))
    if payload is None:
        raise ValueError(f"no learnable LJ scales in {sidecar}")
    names = payload["cgenff_type_names"]
    return (
        {str(n): float(v) for n, v in zip(names, payload["mm_lj_sigma_scale"])},
        {str(n): float(v) for n, v in zip(names, payload["mm_lj_epsilon_scale"])},
    )


def default_parameter_files() -> list[Path]:
    """Every file contributing to the master LJ tables, in load order.

    The bare CGenFF ``.prm`` is not enough: ``load_reference`` merges CHARMM
    stream files on top of it, and those carry chemistry the DES sets actually
    use -- TIP3 water ions plus the noble gases HE/NE/AR/KR/XE, several of which
    receive trained scales. Rewriting only ``par_all36_cgenff.prm`` would deploy
    those types **unscaled** while appearing to succeed.
    """
    from mmml.data.cgenff_dataset import DEF_PRM_PATH, DEF_EXTRA_TOPPAR

    return [Path(DEF_PRM_PATH), *(Path(p) for p in DEF_EXTRA_TOPPAR)]


def write_scaled_cgenff_prm(
    sidecar: str | Path,
    out_dir: str | Path,
    *,
    base_files: list[str | Path] | None = None,
    scale_14: bool = False,
    overwrite: bool = False,
    require_all_scaled: bool = True,
) -> dict[Path, ScaledPrmStats]:
    """Write scaled copies of every CGenFF parameter file into ``out_dir``.

    Returns ``{output_path: stats}``. Output basenames match the inputs, so the
    set can be read back in the same order as the originals.

    With ``require_all_scaled`` (the default) a type carrying a non-unit scale
    that appears in *none* of the files raises, rather than silently deploying
    the unscaled parameter -- the failure this function exists to prevent.
    """
    out_dir = Path(out_dir)
    sigma_map, epsilon_map = _scale_maps(sidecar)

    files = [Path(p) for p in (base_files if base_files is not None
                               else default_parameter_files())]

    results: dict[Path, ScaledPrmStats] = {}
    scaled_anywhere: set[str] = set()
    seen_anywhere: set[str] = set()

    out_dir.mkdir(parents=True, exist_ok=True)
    for src in files:
        dst = out_dir / src.name
        if dst.exists() and not overwrite:
            raise FileExistsError(f"refusing to overwrite {dst} (pass overwrite=True)")
        text, stats = scale_nonbonded_block(
            src.read_text(), sigma_map, epsilon_map, scale_14=scale_14
        )
        dst.write_text(text)
        results[dst] = stats
        scaled_anywhere.update(stats.scaled)
        seen_anywhere.update(stats.scaled)
        seen_anywhere.update(stats.unscaled)

    if require_all_scaled:
        wanted = {
            t for t in set(sigma_map) | set(epsilon_map)
            if sigma_map.get(t, 1.0) != 1.0 or epsilon_map.get(t, 1.0) != 1.0
        }
        missing = sorted(wanted - scaled_anywhere)
        if missing:
            raise ValueError(
                f"{len(missing)} type(s) carry a non-unit LJ scale but were not "
                f"found in any parameter file, so their trained LJ would be "
                f"silently dropped: {', '.join(missing[:12])}"
                + (" ..." if len(missing) > 12 else "")
                + f"\nfiles searched: {', '.join(p.name for p in files)}"
            )
    return results


def deploy_scaled_lj_into_charmm(
    sidecar: str | Path,
    *,
    out_dir: str | Path | None = None,
    scale_14: bool = False,
    verbose: bool = True,
) -> dict[Path, ScaledPrmStats]:
    """Bake ``sidecar``'s LJ scales into CHARMM's live parameters.

    This is what lets ``mm_nonbond_mode=periodic_external`` honour a trained
    sidecar. There CHARMM IMAGE VDW is the LJ backend and the JAX MM pair loop
    is off (``do_mm = include_mm and not periodic_mode``), so per-type scales
    can only reach the energy through the parameter file itself.

    Writes scaled copies of every CGenFF parameter file and reads them back with
    ``append=True`` so the NONBONDED entries override the ones already loaded.
    Returns the per-file stats.
    """
    import tempfile

    from mmml.interfaces.pycharmmInterface.nbonds_config import read_cgenff_prm

    if out_dir is None:
        out_dir = Path(tempfile.mkdtemp(prefix="mmml-scaled-lj-"))
    results = write_scaled_cgenff_prm(
        sidecar, out_dir, scale_14=scale_14, overwrite=True
    )

    # Base file first, then streams, matching the original load order so later
    # files still override earlier ones exactly as they do without scaling.
    order = {p.name: i for i, p in enumerate(default_parameter_files())}
    for path in sorted(results, key=lambda p: order.get(p.name, 999)):
        read_cgenff_prm(path, append=True)

    if verbose:
        total = sum(len(s.scaled) for s in results.values())
        print(
            f"[lj-scales] deployed {total} scaled CGenFF types into CHARMM "
            f"from {Path(sidecar).name} ({len(results)} parameter files, "
            f"written to {out_dir})",
            flush=True,
        )
    return results
