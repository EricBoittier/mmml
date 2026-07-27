"""CHARMM restraints for non-PBC MLpot workflows (MMFP flat-bottom sphere, etc.)."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

import numpy as np

_MMFP_GEO_ACTIVE = False
_NOE_RESTRAINTS_ACTIVE = False
_RESD_RESTRAINTS_ACTIVE = False
_DROFF_MARGIN_A = 1.0e-3
_MXCMSZ = 78
_CHARMM_FAILURE_MARKERS = (
    "unrecognized command",
    "not compiled",
    "bomlev",
    "terminating",
)

# NH3–CH3Cl ADUMB examples: half-harmonic outer walls on traced bond distances.
_ADUMB_RC_WALL_PAIRS: tuple[tuple[str, str], ...] = (
    ("CL1", "C1"),
    ("C1", "N1"),
)
_ADUMB_RC_WALL_PAIR_BY_NAME: dict[str, tuple[tuple[str, str], ...]] = {
    "rcl": (("CL1", "C1"),),
    "r_cl": (("CL1", "C1"),),
    "rcn": (("C1", "N1"),),
    "r_cn": (("C1", "N1"),),
    "r_nc": (("C1", "N1"),),
    # Combination / both-bond windows keep both component walls.
    "rdif": _ADUMB_RC_WALL_PAIRS,
    "rrat": _ADUMB_RC_WALL_PAIRS,
}
# Activate MMFP walls below umbrella ``max`` so UM1RXN does not hard-abort first.
_DEFAULT_ADUMB_RC_WALL_MARGIN_A = 0.75


def _unique_atom_index_by_name(name: str) -> int:
    """Return 0-based PSF atom index for a unique IUPAC ``atype`` label."""
    from pycharmm import select_atoms as sel

    target = str(name or "").strip()
    if not target:
        raise ValueError("atom name must be non-empty")
    atoms = sel.SelectAtoms()
    atoms.by_atom_type(target)
    indexes = [int(i) for i in atoms.get_atom_indexes()]
    if len(indexes) != 1:
        raise ValueError(
            f"expected exactly one atom named {target!r}, found {len(indexes)}"
        )
    return indexes[0]


def _charmm_lingo_atom_num(atom_index: int) -> int:
    """Map 0-based PSF index to 1-based ``sele atom N end`` numbering."""
    return int(atom_index) + 1


def adumb_rc_wall_pairs_for_name(umb_name: str | None) -> tuple[tuple[str, str], ...]:
    """Return RESD wall atom pairs for an ADUMB RXNCOR ``name`` token.

    Distance-only umbrellas (``rcl`` / ``rcn``) must wall **only** that bond.
    Capping both Cl–C and C–N to a tight ``rcl`` max (e.g. 4 Å) puts a huge
    RESDistance on a reactant C⋯N (~4 Å) and aborts before heat.
    """
    key = str(umb_name or "").strip().lower()
    if not key:
        return _ADUMB_RC_WALL_PAIRS
    return _ADUMB_RC_WALL_PAIR_BY_NAME.get(key, _ADUMB_RC_WALL_PAIRS)


def adumb_rc_wall_pairs_for_names(
    umb_names: list[str] | tuple[str, ...] | None,
) -> tuple[tuple[str, str], ...]:
    """Union of wall pairs for every umbrella name (2D ADUMB: ``rcl`` + ``rcn``).

    A single-name lookup kept only Cl–C walls for 2D scripts whose first card
    is ``name rcl``, leaving C–N free to exceed ``max`` → UM1RXN abort.
    """
    names = [str(n).strip().lower() for n in (umb_names or ()) if str(n).strip()]
    if not names:
        return _ADUMB_RC_WALL_PAIRS
    if len(names) == 1:
        return adumb_rc_wall_pairs_for_name(names[0])
    out: list[tuple[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for name in names:
        for pair in adumb_rc_wall_pairs_for_name(name):
            if pair not in seen:
                seen.add(pair)
                out.append(pair)
    return tuple(out) if out else _ADUMB_RC_WALL_PAIRS


def adumb_rc_wall_margin_A() -> float:
    """Inside offset (Å) for MMFP ``droff`` below umbrella ``max``."""
    raw = (os.environ.get("MMML_ADUMB_RC_WALL_MARGIN") or "").strip()
    if raw:
        try:
            margin = float(raw)
        except ValueError:
            margin = _DEFAULT_ADUMB_RC_WALL_MARGIN_A
    else:
        margin = _DEFAULT_ADUMB_RC_WALL_MARGIN_A
    return max(0.05, margin)


def adumb_rc_wall_droff(rcmax: float, *, margin: float | None = None) -> float:
    """MMFP outer-wall ``droff`` — strictly below UM1RXN ``umbmax``."""
    m = adumb_rc_wall_margin_A() if margin is None else max(0.05, float(margin))
    droff = float(rcmax) - m
    if droff <= 0:
        droff = 0.9 * float(rcmax)
    return droff


def _mmfp_rcm_distance_wall_block(
    atom_i: int,
    atom_j: int,
    *,
    droff: float,
    force: float,
) -> list[str]:
    """Return MMFP ``GEO sphere RCM distance`` lines (CHARMM c47 syntax)."""
    return [
        "GEO sphere RCM distance -",
        f"    harmonic outside force {float(force):g} droff {float(droff):g} -",
        f"    sele atom {_charmm_lingo_atom_num(atom_i)} end -",
        f"    sele atom {_charmm_lingo_atom_num(atom_j)} end",
    ]


def adumb_rc_walls_backend() -> str:
    """Return ``resd``, ``noe``, ``mmfp``, or ``off`` for traced-RC outer walls."""
    raw = (os.environ.get("MMML_ADUMB_RC_WALL_BACKEND") or "").strip().lower()
    if raw in ("0", "off", "none", "no"):
        return "off"
    if raw in ("mmfp",):
        return "mmfp"
    if raw in ("noe",):
        return "noe"
    if raw in ("resd", "resdistance"):
        return "resd"
    legacy = (os.environ.get("MMML_ADUMB_RC_MMFP_WALLS") or "").strip().lower()
    if legacy in ("1", "yes", "true", "on"):
        return "mmfp"
    if legacy in ("0", "no", "false", "off"):
        return "off"
    return "resd"


def adumb_rc_mmfp_walls_enabled() -> bool:
    """True when ADUMB RC walls use the MMFP backend (legacy env name)."""
    return adumb_rc_walls_backend() == "mmfp"


def adumb_rc_walls_enabled() -> bool:
    """True when any ADUMB RC outer-wall backend is active."""
    return adumb_rc_walls_backend() != "off"


def _noe_adumb_rc_distance_wall_assign(
    atom_i: int,
    atom_j: int,
    *,
    rmax: float,
    kmax: float,
) -> str:
    """One-line NOE ``assign`` (must fit ``mxcmsz``; no ``-`` continuations)."""
    card = (
        f"assi sele atom {_charmm_lingo_atom_num(atom_i)} end "
        f"sele atom {_charmm_lingo_atom_num(atom_j)} end "
        f"kmin 0 rmin 0 kmax {float(kmax):g} rmax {float(rmax):g}"
    )
    if len(card) > 78:
        raise ValueError(
            f"NOE assign card length {len(card)} exceeds mxcmsz (~80): {card[:60]}…"
        )
    return card


def _noe_adumb_rc_distance_walls_script(
    walls: tuple[tuple[int, int, float, float], ...],
) -> str:
    """NOE upper-bound restraints: flat for ``r <= rmax``, harmonic above."""
    lines = ["noe", "reset"]
    for atom_i, atom_j, rmax, kmax in walls:
        lines.append(
            _noe_adumb_rc_distance_wall_assign(
                atom_i, atom_j, rmax=rmax, kmax=kmax
            )
        )
    lines.append("end")
    return "\n".join(lines) + "\n"


def _resd_adumb_rc_distance_wall_line(
    atom_i: int,
    atom_j: int,
    *,
    rmax: float,
    kmax: float,
) -> str:
    """One-line ``RESDistance POSITIVE`` upper-bound wall (``mxcmsz`` safe).

    Uses ``BYNUMBER`` (``BYNu``) atom indices so tokens work regardless of segid
    naming (Packmol ``CLST``, ``--from-pdb`` ``SYS``, or truncated RESN fields).
    """
    i_num = _charmm_lingo_atom_num(atom_i)
    j_num = _charmm_lingo_atom_num(atom_j)
    card = (
        f"RESDistance KVAL {float(kmax):g} RVAL {float(rmax):g} POSITIVE "
        f"1.0 BYNU {i_num} {j_num}"
    )
    if len(card) > _MXCMSZ:
        raise ValueError(
            f"RESDistance card length {len(card)} exceeds mxcmsz (~80): {card[:60]}…"
        )
    return card


def _resd_adumb_rc_distance_wall_commands(
    walls: tuple[tuple[int, int, float, float], ...],
) -> list[str]:
    """Return single-line ``RESDistance`` commands for upper-bound walls."""
    commands = ["RESDistance RESEt"]
    for atom_i, atom_j, rmax, kmax in walls:
        commands.append(
            _resd_adumb_rc_distance_wall_line(
                atom_i, atom_j, rmax=rmax, kmax=kmax
            )
        )
    return commands


@dataclass(frozen=True)
class AdumbRcGuard:
    """Track traced ADUMB distances vs umbrella max during overlap dynamics.

    Optional ``umb_min`` / ``umb_max`` guard a combination RC
    xi = r(CL1-C1) - r(C1-N1) (Menshutkin difference). Component RESD walls
    alone do **not** keep xi inside a tight window like [-3, 3] when
    ``adumrcmax`` is ~8 A.
    """

    rcmax: float
    rcwall: float
    pairs: tuple[tuple[str, str], ...] = _ADUMB_RC_WALL_PAIRS
    wall_margin: float = _DEFAULT_ADUMB_RC_WALL_MARGIN_A
    umb_min: float | None = None
    umb_max: float | None = None
    # Rewind before UM1RXN hard-abort when |xi| is this close to the edge.
    xi_margin: float = 0.05

    def wall_droff(self) -> float:
        return adumb_rc_wall_droff(self.rcmax, margin=self.wall_margin)


def measure_adumb_rc_distances(
    pairs: tuple[tuple[str, str], ...] | None = None,
) -> dict[str, float]:
    """Return traced bond distances (Å) for ``(name1, name2)`` atom pairs."""
    out: dict[str, float] = {}
    for name1, name2 in pairs or _ADUMB_RC_WALL_PAIRS:
        i = _unique_atom_index_by_name(name1)
        j = _unique_atom_index_by_name(name2)
        x, y, z = _positions_xyz()
        dx = float(x[i] - x[j])
        dy = float(y[i] - y[j])
        dz = float(z[i] - z[j])
        key = f"{name1}-{name2}"
        out[key] = float(np.sqrt(dx * dx + dy * dy + dz * dz))
    return out


def measure_adumb_bond_difference_xi(
    pairs: tuple[tuple[str, str], ...] | None = None,
) -> float | None:
    """Return xi = r(CL1-C1) - r(C1-N1) if both distances exist."""
    dists = measure_adumb_rc_distances(pairs)
    d_cl = dists.get("CL1-C1")
    d_cn = dists.get("C1-N1")
    if d_cl is None or d_cn is None:
        return None
    return float(d_cl) - float(d_cn)


def check_adumb_rc_before_overlap_chunk(
    guard: AdumbRcGuard,
    *,
    overlap_context: str,
    chunk_index: int,
    n_chunks: int,
    warn_fraction: float = 0.92,
) -> None:
    """Fail fast when a traced RC is already at the umbrella hard limit (no restart ladder)."""
    if prepare_adumb_rc_before_overlap_chunk(
        guard,
        overlap_context=overlap_context,
        chunk_index=chunk_index,
        n_chunks=n_chunks,
        final_restart=None,
        warn_fraction=warn_fraction,
    ):
        raise RuntimeError("internal: prepare_adumb_rc returned retry without restart ladder")


def prepare_adumb_rc_before_overlap_chunk(
    guard: AdumbRcGuard,
    *,
    overlap_context: str,
    chunk_index: int,
    n_chunks: int,
    final_restart: "Path | None",
    warn_fraction: float = 0.92,
    max_recovery_lookback: int = 5,
    quiet_walls: bool = False,
) -> bool:
    """Reinstall RC walls, verify range, rewind from numbered restarts if needed.

    Returns ``True`` when the caller should retry the current overlap chunk
    (in-memory coords restored from an earlier ``heat.NNNN.res``).
    """
    from pathlib import Path

    label = f"overlap ({overlap_context}) chunk {chunk_index + 1}/{n_chunks}"
    rcmax = float(guard.rcmax)

    if adumb_rc_walls_enabled():
        install_adumb_rxncor_distance_walls(
            rcmax=guard.rcmax,
            rcwall=guard.rcwall,
            pairs=guard.pairs,
            wall_margin=guard.wall_margin,
            quiet=quiet_walls,
        )

    def _worst_distance() -> tuple[str, float]:
        dists = measure_adumb_rc_distances(guard.pairs)
        worst_name = max(dists, key=dists.get)
        return worst_name, float(dists[worst_name])

    def _xi_out_of_window() -> tuple[float, float, float] | None:
        """Return ``(xi, lo, hi)`` when ξ is outside the soft ADUMB window."""
        if guard.umb_min is None or guard.umb_max is None:
            return None
        xi = measure_adumb_bond_difference_xi(guard.pairs)
        if xi is None:
            return None
        margin = max(0.0, float(guard.xi_margin))
        lo = float(guard.umb_min) + margin
        hi = float(guard.umb_max) - margin
        if lo >= hi:
            lo = float(guard.umb_min)
            hi = float(guard.umb_max)
        if xi < lo or xi > hi:
            return float(xi), lo, hi
        return None

    xi_hit = _xi_out_of_window()
    worst_name, worst = _worst_distance()
    distance_ok = worst < rcmax - 1.0e-4
    if xi_hit is None and distance_ok:
        warn_at = rcmax * float(warn_fraction)
        if worst >= warn_at:
            print(
                f"WARN: {label}: ADUMB RC {worst_name}={worst:.3f} Å "
                f"approaching umbrella max {rcmax:g} Å "
                f"(wall rmax≈{guard.wall_droff():.2f} Å)",
                flush=True,
            )
        return False

    reason = (
        f"ξ=r(ClC)−r(CN)={xi_hit[0]:.3f} Å outside soft window "
        f"[{xi_hit[1]:g}, {xi_hit[2]:g}] (umbrella [{guard.umb_min:g}, {guard.umb_max:g}])"
        if xi_hit is not None
        else (
            f"ADUMB reaction coordinate {worst_name}={worst:.3f} Å "
            f"is at or beyond umbrella max {rcmax:g} Å"
        )
    )

    if final_restart is None:
        raise RuntimeError(
            f"{label}: {reason} — UM1RXN would abort. "
            "Widen umbrella min/max, tighten dynamics, or enable RC walls (default RESD)."
        )

    from mmml.interfaces.pycharmmInterface.mlpot.artifact_paths import (
        overlap_chunk_restart_path,
    )

    stage_restart = Path(final_restart)
    for back in range(1, max(1, int(max_recovery_lookback)) + 1):
        idx = int(chunk_index) - back
        if idx < 0:
            break
        candidate = overlap_chunk_restart_path(stage_restart, idx)
        if not candidate.is_file():
            continue
        print(
            f"ADUMB RC recovery: {reason}; "
            f"restoring {candidate.name} and retrying chunk…",
            flush=True,
        )
        from mmml.interfaces.pycharmmInterface.mlpot.bonded_mm_recovery import (
            restore_charmm_state_from_restart,
        )

        restore_charmm_state_from_restart(candidate)
        if adumb_rc_walls_enabled():
            install_adumb_rxncor_distance_walls(
                rcmax=guard.rcmax,
                rcwall=guard.rcwall,
                pairs=guard.pairs,
                wall_margin=guard.wall_margin,
                quiet=True,
            )
        xi_hit = _xi_out_of_window()
        worst_name, worst = _worst_distance()
        if xi_hit is None and worst < rcmax - 1.0e-4:
            print(
                f"ADUMB RC recovery: restored {candidate.name}; "
                f"{worst_name}={worst:.3f} Å (< {rcmax:g} Å)",
                flush=True,
            )
            return True
        reason = (
            f"ξ=r(ClC)−r(CN)={xi_hit[0]:.3f} Å outside soft window "
            f"[{xi_hit[1]:g}, {xi_hit[2]:g}]"
            if xi_hit is not None
            else f"{worst_name}={worst:.3f} Å ≥ {rcmax:g} Å"
        )

    raise RuntimeError(
        f"{label}: {reason} after rewind "
        f"(lookback={max_recovery_lookback} numbered restarts). "
        "Widen umbrella min/max / adumrcmax or restart from an earlier heat.NNNN.res."
    )


_ENERGY_VERIFY_TOL_KCAL = 1.0e-4
_DROFF_TUNE_MAX_ATTEMPTS = 8


@dataclass
class FlatBottomSphereConfig:
    """Flat-bottom spherical MMFP wall; inside ``radius`` has no restraint."""

    radius: float = 20.0
    force: float = 1.0
    xref: float = 0.0
    yref: float = 0.0
    zref: float = 0.0
    selection: str = "all"


def _import_pycharmm():
    import mmml.interfaces.pycharmmInterface.import_pycharmm  # noqa: F401 — CHARMM env
    import pycharmm

    return pycharmm


def _positions_xyz() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    import pycharmm.coor as coor

    return coor.get_positions_array()


def _set_positions_xyz(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> None:
    import pycharmm.coor as coor

    coor.set_positions_array(x, y, z)


def center_cluster_at_origin(*, orient: bool = False) -> None:
    """Translate so the cluster geometric COM is at the origin (non-PBC).

    Uses the PyCHARMM ``coor`` API rather than lingo ``coor …`` scripts.
    Some CHARMM/PyCHARMM builds reject ``coor`` as an unrecognized lingo
    command (LEVEL 0 warning, no Python exception) even when the coordinate
    API works — that previously left MMFP droff untuned and walls fighting
    elongated Packmol clouds. ``orient`` is off by default for the same reason.
    """
    if orient:
        pycharmm = _import_pycharmm()
        try:
            pycharmm.lingo.charmm_script("coor orient sele all end")
        except Exception:
            pass

    x, y, z = _positions_xyz()
    cx = float(np.mean(x))
    cy = float(np.mean(y))
    cz = float(np.mean(z))
    _set_positions_xyz(x - cx, y - cy, z - cz)


def setup_flat_bottom_sphere_mmfp(config: FlatBottomSphereConfig) -> None:
    """Install CHARMM MMFP flat-bottom sphere restraint (inside ``radius``: no force).

    Matches::

        MMFP
        GEO sphere harm -
            xref … yref … zref … -
            droff <radius> force <force> -
            sele all end
        END
    """
    if config.radius <= 0:
        raise ValueError(f"flat-bottom radius must be > 0, got {config.radius}")
    if config.force <= 0:
        raise ValueError(f"flat-bottom force must be > 0, got {config.force}")

    clear_mmfp_restraints()
    pycharmm = _import_pycharmm()
    sel = config.selection.strip() or "all"
    script = f"""
MMFP
GEO sphere harm -
    xref {float(config.xref):.6f} yref {float(config.yref):.6f} zref {float(config.zref):.6f} -
    droff {float(config.radius):.6f} force {float(config.force):.6f} -
    sele {sel} end
END
"""
    pycharmm.lingo.charmm_script(script)
    global _MMFP_GEO_ACTIVE
    _MMFP_GEO_ACTIVE = True


def _selected_max_radius(selection: str, *, xref: float, yref: float, zref: float) -> float | None:
    """Conservative max selected distance from the MMFP reference point."""
    sel = (selection or "all").strip() or "all"
    if sel.lower() == "all":
        try:
            x, y, z = _positions_xyz()
            r2 = (x - float(xref)) ** 2 + (y - float(yref)) ** 2 + (z - float(zref)) ** 2
            return float(np.sqrt(np.max(r2)))
        except Exception as exc:
            print(
                f"WARN: could not estimate MMFP droff from selection {sel!r}: {exc}",
                flush=True,
            )
            return None

    # Non-trivial selections: fall back to CHARMM ``coor stat`` substitutions.
    pycharmm = _import_pycharmm()
    try:
        pycharmm.lingo.charmm_script(f"coor stat sele {sel} end")
        xmin = float(pycharmm.lingo.get_energy_value("XMIN"))
        xmax = float(pycharmm.lingo.get_energy_value("XMAX"))
        ymin = float(pycharmm.lingo.get_energy_value("YMIN"))
        ymax = float(pycharmm.lingo.get_energy_value("YMAX"))
        zmin = float(pycharmm.lingo.get_energy_value("ZMIN"))
        zmax = float(pycharmm.lingo.get_energy_value("ZMAX"))
    except Exception as exc:
        print(
            f"WARN: could not estimate MMFP droff from selection {sel!r}: {exc}",
            flush=True,
        )
        return None

    dx = max(abs(xmin - float(xref)), abs(xmax - float(xref)))
    dy = max(abs(ymin - float(yref)), abs(ymax - float(yref)))
    dz = max(abs(zmin - float(zref)), abs(zmax - float(zref)))
    return float(np.sqrt(dx * dx + dy * dy + dz * dz))


def _skip_mmfp_energy_verify() -> bool:
    """Skip post-MMFP ``ENER`` on MPI-linked CHARMM (can hang after GEO install)."""
    import os

    flag = (os.environ.get("MMML_MMFP_ENERGY_VERIFY") or "").strip().lower()
    if flag in ("0", "no", "false", "off"):
        return True
    if flag in ("1", "yes", "true", "on"):
        return False
    try:
        from mmml.interfaces.pycharmmInterface.charmm_mpi import (
            _under_mpirun,
            charmm_lib_links_mpi,
        )

        return bool(charmm_lib_links_mpi() and _under_mpirun())
    except Exception:
        return False


def _current_charmm_energy_kcalmol() -> float | None:
    if _skip_mmfp_energy_verify():
        return None
    try:
        import pycharmm.energy as energy

        from mmml.interfaces.pycharmmInterface.charmm_levels import run_charmm_script_quiet

        print("MMFP: running CHARMM ENER for zero-wall check…", flush=True)
        run_charmm_script_quiet("ENER")
        row = energy.get_energy().iloc[0].to_dict()
        for key in ("ENER", "ENERgy", "ENERGY"):
            if key in row:
                return float(row[key])
    except Exception as exc:
        print(
            f"WARN: could not verify MMFP zero-energy install: {exc}",
            flush=True,
        )
    return None


def _energy_delta_after_install(before: float | None) -> float | None:
    if before is None:
        return None
    after = _current_charmm_energy_kcalmol()
    if after is None:
        return None
    return after - before


def _next_droff_increment(radius: float, attempt: int) -> float:
    base = max(0.05, 0.01 * float(radius))
    return base * (2 ** max(0, attempt - 1))


def _charmm_output_indicates_failure(log_text: str) -> str | None:
    """Return a short reason when captured CHARMM output shows install failure."""
    lower = str(log_text or "").lower()
    if "unrecognized command" in lower:
        return "CHARMM reported unrecognized command(s)"
    if "not compiled" in lower:
        return "required CHARMM module is not compiled"
    if "unrecognizable segid" in lower or "error in nxtatm" in lower:
        return "CHARMM could not parse restraint atom tokens"
    if "wrong number of atoms specified" in lower:
        return "CHARMM RESDistance atom syntax rejected"
    if "bomlev" in lower and "terminat" in lower:
        return "CHARMM aborted (BOMLEV)"
    return None


def _resd_restraint_count_from_log(log_text: str) -> int | None:
    """Parse the last ``RESDIST: Current number of restraints=`` line."""
    import re

    matches = re.findall(
        r"RESDIST:\s*Current number of restraints\s*=\s*(\d+)",
        str(log_text or ""),
        flags=re.IGNORECASE,
    )
    if not matches:
        return None
    return int(matches[-1])


def _skip_adumb_rc_wall_verify() -> bool:
    """Skip captured-output verify on MPI-linked CHARMM (RESD can trigger ENER)."""
    flag = (os.environ.get("MMML_ADUMB_RC_WALL_VERIFY") or "").strip().lower()
    if flag in ("0", "no", "false", "off"):
        return True
    if flag in ("1", "yes", "true", "on"):
        return False
    try:
        from mmml.interfaces.pycharmmInterface.charmm_mpi import (
            _under_mpirun,
            charmm_lib_links_mpi,
        )

        return bool(charmm_lib_links_mpi() and _under_mpirun())
    except Exception:
        return False


def _run_charmm_commands_verified(
    commands: list[str],
    *,
    label: str,
    verify: bool | None = None,
    expect_resd_restraints: int | None = None,
) -> None:
    """Run one CHARMM card per ``charmm_script`` call and fail on WRNLEV noise."""
    from pathlib import Path

    pycharmm = _import_pycharmm()
    if not commands:
        return
    do_verify = not _skip_adumb_rc_wall_verify() if verify is None else bool(verify)
    if do_verify:
        from mmml.interfaces.pycharmmInterface.charmm_levels import capture_fortran_stdio

        with capture_fortran_stdio() as tmp_path:
            for cmd in commands:
                pycharmm.lingo.charmm_script(cmd)
            log_text = Path(tmp_path).read_text(encoding="utf-8", errors="replace")
        try:
            Path(tmp_path).unlink(missing_ok=True)
        except OSError:
            pass
        reason = _charmm_output_indicates_failure(log_text)
        if reason is not None:
            snippet = next(
                (
                    line.strip()
                    for line in log_text.splitlines()
                    if any(
                        m in line.lower()
                        for m in _CHARMM_FAILURE_MARKERS
                        + ("nxtatm", "wrong number of atoms", "resdist")
                    )
                ),
                "",
            )
            detail = f" ({snippet})" if snippet else ""
            raise RuntimeError(f"{label}: {reason}{detail}")
        if expect_resd_restraints is not None:
            count = _resd_restraint_count_from_log(log_text)
            if count is not None:
                if count < int(expect_resd_restraints):
                    raise RuntimeError(
                        f"{label}: expected {expect_resd_restraints} RESDistance "
                        f"restraint(s), CHARMM reports {count}"
                    )
            elif "resdist:" in log_text.lower():
                raise RuntimeError(
                    f"{label}: could not parse RESDistance restraint count from "
                    "captured CHARMM output"
                )
            # MPI-linked builds often echo RESDIST to the terminal but not the
            # capture file — skip the count gate when the log is empty.
        return
    for cmd in commands:
        pycharmm.lingo.charmm_script(cmd)


def _run_charmm_lingo_block(script: str, *, label: str = "CHARMM") -> None:
    """Execute a multi-line CHARMM lingo block (NOE / MMFP).

    Uses ``pycharmm.lingo.charmm_script`` directly (same as flat-bottom MMFP).
    ``mpi_charmm_script`` uppercases the whole blob and can hang in ``NOESET`` /
    ``MMFP`` on MPI-linked builds when cards use ``-`` continuations.
    """
    from mmml.interfaces.pycharmmInterface.mlpot.cli_common import (
        split_charmm_lingo_commands,
    )

    commands = split_charmm_lingo_commands(script)
    _run_charmm_commands_verified(commands, label=label)


def _mmfp_adumb_rc_distance_walls_script(
    walls: tuple[tuple[int, int, float, float], ...],
) -> str:
    """Multi-line MMFP script installing all ADUMB RC outer walls."""
    lines = ["MMFP"]
    for atom_i, atom_j, droff, force in walls:
        lines.extend(
            _mmfp_rcm_distance_wall_block(
                atom_i,
                atom_j,
                droff=droff,
                force=force,
            )
        )
    lines.append("END")
    return "\n".join(lines) + "\n"


def setup_distance_wall_mmfp(
    sel1: str,
    sel2: str,
    *,
    max_dist: float,
    force: float,
    atom_i: int | None = None,
    atom_j: int | None = None,
) -> None:
    """Half-harmonic MMFP outer wall on an atom–atom distance (``droff`` = onset)."""
    if max_dist <= 0:
        raise ValueError(f"distance-wall droff must be > 0, got {max_dist}")
    if force <= 0:
        raise ValueError(f"distance-wall force must be > 0, got {force}")
    if atom_i is None or atom_j is None:
        atom_i = _unique_atom_index_by_name(sel1)
        atom_j = _unique_atom_index_by_name(sel2)
    script = _mmfp_adumb_rc_distance_walls_script(
        ((int(atom_i), int(atom_j), float(max_dist), float(force)),),
    )
    print(
        f"MMFP: GEO sphere RCM distance wall droff={float(max_dist):g} Å "
        f"(atoms {int(atom_i)}–{int(atom_j)}; {sel1} ↔ {sel2})…",
        flush=True,
    )
    _run_charmm_lingo_block(script)
    print("MMFP: distance wall install returned to Python", flush=True)
    global _MMFP_GEO_ACTIVE
    _MMFP_GEO_ACTIVE = True


def reinstall_adumb_rxncor_walls_from_workflow_args(args: Any) -> None:
    """Reapply NOE/MMFP outer walls after overlap rescue cleared them."""
    guard = getattr(args, "_adumb_rc_guard", None)
    if guard is None or not adumb_rc_walls_enabled():
        return
    install_adumb_rxncor_distance_walls(
        rcmax=guard.rcmax,
        rcwall=guard.rcwall,
        pairs=guard.pairs,
        wall_margin=guard.wall_margin,
        quiet=True,
    )


def install_adumb_rxncor_distance_walls(
    *,
    rcmax: float,
    rcwall: float,
    pairs: tuple[tuple[str, str], ...] | None = None,
    wall_margin: float | None = None,
    backend: str | None = None,
    quiet: bool = False,
) -> None:
    """Install soft outer walls on traced RXNCOR distances before ``umbrella rxncor``.

    Default backend is CHARMM **RESDistance POSITIVE** half-harmonic walls (works
    without the NOE module).  Legacy backends: ``noe`` (needs KEY_NOE), ``mmfp``
    (``MMML_ADUMB_RC_WALL_BACKEND=mmfp``; can hang on MPI-linked builds).
    ``rmax`` / ``droff`` is set below umbrella ``max`` so the wall activates
    before UM1RXN hard-aborts.
    """
    wall_backend = (backend or adumb_rc_walls_backend()).strip().lower()
    if wall_backend == "off":
        return

    wall_pairs = pairs if pairs is not None else _ADUMB_RC_WALL_PAIRS
    rmax_wall = adumb_rc_wall_droff(rcmax, margin=wall_margin)
    walls: list[tuple[int, int, float, float]] = []
    for name1, name2 in wall_pairs:
        walls.append(
            (
                _unique_atom_index_by_name(name1),
                _unique_atom_index_by_name(name2),
                rmax_wall,
                float(rcwall),
            )
        )
    if wall_backend == "mmfp":
        if not quiet:
            print(
                "MMFP: installing ADUMB RC distance walls "
                f"(droff={rmax_wall:g} Å, umbrella max={float(rcmax):g} Å, "
                f"force={float(rcwall):g})…",
                flush=True,
            )
        _run_charmm_lingo_block(
            _mmfp_adumb_rc_distance_walls_script(tuple(walls)),
            label="MMFP ADUMB RC walls",
        )
        if not quiet:
            print(f"MMFP: {len(wall_pairs)} ADUMB distance wall(s) installed", flush=True)
        global _MMFP_GEO_ACTIVE
        _MMFP_GEO_ACTIVE = True
        return

    if wall_backend == "noe":
        if not quiet:
            print(
                "NOE: installing ADUMB RC upper-bound walls "
                f"(rmax={rmax_wall:g} Å, umbrella max={float(rcmax):g} Å, "
                f"kmax={float(rcwall):g})…",
                flush=True,
            )
        _run_charmm_lingo_block(
            _noe_adumb_rc_distance_walls_script(tuple(walls)),
            label="NOE ADUMB RC walls",
        )
        if not quiet:
            print(f"NOE: {len(wall_pairs)} ADUMB distance wall(s) installed", flush=True)
        global _NOE_RESTRAINTS_ACTIVE
        _NOE_RESTRAINTS_ACTIVE = True
        return

    if not quiet:
        print(
            "RESD: installing ADUMB RC upper-bound walls "
            f"(rmax={rmax_wall:g} Å, umbrella max={float(rcmax):g} Å, "
            f"kmax={float(rcwall):g})…",
            flush=True,
        )
    _run_charmm_commands_verified(
        _resd_adumb_rc_distance_wall_commands(tuple(walls)),
        label="RESD ADUMB RC walls",
        verify=True,
        expect_resd_restraints=len(wall_pairs),
    )
    if not quiet:
        print(f"RESD: {len(wall_pairs)} ADUMB distance wall(s) installed", flush=True)
    global _RESD_RESTRAINTS_ACTIVE
    _RESD_RESTRAINTS_ACTIVE = True


def clear_noe_restraints() -> None:
    """Remove NOE restraints (safe to call if none were defined)."""
    global _NOE_RESTRAINTS_ACTIVE
    if not _NOE_RESTRAINTS_ACTIVE:
        return
    _run_charmm_lingo_block("noe\nreset\nend\n", label="NOE clear")
    _NOE_RESTRAINTS_ACTIVE = False


def clear_resd_restraints() -> None:
    """Remove RESDistance restraints (safe to call if none were defined)."""
    global _RESD_RESTRAINTS_ACTIVE
    if not _RESD_RESTRAINTS_ACTIVE:
        return
    _run_charmm_commands_verified(
        ["RESDistance RESEt"],
        label="RESD clear",
        verify=False,
    )
    _RESD_RESTRAINTS_ACTIVE = False


def clear_adumb_rxncor_restraints() -> None:
    """Remove ADUMB RC outer walls (RESD, NOE, and/or MMFP)."""
    clear_resd_restraints()
    clear_noe_restraints()
    clear_mmfp_restraints()


def clear_mmfp_restraints() -> None:
    """Remove MMFP terms (safe to call if none were defined)."""
    global _MMFP_GEO_ACTIVE
    if not _MMFP_GEO_ACTIVE:
        return
    pycharmm = _import_pycharmm()
    pycharmm.lingo.charmm_script(
        """
MMFP
GEO RESET
END
"""
    )
    _MMFP_GEO_ACTIVE = False


def apply_flat_bottom_workflow(
    *,
    radius: float | None,
    force: float = 1.0,
    center_at_origin: bool = True,
    xref: float = 0.0,
    yref: float = 0.0,
    zref: float = 0.0,
    selection: str = "all",
) -> FlatBottomSphereConfig | None:
    """Optionally center the cluster and set up MMFP flat-bottom sphere."""
    if radius is None or radius <= 0:
        return None
    print(
        f"MMFP: installing flat-bottom sphere (requested droff={float(radius):.2f} Å)…",
        flush=True,
    )
    if center_at_origin:
        print("MMFP: centering cluster COM at origin…", flush=True)
        center_cluster_at_origin()
    skip_ener = _skip_mmfp_energy_verify()
    if skip_ener:
        print(
            "MMFP: skipping CHARMM ENER zero-wall verify "
            "(MPI-linked CHARMM under mpirun; set MMML_MMFP_ENERGY_VERIFY=1 to force)",
            flush=True,
        )
        energy_before = None
    else:
        energy_before = _current_charmm_energy_kcalmol()
    requested_radius = float(radius)
    current_radius = _selected_max_radius(selection, xref=xref, yref=yref, zref=zref)
    effective_radius = requested_radius
    if current_radius is not None:
        effective_radius = max(requested_radius, current_radius + _DROFF_MARGIN_A)
        if effective_radius > requested_radius:
            print(
                "MMFP flat-bottom droff adjusted "
                f"{requested_radius:.3f} -> {effective_radius:.3f} Å "
                f"so initial {selection!r} wall energy is zero",
                flush=True,
            )
    cfg = FlatBottomSphereConfig(
        radius=effective_radius,
        force=float(force),
        xref=xref,
        yref=yref,
        zref=zref,
        selection=selection,
    )
    for attempt in range(1, _DROFF_TUNE_MAX_ATTEMPTS + 1):
        print(
            f"MMFP: GEO sphere harm droff={cfg.radius:.3f} Å "
            f"(attempt {attempt}/{_DROFF_TUNE_MAX_ATTEMPTS})…",
            flush=True,
        )
        setup_flat_bottom_sphere_mmfp(cfg)
        print("MMFP: GEO install returned to Python", flush=True)
        delta = _energy_delta_after_install(energy_before)
        if delta is None:
            print(
                "MMFP: flat-bottom installed "
                f"(droff={cfg.radius:.2f} Å, force={cfg.force:.2f}; no ENER verify)",
                flush=True,
            )
            return cfg
        if abs(delta) <= _ENERGY_VERIFY_TOL_KCAL:
            print(
                f"MMFP flat-bottom zero-energy check OK: ΔE={delta:+.6f} kcal/mol",
                flush=True,
            )
            return cfg
        if attempt == _DROFF_TUNE_MAX_ATTEMPTS:
            print(
                "WARN: MMFP flat-bottom changed energy at install "
                f"by {delta:+.6f} kcal/mol after {attempt} droff tuning attempt(s) "
                f"(droff={cfg.radius:.6f} Å)",
                flush=True,
            )
            return cfg
        old_radius = cfg.radius
        cfg.radius = old_radius + _next_droff_increment(old_radius, attempt)
        print(
            "MMFP flat-bottom ΔE not zero "
            f"({delta:+.6f} kcal/mol); increasing droff "
            f"{old_radius:.3f} -> {cfg.radius:.3f} Å and retrying",
            flush=True,
        )
    return cfg
