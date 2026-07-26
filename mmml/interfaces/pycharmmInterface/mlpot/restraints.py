"""CHARMM restraints for non-PBC MLpot workflows (MMFP flat-bottom sphere, etc.)."""

from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np

_MMFP_GEO_ACTIVE = False
_DROFF_MARGIN_A = 1.0e-3

# NH3–CH3Cl ADUMB examples: half-harmonic outer walls on traced bond distances.
_ADUMB_RC_WALL_PAIRS: tuple[tuple[str, str], ...] = (
    ("CL1", "C1"),
    ("C1", "N1"),
)
# Activate MMFP walls below umbrella ``max`` so UM1RXN does not hard-abort first.
_DEFAULT_ADUMB_RC_WALL_MARGIN_A = 0.75


def _unique_atom_index_by_name(name: str) -> int:
    """Return 1-based CHARMM atom index for a unique IUPAC ``atype`` label."""
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
        f"    sele atom {int(atom_i)} end -",
        f"    sele atom {int(atom_j)} end",
    ]


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


@dataclass(frozen=True)
class AdumbRcGuard:
    """Track traced ADUMB distances vs umbrella ``max`` during overlap dynamics."""

    rcmax: float
    rcwall: float
    pairs: tuple[tuple[str, str], ...] = _ADUMB_RC_WALL_PAIRS
    wall_margin: float = _DEFAULT_ADUMB_RC_WALL_MARGIN_A

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
        dx = float(x[i - 1] - x[j - 1])
        dy = float(y[i - 1] - y[j - 1])
        dz = float(z[i - 1] - z[j - 1])
        key = f"{name1}-{name2}"
        out[key] = float(np.sqrt(dx * dx + dy * dy + dz * dz))
    return out


def check_adumb_rc_before_overlap_chunk(
    guard: AdumbRcGuard,
    *,
    overlap_context: str,
    chunk_index: int,
    n_chunks: int,
    warn_fraction: float = 0.92,
) -> None:
    """Fail fast when a traced RC is already at the umbrella hard limit."""
    dists = measure_adumb_rc_distances(guard.pairs)
    rcmax = float(guard.rcmax)
    worst_name = max(dists, key=dists.get)
    worst = float(dists[worst_name])
    label = (
        f"overlap ({overlap_context}) chunk {chunk_index + 1}/{n_chunks}"
    )
    if worst >= rcmax - 1.0e-4:
        raise RuntimeError(
            f"{label}: ADUMB reaction coordinate {worst_name}={worst:.3f} Å "
            f"is at or beyond umbrella max {rcmax:g} Å — UM1RXN would abort. "
            "Enable MMFP walls (default), increase adumrcmax, or restart from "
            "an earlier checkpoint."
        )
    warn_at = rcmax * float(warn_fraction)
    if worst >= warn_at:
        print(
            f"WARN: {label}: ADUMB RC {worst_name}={worst:.3f} Å "
            f"approaching umbrella max {rcmax:g} Å "
            f"(MMFP wall droff≈{guard.wall_droff():.2f} Å)",
            flush=True,
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


def _run_mmfp_charmm_script(script: str) -> None:
    """Execute MMFP lingo under MPI when md-system runs under mpirun."""
    try:
        from mmml.interfaces.pycharmmInterface.charmm_mpi import mpi_charmm_script

        mpi_charmm_script(script, barriers="none")
        return
    except Exception:
        pass
    pycharmm = _import_pycharmm()
    pycharmm.lingo.charmm_script(script)


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
    _run_mmfp_charmm_script(script)
    print("MMFP: distance wall install returned to Python", flush=True)
    global _MMFP_GEO_ACTIVE
    _MMFP_GEO_ACTIVE = True


def install_adumb_rxncor_distance_walls(
    *,
    rcmax: float,
    rcwall: float,
    pairs: tuple[tuple[str, str], ...] | None = None,
    wall_margin: float | None = None,
) -> None:
    """Install soft outer walls on traced RXNCOR distances before ``umbrella rxncor``.

    Uses multi-line ``MMFP GEO sphere RCM distance harmonic outside`` cards
    (see ``setup/charmm/test/c47test/omm_mmfp_rcm_distance.inp``).  ``droff`` is
    set below umbrella ``max`` so the wall activates before UM1RXN hard-aborts.
    """
    wall_pairs = pairs if pairs is not None else _ADUMB_RC_WALL_PAIRS
    droff = adumb_rc_wall_droff(rcmax, margin=wall_margin)
    walls: list[tuple[int, int, float, float]] = []
    for name1, name2 in wall_pairs:
        walls.append(
            (
                _unique_atom_index_by_name(name1),
                _unique_atom_index_by_name(name2),
                droff,
                float(rcwall),
            )
        )
    print(
        "MMFP: installing ADUMB RC distance walls "
        f"(droff={droff:g} Å, umbrella max={float(rcmax):g} Å, "
        f"force={float(rcwall):g})…",
        flush=True,
    )
    _run_mmfp_charmm_script(_mmfp_adumb_rc_distance_walls_script(tuple(walls)))
    print(f"MMFP: {len(wall_pairs)} ADUMB distance wall(s) installed", flush=True)
    global _MMFP_GEO_ACTIVE
    _MMFP_GEO_ACTIVE = True


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
