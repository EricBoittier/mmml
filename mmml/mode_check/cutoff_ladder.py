"""COM distances that sample every hybrid ML/MM switch region."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class CutoffStation:
    """One labeled COM separation on the hybrid handoff ruler."""

    label: str
    com_A: float
    region: str
    description: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def cutoff_region_stations(
    *,
    ml_switch_width: float,
    mm_switch_on: float,
    mm_switch_width: float,
    beyond_pad_A: float = 4.0,
    ml_interior_floor_A: float = 3.5,
) -> list[CutoffStation]:
    """Return COM stations covering pure-ML → handoff → MM-tail → beyond.

    Boundaries (complementary handoff)::

        r_ml_edge   = mm_switch_on - ml_switch_width   # ML fully on for r ≤ this
        r_mm_on     = mm_switch_on                     # ML→0, MM→1
        r_mm_off    = mm_switch_on + mm_switch_width   # switched MM → 0

    Stations that would fall below ``ml_interior_floor_A`` are raised to that
    floor (unoriented templates clash at very short COM).
    """
    ml_w = float(ml_switch_width)
    mm_on = float(mm_switch_on)
    mm_w = float(mm_switch_width)
    if ml_w <= 0.0 or mm_on <= 0.0 or mm_w <= 0.0:
        raise ValueError("cutoff widths / mm_switch_on must be positive")

    r_ml_edge = mm_on - ml_w
    r_mm_off = mm_on + mm_w
    r_beyond = r_mm_off + float(beyond_pad_A)
    floor = float(ml_interior_floor_A)

    # Mid of pure-ML band [floor, r_ml_edge], clamped.
    if r_ml_edge <= floor:
        r_ml_interior = floor
    else:
        r_ml_interior = 0.5 * (floor + r_ml_edge)

    r_handoff_mid = 0.5 * (r_ml_edge + mm_on)
    r_mm_tail_mid = mm_on + 0.5 * mm_w

    raw: list[tuple[str, float, str, str]] = [
        (
            "ml_interior",
            r_ml_interior,
            "pure_ml",
            "Interior of ML-full region (s_ML≈1, MM off)",
        ),
        (
            "ml_edge",
            max(r_ml_edge, floor),
            "pure_ml_edge",
            "End of ML-full / start of complementary handoff",
        ),
        (
            "handoff_mid",
            r_handoff_mid,
            "handoff",
            "Mid complementary handoff (s_ML + s_MM = 1)",
        ),
        (
            "mm_switch_on",
            mm_on,
            "handoff_end",
            "Handoff end: ML→0, MM→1",
        ),
        (
            "mm_tail_mid",
            r_mm_tail_mid,
            "mm_tail",
            "Mid MM outer taper (ML off)",
        ),
        (
            "mm_off",
            r_mm_off,
            "mm_off",
            "Switched MM reaches zero",
        ),
        (
            "beyond",
            r_beyond,
            "beyond",
            "Past all switched two-body cutoffs (numerical / monomer-parity)",
        ),
    ]

    # Deduplicate by rounded COM (custom cutoffs can collapse stations).
    out: list[CutoffStation] = []
    seen: set[float] = set()
    for label, com, region, desc in raw:
        key = round(float(com), 6)
        if key in seen:
            continue
        seen.add(key)
        out.append(
            CutoffStation(
                label=label,
                com_A=float(com),
                region=region,
                description=desc,
            )
        )
    return out


def region_boundaries(
    *,
    ml_switch_width: float,
    mm_switch_on: float,
    mm_switch_width: float,
) -> dict[str, float]:
    """Named boundary radii for summaries / docs."""
    ml_w = float(ml_switch_width)
    mm_on = float(mm_switch_on)
    mm_w = float(mm_switch_width)
    return {
        "ml_switch_width": ml_w,
        "mm_switch_on": mm_on,
        "mm_switch_width": mm_w,
        "r_ml_edge_A": mm_on - ml_w,
        "r_mm_off_A": mm_on + mm_w,
    }
