"""
Cutoff and switching parameters for ML/MM hybrid potentials.

Energy-conserving force shifting (what to consider)
--------------------------------------------------
For a single combined potential E_hybrid(r) that switches between ML and MM,
energy is conserved only if forces are F = -dE_hybrid/dR.

1. Complementary windows: ML two-body should go to 0 as MM goes to 1.
   Use s_ML(r) + s_MM(r) = 1 over the handoff region so there is no gap
   (double-counting or missing energy). Define:
     E_hybrid = s_ML(r) * E_ML + s_MM(r) * E_MM

2. Forces from E_hybrid (product rule):
     F = -dE_hybrid/dR = s_ML * F_ML + s_MM * F_MM + (E_MM - E_ML) * (ds_MM/dR)
   So you need:
   - scaled ML/MM forces: s_ML*F_ML, s_MM*F_MM
   - correction term: (E_MM - E_ML) * (ds_MM/dR)  [or equivalently -(E_MM - E_ML)*(ds_ML/dR)]
   Naive force blending F = s*F_ML + (1-s)*F_MM without that correction is
   not the gradient of any E_hybrid unless E_ML = E_MM in the switch zone.

3. Per-term switching (current style): ML and MM are switched separately
   (E_ML_sw = s_ML*E_ML, E_MM_sw = s_MM*E_MM). Total E = E_ML_sw + E_MM_sw is
   conservative only if forces are F = -d(E_ML_sw + E_MM_sw)/dR. That is
   already done when each term uses F = s*F_term - E_term*(ds/dR). For
   complementary handoff, use the same r-interval and s_MM = 1 - s_ML so
   E_hybrid = s_ML*E_ML + (1-s_ML)*E_MM has no overlap/gap.
"""
import warnings
from pathlib import Path

import numpy as np

# Switching-function exponents (canonical values used by both calculators).
# Defined here to avoid circular imports between cutoffs ↔ mmml_calculator.
GAMMA_ON: float = 1.0
GAMMA_OFF: float = 3.0


def handoff_widths_from_args(args) -> tuple[float, float, float]:
    """Return (ml_switch_width, mm_switch_on, mm_switch_width) from CLI/config namespace."""
    ml_w = getattr(args, "ml_switch_width", None)
    if ml_w is None:
        ml_w = getattr(args, "ml_cutoff", 0.1)
    mm_on = float(getattr(args, "mm_switch_on", 5.0))
    mm_w = getattr(args, "mm_switch_width", None)
    if mm_w is None:
        mm_w = getattr(args, "mm_cutoff", 1.0)
    return float(ml_w), mm_on, float(mm_w)


def _resolve_ml_switch_width(
    ml_switch_width: float,
    *,
    ml_cutoff: float | None = None,
    ml_cutoff_distance: float | None = None,
) -> float:
    if ml_cutoff is not None:
        return float(ml_cutoff)
    if ml_cutoff_distance is not None:
        return float(ml_cutoff_distance)
    return float(ml_switch_width)


def _resolve_mm_switch_width(
    mm_switch_width: float,
    *,
    mm_cutoff: float | None = None,
) -> float:
    if mm_cutoff is not None:
        return float(mm_cutoff)
    return float(mm_switch_width)


class CutoffParameters:
    """Parameters for ML/MM switching (widths and handoff distance)."""

    def __init__(
        self,
        ml_switch_width: float = 0.1,
        mm_switch_on: float = 5.0,
        mm_switch_width: float = 1.0,
        *,
        complementary_handoff: bool = True,
        # Deprecated aliases (same semantics as the canonical names above).
        ml_cutoff: float | None = None,
        mm_cutoff: float | None = None,
        ml_cutoff_distance: float | None = None,
    ):
        """
        Args:
            ml_switch_width: Width (Å) of the ML taper; handoff runs
                [mm_switch_on - ml_switch_width, mm_switch_on].
            mm_switch_on: Distance (Å) where ML reaches 0 and MM reaches 1
                in complementary mode.
            mm_switch_width: Width (Å) of the MM outer taper past mm_switch_on
                (and MM ramp width in legacy mode).
            complementary_handoff: If True, use s_MM = 1 - s_ML over the handoff
                interval [mm_switch_on - ml_switch_width, mm_switch_on].
        """
        if ml_cutoff is not None or ml_cutoff_distance is not None:
            warnings.warn(
                "ml_cutoff / ml_cutoff_distance are deprecated; use ml_switch_width",
                DeprecationWarning,
                stacklevel=2,
            )
        if mm_cutoff is not None:
            warnings.warn(
                "mm_cutoff is deprecated; use mm_switch_width",
                DeprecationWarning,
                stacklevel=2,
            )
        self.ml_switch_width = _resolve_ml_switch_width(
            ml_switch_width,
            ml_cutoff=ml_cutoff,
            ml_cutoff_distance=ml_cutoff_distance,
        )
        self.mm_switch_on = float(mm_switch_on)
        self.mm_switch_width = _resolve_mm_switch_width(
            mm_switch_width, mm_cutoff=mm_cutoff
        )
        self.complementary_handoff = complementary_handoff

    @property
    def ml_cutoff(self) -> float:
        """Deprecated alias for :attr:`ml_switch_width`."""
        return self.ml_switch_width

    @ml_cutoff.setter
    def ml_cutoff(self, value: float) -> None:
        self.ml_switch_width = float(value)

    @property
    def mm_cutoff(self) -> float:
        """Deprecated alias for :attr:`mm_switch_width`."""
        return self.mm_switch_width

    @mm_cutoff.setter
    def mm_cutoff(self, value: float) -> None:
        self.mm_switch_width = float(value)

    def __str__(self):
        return (
            f"CutoffParameters(ml_switch_width={self.ml_switch_width}, "
            f"mm_switch_on={self.mm_switch_on}, "
            f"mm_switch_width={self.mm_switch_width}, "
            f"complementary_handoff={self.complementary_handoff})"
        )

    def __repr__(self):
        return self.__str__()

    def __eq__(self, other):
        if not isinstance(other, CutoffParameters):
            return False
        ml_w_self = float(self.ml_switch_width)
        mm_on_self = float(self.mm_switch_on)
        mm_w_self = float(self.mm_switch_width)
        ml_w_other = float(other.ml_switch_width)
        mm_on_other = float(other.mm_switch_on)
        mm_w_other = float(other.mm_switch_width)
        return (
            ml_w_self == ml_w_other
            and mm_on_self == mm_on_other
            and mm_w_self == mm_w_other
            and getattr(self, "complementary_handoff", True)
            == getattr(other, "complementary_handoff", True)
        )

    def __ne__(self, other):
        return not self.__eq__(other)

    def __hash__(self):
        comp = getattr(self, "complementary_handoff", True)
        return hash(
            (
                float(self.ml_switch_width),
                float(self.mm_switch_on),
                float(self.mm_switch_width),
                comp,
            )
        )

    # --- Switching functions (must match mmml_calculator implementation) ---
    @staticmethod
    def _smoothstep01(s):
        return s * s * (3.0 - 2.0 * s)

    @staticmethod
    def _sharpstep(r, x0, x1, gamma=3.0):
        # Match _sharpstep in mmml_calculator: clip -> power -> smoothstep
        s = np.clip((r - x0) / np.maximum(x1 - x0, 1e-12), 0.0, 1.0)
        s = s ** gamma
        return CutoffParameters._smoothstep01(s)

    def ml_scale(self, r, gamma_ml: float = 5.0):
        """ML taper: 1→0 over [mm_switch_on - ml_switch_width, mm_switch_on]."""
        r = np.asarray(r, dtype=float)
        start = float(self.mm_switch_on) - float(self.ml_switch_width)
        stop = float(self.mm_switch_on)
        return 1.0 - self._sharpstep(r, start, stop, gamma=gamma_ml)

    def ml_mm_scales_complementary(self, r, gamma_ml: float = 5.0, gamma_mm_off: float = 3.0):
        """(s_ML, s_MM): s_ML + s_MM = 1 over handoff; s_MM tapers to 0 at mm_switch_on + mm_switch_width."""
        s_ml = self.ml_scale(r, gamma_ml=gamma_ml)
        handoff = 1.0 - s_ml
        mm_taper = 1.0 - self._sharpstep(
            np.asarray(r, dtype=float),
            float(self.mm_switch_on),
            float(self.mm_switch_on) + float(self.mm_switch_width),
            gamma=gamma_mm_off,
        )
        s_mm = handoff * np.asarray(mm_taper, dtype=float)
        return s_ml, s_mm

    def mm_scale_complementary(self, r, gamma_ml: float = 5.0, gamma_mm_off: float = 3.0):
        """MM scale: (1 - s_ML) over handoff, tapered to 0 at mm_switch_on + mm_switch_width."""
        _, s_mm = self.ml_mm_scales_complementary(
            r, gamma_ml=gamma_ml, gamma_mm_off=gamma_mm_off
        )
        return s_mm

    def mm_scale(self, r, gamma_on: float = 0.001, gamma_off: float = 3.0):
        """MM window (legacy): 0→1 over [mm_switch_on, mm_switch_on+mm_switch_width], then 1→0."""
        r = np.asarray(r, dtype=float)
        mm_on = self._sharpstep(
            r,
            float(self.mm_switch_on),
            float(self.mm_switch_on) + float(self.mm_switch_width),
            gamma=gamma_on,
        )
        mm_off = self._sharpstep(
            r,
            float(self.mm_switch_on) + float(self.mm_switch_width),
            float(self.mm_switch_on) + 2.0 * float(self.mm_switch_width),
            gamma=gamma_off,
        )
        return mm_on * (1.0 - mm_off)

    def to_dict(self):
        return {
            "ml_switch_width": self.ml_switch_width,
            "mm_switch_on": self.mm_switch_on,
            "mm_switch_width": self.mm_switch_width,
            "complementary_handoff": getattr(self, "complementary_handoff", True),
            # Legacy keys for saved configs
            "ml_cutoff": self.ml_switch_width,
            "mm_cutoff": self.mm_switch_width,
        }

    @classmethod
    def from_dict(cls, d):
        return cls(
            ml_switch_width=d.get(
                "ml_switch_width", d.get("ml_cutoff", d.get("ml_cutoff_distance", 0.1))
            ),
            mm_switch_on=d["mm_switch_on"],
            mm_switch_width=d.get("mm_switch_width", d.get("mm_cutoff", 1.0)),
            complementary_handoff=d.get("complementary_handoff", True),
        )

    def plot_cutoff_parameters(self, save_dir: Path | None = None):
        import matplotlib.pyplot as plt

        ml_w = float(self.ml_switch_width)
        mm_switch_on = float(self.mm_switch_on)
        mm_w = float(self.mm_switch_width)
        comp = getattr(self, "complementary_handoff", True)
        r0 = mm_switch_on - ml_w

        r_max = float(max(ml_w, mm_switch_on + 2.0 * mm_w) * 1.5 + 2.0)
        r = np.linspace(0.01, r_max, 600)

        ml_scale = self.ml_scale(r, gamma_ml=GAMMA_ON)
        mm_comp = self.mm_scale_complementary(r, gamma_ml=GAMMA_ON, gamma_mm_off=GAMMA_OFF)
        mm_legacy = self.mm_scale(r, gamma_on=GAMMA_ON, gamma_off=GAMMA_OFF)

        fig, ax = plt.subplots(1, 1, figsize=(9, 5))
        ax.plot(r, ml_scale, label=r"$s_{\mathrm{ML}}$", lw=2, color="C0")
        ax.plot(
            r,
            mm_comp,
            label=r"$s_{\mathrm{MM}}$ (complementary)" if comp else r"$s_{\mathrm{MM}}$",
            lw=2,
            color="C1",
        )
        ax.plot(
            r,
            ml_scale + mm_comp,
            "k--",
            lw=1.5,
            alpha=0.8,
            label=r"$s_{\mathrm{ML}}+s_{\mathrm{MM}}$",
        )
        ax.plot(r, mm_legacy, ":", lw=1.5, color="C1", alpha=0.7, label=r"MM (legacy)")

        ax.axvline(
            r0,
            color="C0",
            linestyle="--",
            lw=1,
            alpha=0.7,
            label=f"handoff start {r0:.2f} Å",
        )
        ax.axvline(
            mm_switch_on,
            color="k",
            linestyle="-.",
            lw=1.5,
            label=f"handoff end {mm_switch_on:.2f} Å",
        )
        ax.axvline(
            mm_switch_on + mm_w,
            color="C1",
            linestyle="--",
            lw=1,
            alpha=0.8,
            label=f"MM taper end {mm_switch_on + mm_w:.2f} Å",
        )

        ax.set_xlabel("COM distance r (Å)")
        ax.set_ylabel("Scale factor")
        ax.set_ylim(-0.05, 1.2)
        title = (
            "ML/MM handoff (complementary: s_MM→0 at mm_on+mm_width)"
            if comp
            else "ML/MM handoff (legacy)"
        )
        ax.set_title(
            f"{title} | ml_w={ml_w:.2f}, mm_on={mm_switch_on:.2f}, mm_w={mm_w:.2f}"
        )
        ax.legend(loc="best", fontsize=8)
        ax.grid(alpha=0.3)

        fig.tight_layout()
        out_dir = save_dir if save_dir is not None else Path.cwd()
        out_dir.mkdir(parents=True, exist_ok=True)
        suffix = "complementary" if comp else "legacy"
        out_path = (
            out_dir
            / f"cutoffs_schematic_{ml_w:.2f}_{mm_switch_on:.2f}_{mm_w:.2f}_{suffix}.png"
        )
        fig.savefig(out_path, dpi=150)
        return ax
