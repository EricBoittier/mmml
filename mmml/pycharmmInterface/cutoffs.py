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
import numpy as np
from pathlib import Path

from mmml.pycharmmInterface.mmml_calculator import GAMMA_ON, GAMMA_OFF

class CutoffParameters:
    """Parameters for ML and MM cutoffs and switching functions"""
    def __init__(
        self,
        ml_cutoff: float = 2.0,
        mm_switch_on: float = 5.0,
        mm_cutoff: float = 1.0,
        *,
        complementary_handoff: bool = True,
    ):
        """
        Args:
            ml_cutoff: Distance where ML potential is cut off (width of ML taper)
            mm_switch_on: Distance where ML reaches 0 and MM reaches 1 in complementary mode
            mm_cutoff: Width of MM ramp in legacy mode; unused when complementary_handoff=True
            complementary_handoff: If True, use s_MM = 1 - s_ML over [mm_switch_on - ml_cutoff, mm_switch_on]
        """
        self.ml_cutoff = ml_cutoff
        self.mm_switch_on = mm_switch_on
        self.mm_cutoff = mm_cutoff
        self.complementary_handoff = complementary_handoff

    def __str__(self):
        return f"CutoffParameters(ml_cutoff={self.ml_cutoff}, mm_switch_on={self.mm_switch_on}, mm_cutoff={self.mm_cutoff}, complementary_handoff={self.complementary_handoff})"
    
    def __repr__(self):
        return self.__str__()

    def __eq__(self, other):
        if not isinstance(other, CutoffParameters):
            return False
        ml_cutoff_self = float(self.ml_cutoff) if hasattr(self.ml_cutoff, '__float__') else self.ml_cutoff
        mm_switch_on_self = float(self.mm_switch_on) if hasattr(self.mm_switch_on, '__float__') else self.mm_switch_on
        mm_cutoff_self = float(self.mm_cutoff) if hasattr(self.mm_cutoff, '__float__') else self.mm_cutoff
        ml_cutoff_other = float(other.ml_cutoff) if hasattr(other.ml_cutoff, '__float__') else other.ml_cutoff
        mm_switch_on_other = float(other.mm_switch_on) if hasattr(other.mm_switch_on, '__float__') else other.mm_switch_on
        mm_cutoff_other = float(other.mm_cutoff) if hasattr(other.mm_cutoff, '__float__') else other.mm_cutoff
        return (ml_cutoff_self == ml_cutoff_other and
                mm_switch_on_self == mm_switch_on_other and
                mm_cutoff_self == mm_cutoff_other and
                getattr(self, "complementary_handoff", True) == getattr(other, "complementary_handoff", True))
    
    def __ne__(self, other):
        return not self.__eq__(other)
    
    def __hash__(self):
        ml_cutoff_val = float(self.ml_cutoff) if hasattr(self.ml_cutoff, '__float__') else self.ml_cutoff
        mm_switch_on_val = float(self.mm_switch_on) if hasattr(self.mm_switch_on, '__float__') else self.mm_switch_on
        mm_cutoff_val = float(self.mm_cutoff) if hasattr(self.mm_cutoff, '__float__') else self.mm_cutoff
        comp = getattr(self, "complementary_handoff", True)
        return hash((ml_cutoff_val, mm_switch_on_val, mm_cutoff_val, comp))

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
        """ML taper: 1→0 over [mm_switch_on - ml_cutoff, mm_switch_on]."""
        r = np.asarray(r, dtype=float)
        start = float(self.mm_switch_on) - float(self.ml_cutoff)
        stop = float(self.mm_switch_on)
        return 1.0 - self._sharpstep(r, start, stop, gamma=gamma_ml)

    def ml_mm_scales_complementary(self, r, gamma_ml: float = 5.0):
        """(s_ML, s_MM) with s_ML + s_MM = 1 over the handoff.
        ML 1→0 and MM 0→1 over [mm_switch_on - ml_cutoff, mm_switch_on].
        Use when building E_hybrid = s_ML*E_ML + s_MM*E_MM for energy-conserving switching."""
        s_ml = self.ml_scale(r, gamma_ml=gamma_ml)
        return s_ml, 1.0 - s_ml

    def mm_scale_complementary(self, r, gamma_ml: float = 5.0):
        """MM scale with s_MM = 1 - s_ML over handoff [mm_switch_on - ml_cutoff, mm_switch_on]."""
        s_ml, s_mm = self.ml_mm_scales_complementary(r, gamma_ml=gamma_ml)
        return s_mm

    def mm_scale(self, r, gamma_on: float = 0.001, gamma_off: float = 3.0):
        """MM window (legacy): 0→1 over [mm_switch_on, mm_switch_on+mm_cutoff], then 1→0."""
        r = np.asarray(r, dtype=float)
        mm_on = self._sharpstep(r,
                                float(self.mm_switch_on),
                                float(self.mm_switch_on) + float(self.mm_cutoff),
                                gamma=gamma_on)
        mm_off = self._sharpstep(r,
                                 float(self.mm_switch_on) + float(self.mm_cutoff),
                                 float(self.mm_switch_on) + 2.0 * float(self.mm_cutoff),
                                 gamma=gamma_off)
        return mm_on * (1.0 - mm_off)

    def to_dict(self):
        return {
            "ml_cutoff": self.ml_cutoff,
            "mm_switch_on": self.mm_switch_on,
            "mm_cutoff": self.mm_cutoff,
            "complementary_handoff": getattr(self, "complementary_handoff", True),
        }

    def from_dict(self, d):
        return CutoffParameters(
            ml_cutoff=d["ml_cutoff"],
            mm_switch_on=d["mm_switch_on"],
            mm_cutoff=d["mm_cutoff"],
            complementary_handoff=d.get("complementary_handoff", True),
        )

    def plot_cutoff_parameters(self, save_dir: Path | None = None):
        import numpy as np
        import matplotlib.pyplot as plt

        ml_cutoff = float(self.ml_cutoff)
        mm_switch_on = float(self.mm_switch_on)
        mm_cutoff = float(self.mm_cutoff)
        comp = getattr(self, "complementary_handoff", True)
        r0 = mm_switch_on - ml_cutoff

        r_max = float(max(ml_cutoff, mm_switch_on + 2.0 * mm_cutoff) * 1.5 + 2.0)
        r = np.linspace(0.01, r_max, 600)

        ml_scale = self.ml_scale(r, gamma_ml=GAMMA_ON)
        mm_comp = self.mm_scale_complementary(r, gamma_ml=GAMMA_ON)
        mm_legacy = self.mm_scale(r, gamma_on=GAMMA_ON, gamma_off=GAMMA_OFF)

        fig, ax = plt.subplots(1, 1, figsize=(9, 5))
        ax.plot(r, ml_scale, label=r"$s_{\mathrm{ML}}$", lw=2, color="C0")
        ax.plot(r, mm_comp, label=r"$s_{\mathrm{MM}}$ (complementary)" if comp else r"$s_{\mathrm{MM}}$", lw=2, color="C1")
        ax.plot(r, ml_scale + mm_comp, "k--", lw=1.5, alpha=0.8, label=r"$s_{\mathrm{ML}}+s_{\mathrm{MM}}$")
        ax.plot(r, mm_legacy, ":", lw=1.5, color="C1", alpha=0.7, label=r"MM (legacy)")

        ax.axvline(r0, color="C0", linestyle="--", lw=1, alpha=0.7, label=f"handoff start {r0:.2f} Å")
        ax.axvline(mm_switch_on, color="k", linestyle="-.", lw=1.5, label=f"handoff end {mm_switch_on:.2f} Å")
        ax.axvline(mm_switch_on + mm_cutoff, color="gray", linestyle=":", lw=1, alpha=0.6, label=f"MM legacy full {mm_switch_on + mm_cutoff:.2f} Å")

        ax.set_xlabel("COM distance r (Å)")
        ax.set_ylabel("Scale factor")
        ax.set_ylim(-0.05, 1.2)
        title = f"ML/MM handoff (complementary s_MM=1-s_ML)" if comp else "ML/MM handoff (legacy)"
        ax.set_title(f"{title} | ml_cut={ml_cutoff:.2f}, mm_on={mm_switch_on:.2f}, mm_cut={mm_cutoff:.2f}")
        ax.legend(loc="best", fontsize=8)
        ax.grid(alpha=0.3)

        fig.tight_layout()
        out_dir = save_dir if save_dir is not None else Path.cwd()
        out_dir.mkdir(parents=True, exist_ok=True)
        suffix = "complementary" if comp else "legacy"
        out_path = out_dir / f"cutoffs_schematic_{ml_cutoff:.2f}_{mm_switch_on:.2f}_{mm_cutoff:.2f}_{suffix}.png"
        fig.savefig(out_path, dpi=150)
        return ax

