import numpy as np
from pathlib import Path

class CutoffParameters:
    """Parameters for ML and MM cutoffs and switching functions"""
    def __init__(
        self,
        ml_cutoff: float = 2.0,
        mm_switch_on: float = 5.0,
        mm_cutoff: float = 1.0
    ):
        """
        Args:
            ml_cutoff: Distance where ML potential is cut off
            mm_switch_on: Distance where MM potential starts switching on
            mm_cutoff: Final cutoff for MM potential
        """
        self.ml_cutoff =  ml_cutoff 
        self.mm_switch_on = mm_switch_on
        self.mm_cutoff = mm_cutoff


    def __str__(self):
        return f"CutoffParameters(ml_cutoff={self.ml_cutoff}, mm_switch_on={self.mm_switch_on}, mm_cutoff={self.mm_cutoff})"
    
    def __repr__(self):
        return self.__str__()

    def __eq__(self, other):
        if not isinstance(other, CutoffParameters):
            return False
        # Convert to floats for comparison (handles JAX arrays)
        ml_cutoff_self = float(self.ml_cutoff) if hasattr(self.ml_cutoff, '__float__') else self.ml_cutoff
        mm_switch_on_self = float(self.mm_switch_on) if hasattr(self.mm_switch_on, '__float__') else self.mm_switch_on
        mm_cutoff_self = float(self.mm_cutoff) if hasattr(self.mm_cutoff, '__float__') else self.mm_cutoff
        ml_cutoff_other = float(other.ml_cutoff) if hasattr(other.ml_cutoff, '__float__') else other.ml_cutoff
        mm_switch_on_other = float(other.mm_switch_on) if hasattr(other.mm_switch_on, '__float__') else other.mm_switch_on
        mm_cutoff_other = float(other.mm_cutoff) if hasattr(other.mm_cutoff, '__float__') else other.mm_cutoff
        return (ml_cutoff_self == ml_cutoff_other and 
                mm_switch_on_self == mm_switch_on_other and 
                mm_cutoff_self == mm_cutoff_other)
    
    def __ne__(self, other):
        return not self.__eq__(other)
    
    def __hash__(self):
        # Convert to Python floats if they're JAX arrays (for hashability)
        ml_cutoff_val = float(self.ml_cutoff) if hasattr(self.ml_cutoff, '__float__') else self.ml_cutoff
        mm_switch_on_val = float(self.mm_switch_on) if hasattr(self.mm_switch_on, '__float__') else self.mm_switch_on
        mm_cutoff_val = float(self.mm_cutoff) if hasattr(self.mm_cutoff, '__float__') else self.mm_cutoff
        return hash((ml_cutoff_val, mm_switch_on_val, mm_cutoff_val))

    def to_dict(self):
        return {
            "ml_cutoff": self.ml_cutoff,
            "mm_switch_on": self.mm_switch_on,
            "mm_cutoff": self.mm_cutoff
        }
    
    def from_dict(self, d):
        return CutoffParameters(
            ml_cutoff=d["ml_cutoff"],
            mm_switch_on=d["mm_switch_on"],
            mm_cutoff=d["mm_cutoff"]
        )

    def plot_cutoff_parameters(self, save_dir: Path | None = None):
        import numpy as np
        import matplotlib.pyplot as plt

        ml_cutoff = float(self.ml_cutoff)
        mm_switch_on = float(self.mm_switch_on)
        mm_cutoff = float(self.mm_cutoff)

        r_max = float(max(ml_cutoff, mm_switch_on + 2.0 * mm_cutoff) * 1.5 + 2.0)
        r = np.linspace(0.01, r_max, 600)

        def _np_smoothstep01(s): return s * s * (3.0 - 2.0 * s)
        def _np_sharpstep(r, x0, x1, gamma=3.0):
            s = np.clip((r - x0) / max(x1 - x0, 1e-12), 0.0, 1.0)
            s = s ** gamma
            return _np_smoothstep01(s)

        gamma_ml = 5.0     # your steeper ML taper
        gamma_on = 0.001    # faster MM turn-on
        gamma_off = 3.0    # smooth MM turn-off

        ml_scale = 1.0 - _np_sharpstep(r, mm_switch_on - ml_cutoff, mm_switch_on, gamma=gamma_ml)
        mm_on    = _np_sharpstep(r, mm_switch_on, mm_switch_on + mm_cutoff, gamma=gamma_on)
        mm_off   = _np_sharpstep(r, mm_switch_on + mm_cutoff, mm_switch_on + 2.0 * mm_cutoff, gamma=gamma_off)
        mm_scale = mm_on * (1.0 - mm_off)


        fig, ax = plt.subplots(1, 1, figsize=(8, 5))
        ax.plot(r, ml_scale, label="ML scale", lw=2, color="C0")
        ax.plot(r, mm_scale, label="MM scale", lw=2, color="C1")
        ax.plot(r, ml_scale + mm_scale, "--", lw=1, color="gray", alpha=0.7, label="ML+MM")

        # ax.axvline(taper_start, color="C0", linestyle="--", lw=1, alpha=0.7, label=f"ML start {taper_start:.2f} Å")
        ax.axvline(mm_switch_on, color="k", linestyle=":", lw=1.5, label=f"handoff {mm_switch_on:.2f} Å")
        ax.axvline(mm_switch_on + mm_cutoff, color="C1", linestyle="--", lw=1, alpha=0.7, label=f"MM full-on {mm_switch_on + mm_cutoff:.2f} Å")
        ax.axvline(mm_switch_on + 2.0 * mm_cutoff, color="C1", linestyle="-.", lw=1, alpha=0.7, label=f"MM off {mm_switch_on + 2.0 * mm_cutoff:.2f} Å")

        ax.set_xlabel("COM distance r (Å)")
        ax.set_ylabel("Scale factor")
        ax.set_ylim(-0.05, 1.15)
        ax.set_title(f"ML/MM Handoff (force-switched MM) | ml={ml_cutoff:.2f}, mm_on={mm_switch_on:.2f}, mm_cut={mm_cutoff:.2f}")
        ax.legend(loc="best")
        ax.grid(alpha=0.3)

        fig.tight_layout()
        out_dir = save_dir if save_dir is not None else Path.cwd()
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"cutoffs_schematic_{self.ml_cutoff:.2f}_{self.mm_switch_on:.2f}_{self.mm_cutoff:.2f}.png"
        fig.savefig(out_path, dpi=150)
        try:
            plt.show()
        except Exception:
            pass
        print(f"Saved cutoff schematic to {out_path}")

