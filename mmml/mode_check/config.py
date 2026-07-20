"""Configuration for monomer / small-cluster mode checks."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal

RESULT_SCHEMA_VERSION = "1.0"

CheckName = Literal["fd", "bond-scan", "vibrations", "kick", "minimize"]

# Inclusive stretch grid used by the hybrid Hessian diagnostics (Å).
DEFAULT_BOND_DELTAS: tuple[float, ...] = tuple(
    round(-0.08 + i * 0.01, 12) for i in range(17)
)


@dataclass(frozen=True)
class ModeCheckConfig:
    """Scientific / numerical knobs for ``run_mode_check`` (calculator-agnostic)."""

    checks: tuple[CheckName, ...] = ("fd", "bond-scan", "vibrations")
    fd_atoms: int = 3
    fd_dx_A: float = 1e-3
    bond_deltas: tuple[float, ...] = DEFAULT_BOND_DELTAS
    bond_fit_abs_delta_max: float = 0.03
    minimize_fmax: float = 0.05
    minimize_steps: int = 400
    vib_delta_A: float = 0.01
    vib_nfree: int = 2
    kick_delta_A: float = 0.03
    kick_timestep_fs: float = 0.1
    kick_steps: int = 500
    atoms_per_monomer: tuple[int, ...] | None = None

    def __post_init__(self) -> None:
        allowed = {"fd", "bond-scan", "vibrations", "kick", "minimize"}
        checks = tuple(str(c) for c in self.checks)
        unknown = [c for c in checks if c not in allowed]
        if unknown:
            raise ValueError(f"unknown checks: {unknown}; allowed={sorted(allowed)}")
        if not checks:
            raise ValueError("checks must be non-empty")
        object.__setattr__(self, "checks", checks)  # type: ignore[arg-type]
        if self.atoms_per_monomer is not None:
            object.__setattr__(
                self,
                "atoms_per_monomer",
                tuple(int(n) for n in self.atoms_per_monomer),
            )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class HybridModeCheckSetup:
    """CHARMM/hybrid calculator setup for vacuum monomer or small clusters."""

    composition: tuple[tuple[str, int], ...]
    checkpoint: Path
    do_mm: bool = True
    do_ml: bool = True
    do_ml_dimer: bool | None = None
    ml_switch_width: float = 1.5
    mm_switch_on: float = 6.0
    mm_switch_width: float = 5.0
    mm_charge_mode: str = "q0"
    lr_solver: str = "mic"
    monomer_separation_A: float = 2.8
    xyz: Path | None = None
    max_pairs: int = 20_000

    def __post_init__(self) -> None:
        object.__setattr__(self, "checkpoint", Path(self.checkpoint).expanduser())
        if self.xyz is not None:
            object.__setattr__(self, "xyz", Path(self.xyz).expanduser())
        n_mol = sum(int(c) for _, c in self.composition)
        if n_mol < 1:
            raise ValueError("composition must include at least one monomer")
        if self.do_ml_dimer is None:
            object.__setattr__(self, "do_ml_dimer", n_mol >= 2)
        # MM is cross-monomer only; n=1 must not request doMM.
        if n_mol < 2 and self.do_mm:
            object.__setattr__(self, "do_mm", False)


@dataclass
class ModeCheckPaths:
    """Output layout under ``--output-dir``."""

    output_dir: Path
    summary_json: Path = field(init=False)
    vib_dir: Path = field(init=False)
    kick_r_npy: Path = field(init=False)

    def __post_init__(self) -> None:
        self.output_dir = Path(self.output_dir)
        self.summary_json = self.output_dir / "mode_check_summary.json"
        self.vib_dir = self.output_dir / "vibrations"
        self.kick_r_npy = self.output_dir / "kick_r.npy"
