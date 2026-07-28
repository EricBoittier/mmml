"""Configuration for batched umbrella sampling (NVT + MBAR)."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal, Mapping, Sequence


SeedMode = Literal["stretch", "tile", "frames"]


@dataclass(frozen=True)
class WindowSchedule:
    """Resolved umbrella windows (1D or 2D distance CVs)."""

    ndim: int
    atom_pairs: tuple[tuple[int, int], ...]
    xi0: tuple[float, ...]
    yi0: tuple[float, ...] | None
    k_x: tuple[float, ...]
    k_y: tuple[float, ...] | None
    grid_shape: tuple[int, ...]

    @property
    def n_windows(self) -> int:
        return len(self.xi0)


def _linspace_or_list(
    *,
    explicit: tuple[float, ...],
    lo: float | None,
    hi: float | None,
    n: int | None,
    label: str,
) -> tuple[float, ...]:
    if explicit:
        return tuple(float(x) for x in explicit)
    if lo is None or hi is None or n is None:
        return ()
    if n < 1:
        raise ValueError(f"{label} n_windows must be >= 1 (got {n})")
    if n == 1:
        return (float(lo),)
    import numpy as np

    return tuple(float(x) for x in np.linspace(float(lo), float(hi), int(n)))


def _broadcast_k(k: float | Sequence[float], n: int) -> tuple[float, ...]:
    if isinstance(k, (int, float)):
        return tuple(float(k) for _ in range(n))
    out = tuple(float(x) for x in k)
    if len(out) != n:
        raise ValueError(f"force-constant length {len(out)} must match n_windows={n}")
    return out


@dataclass(frozen=True)
class UmbrellaConfig:
    """Inputs for :func:`mmml.umbrella.sample.run_umbrella_nvt`."""

    checkpoint: Path
    structure: Path
    output_dir: Path
    atom_i: int
    atom_j: int
    atom_k: int | None = None
    atom_l: int | None = None
    targets_A: tuple[float, ...] = ()
    targets_y_A: tuple[float, ...] = ()
    xi_min: float | None = None
    xi_max: float | None = None
    n_windows: int | None = None
    yi_min: float | None = None
    yi_max: float | None = None
    n_windows_y: int | None = None
    k_ev_A2: float | tuple[float, ...] = 10.0
    k_y_ev_A2: float | tuple[float, ...] | None = None
    temperature_K: float = 300.0
    timestep_fs: float = 0.1
    nsteps: int = 1000
    printfreq: int = 100
    savefreq: int | None = None
    seed: int = 42
    use_ema: bool = True
    model: str | None = None
    overwrite: bool = False
    write_window_xyz: bool = False
    structure_index: int = 0
    seed_mode: SeedMode = "stretch"
    move_with: tuple[int, ...] = ()
    move_with2: tuple[int, ...] = ()
    invert_with: tuple[int, ...] = ()
    max_seed_force: float = 15.0
    thermostat: Literal["langevin", "nose-hoover"] = "langevin"
    langevin_gamma: float = 0.1
    max_window_temp_K: float | None = None
    replica_exchange: bool = False
    rex_freq: int = 100

    def __post_init__(self) -> None:
        if self.atom_i == self.atom_j or min(self.atom_i, self.atom_j) < 0:
            raise ValueError("atom_i and atom_j must be distinct non-negative indices")
        has_k = self.atom_k is not None
        has_l = self.atom_l is not None
        if has_k != has_l:
            raise ValueError("atom_k and atom_l must both be set for 2D umbrella")
        if has_k and has_l:
            assert self.atom_k is not None and self.atom_l is not None
            if self.atom_k == self.atom_l or min(self.atom_k, self.atom_l) < 0:
                raise ValueError("atom_k and atom_l must be distinct non-negative indices")
        if self.temperature_K <= 0:
            raise ValueError(f"temperature_K must be > 0 (got {self.temperature_K})")
        if self.timestep_fs <= 0:
            raise ValueError(f"timestep_fs must be > 0 (got {self.timestep_fs})")
        if self.nsteps < 1:
            raise ValueError(f"nsteps must be >= 1 (got {self.nsteps})")
        if self.printfreq < 1:
            raise ValueError(f"printfreq must be >= 1 (got {self.printfreq})")
        if self.savefreq is not None and self.savefreq < 1:
            raise ValueError(f"savefreq must be >= 1 (got {self.savefreq})")
        if self.structure_index < 0:
            raise ValueError(f"structure_index must be >= 0 (got {self.structure_index})")
        if self.seed_mode not in ("stretch", "tile", "frames"):
            raise ValueError(
                f"seed_mode must be stretch|tile|frames (got {self.seed_mode!r})"
            )
        if self.max_seed_force <= 0:
            raise ValueError(f"max_seed_force must be > 0 (got {self.max_seed_force})")
        if self.thermostat not in ("langevin", "nose-hoover"):
            raise ValueError(
                f"thermostat must be langevin|nose-hoover (got {self.thermostat!r})"
            )
        if self.langevin_gamma <= 0:
            raise ValueError(f"langevin_gamma must be > 0 (got {self.langevin_gamma})")
        if self.max_window_temp_K is not None and self.max_window_temp_K <= 0:
            raise ValueError(
                f"max_window_temp_K must be > 0 (got {self.max_window_temp_K})"
            )
        if self.rex_freq < 1:
            raise ValueError(f"rex_freq must be >= 1 (got {self.rex_freq})")
        # Force validation via schedule construction
        sched = self.resolve_schedule()
        if sched.n_windows < 1:
            raise ValueError("need at least one umbrella window target")
        if any(k < 0 for k in sched.k_x):
            raise ValueError("force constants must be non-negative")
        if sched.k_y is not None and any(k < 0 for k in sched.k_y):
            raise ValueError("force constants must be non-negative")

    @property
    def is_2d(self) -> bool:
        return self.atom_k is not None and self.atom_l is not None

    def resolve_targets(self) -> tuple[float, ...]:
        """CV1 window centers (length ``K`` after product expansion)."""
        return self.resolve_schedule().xi0

    def resolve_force_constants(self) -> tuple[float, ...]:
        """CV1 force constants (length ``K``)."""
        return self.resolve_schedule().k_x

    def resolve_schedule(self) -> WindowSchedule:
        """Resolve 1D or 2D window centers and force constants."""
        x_centers = _linspace_or_list(
            explicit=self.targets_A,
            lo=self.xi_min,
            hi=self.xi_max,
            n=self.n_windows,
            label="x",
        )
        if not self.is_2d:
            if not x_centers:
                return WindowSchedule(
                    ndim=1,
                    atom_pairs=((self.atom_i, self.atom_j),),
                    xi0=(),
                    yi0=None,
                    k_x=(),
                    k_y=None,
                    grid_shape=(0,),
                )
            kx = _broadcast_k(self.k_ev_A2, len(x_centers))
            return WindowSchedule(
                ndim=1,
                atom_pairs=((self.atom_i, self.atom_j),),
                xi0=x_centers,
                yi0=None,
                k_x=kx,
                k_y=None,
                grid_shape=(len(x_centers),),
            )

        y_centers = _linspace_or_list(
            explicit=self.targets_y_A,
            lo=self.yi_min,
            hi=self.yi_max,
            n=self.n_windows_y,
            label="y",
        )
        if not x_centers or not y_centers:
            raise ValueError(
                "2D umbrella requires CV1 centers (targets_A or xi-min/max/n-windows) "
                "and CV2 centers (targets_y_A or yi-min/max/n-windows-y)"
            )
        import numpy as np

        xx, yy = np.meshgrid(np.asarray(x_centers), np.asarray(y_centers), indexing="ij")
        xi0 = tuple(float(x) for x in xx.ravel())
        yi0 = tuple(float(y) for y in yy.ravel())
        n = len(xi0)
        kx = _broadcast_k(self.k_ev_A2, n)
        ky_src = self.k_ev_A2 if self.k_y_ev_A2 is None else self.k_y_ev_A2
        ky = _broadcast_k(ky_src, n)
        assert self.atom_k is not None and self.atom_l is not None
        return WindowSchedule(
            ndim=2,
            atom_pairs=((self.atom_i, self.atom_j), (self.atom_k, self.atom_l)),
            xi0=xi0,
            yi0=yi0,
            k_x=kx,
            k_y=ky,
            grid_shape=(len(x_centers), len(y_centers)),
        )

    def effective_savefreq(self) -> int:
        return int(self.savefreq if self.savefreq is not None else self.printfreq)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> UmbrellaConfig:
        raw = dict(data)
        for key in ("checkpoint", "structure", "output_dir"):
            if key in raw and raw[key] is not None:
                raw[key] = Path(raw[key])
        for key in ("targets_A", "targets_y_A", "move_with", "move_with2", "invert_with"):
            if key in raw and raw[key] is not None:
                if key.startswith("move_") or key == "invert_with":
                    raw[key] = tuple(int(x) for x in raw[key])
                else:
                    raw[key] = tuple(float(x) for x in raw[key])
        for key in ("k_ev_A2", "k_y_ev_A2"):
            if key in raw and raw[key] is not None:
                k = raw[key]
                if isinstance(k, (list, tuple)):
                    raw[key] = tuple(float(x) for x in k)
                else:
                    raw[key] = float(k)
        return cls(**raw)

    def to_dict(self) -> dict[str, Any]:
        out = asdict(self)
        for key in ("checkpoint", "structure", "output_dir"):
            out[key] = str(out[key])
        out["targets_A"] = list(out["targets_A"])
        out["targets_y_A"] = list(out["targets_y_A"])
        out["move_with"] = list(out["move_with"])
        out["move_with2"] = list(out["move_with2"])
        out["invert_with"] = list(out["invert_with"])
        if isinstance(self.k_ev_A2, tuple):
            out["k_ev_A2"] = list(self.k_ev_A2)
        if isinstance(self.k_y_ev_A2, tuple):
            out["k_y_ev_A2"] = list(self.k_y_ev_A2)
        return out


@dataclass(frozen=True)
class UmbrellaMbarConfig:
    """Inputs for :func:`mmml.umbrella.mbar.run_umbrella_mbar`."""

    run_dir: Path
    checkpoint: Path | None = None
    temperature_K: float | None = None
    mbar_verbose: bool = False
    ml_batch_size: int = 32

    def __post_init__(self) -> None:
        if self.temperature_K is not None and self.temperature_K <= 0:
            raise ValueError(f"temperature_K must be > 0 (got {self.temperature_K})")
        if self.ml_batch_size < 1:
            raise ValueError(f"ml_batch_size must be >= 1 (got {self.ml_batch_size})")
