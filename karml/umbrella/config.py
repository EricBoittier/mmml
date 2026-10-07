"""Configuration for batched umbrella sampling (NVT + MBAR)."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal, Mapping, Sequence

from mmml.md.restraints import (
    AngleWall,
    BondRetentionWall,
    DihedralCV,
    FlatBottomWall,
    LinearDistanceCV,
    cv_from_spec,
)
from mmml.md.restraints.linear_distance import ReactionChannelRestraint


SeedMode = Literal["stretch", "tile", "frames"]
UmbrellaEngine = Literal["packed_ml", "hybrid_jaxmd"]


def _atom_ref_is_name(ref: Any) -> bool:
    """True when a CV/wall atom reference still needs PSF name → index binding."""
    if isinstance(ref, str):
        s = ref.strip()
        if not s:
            return False
        try:
            int(s)
            return False
        except ValueError:
            return True
    return False


def _pairs_need_name_bind(pairs: Any) -> bool:
    for pair in pairs or ():
        if any(_atom_ref_is_name(x) for x in pair):
            return True
    return False


def _spec_needs_name_bind(spec: Any) -> bool:
    """True when ``cv_x`` / wall YAML still carries atom *names* (e.g. ``C1``)."""
    if spec is None or isinstance(
        spec, (LinearDistanceCV, DihedralCV, FlatBottomWall, BondRetentionWall, AngleWall)
    ):
        return False
    if not isinstance(spec, dict):
        return False
    if "pairs" in spec and _pairs_need_name_bind(spec["pairs"]):
        return True
    if "atoms" in spec and any(_atom_ref_is_name(x) for x in spec["atoms"]):
        return True
    if "dihedral" in spec and any(_atom_ref_is_name(x) for x in spec["dihedral"]):
        return True
    if "cv" in spec and _spec_needs_name_bind(spec["cv"]):
        return True
    return False


def _resolve_wall(spec):
    """Build whichever wall kind the spec describes.

    A mapping carrying ``r_max`` is a :class:`BondRetentionWall` (bound on the
    shortest of several competing distances); one carrying ``xi_grid`` is a
    :class:`ReactionChannelRestraint` (a flat bottom that FOLLOWS a reference
    path rather than sitting at a fixed value); anything else is a
    :class:`FlatBottomWall` on a linear CV. Both expose ``energy_batched`` and
    ``forces_batched``, so the sampler does not care which it has.

    Specs that still use atom *names* are left as dicts for hybrid name binding.
    """
    if isinstance(spec, (FlatBottomWall, BondRetentionWall, AngleWall,
                         ReactionChannelRestraint)):
        return spec
    if isinstance(spec, dict) and _spec_needs_name_bind(spec):
        return spec
    if isinstance(spec, dict) and "atoms" in spec:
        return AngleWall.from_spec(spec)
    if isinstance(spec, dict) and "r_max" in spec:
        return BondRetentionWall.from_spec(spec)
    if isinstance(spec, dict) and "xi_grid" in spec:
        return ReactionChannelRestraint.from_spec(spec)
    return FlatBottomWall.from_spec(spec)


@dataclass(frozen=True)
class WindowSchedule:
    """Resolved umbrella windows (1D or 2D CVs).

    ``cvs`` holds the authoritative collective variables; ``atom_pairs`` is the
    legacy view and only describes plain-distance CVs faithfully. Code that must
    handle antisymmetric-stretch reaction coordinates reads ``cvs``.
    """

    ndim: int
    atom_pairs: tuple[tuple[int, int], ...]
    xi0: tuple[float, ...]
    yi0: tuple[float, ...] | None
    k_x: tuple[float, ...]
    k_y: tuple[float, ...] | None
    grid_shape: tuple[int, ...]
    cvs: tuple[Any, ...] = ()
    walls: tuple[FlatBottomWall, ...] = ()

    def __post_init__(self) -> None:
        if not self.cvs:
            # Legacy construction from atom pairs alone: every CV is a distance.
            object.__setattr__(
                self,
                "cvs",
                tuple(LinearDistanceCV.distance(i, j) for i, j in self.atom_pairs),
            )
        elif len(self.cvs) != self.ndim:
            raise ValueError(
                f"schedule has ndim={self.ndim} but {len(self.cvs)} CVs"
            )

    @property
    def n_windows(self) -> int:
        return len(self.xi0)

    @property
    def targets_per_cv(self) -> tuple[tuple[float, ...], ...]:
        """Window centers as ``(ndim, K)``."""
        if self.ndim == 1 or self.yi0 is None:
            return (self.xi0,)
        return (self.xi0, self.yi0)

    @property
    def k_per_cv(self) -> tuple[tuple[float, ...], ...]:
        """Force constants as ``(ndim, K)``."""
        if self.ndim == 1 or self.k_y is None:
            return (self.k_x,)
        return (self.k_x, self.k_y)

    def wall_specs(self) -> list[dict[str, Any]]:
        """JSON-serialisable wall descriptions, for snapshots and summaries."""
        return [w.to_spec() for w in self.walls]

    def cv_specs(self) -> list[dict[str, Any]]:
        """JSON-serialisable CV descriptions, for snapshots and summaries."""
        out: list[dict[str, Any]] = []
        for cv in self.cvs:
            if isinstance(cv, DihedralCV):
                out.append(cv.to_spec())
            elif hasattr(cv, "to_spec"):
                out.append(cv.to_spec())
            else:
                out.append(
                    {
                        "pairs": [list(p) for p in cv.pairs],
                        "coefficients": list(cv.coefficients),
                    }
                )
        return out


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
    output_dir: Path
    # Optional: a combination CV (``cv_x``) can define the coordinate instead,
    # and the hybrid engine can name atoms by ``atom_name_i``/``atom_name_j``.
    atom_i: int | None = None
    atom_j: int | None = None
    structure: Path | None = None
    atom_k: int | None = None
    atom_l: int | None = None
    cv_x: Any = None
    """CV1 override: a :class:`LinearDistanceCV`, a ``(i, j)`` pair, or a
    ``{"pairs": ..., "coefficients": ...}`` mapping. Takes precedence over
    ``atom_i``/``atom_j`` and is how antisymmetric-stretch reaction coordinates
    such as ``xi = r(C-Cl) - r(C-N)`` are declared."""
    cv_y: Any = None
    """CV2 override; takes precedence over ``atom_k``/``atom_l``."""
    walls: tuple[Any, ...] = ()
    """Flat-bottom confinement restraints (see
    :class:`~mmml.md.restraints.FlatBottomWall`). A reaction coordinate built as
    a difference of distances does not bound the system -- a dissociated complex
    can satisfy the same ``xi`` -- and a fitted potential is typically unbounded
    below outside its training data, so the dissociated branch is downhill.
    Walling the *sum* of the same two distances removes that escape route
    without biasing the reaction path."""
    targets_A: tuple[float, ...] = ()
    targets_y_A: tuple[float, ...] = ()
    #: Explicit ``((x0, y0), (x1, y1), ...)`` window centres for a 2D umbrella,
    #: instead of the full ``targets_A x targets_y_A`` grid.
    #:
    #: A reaction is a one-dimensional path through a two-dimensional space, so
    #: the grid is almost entirely wasted: for a methyl transfer, windows with
    #: both r(C-Cl) and r(C-N) large are a dissociated methyl, and windows with
    #: both small are a five-coordinate carbon. Neither is on the path, neither
    #: is in the training data, and sampling them is not just wasted time -- it
    #: puts the model far outside the region it was fitted on.
    #:
    #: Why 2D at all: biasing xi = r(C-Cl) - r(C-N) alone fixes the DIFFERENCE
    #: and leaves the SUM free, and on this surface the dissociated branch is
    #: downhill. A 1D run measured r(C-Cl) 3.2-5.3 A and r(C-N) 3.1-3.7 A
    #: simultaneously -- a perfect reaction coordinate with no bond ever formed.
    #: Restraining both distances removes that degeneracy by construction rather
    #: than patching it with walls afterwards.
    #:
    #: The natural source of centres is the reaction path the model was trained
    #: on. The free energy is then reconstructed on the sampled band and
    #: projected onto whatever coordinate is wanted for plotting.
    targets_xy: tuple[tuple[float, float], ...] = ()
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
    equilibration_steps: int = 0
    """Leading steps discarded before any frame is recorded. Window seeds come
    from optimised geometries with no kinetic energy, so the first part of each
    trajectory is a heating transient rather than equilibrium sampling."""
    printfreq: int = 100
    savefreq: int | None = None
    seed: int = 42
    use_ema: bool = True
    model: str | None = None
    overwrite: bool = False
    resume: bool = False
    """Hybrid only: keep existing ``windows/wXXX.npz`` and re-run missing/failed."""
    resume_failed: bool = True
    """When ``resume``, also re-run windows previously marked failed."""
    only_windows: tuple[int, ...] = ()
    """Optional hybrid subset (0-based). Empty means all windows (subject to resume)."""
    write_window_xyz: bool = False
    structure_index: int = 0
    seed_mode: SeedMode = "stretch"
    move_with: tuple[int, ...] = ()
    move_with2: tuple[int, ...] = ()
    invert_with: tuple[int, ...] = ()
    max_seed_force: float = 15.0
    """Reject a seed whose **ML-region** max |F| exceeds this (eV/Å).

    Gating on the whole-system maximum is meaningless in explicit solvent: a raw
    Packmol box pins it at a window-independent value (~37 eV/Å for dense ACN at
    the default 2.0 Å packing tolerance) coming from solvent contacts the seeding
    never touches, so every window reports the same number and no strained seed
    is ever caught."""
    relax_seed_steps: int = 0
    """FIRE steps relaxing the surroundings around a frozen seeded solute (0 = off).

    Windows are seeded by rigidly displacing the solute inside a box packed for a
    different ξ, so the solvent has to accommodate before dynamics can start.
    Without it the far windows integrate straight into an overlap and abort
    non-finite within ~1 ps."""
    relax_seed_fmax: float = 1.0
    """Convergence target (eV/Å) on the mobile atoms of that relaxation."""
    thermostat: Literal["langevin", "nose-hoover"] = "langevin"
    langevin_gamma: float = 0.1
    max_window_temp_K: float | None = None
    replica_exchange: bool = False
    rex_freq: int = 100
    engine: UmbrellaEngine = "packed_ml"
    from_psf: Path | None = None
    from_pdb: Path | None = None
    from_crd: Path | None = None
    composition: str | None = None
    box_size: float | None = None
    ml_resnames: tuple[str, ...] = ("AMM1", "CH3CL")
    #: Use a complete static on-device pair list instead of rebuilding a padded
    #: neighbour list on the host each block.
    #:
    #: Correct because the switched force field makes pairs beyond ``ctofnb``
    #: contribute exactly zero: benchmarked 300-15000 atoms, the two agree to
    #: |dE| <= 2.5e-12 eV on totals of order 200 eV. So this is a cost choice,
    #: not an accuracy one.
    #:
    #: Measured 1.5x / 2.5x / 2.3x faster at 600 / 1200 / 2625 atoms (A100,
    #: host rebuild amortised over a 20-step block), level at 4800 and 1.4x
    #: slower at 7200. Turn off above **~4800 atoms** on GPU, or ~2600 on CPU,
    #: where the O(N^2) energy costs more than the host rebuild saves; the
    #: sampler prints a note if a run starts past that with this still on.
    #: See ``scripts/bench_static_vs_neighbor_pairs.py``.
    static_pairs: bool = True
    atom_name_i: str | None = None
    atom_name_j: str | None = None
    lr_solver: str = "mic"
    # Verlet skin (Å) for the intermolecular MM pair list. > 0 builds the list
    # at cutoff + skin and reuses it while every atom has moved < skin/2, which
    # turns the per-block host rebuild into a scalar check. Pairs inside the
    # skin are inert: mm_nonbonded zeroes everything beyond ctofnb.
    nl_skin_A: float = 0.0
    # MD steps per compiled block / MM pair refresh. None uses savefreq (the
    # previous behavior, one refresh per recorded frame).
    nl_update_interval: int | None = None

    #: Picoseconds of NVT run on the packed box *before* any window starts, at
    #: the schedule point nearest the base geometry, to relax the liquid.
    #:
    #: A Packmol box is at the right density from the first step but its liquid
    #: structure is not relaxed at all, and the first solvation shell around a
    #: charged solute takes tens to hundreds of picoseconds to form. Turan et
    #: al. use 500 ps NpT + 2 ns NVT before sampling; this pipeline was using
    #: none, because ``equilibration_steps`` only discards frames from a window
    #: that has already started. Sampling a solvated barrier from an unrelaxed
    #: box under-solvates the ion pair and biases the barrier high.
    #:
    #: The relaxed structure is cached under ``output_dir/../equilibrated_*.npz``
    #: keyed by system size and seed, so the cost is paid once per box.
    pre_equilibrate_ps: float = 0.0
    #: Seed each window from the previous window's final frame instead of
    #: re-stretching the solute from the one base structure every time.
    #:
    #: With ``seed_mode="stretch"`` alone every window starts from the same
    #: solvent configuration, so no window ever inherits its neighbour's
    #: relaxed solvation shell and each must re-form it from scratch inside its
    #: own short trajectory. Turan et al. restart each window from the previous
    #: window's structure for exactly this reason; consecutive centres differ
    #: by ~0.1 A, so the carried-over shell is nearly correct already. The
    #: solute is still stretched to the new centre -- only the solvent is
    #: inherited.
    #:
    #: Tell-tale that this is off: an identical ``seed_max|F|`` on every window.
    seed_from_previous_window: bool = False
    #: Stages over which to raise the temperature from ``heat_start_fraction *
    #: temperature_K`` to the target during pre-equilibration. Minimised or
    #: freshly packed boxes have no kinetic energy, and assigning full-target
    #: velocities in one step is a thermal shock; 0 disables staging.
    heat_stages: int = 0
    heat_start_fraction: float = 0.2

    def __post_init__(self) -> None:
        if self.pre_equilibrate_ps < 0:
            raise ValueError(
                f"pre_equilibrate_ps must be >= 0 (got {self.pre_equilibrate_ps})"
            )
        if self.heat_stages < 0:
            raise ValueError(f"heat_stages must be >= 0 (got {self.heat_stages})")
        if not 0.0 < self.heat_start_fraction <= 1.0:
            raise ValueError(
                "heat_start_fraction must be in (0, 1] (got "
                f"{self.heat_start_fraction}) -- a 0 K start is the thermal "
                "shock the staging exists to prevent"
            )
        self._post_init_rest()

    def _post_init_rest(self) -> None:
        if self.engine not in ("packed_ml", "hybrid_jaxmd"):
            raise ValueError(
                f"engine must be packed_ml|hybrid_jaxmd (got {self.engine!r})"
            )
        if self.cv_x is None:
            if self.atom_i is None or self.atom_j is None:
                raise ValueError("provide either cv_x or both atom_i and atom_j")
            if self.atom_i == self.atom_j or min(self.atom_i, self.atom_j) < 0:
                raise ValueError("atom_i and atom_j must be distinct non-negative indices")
        has_k = self.atom_k is not None
        has_l = self.atom_l is not None
        if self.cv_y is None:
            if has_k != has_l:
                raise ValueError("atom_k and atom_l must both be set for 2D umbrella")
            if has_k and has_l:
                assert self.atom_k is not None and self.atom_l is not None
                if self.atom_k == self.atom_l or min(self.atom_k, self.atom_l) < 0:
                    raise ValueError(
                        "atom_k and atom_l must be distinct non-negative indices"
                    )
        # Named-atom CVs (``C1``/``CL1``/…) are bound after the PSF is loaded.
        needs_name_bind = (
            _spec_needs_name_bind(self.cv_x)
            or _spec_needs_name_bind(self.cv_y)
            or any(_spec_needs_name_bind(w) for w in self.walls)
        )
        if not needs_name_bind:
            # Fail here rather than deep inside the sampler on a malformed spec.
            self.resolve_cvs()
            self.resolve_walls()
        if self.temperature_K <= 0:
            raise ValueError(f"temperature_K must be > 0 (got {self.temperature_K})")
        if self.timestep_fs <= 0:
            raise ValueError(f"timestep_fs must be > 0 (got {self.timestep_fs})")
        if self.nsteps < 1:
            raise ValueError(f"nsteps must be >= 1 (got {self.nsteps})")
        if self.equilibration_steps < 0:
            raise ValueError(
                f"equilibration_steps must be >= 0 (got {self.equilibration_steps})"
            )
        if self.equilibration_steps >= self.nsteps:
            raise ValueError(
                f"equilibration_steps ({self.equilibration_steps}) must be less "
                f"than nsteps ({self.nsteps}); no production frames would remain"
            )
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
        if self.relax_seed_steps < 0:
            raise ValueError(
                f"relax_seed_steps must be >= 0 (got {self.relax_seed_steps})"
            )
        if self.relax_seed_fmax <= 0:
            raise ValueError(
                f"relax_seed_fmax must be > 0 (got {self.relax_seed_fmax})"
            )
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
        if self.box_size is not None and float(self.box_size) <= 0:
            raise ValueError(f"box_size must be > 0 (got {self.box_size})")
        if not self.ml_resnames:
            raise ValueError("ml_resnames must be non-empty")
        if self.engine == "packed_ml":
            if self.structure is None:
                raise ValueError("packed_ml engine requires structure")
        else:
            if self.is_2d:
                raise ValueError(
                    "hybrid_jaxmd engine supports 1D umbrellas only in v1 "
                    "(omit --atoms2 / atom_k/atom_l)"
                )
            if self.replica_exchange:
                raise ValueError(
                    "hybrid_jaxmd engine does not support replica_exchange in v1"
                )
            has_psf = self.from_psf is not None
            has_coords = (
                self.from_pdb is not None
                or self.from_crd is not None
                or self.structure is not None
            )
            has_comp = self.composition is not None
            if has_psf and not has_coords:
                raise ValueError(
                    "hybrid_jaxmd with from_psf also needs from_pdb, from_crd, or structure"
                )
            if not has_psf and not has_comp:
                raise ValueError(
                    "hybrid_jaxmd requires from_psf (+ coords) or composition (+ box_size)"
                )
            if has_comp and self.box_size is None and self.from_psf is None:
                raise ValueError("hybrid_jaxmd composition path requires box_size")
        if needs_name_bind:
            n_win = int(self.n_windows or len(self.targets_A) or 0)
            if n_win < 1:
                raise ValueError("need at least one umbrella window target")
        else:
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
        return self.cv_y is not None or (
            self.atom_k is not None and self.atom_l is not None
        )

    @property
    def uses_paired_windows(self) -> bool:
        return bool(self.targets_xy)

    def resolve_cvs(self) -> tuple[LinearDistanceCV, ...]:
        """Resolve CV1 (and CV2 when 2D) from ``cv_*`` or the atom-index fields."""
        if self.cv_x is not None:
            cv_x = cv_from_spec(self.cv_x)
        else:
            cv_x = LinearDistanceCV.distance(int(self.atom_i), int(self.atom_j))
        if not self.is_2d:
            return (cv_x,)
        if self.cv_y is not None:
            cv_y = cv_from_spec(self.cv_y)
        else:
            cv_y = LinearDistanceCV.distance(int(self.atom_k), int(self.atom_l))
        return (cv_x, cv_y)

    def resolve_walls(self) -> tuple[FlatBottomWall, ...]:
        """Resolve the configured confinement walls."""
        return tuple(_resolve_wall(w) for w in self.walls)

    def _legacy_atom_pairs(
        self, cvs: tuple[Any, ...]
    ) -> tuple[tuple[int, int], ...]:
        """First pair of each CV -- the backward-compatible ``atom_pairs`` view.

        Only faithful for plain-distance CVs; combination / dihedral CVs carry
        their real definition on ``WindowSchedule.cvs``.
        """
        return tuple(cv.pairs[0] for cv in cvs)

    def resolve_targets(self) -> tuple[float, ...]:
        """CV1 window centers (length ``K`` after product expansion)."""
        return self.resolve_schedule().xi0

    def resolve_force_constants(self) -> tuple[float, ...]:
        """CV1 force constants (length ``K``)."""
        return self.resolve_schedule().k_x

    def resolve_schedule(self) -> WindowSchedule:
        """Resolve 1D or 2D window centers and force constants."""
        cvs = self.resolve_cvs()
        walls = self.resolve_walls()
        atom_pairs = self._legacy_atom_pairs(cvs)

        # Explicit paired centres: a band of windows following the reaction
        # path, not the full outer product of the two axes.
        if self.targets_xy:
            if len(cvs) != 2:
                raise ValueError(
                    "targets_xy needs two CVs; give cv_y (or atom_k/atom_l) "
                    "as well as cv_x"
                )
            import numpy as np

            pairs = [tuple(float(v) for v in p) for p in self.targets_xy]
            bad = [p for p in pairs if len(p) != 2]
            if bad:
                raise ValueError(
                    f"targets_xy entries must be (x, y) pairs; got {bad[0]!r}"
                )
            xi0 = tuple(p[0] for p in pairs)
            yi0 = tuple(p[1] for p in pairs)
            n = len(xi0)
            ky_src = self.k_ev_A2 if self.k_y_ev_A2 is None else self.k_y_ev_A2
            return WindowSchedule(
                ndim=2,
                atom_pairs=atom_pairs,
                xi0=xi0,
                yi0=yi0,
                k_x=_broadcast_k(self.k_ev_A2, n),
                k_y=_broadcast_k(ky_src, n),
                # A path, not a lattice: one window per centre.
                grid_shape=(n,),
                cvs=cvs,
                walls=walls,
            )

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
                    atom_pairs=atom_pairs,
                    xi0=(),
                    yi0=None,
                    k_x=(),
                    k_y=None,
                    grid_shape=(0,),
                    cvs=cvs,
                    walls=walls,
                )
            kx = _broadcast_k(self.k_ev_A2, len(x_centers))
            return WindowSchedule(
                ndim=1,
                atom_pairs=atom_pairs,
                xi0=x_centers,
                yi0=None,
                k_x=kx,
                k_y=None,
                grid_shape=(len(x_centers),),
                cvs=cvs,
                walls=walls,
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
        return WindowSchedule(
            ndim=2,
            atom_pairs=atom_pairs,
            xi0=xi0,
            yi0=yi0,
            k_x=kx,
            k_y=ky,
            grid_shape=(len(x_centers), len(y_centers)),
            cvs=cvs,
            walls=walls,
        )

    def effective_savefreq(self) -> int:
        return int(self.savefreq if self.savefreq is not None else self.printfreq)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> UmbrellaConfig:
        raw = dict(data)
        for key in (
            "checkpoint",
            "structure",
            "output_dir",
            "from_psf",
            "from_pdb",
            "from_crd",
        ):
            if key in raw and raw[key] is not None:
                raw[key] = Path(raw[key])
        for key in (
            "targets_A",
            "targets_y_A",
            "move_with",
            "move_with2",
            "invert_with",
            "only_windows",
        ):
            if key in raw and raw[key] is not None:
                if key.startswith("move_") or key in {"invert_with", "only_windows"}:
                    raw[key] = tuple(int(x) for x in raw[key])
                else:
                    raw[key] = tuple(float(x) for x in raw[key])
        if "ml_resnames" in raw and raw["ml_resnames"] is not None:
            raw["ml_resnames"] = tuple(str(x) for x in raw["ml_resnames"])
        for key in ("k_ev_A2", "k_y_ev_A2"):
            if key in raw and raw[key] is not None:
                k = raw[key]
                if isinstance(k, (list, tuple)):
                    raw[key] = tuple(float(x) for x in k)
                else:
                    raw[key] = float(k)
        for key in ("cv_x", "cv_y"):
            if raw.get(key) is not None:
                if _spec_needs_name_bind(raw[key]):
                    # Keep YAML atom names; hybrid binds them to PSF indices.
                    raw[key] = dict(raw[key])
                else:
                    raw[key] = cv_from_spec(raw[key])
        if raw.get("walls"):
            raw["walls"] = tuple(_resolve_wall(w) for w in raw["walls"])
        return cls(**raw)

    def to_dict(self) -> dict[str, Any]:
        out = asdict(self)
        for key in (
            "checkpoint",
            "structure",
            "output_dir",
            "from_psf",
            "from_pdb",
            "from_crd",
        ):
            if out[key] is not None:
                out[key] = str(out[key])
        for key in ("cv_x", "cv_y"):
            cv = getattr(self, key)
            if cv is None:
                out[key] = None
            else:
                resolved = cv_from_spec(cv)
                if isinstance(resolved, DihedralCV):
                    out[key] = {"kind": "dihedral", "atoms": list(resolved.atoms)}
                else:
                    out[key] = {
                        "pairs": [list(p) for p in resolved.pairs],
                        "coefficients": list(resolved.coefficients),
                    }
        out["targets_A"] = list(out["targets_A"])
        out["targets_y_A"] = list(out["targets_y_A"])
        out["move_with"] = list(out["move_with"])
        out["move_with2"] = list(out["move_with2"])
        out["invert_with"] = list(out["invert_with"])
        out["ml_resnames"] = list(out["ml_resnames"])
        out["walls"] = [w.to_spec() for w in self.resolve_walls()]
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
