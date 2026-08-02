"""Explicit solvent force-field parameters, so no CHARMM is needed at runtime.

Why hard-code these rather than read a PSF
------------------------------------------
The CHARMM-backed builder gave repeated, opaque failures: it can only build one
system per process, its Packmol stage returned configurations whose energy was
-9.5e5 eV for 1038 atoms, and the diagnostics disagreed about whether the
coordinates were even overlapping. Every number below is instead written out
explicitly, so the starting configuration and the energy that scores it are both
fully inspectable and reproducible without live CHARMM state.

Values are the standard CHARMM ones. Note CHARMM's conventions:

- Bond and angle force constants already absorb the factor of 1/2, i.e.
  E = k(r-r0)^2 and k(theta-theta0)^2, matching ``mm_bonded``'s explicit path.
- LJ is specified as (epsilon, Rmin/2), with epsilon quoted negative in CHARMM
  parameter files; magnitudes are used here.

TIP3 is the CHARMM-modified TIP3P water in ``toppar_water_ions.str``: unlike the
original Jorgensen TIP3P it carries LJ on the hydrogens, which matters for the
ML/MM solute-solvent contact and is why it is reproduced faithfully here.

Only water is populated for now. The remaining four Turan solvents follow the
same schema; fill them in from ``top_all36_cgenff.rtf`` /
``par_all36_cgenff.prm`` (residues MEOH, ACN, BENZ, and the CGenFF-typed CHEX in
``top_chex.rtf``) before running those legs.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

__all__ = ["SolventModel", "SOLVENT_MODELS", "get_solvent_model", "from_json"]


@dataclass(frozen=True)
class SolventModel:
    """One rigid-topology solvent molecule, replicated to build a box."""

    name: str
    residue: str
    #: Atomic numbers, in the order coordinates are laid out.
    Z: np.ndarray
    #: Reference geometry (n_atoms, 3) in Angstrom, centred on the first atom.
    geometry: np.ndarray
    charges: np.ndarray          # e
    epsilon: np.ndarray          # kcal/mol, positive magnitude
    rmin_half: np.ndarray        # Angstrom
    bonds: np.ndarray            # (n_bonds, 2) local indices
    bond_k: np.ndarray           # kcal/mol/A^2, CHARMM convention
    bond_r0: np.ndarray          # Angstrom
    angles: np.ndarray           # (n_angles, 3) local indices
    angle_k: np.ndarray          # kcal/mol/rad^2, CHARMM convention
    angle_theta0: np.ndarray     # degrees
    density_kg_m3: float
    box_side_A: float
    metadata: dict = field(default_factory=dict)

    @property
    def n_atoms(self) -> int:
        return int(self.Z.shape[0])

    @property
    def molar_mass(self) -> float:
        from ase.data import atomic_masses

        return float(atomic_masses[self.Z.astype(int)].sum())

    def n_for_density(self, box_side_A: float) -> int:
        """Molecules that fill a cube of this side at the experimental density."""
        volume = box_side_A**3
        return int(round(
            self.density_kg_m3 * 1e-3 * volume * 6.02214076e23
            / (1e24 * self.molar_mass)
        ))

    def validate(self) -> None:
        n = self.n_atoms
        for name, arr in (("geometry", self.geometry), ("charges", self.charges),
                          ("epsilon", self.epsilon), ("rmin_half", self.rmin_half)):
            if arr.shape[0] != n:
                raise ValueError(f"{self.name}: {name} has {arr.shape[0]} rows, expected {n}")
        if abs(float(self.charges.sum())) > 1e-6:
            raise ValueError(
                f"{self.name}: charges sum to {self.charges.sum():+.6f} e, expected neutral"
            )
        if self.bonds.size and int(self.bonds.max()) >= n:
            raise ValueError(f"{self.name}: bond index out of range")
        if self.angles.size and int(self.angles.max()) >= n:
            raise ValueError(f"{self.name}: angle index out of range")
        for rows, *params in ((self.bonds, self.bond_k, self.bond_r0),
                              (self.angles, self.angle_k, self.angle_theta0)):
            for p in params:
                if p.shape[0] != rows.shape[0]:
                    raise ValueError(f"{self.name}: bonded parameter length mismatch")


def _tip3() -> SolventModel:
    """CHARMM-modified TIP3P water (toppar_water_ions.str, RESI TIP3).

    r(OH) = 0.9572 A, angle(HOH) = 104.52 deg. CHARMM's TIP3 puts LJ on the
    hydrogens as well as the oxygen.
    """
    r_oh, theta = 0.9572, np.deg2rad(104.52)
    geom = np.array([
        [0.0, 0.0, 0.0],
        [r_oh, 0.0, 0.0],
        [r_oh * np.cos(theta), r_oh * np.sin(theta), 0.0],
    ])
    return SolventModel(
        name="water",
        residue="TIP3",
        Z=np.array([8, 1, 1]),
        geometry=geom,
        charges=np.array([-0.834, 0.417, 0.417]),
        epsilon=np.array([0.1521, 0.0460, 0.0460]),
        rmin_half=np.array([1.7682, 0.2245, 0.2245]),
        bonds=np.array([[0, 1], [0, 2]]),
        bond_k=np.array([450.0, 450.0]),
        bond_r0=np.array([0.9572, 0.9572]),
        angles=np.array([[1, 0, 2]]),
        angle_k=np.array([55.0]),
        angle_theta0=np.array([104.52]),
        density_kg_m3=997.0,
        box_side_A=30.0,
        metadata={"source": "CHARMM toppar_water_ions.str, RESI TIP3"},
    )


def from_json(path) -> SolventModel:
    """Load a model emitted by ``10_extract_solvent_params.py``."""
    import json

    d = json.loads(Path(path).read_text())
    return SolventModel(
        name=d["name"],
        residue=d["residue"],
        Z=np.asarray(d["Z"], dtype=np.int32),
        geometry=np.asarray(d["geometry"], dtype=np.float64),
        charges=np.asarray(d["charges"], dtype=np.float64),
        epsilon=np.asarray(d["epsilon"], dtype=np.float64),
        rmin_half=np.asarray(d["rmin_half"], dtype=np.float64),
        bonds=np.asarray(d["bonds"], dtype=np.int32).reshape(-1, 2),
        bond_k=np.asarray(d["bond_k"], dtype=np.float64),
        bond_r0=np.asarray(d["bond_r0"], dtype=np.float64),
        angles=np.asarray(d["angles"], dtype=np.int32).reshape(-1, 3),
        angle_k=np.asarray(d["angle_k"], dtype=np.float64),
        angle_theta0=np.asarray(d["angle_theta0"], dtype=np.float64),
        density_kg_m3=float(d["density_kg_m3"]),
        box_side_A=float(d["box_side_A"]),
        metadata={"source": d.get("source", str(path))},
    )


SOLVENT_MODELS: dict[str, SolventModel] = {}
for _m in (_tip3(),):
    _m.validate()
    SOLVENT_MODELS[_m.name] = _m

# Anything extracted with 10_extract_solvent_params.py is picked up here, so
# adding a solvent needs no edit to this file. The hard-coded TIP3 above wins if
# both exist, since it is the reference implementation checked against CHARMM.
_PARAM_DIR = Path(__file__).resolve().parent / "solvent_params"
if _PARAM_DIR.is_dir():
    for _p in sorted(_PARAM_DIR.glob("*.json")):
        try:
            _model = from_json(_p)
        except Exception as _exc:  # a malformed file must not hide the good ones
            import warnings

            warnings.warn(f"skipping solvent params {_p.name}: {_exc}", stacklevel=2)
            continue
        _model.validate()
        SOLVENT_MODELS.setdefault(_model.name, _model)


def get_solvent_model(name: str) -> SolventModel:
    try:
        return SOLVENT_MODELS[name]
    except KeyError:
        raise SystemExit(
            f"no explicit parameters for solvent {name!r}; available: "
            f"{sorted(SOLVENT_MODELS)}. Add it to solvent_models.py from the "
            "CGenFF topology/parameter files."
        ) from None
