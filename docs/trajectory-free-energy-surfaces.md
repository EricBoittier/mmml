# Free-energy surfaces from an ASE trajectory

MMML can estimate a one- or two-dimensional free-energy surface (FES) from
scalar coordinates evaluated on an ASE trajectory. The estimator constructs a
histogram and calculates

\[
F(q) = -R T \ln P(q),
\]

with the minimum occupied-bin energy shifted to zero. This is an empirical FES
of the sampled trajectory; reliable barriers require equilibrated sampling and
adequate transitions between metastable states.

## Load a trajectory

```python
from ase.io import read

frames = read("/Users/ericboittier/test4.pdb", index=":")
print(len(frames), "frames")
print(len(frames[0]), "atoms per frame")
```

`test4.pdb` contains 288 frames. Each frame has a capped peptide in a periodic
water box. ASE uses zero-based atom indices. For this topology, the central
backbone coordinates are:

- phi: `C-N-CA-C`, atoms `(4, 6, 8, 14)`
- psi: `N-CA-C-N`, atoms `(6, 8, 14, 16)`

## Calculate a two-dimensional phi/psi FES

ASE reports dihedrals in the interval 0–360 degrees. Coordinate functions can
be arbitrary Python callables that accept one `ase.Atoms` frame and return one
scalar.

```python
from mmml.utils.plotting import fes_from_trajectory, plot_fes

phi = lambda atoms: atoms.get_dihedral(4, 6, 8, 14, mic=True)
psi = lambda atoms: atoms.get_dihedral(6, 8, 14, 16, mic=True)

phi_psi_fes = fes_from_trajectory(
    frames,
    [phi, psi],
    temperature_k=300.0,
    bins=(36, 36),
    ranges=((0.0, 360.0), (0.0, 360.0)),
    energy_unit="kcal/mol",
)

fig, ax = plot_fes(
    phi_psi_fes,
    labels=(r"$\phi$ (degrees)", r"$\psi$ (degrees)"),
    max_free_energy=5.0,
)
```

![Phi/psi free-energy surface](images/plots/test4_phi_psi_fes.png)

## Calculate a one-dimensional profile

Pass one coordinate to obtain a free-energy profile.

```python
phi_fes = fes_from_trajectory(
    frames,
    [phi],
    temperature_k=300.0,
    bins=36,
    ranges=((0.0, 360.0),),
)

fig, ax = plot_fes(
    phi_fes,
    labels=(r"$\phi$ (degrees)",),
    max_free_energy=5.0,
)
```

![Phi free-energy profile](images/plots/test4_phi_fes.png)

The same pattern works for any scalar coordinate, for example:

```python
distance = lambda atoms: atoms.get_distance(0, 18, mic=True)
angle = lambda atoms: atoms.get_angle(4, 6, 8, mic=True)
radius_of_gyration = lambda atoms: atoms[:22].get_moments_of_inertia().sum()
```

For biased or reweighted trajectories, evaluate coordinates explicitly and
pass one statistical weight per frame to `calculate_fes`:

```python
from mmml.utils.plotting import calculate_fes, evaluate_coordinates

samples = evaluate_coordinates(frames, [phi, psi])
surface = calculate_fes(samples, weights=frame_weights, bins=(36, 36))
```

Empty histogram bins are assigned a finite display floor. They are not sampled
free-energy estimates and should not be interpreted as physical barriers.
