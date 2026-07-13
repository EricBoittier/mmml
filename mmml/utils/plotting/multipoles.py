"""Parametric-surface visualizations of learned multipoles and MBD response.

House style (see ``docs/plot-style-gallery.md`` "Periodic landscape on a
torus"): a quantity is wrapped onto a parametric surface with
``ax.plot_surface(..., facecolors=cmap(scalar))`` rather than shown as a flat
heatmap. Here the natural surface is a **sphere around each source** whose
angular colour/radius encodes the field that source produces.

Three related views, all driven by the *same* physics already in
:mod:`mmml.models.multipoles.electrostatics`:

* :func:`plot_multipole_surfaces` -- one deformed sphere per atom/fragment,
  radius and colour set by the angular electrostatic potential of its point
  multipole (charge + dipole + quadrupole + octupole). A bare charge is a round
  sphere; a dipole shows the familiar +/- lobes; higher poles add structure.
* :func:`plot_field_slice` -- the potential + electric field of the whole set
  on a 2D plane (filled potential contours + streamlines). This wraps the
  existing :func:`~mmml.models.multipoles.electrostatics._point_multipole_potential_field_au`;
  reach for it when you want *the field itself*, not the per-source surfaces.
* :func:`plot_mbd_surfaces` / :func:`plot_dispersion_field_slice` -- the same
  language extended to the learned MBD term: per-atom polarizability spheres
  (radius) coloured by C6, and the ``-C6/r^6`` dispersion potential on a plane.

Every function takes plain arrays so it works with either a model's predictions
or hand-set analytic moments; the ``examples`` at the bottom render both.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (registers 3d projection)

from mmml.models.multipoles.electrostatics import (
    BOHR_TO_ANGSTROM,
    _point_multipole_potential_field_au,
)
from mmml.utils.plotting.styles import apply_plot_style, default_cmap

__all__ = [
    "plot_multipole_surfaces",
    "plot_field_slice",
    "plot_mbd_surfaces",
    "plot_dispersion_field_slice",
]


def _unit_sphere(n: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(x, y, z) direction cosines of an n*n theta/phi parametric sphere."""
    theta = np.linspace(0.0, np.pi, n)          # polar
    phi = np.linspace(0.0, 2.0 * np.pi, n)      # azimuth
    Theta, Phi = np.meshgrid(theta, phi)
    x = np.sin(Theta) * np.cos(Phi)
    y = np.sin(Theta) * np.sin(Phi)
    z = np.cos(Theta)
    return x, y, z


def _one_source_potential_on_sphere(
    origin_bohr: np.ndarray,
    charge: float,
    dipole_bohr: np.ndarray,
    quadrupole_bohr: np.ndarray | None,
    octupole_bohr: np.ndarray | None,
    probe_radius_bohr: float,
    n: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Angular potential of a single point multipole on a probe sphere.

    Returns the sphere's direction cosines and the potential sampled on it (so
    the caller can map potential -> radius and -> colour).
    """
    dx, dy, dz = _unit_sphere(n)
    dirs = np.column_stack([dx.ravel(), dy.ravel(), dz.ravel()])
    points = origin_bohr[None, :] + probe_radius_bohr * dirs
    quad = None if quadrupole_bohr is None else quadrupole_bohr[None]
    octu = None if octupole_bohr is None else octupole_bohr[None]
    potential, _ = _point_multipole_potential_field_au(
        points,
        origin_bohr[None, :],
        np.array([charge]),
        np.asarray(dipole_bohr).reshape(1, 3),
        quad,
        octu,
    )
    return dx, dy, dz, potential.reshape(dx.shape)


def plot_multipole_surfaces(
    origins_bohr: np.ndarray,
    charges: np.ndarray,
    dipoles_bohr: np.ndarray,
    quadrupoles_bohr: np.ndarray | None = None,
    octupoles_bohr: np.ndarray | None = None,
    *,
    atomic_numbers: Sequence[int] | None = None,
    probe_radius_angstrom: float = 1.2,
    radius_gain: float = 0.6,
    out: str | Path | None = None,
    title: str = "Learned molecular multipoles (angular potential per source)",
    style: str = "icml",
    ax=None,
):
    """One deformed sphere per source, radius+colour = its angular potential.

    The potential is signed (a dipole has + and - lobes), so colour uses the
    house **diverging** map about zero; the radius uses ``|V|`` so lobes bulge
    out. Positions are in Bohr in, drawn in Angstrom.
    """
    apply_plot_style(style)
    origins_bohr = np.asarray(origins_bohr, dtype=np.float64).reshape(-1, 3)
    charges = np.asarray(charges, dtype=np.float64).reshape(-1)
    dipoles_bohr = np.asarray(dipoles_bohr, dtype=np.float64).reshape(-1, 3)
    n_src = len(origins_bohr)
    probe_bohr = probe_radius_angstrom / BOHR_TO_ANGSTROM

    made_fig = ax is None
    if made_fig:
        fig = plt.figure(figsize=(9.0, 6.0))
        ax = fig.add_subplot(projection="3d")
    else:
        fig = ax.figure

    cmap = default_cmap("diverging")
    n = 60
    # Shared symmetric colour scale so sources are comparable.
    all_v = []
    per_source = []
    for i in range(n_src):
        q = None if quadrupoles_bohr is None else quadrupoles_bohr[i]
        o = None if octupoles_bohr is None else octupoles_bohr[i]
        dx, dy, dz, V = _one_source_potential_on_sphere(
            origins_bohr[i], charges[i], dipoles_bohr[i], q, o, probe_bohr, n
        )
        per_source.append((dx, dy, dz, V))
        all_v.append(np.abs(V))
    vmax = max(1e-9, float(np.percentile(np.concatenate([v.ravel() for v in all_v]), 98)))

    origins_ang = origins_bohr * BOHR_TO_ANGSTROM
    for i, (dx, dy, dz, V) in enumerate(per_source):
        # radius modulation: base sphere + |V|-driven bulge (normalised)
        r = 0.35 + radius_gain * np.clip(np.abs(V) / vmax, 0, 1)
        cx = origins_ang[i, 0] + r * dx
        cy = origins_ang[i, 1] + r * dy
        cz = origins_ang[i, 2] + r * dz
        facecolors = cmap(0.5 + 0.5 * np.clip(V / vmax, -1, 1))
        ax.plot_surface(
            cx, cy, cz, facecolors=facecolors, rstride=1, cstride=1,
            linewidth=0, antialiased=False, shade=False,
        )
        if atomic_numbers is not None:
            ax.text(origins_ang[i, 0], origins_ang[i, 1], origins_ang[i, 2] + r.max() + 0.15,
                    str(int(atomic_numbers[i])), ha="center", fontsize=9, color="#444")

    _equal_3d(ax, origins_ang)
    ax.set_proj_type("ortho")
    ax.set_axis_off()
    ax.set_title(title)
    import matplotlib.cm as mcm
    from matplotlib.colors import Normalize
    mappable = mcm.ScalarMappable(norm=Normalize(-vmax, vmax), cmap=cmap)
    fig.colorbar(mappable, ax=ax, shrink=0.6, pad=0.02,
                 label="point-multipole potential on probe sphere (a.u.)")
    if made_fig:
        fig.tight_layout()
        if out is not None:
            fig.savefig(out, dpi=200, bbox_inches="tight")
            plt.close(fig)
    return ax


def plot_field_slice(
    origins_bohr: np.ndarray,
    charges: np.ndarray,
    dipoles_bohr: np.ndarray,
    quadrupoles_bohr: np.ndarray | None = None,
    octupoles_bohr: np.ndarray | None = None,
    *,
    plane: str = "xy",
    span_angstrom: float = 6.0,
    grid: int = 220,
    out: str | Path | None = None,
    title: str = "Electrostatic potential + field (multipole sources)",
    style: str = "icml",
):
    """Potential contours + electric-field streamlines on a 2D plane.

    This is the "plot the field" entry point. It wraps the existing physics
    :func:`~mmml.models.multipoles.electrostatics._point_multipole_potential_field_au`
    (potential in Hartree/e, field in a.u.); nothing new is computed here beyond
    laying a grid on ``plane`` through the sources' centroid and drawing it.
    Potential has a real zero -> diverging colour; streamlines show E = -grad V.
    """
    apply_plot_style(style)
    origins_bohr = np.asarray(origins_bohr, dtype=np.float64).reshape(-1, 3)
    charges = np.asarray(charges, dtype=np.float64).reshape(-1)
    dipoles_bohr = np.asarray(dipoles_bohr, dtype=np.float64).reshape(-1, 3)

    axes = {"xy": (0, 1, 2), "xz": (0, 2, 1), "yz": (1, 2, 0)}[plane]
    a0, a1, a2 = axes
    centroid = origins_bohr.mean(axis=0)
    span_bohr = span_angstrom / BOHR_TO_ANGSTROM
    u = np.linspace(-span_bohr, span_bohr, grid)
    v = np.linspace(-span_bohr, span_bohr, grid)
    U, V = np.meshgrid(u, v)
    pts = np.zeros((U.size, 3))
    pts[:, a0] = centroid[a0] + U.ravel()
    pts[:, a1] = centroid[a1] + V.ravel()
    pts[:, a2] = centroid[a2]

    quad = quadrupoles_bohr
    octu = octupoles_bohr
    potential, field = _point_multipole_potential_field_au(
        pts, origins_bohr, charges, dipoles_bohr,
        None if quad is None else np.asarray(quad),
        None if octu is None else np.asarray(octu),
        softening_bohr=0.3,
    )
    P = potential.reshape(U.shape)
    Fx = field[:, a0].reshape(U.shape)
    Fy = field[:, a1].reshape(U.shape)

    Ua = U * BOHR_TO_ANGSTROM
    Va = V * BOHR_TO_ANGSTROM
    vmax = float(np.percentile(np.abs(P), 97))
    fig, ax = plt.subplots(figsize=(7.5, 6.6))
    cmap = default_cmap("diverging")
    cf = ax.contourf(Ua, Va, np.clip(P, -vmax, vmax),
                     levels=np.linspace(-vmax, vmax, 31), cmap=cmap, extend="both")
    speed = np.hypot(Fx, Fy)
    lw = 0.4 + 2.2 * (speed / (np.percentile(speed, 95) + 1e-12))
    ax.streamplot(Ua, Va, Fx, Fy, color="#20202090",
                  density=1.3, linewidth=np.clip(lw, 0.3, 2.2), arrowsize=0.8)
    for o in origins_bohr * BOHR_TO_ANGSTROM:
        ax.plot(o[a0] - centroid[a0] * BOHR_TO_ANGSTROM,
                o[a1] - centroid[a1] * BOHR_TO_ANGSTROM,
                "o", ms=7, mfc="white", mec="#111", mew=1.2)
    ax.set_aspect("equal")
    ax.set_xlabel(f"{'xyz'[a0]} (Å)")
    ax.set_ylabel(f"{'xyz'[a1]} (Å)")
    ax.set_title(title)
    fig.colorbar(cf, ax=ax, label="electrostatic potential (a.u.)")
    fig.tight_layout()
    if out is not None:
        fig.savefig(out, dpi=200, bbox_inches="tight")
        plt.close(fig)
    return ax


def plot_mbd_surfaces(
    positions_angstrom: np.ndarray,
    polarizabilities: np.ndarray,
    c6_coefficients: np.ndarray | None = None,
    *,
    atomic_numbers: Sequence[int] | None = None,
    radius_gain: float = 0.9,
    out: str | Path | None = None,
    title: str = "Learned MBD response: per-atom polarizability spheres",
    style: str = "icml",
    ax=None,
):
    """MBD term in the same surface language: one sphere per atom.

    Radius ~ ``alpha^(1/3)`` (polarizability volume), colour = C6 dispersion
    coefficient. Both are strictly positive -> **sequential** colour. This is
    the dispersion analogue of :func:`plot_multipole_surfaces`; feed it the
    ``polarizabilities`` / ``c6_coefficients`` from
    :func:`mmml.models.mbd.calculator.predict_mbd_from_atoms`.
    """
    apply_plot_style(style)
    pos = np.asarray(positions_angstrom, dtype=np.float64).reshape(-1, 3)
    alpha = np.asarray(polarizabilities, dtype=np.float64).reshape(-1)
    c6 = (np.asarray(c6_coefficients, dtype=np.float64).reshape(-1)
          if c6_coefficients is not None else alpha)

    made_fig = ax is None
    if made_fig:
        fig = plt.figure(figsize=(9.0, 6.0))
        ax = fig.add_subplot(projection="3d")
    else:
        fig = ax.figure

    cmap = default_cmap("sequential")
    dx, dy, dz = _unit_sphere(48)
    a_scale = np.cbrt(np.maximum(alpha, 1e-9))
    a_scale = a_scale / (a_scale.max() + 1e-12)
    c6n = (c6 - c6.min()) / (np.ptp(c6) + 1e-12)
    for i in range(len(pos)):
        r = 0.25 + radius_gain * a_scale[i]
        ax.plot_surface(
            pos[i, 0] + r * dx, pos[i, 1] + r * dy, pos[i, 2] + r * dz,
            color=cmap(c6n[i]), rstride=2, cstride=2,
            linewidth=0, antialiased=True, shade=True,
        )
        if atomic_numbers is not None:
            ax.text(pos[i, 0], pos[i, 1], pos[i, 2] + r + 0.15,
                    str(int(atomic_numbers[i])), ha="center", fontsize=9, color="#444")

    _equal_3d(ax, pos)
    ax.set_proj_type("ortho")
    ax.set_axis_off()
    ax.set_title(title)
    import matplotlib.cm as mcm
    from matplotlib.colors import Normalize
    mappable = mcm.ScalarMappable(norm=Normalize(float(c6.min()), float(c6.max())), cmap=cmap)
    fig.colorbar(mappable, ax=ax, shrink=0.6, pad=0.02, label="C$_6$ coefficient (a.u.)")
    if made_fig:
        fig.tight_layout()
        if out is not None:
            fig.savefig(out, dpi=200, bbox_inches="tight")
            plt.close(fig)
    return ax


def plot_dispersion_field_slice(
    positions_angstrom: np.ndarray,
    c6_coefficients: np.ndarray,
    *,
    plane: str = "xy",
    span_angstrom: float = 6.0,
    grid: int = 240,
    softening_angstrom: float = 0.6,
    out: str | Path | None = None,
    title: str = "MBD dispersion potential  −Σ C$_6$/r$^6$",
    style: str = "icml",
):
    """The ``-C6/r^6`` dispersion potential of the atoms on a 2D plane.

    A scalar field analogous to :func:`plot_field_slice`, but for the MBD term:
    the pairwise ``-C6/r^6`` attraction summed over atoms. Strictly negative
    (attractive) -> sequential colour on the magnitude.
    """
    apply_plot_style(style)
    pos = np.asarray(positions_angstrom, dtype=np.float64).reshape(-1, 3)
    c6 = np.asarray(c6_coefficients, dtype=np.float64).reshape(-1)
    axes = {"xy": (0, 1, 2), "xz": (0, 2, 1), "yz": (1, 2, 0)}[plane]
    a0, a1, a2 = axes
    centroid = pos.mean(axis=0)
    u = np.linspace(-span_angstrom, span_angstrom, grid)
    v = np.linspace(-span_angstrom, span_angstrom, grid)
    U, V = np.meshgrid(u, v)
    soft2 = softening_angstrom ** 2
    field = np.zeros_like(U)
    for i in range(len(pos)):
        du = U - (pos[i, a0] - centroid[a0])
        dv = V - (pos[i, a1] - centroid[a1])
        dw = centroid[a2] - pos[i, a2]
        r2 = du * du + dv * dv + dw * dw + soft2
        field += -c6[i] / r2 ** 3

    fig, ax = plt.subplots(figsize=(7.5, 6.6))
    cmap = default_cmap("sequential")
    mag = -field  # positive magnitude of an attractive potential
    vmax = float(np.percentile(mag, 99))
    cf = ax.contourf(U, V, np.clip(mag, 0, vmax),
                     levels=np.linspace(0, vmax, 28), cmap=cmap, extend="max")
    ax.contour(U, V, mag, levels=8, colors="#ffffff", linewidths=0.3, alpha=0.35)
    for i in range(len(pos)):
        ax.plot(pos[i, a0] - centroid[a0], pos[i, a1] - centroid[a1],
                "o", ms=7, mfc="white", mec="#111", mew=1.2)
    ax.set_aspect("equal")
    ax.set_xlabel(f"{'xyz'[a0]} (Å)")
    ax.set_ylabel(f"{'xyz'[a1]} (Å)")
    ax.set_title(title)
    fig.colorbar(cf, ax=ax, label=r"|dispersion potential|  $\sum C_6/r^6$ (a.u.)")
    fig.tight_layout()
    if out is not None:
        fig.savefig(out, dpi=200, bbox_inches="tight")
        plt.close(fig)
    return ax


def _equal_3d(ax, points: np.ndarray) -> None:
    """Cubic aspect box around ``points`` so spheres are not squashed."""
    c = points.mean(axis=0)
    span = max(1.0, float(np.ptp(points, axis=0).max())) * 0.5 + 1.2
    ax.set_xlim(c[0] - span, c[0] + span)
    ax.set_ylim(c[1] - span, c[1] + span)
    ax.set_zlim(c[2] - span, c[2] + span)
    try:  # matplotlib >= 3.6 can zoom the 3D box to reclaim whitespace
        ax.set_box_aspect([1, 1, 1], zoom=1.4)
    except TypeError:
        ax.set_box_aspect([1, 1, 1])
    ax.view_init(elev=22, azim=45)
