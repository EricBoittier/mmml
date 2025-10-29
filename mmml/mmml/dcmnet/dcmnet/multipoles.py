from pathlib import Path

import ase.visualize
import matplotlib.pyplot as plt
import numpy as np
import patchworklib as pw
from ase.visualize.plot import plot_atoms
from patchworklib import Brick

# set colormap
plt.set_cmap("bwr")


def check_symmetric_and_traceless(Q):
    """
    Check if the 3x3 matrix Q is symmetric and traceless.

    Parameters:
    Q : numpy.ndarray
        The 3x3 matrix to be checked.

    Returns:
    tuple of bool
        Returns (is_symmetric, is_traceless), where each is a boolean indicating the property.
    """
    # Check symmetry: Q should be equal to its transpose
    is_symmetric = np.allclose(Q, Q.T)
    # Check traceless: Trace of Q should be zero
    is_traceless = np.isclose(np.trace(Q), 0)

    return (is_symmetric, is_traceless)


def make_traceless(Q):
    """
    Adjust the diagonal elements of a 3x3 matrix Q to make it traceless.

    Parameters:
    Q : numpy.ndarray
        The 3x3 matrix to be adjusted.

    Returns:
    numpy.ndarray
        The adjusted 3x3 matrix that is traceless.
    """
    trace_Q = np.trace(Q)
    correction = trace_Q / 3
    Q_traceless = Q.copy()
    np.fill_diagonal(Q_traceless, Q_traceless.diagonal() - correction)

    return Q_traceless


def plot_3d(grid, esp, atoms=None):
    scat_size = 10
    fig = plt.figure(figsize=(10, 10))
    ax1 = fig.add_subplot(321, projection="3d")
    ax1.scatter(*grid.T, c=esp, vmin=-0.015, vmax=0.015, s=scat_size, alpha=1)
    # ax1.scatter(*atoms.positions.T,
    # c=atoms.numbers, cmap='Paired'
    # , s=100)
    ax1.set_xlim(-10, 10)
    ax1.set_ylim(-10, 10)
    ax1.set_zlim(-10, 10)
    # label axes
    ax1.set_xlabel("X")
    ax1.set_ylabel("Y")
    # ax1.set_zlabel('Z')
    ax1.set_zticks([])
    # set rotation angle
    ax1.view_init(elev=90, azim=-90)
    # ax1.axis('off')
    ax1.grid("off")

    if atoms is not None:
        ax2 = fig.add_subplot(322)
        plot_atoms(atoms, ax2, rotation=("0x,0y,0z"), scale=0.05)
        # turn off axis labels, ticks, and grid
        ax2.axis("off")

        ax3 = fig.add_subplot(323, projection="3d")
        ax3.scatter(*grid.T, c=esp, vmin=-0.015, vmax=0.015, s=scat_size, alpha=1)
        # ax3.scatter(*atoms.positions.T,
        #             c=atoms.numbers, cmap='Paired'
        #             , s=100)
        ax3.view_init(elev=0, azim=0)
        ax3.set_xlim(-10, 10)
        ax3.set_ylim(-10, 10)
        ax3.set_zlim(-10, 10)
        # ax3.set_xlabel('X')
        ax3.set_xticks([])
        ax3.set_ylabel("Y")
        ax3.set_zlabel("Z")
        # ax3.axis('off')
        # atoms
        ax4 = fig.add_subplot(324)
        plot_atoms(atoms, ax4, rotation=("-90x,-90y,0z"), scale=0.05)
        ax4.axis("off")

        ax5 = fig.add_subplot(325, projection="3d")
        ax5.scatter(*grid.T, c=esp, vmin=-0.015, vmax=0.015, s=scat_size, alpha=1)
        # ax5.scatter(*atoms.positions.T,
        #             c=atoms.numbers, cmap='Paired'
        #             , s=100)
        ax5.view_init(elev=0, azim=90)
        ax5.set_xlim(-10, 10)
        ax5.set_ylim(-10, 10)
        ax5.set_zlim(-10, 10)
        ax5.set_xlabel("X")
        # ax5.set_ylabel('Y')
        ax5.set_yticks([])
        ax5.set_zlabel("Z")
        # ax5.axis('off')
        # atoms
        ax6 = fig.add_subplot(326)
        plot_atoms(atoms, ax6, rotation=("-90x,180y,0z"), scale=0.5)
        ax6.axis("off")

        # make a horizontal colorbar that spans two columns
        cbar_ax = fig.add_axes([0.8, 0.8, 0.1, 0.1])
        fig.colorbar(
            ax1.get_children()[0], cax=cbar_ax, label="Electrostatic Potential (a.u.)"
        )

        spec5 = fig.add_gridspec(
            ncols=2, nrows=3, width_ratios=[1, 0.5], height_ratios=[1, 1, 1]
        )

        plt.subplots_adjust(wspace=-0.55, hspace=0.0)

        plt.show()


def read_xyz(testdata: Path):
    """
    Read the xyz file and return the coordinates and atom types.

    Parameters:
    """
    # read xyz file
    xyz = np.genfromtxt(testdata / "val-ala_0_0.xyz", skip_header=2, usecols=(1, 2, 3))
    atom_types = np.genfromtxt(
        testdata / "val-ala_0_0.xyz", skip_header=2, usecols=(0), dtype=str
    )


def load_data(
    testdata: Path, cull: bool = False, cull_min: float = 3.0, cull_max: float = 4.5
) -> dict:
    # print(testdata)
    # read npz file
    npz = np.load(testdata / "multipoles.npz")
    xyz = npz["monomer_coords"]
    atom_types = npz["elements"]
    grid = np.genfromtxt(testdata / "grid.dat")
    esp = np.genfromtxt(testdata / "grid_esp.dat")

    if cull:
        from scipy.spatial.distance import cdist

        grid_idx = np.where(np.all(cdist(grid, xyz) >= (cull_min - 1e-10), axis=-1))[0]
        grid = grid[grid_idx]
        esp = esp[grid_idx]
        grid_idx = np.where(np.all(cdist(grid, xyz) >= (cull_max - 1e-10), axis=-1))[0]
        grid_idx = [_ for _ in range(grid.shape[0]) if _ not in grid_idx]
        grid = grid[grid_idx]
        esp = esp[grid_idx]

    npz = np.load(testdata / "multipoles.npz")
    monopoles = npz["monopoles"]
    dipoles = npz["dipoles"]
    quadrupoles = npz["quadrupoles"]
    print("npz keys: ", npz.keys())

    return {
        "xyz": xyz,
        "atom_types": atom_types,
        "grid": grid,
        "esp": esp,
        "monopoles": monopoles,
        "dipoles": dipoles,
        "quadrupoles": quadrupoles,
    }


def calc_mono_esp(data):
    monopole_esp = np.zeros(data["esp"].shape)
    for i in range(data["monopoles"].shape[0]):
        for grid_point in range(data["grid"].shape[0]):
            r = (data["grid"][grid_point] - data["xyz"][i]) * 1.8897259886
            r_norm = np.linalg.norm(r)
            monopole_esp[grid_point] += data["monopoles"][i] / r_norm
    return monopole_esp


def calc_dipole_esp(data):
    calc_esp = np.zeros(data["esp"].shape)
    for i in range(data["dipoles"].shape[0]):
        for grid_point in range(data["grid"].shape[0]):
            r = (data["grid"][grid_point] - data["xyz"][i]) * 1.8897259886
            r_norm = np.linalg.norm(r)
            calc_esp[grid_point] += np.dot(data["dipoles"][i], r) / r_norm**3
    return calc_esp


def calc_quad_esp(data):
    calc_esp = np.zeros(data["esp"].shape)
    for i in range(data["quadrupoles"].shape[0]):
        for grid_point in range(data["grid"].shape[0]):
            r = (data["grid"][grid_point] - data["xyz"][i]) * 1.8897259886
            r_norm = np.linalg.norm(r)
            v = np.dot(r, np.dot(data["quadrupoles"][i], r)) / (2 * r_norm**5)
            calc_esp[grid_point] += v
    return calc_esp


def plot_esp(esp, calc_esp, title=""):
    print(len(esp))
    rmse = np.sqrt(np.mean((esp * 627.509 - calc_esp * 627.509) ** 2))
    plt.scatter(esp, calc_esp)
    plt.xlabel("ESP")
    plt.ylabel("Calculated ESP")
    plt.title(title + "\n" "ESP vs Calculated ESP (RMSE = {:.4f})".format(rmse))
    plt.plot([-0.05, 0.05], [-0.05, 0.05], color="red")
    ax = plt.gca()
    ax.set_aspect("equal", adjustable="box")
    plt.show()


def calc_esp_from_multipoles(data):
    monopole_esp = calc_mono_esp(data)
    dipole_esp = calc_dipole_esp(data)
    quad_esp = calc_quad_esp(data)
    return monopole_esp, dipole_esp, quad_esp


def plot_moments(data):
    # 3 panel figure
    fig, axs = plt.subplots(1, 3, figsize=(5, 5), sharey=True, sharex=True)
    axs[0].imshow(np.repeat(data["monopoles"], 9, axis=1), cmap="bwr", vmin=-1, vmax=1)
    # print(data["monopoles"].shape)
    axs[0].set_title("$l=0$")
    axs[1].imshow(np.repeat(data["dipoles"], 3, axis=1), cmap="bwr", vmin=-1, vmax=1)
    # print(data["dipoles"].shape)
    axs[1].set_title("$l=1$")
    # print(data["quadrupoles"].shape)
    # print(data["quadrupoles"])
    # print(data["quadrupoles"].reshape(data["quadrupoles"].shape[0], 9))
    # for i in range(data["quadrupoles"].shape[0]):
    #     data["quadrupoles"][i] = make_traceless(data["quadrupoles"][i])

    axs[2].imshow(
        data["quadrupoles"].reshape(data["quadrupoles"].shape[0], 9),
        cmap="bwr",
        vmin=-1,
        vmax=1,
    )
    axs[2].set_title("$l=2$")
    # set wspace
    for i in range(3):
        axs[i].set_xticks([])
        axs[i].set_yticks([])
    # print(data["atom_types"].tolist())
    axs[0].set_yticks(
        range(len(data["elements"].tolist())),
        # data["atom_types"].tolist()
    )
    atom_types = data["elements"].tolist()
    axs[0].set_yticklabels(atom_types)

    plt.subplots_adjust(wspace=0.0, hspace=0.0)
    plt.show()


def plot_esp_slices(data):
    """
    Three panel figure with xy slice, xz slice, and yz slice of the ESP.
    :param data:
    :return:
    """
    fig, axs = plt.subplots(1, 3, figsize=(5, 5), sharey=True, sharex=True)
    # print(data["grid"].shape)
    axs[0].scatter(
        data["grid"][:, 0],
        data["grid"][:, 1],
        c=data["esp"],
        vmin=-0.05,
        vmax=0.05,
        s=1,
    )
    axs[0].set_title("XY Slice")
    axs[1].scatter(
        data["grid"][:, 0],
        data["grid"][:, 2],
        c=data["esp"],
        vmin=-0.05,
        vmax=0.05,
        s=1,
    )
    axs[1].set_title("XZ Slice")
    axs[2].scatter(
        data["grid"][:, 1],
        data["grid"][:, 2],
        c=data["esp"],
        vmin=-0.05,
        vmax=0.05,
        s=1,
    )
    axs[2].set_title("YZ Slice")
    for i in range(3):
        axs[i].set_xticks([])
        axs[i].set_yticks([])
    plt.subplots_adjust(wspace=0.0, hspace=0.0)
    plt.show()


if __name__ == "__main__":
    # testdata = Path("/home/boittier/Documents/phd/pythonProject/"
    #                 "data/ala-gly_1_0")
    testdata = Path("/Users/ericboittier/jaxeq/ala-gly_1_0")
    #                 "data/ala-gly_1_0")
    data = load_data(testdata, cull=True, cull_max=5.0)

    print("Quad. shape", data["quadrupoles"].shape)

    monopole_esp, dipole_esp, quad_esp = calc_esp_from_multipoles(data)

    plot_esp(data["esp"], monopole_esp, title="Monopole")
    np.savetxt("grid.csv", data["grid"], delimiter=",")
    np.savetxt("mono.csv", monopole_esp, delimiter=",")

    # plot_esp(data["esp"], dipole_esp, title="Dipole")
    plot_esp(data["esp"], monopole_esp + dipole_esp, title="Monopole + Dipole")
    np.savetxt("dipole.csv", dipole_esp, delimiter=",")

    plot_esp(
        data["esp"],
        monopole_esp + dipole_esp + quad_esp,
        title="Monopole + Dipole + Quadrupole",
    )
    np.savetxt("quadrupole.csv", quad_esp, delimiter=",")

    plot_moments(data)

    atoms = ase.Atoms(data["atom_types"], data["xyz"])
    # add unit cell
    atoms.set_cell(10 * np.eye(3))
    # center the atoms
    atoms.center()
    atoms.set_pbc([True, True, True])

    plot_3d(data["grid"], (monopole_esp), atoms=atoms)

    plot_3d(data["grid"], (monopole_esp + dipole_esp), atoms=atoms)

    plot_3d(data["grid"], (monopole_esp + dipole_esp + quad_esp), atoms=atoms)

    plt.show()
