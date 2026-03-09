import os
import shutil
from io import BytesIO
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from ase import Atoms, io
from ase.data import covalent_radii
from ase.io.pov import get_bondpairs, set_high_bondorder_pairs
from ase.visualize.plot import plot_atoms
from matplotlib.offsetbox import AnnotationBbox, OffsetImage
from sklearn.decomposition import PCA

default_color_dict = {
    "Cl": [102, 227, 115],
    "C": [61, 61, 64],
    "O": [240, 10, 10],
    "N": [10, 10, 240],
    "F": [0, 232, 0],
    "H": [232, 206, 202],
    "K": [128, 50, 100],
    "X": [200, 200, 200],
}


def render_povray(
    atoms, pov_name, rotation="0x, 0y, 0z", radius_scale=0.25, color_dict=None
):

    # align the molecule to the principal axes
    pca = PCA(n_components=3)
    pca.fit(atoms.get_positions())
    atoms.set_positions(pca.transform(atoms.get_positions()))

    if color_dict is None:
        color_dict = default_color_dict

    path = Path(pov_name)
    pov_name = path.name
    base = path.parent

    radius_list = []
    for atomic_number in atoms.get_atomic_numbers():
        radius_list.append(radius_scale * covalent_radii[atomic_number])

    colors = np.array([color_dict[atom.symbol] for atom in atoms]) / 255

    bondpairs = get_bondpairs(atoms, radius=0.9)
    good_bonds = []
    good_bond_keys = []
    for _ in bondpairs:
        #  remove the Cl-Cl bonds
        if not (atoms[_[0]].symbol == "Cl" and atoms[_[1]].symbol == "Cl"):
            good_bonds.append(_)
            good_bond_keys.append((_[0], _[1]))
            good_bond_keys.append((_[1], _[0]))

    # create hydrogen bonds
    _pos = atoms.get_positions()
    _z = atoms.get_atomic_numbers()
    idx_onh = (_z == 8) | (_z == 1) | (_z == 7)
    idxs = np.where(idx_onh)[0]

    # create a mapping between atom idxs in the first atoms object
    # and the idxs in the new atoms object
    map = {}
    for i, idx in enumerate(idxs):
        map[i] = idx
    # create a new atoms object with only N, O and H atoms
    atoms_onh = Atoms(_z[idxs], _pos[idxs])
    bondpairs_onh = get_bondpairs(atoms_onh, radius=1.5)
    for _ in bondpairs_onh:
        if (map[_[0]], map[_[1]]) not in good_bond_keys:
            distance = np.linalg.norm(_pos[_[0]] - _pos[_[1]])
            # check that atom1 is H or N/O and atom2 is N/O or H
            if (_z[_[0]] == 1 and (_z[_[1]] == 7 or _z[_[1]] == 8)) or (
                _z[_[1]] == 1 and (_z[_[0]] == 7 or _z[_[0]] == 8)
            ):
                if 1.0 < distance < 3.5:
                    print(f"Adding bond between", map[_[0]], map[_[1]])
                    good_bonds.append(_)

    good_bonds = set_high_bondorder_pairs(good_bonds)

    kwargs = {  # For povray files only
        "transparent": True,  # Transparent background
        "canvas_width": 1028,  # Width of canvas in pixels
        "canvas_height": None,  # None,  # Height of canvas in pixels
        "camera_dist": 50.0,  # Distance from camera to front atom,
        "camera_type": "orthographic angle 0",  # 'perspective angle 20'
        "depth_cueing": False,
        "colors": colors,
        "bondatoms": good_bonds,
        "textures": ["jmol"] * len(atoms),
    }

    generic_projection_settings = {
        "rotation": rotation,
        "radii": radius_list,
    }

    povobj = io.write(
        pov_name, atoms, **generic_projection_settings, povray_settings=kwargs
    )

    povobj.render(
        povray_executable="/pchem-data/meuwly/boittier/home/miniforge3/envs/jaxphyscharmm/bin/povray"
    )
    png_name = pov_name.replace(".pov", ".png")
    shutil.move(png_name, base / png_name)
    return png_name


def annotate_ebc(ebc, energies, ase_atoms):
    range_ = max(
        max(ebc._proto_2D[:, 0].flatten()), max(ebc._proto_2D[:, 1].flatten())
    ) - min(min(ebc._proto_2D[:, 0].flatten()), min(ebc._proto_2D[:, 1].flatten()))
    range_ *= 1.5

    for idk, cluster_key in enumerate(ebc._cluster_ids[:]):
        cluster_members = ebc.get_cluster_members(cluster_key)
        coords = ebc._proto_2D[cluster_members]
        ccolor = ebc.cluster_colormap[idk]
        fig, ax = plt.subplots()
        plt.title(f"Cluster {cluster_key + 1}", color=ccolor)

        ax.scatter(
            coords[:, 0],
            coords[:, 1],
            c=energies[cluster_members],
            s=1,
            cmap="jet",
            vmin=0,
            vmax=100,
        )
        ax.scatter(
            ebc._proto_2D[::, 0],
            ebc._proto_2D[::, 1],
            alpha=0.01,
            color="gray",
            zorder=-1,
        )
        ax.set_xlim(-range_ / 1.2, range_ / 1.2)
        ax.set_ylim(-range_ / 1.2, range_ / 1.2)

        plot_coords = np.array([[0.2, 0.2], [0.2, 0.8], [0.8, 0.8], [0.8, 0.2]])

        cluster_members_energy = {_: float(energies[_]) for _ in cluster_members[0]}
        cm = list(cluster_members[0].copy())
        cm.sort(key=lambda x: abs(cluster_members_energy[x] - energies.mean()))

        for idx, i in enumerate(cm[:4]):
            # Use OffsetImage to embed the image into the PCA plot
            png_name = render_povray(
                ase_atoms[i], f"test-{i}-{idx}-{idk}-{cluster_key}.pov"
            )
            image = plt.imread(f"test-{i}-{idx}-{idk}-{cluster_key}.png")
            offset_image = OffsetImage(image, zoom=0.1)
            annotation = AnnotationBbox(
                offset_image,
                (plot_coords[idx][0], plot_coords[idx][1]),
                frameon=False,
                boxcoords=ax.transAxes,
                zorder=10,  # Relative to the axes of the plot
            )
            ax.add_artist(annotation)
            plot_coords[idx] = plot_coords[idx] * range_ - range_ / 2
            data_points = [ebc._proto_2D[i][0], ebc._proto_2D[i][1]]
            linexs = np.array([plot_coords[idx][0], data_points[0]]).flatten()
            lineys = np.array([plot_coords[idx][1], data_points[1]]).flatten()
            ax.plot(linexs, lineys, "--", color="k")
            del image
        plt.show()

        # clean up povray files (.ini, .pov) and png images
        for file in Path().rglob("*.pov"):
            os.remove(file)
        for file in Path().rglob("*.ini"):
            os.remove(file)
        for file in Path().rglob("*.png"):
            os.remove(file)

    return True
