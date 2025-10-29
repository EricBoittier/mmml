import os
import psi4

from pathlib import Path
# get current directory
cwd = Path.cwd()
cube_path = Path("/home/boittier/jaxeq/cubes")
psi4_path = Path( "/home/boittier/jaxeq/psi4")

import jax.numpy as jnp
import numpy as np
from scipy.spatial.distance import cdist

psi4.set_options({'basis': 'def2-TZVP', })


def make_grid_data(surface_points):
    """
    create a grid.dat file for psi4
    :param surface_points:
    :return:
    """
    with open('grid.dat', 'w') as file:
        for xyz in surface_points:
            for c in xyz:
                file.write(str(c) + ' ')
            file.write('\n')


def get_surface_points(coordinates):
    """
    return surface points for a given molecule
    :param coordinates:
    :return:
    """
    N_points, CUTOFF = 1000, 4.0
    cutoffs = [3.0]
    surface_points_all = []
    for cutoff in cutoffs:
        monomer_coords = coordinates.copy()
        surface_points = np.random.normal(size=[N_points, 3])
        surface_points = (surface_points / np.linalg.norm(
            surface_points, axis=-1, keepdims=True)) * cutoff
        surface_points = np.reshape(
            surface_points[None] + monomer_coords[:, None], [-1, 3])
        surface_points = surface_points[
            np.where(np.all(cdist(surface_points, monomer_coords
                                  ) >= (cutoff - 1e-1), axis=-1))[0]]
        surface_points_all.append(surface_points)
    surface_points = np.concatenate(surface_points_all, axis=0)
    return surface_points


def get_points_from_cube(filename):
    pass


def get_grid_points(coordinates):
    """
    create a uniform grid of points around the molecule,
    starting from minimum and maximum coordinates of the molecule (plus minus some padding)
    :param coordinates:
    :return:
    """
    bounds = np.array([np.min(coordinates, axis=0),
                       np.max(coordinates, axis=0)])
    padding = 3.0
    bounds = bounds + np.array([-1, 1])[:, None] * padding
    grid_points = np.meshgrid(*[np.linspace(a, b, 15)
                                for a, b in zip(bounds[0], bounds[1])])

    grid_points = np.stack(grid_points, axis=0)
    grid_points = np.reshape(grid_points.T, [-1, 3])
    #  exclude points that are too close to the molecule
    grid_points = grid_points[
        #np.where(np.all(cdist(grid_points, coordinates) >= (2.0 - 1e-1), axis=-1))[0]]
        np.where(np.all(cdist(grid_points, coordinates) >= (2.5 - 1e-1), axis=-1))[0]]

    return grid_points

def esp_calc(surface_points, monomer_coords, elements):
    make_grid_data(surface_points)
    psi4_mol = psi4.core.Molecule.from_arrays(monomer_coords, elem=elements,
                                              fix_orientation=True,
                                              fix_com=True, )
    psi4.core.set_output_file('output.dat', False)
    e, wfn = psi4.energy('PBE0', molecule=psi4_mol, return_wfn=True)
    psi4.oeprop(wfn, 'GRID_ESP', 'MBIS_CHARGES', title='MBIS Multipoles')
    wfn_variables = wfn.variables()
    monopoles_ref = wfn_variables['MBIS CHARGES']
    dipoles_ref = wfn_variables['MBIS DIPOLES']
    quadrupoles_ref = wfn_variables['MBIS QUADRUPOLES']
    # multipoles dict
    multipoles = {'monopoles': monopoles_ref, 'dipoles': dipoles_ref, 'quadrupoles': quadrupoles_ref,
                  'surface_points': surface_points, 'monomer_coords': monomer_coords, 'elements': elements}
    # write to npz file
    np.savez('multipoles.npz', **multipoles)
    print("Finished ESP")



def make_psi4_dir(filename):
    """
    create a psi4 directory and change to it
    :param filename:
    :return:
    """
    print(filename)
    psi4_dir = psi4_path / filename
    if not psi4_dir.exists():
        psi4_dir.mkdir()
    os.chdir(psi4_dir)
    return psi4_dir


def read_grid(filename):
    """
    read grid.dat file
    :param filename:
    :return:
    """
    psi4_dir = psi4_path / filename.stem
    filename = psi4_dir / "grid.dat"
    with open(filename, 'r') as file:
        lines = file.readlines()
    grid = []
    for line in lines:
        grid.append([float(x) for x in line.split()])
    return np.array(grid)


def read_ref_esp(filename):
    psi4_dir = psi4_path / filename.stem
    filename = psi4_dir / "grid_esp.dat"
    with open(filename, 'r') as file:
        lines = file.readlines()
    grid = []
    for line in lines:
        grid.append([float(x) for x in line.split()])
    return np.array(grid)


# if __name__ == "__main__":
#     import argparse
#     from pathlib import Path
#
#     parser = argparse.ArgumentParser()
#     parser.add_argument("-p", "--pdb", type=str, default=None)
#     args = parser.parse_args()
#     if args.pdb is not None:
#         pdb = Path(args.pdb)
#         print(type(pdb))
#         make_psi4_dir(pdb)
#     else:
#         raise NotImplementedError
#     elements, coords = get_pdb_data(pdb)
# # <<<<<<< HEAD
#
#     # from cubes_ import cube
#     # import numpy as np
#     # cube_file = "/Users/ericboittier/Documents/github/pythonProject/cubes/gaussian/testjax.chk.p.cube"
#     # cube1 = cube(cube_file)
#     # surface_points = cube1.get_grid() * 0.529177 #get_grid_points(coords)
#     # surface_points = get_grid_points(coords)
#     # surface_points = get_surface_points(coords)
# # =======
# #     print(elements)
# #     print(coords)
#     # from project.cubes_ import cube
#     # cube = cube("/home/boittier/Documents/phd/pythonProject/cubes/gaussian/testjax.chk.p.cube")
#     # surface_points = cube.get_grid()
#     # print(surface_points.shape)
#     surface_points = get_grid_points(coords)
#     print("surface:", surface_points.shape)
# # >>>>>>> 32f69ce81791a04fb27f1ee1f2a7949a7eb5b08c
#     esp_calc(surface_points, coords, elements)
