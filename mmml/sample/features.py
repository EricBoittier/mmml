from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import MDAnalysis as mda
from rdkit import Chem
from rdkit.Chem import Draw, rdDepictor
from rdkit.Chem.Draw import rdMolDraw2D
from rdkit.Chem.inchi import MolFromInchi
import ase
from ase import io as ase_io
from ase.visualize import view as viewmol
from ase.data import atomic_masses as ase_data_masses
from dscribe.descriptors import MBTR
from MDAnalysis.analysis.distances import dist
import pandas as pd
from sklearn.linear_model import LinearRegression
from tqdm import tqdm
import os
import requests
from html.parser import HTMLParser
import random
import argparse
from PIL import Image
import io


def show_atom_number(mol, labels, type_label=True):
    for i, atom in enumerate(mol.GetAtoms()):
        l = str(labels[i].type) if type_label else labels[i].name
        atom.SetProp("atomLabel", l)
    return mol


def draw_mol(mol, fn, fs=22, w=350, h=300):
    # Do the drawing.
    d = rdMolDraw2D.MolDraw2DCairo(w, h)
    d.drawOptions().minFontSize = fs
    d.DrawMolecule(mol)
    d.FinishDrawing()
    # d.WriteDrawingText(f'{fn}.png')
    # Change the last line of the above to get a byte string.
    png = d.GetDrawingText()

    # Now read into PIL.
    img = Image.open(io.BytesIO(png))
    return img


def plot_timeseries(data, columns, xcol):
    fig, axes = plt.subplots(len(columns) - 1, 1, sharex=True, figsize=(8, 12))
    fig.suptitle("Simulation Data Over Time")

    # columns = columns.to_numpy()

    for ax, col in zip(axes, columns[1:]):
        ax.plot(data[xcol].values, data[col].values, label=col)
        colname = "\n".join(col.split(" "))
        ax.set_ylabel(colname)
        ax.legend()
        ax.grid(True)

    axes[-1].set_xlabel(xcol)
    plt.tight_layout(rect=[0, 0, 1, 0.98])
    plt.savefig("output.png")
    plt.show()


def plot_temperature_density(data, fitdf=None):
    data = data[data["Temperature (K)"] > 175.5]

    X = data["Temperature (K)"].values.reshape(-1, 1)
    y = data["Density (g/mL)"].values

    # fit polynomial degree 2
    fit = np.polyfit(X.ravel(), y, 2)
    print(fit)
    # scatter plot
    plt.figure(figsize=(8, 6))
    plt.scatter(X, y, label="Data", alpha=0.01, color="gray")

    def func(X):
        return fit[0] * X**2 + fit[1] * X + fit[2]

    plt.plot(X, func(X), "--", label="Quadratic Fit", color="r")
    # plt.plot(X, fit.predict(X), label="Linear Fit")
    plt.xlabel("Temperature (K)")
    plt.ylabel("Density (g/mL)")
    plt.legend()
    plt.grid(True)
    # diagonal line
    # plt.plot([0, 1], [0, 1], "--", color="k", transform=plt.gca().transAxes)
    plt.title("Temperature vs Density")
    plt.text(
        0.1,
        0.1,
        f"y = {fit[0]:.2e}x^2 + {fit[1]:.2e}x + {fit[2]:.2e}",
        transform=plt.gca().transAxes,
    )

    exp_df = None
    if fitdf is not None:
        exp_df = pd.read_json(fitdf)  # "../../fitdata_meoh.json"
        dens_key = "Mass density, kg/m3"
        temp_key = "Temperature, K"
        pressure_key = "Pressure, kPa"
        exp_df["simDens"] = func(exp_df[temp_key]) * 1000
        exp_df = exp_df[exp_df[pressure_key] < 102]
        exp_df = exp_df[exp_df[pressure_key] > 101]
        _exp_df = exp_df[[dens_key, temp_key]]
        _exp_df = _exp_df.dropna()
        _exp_df[dens_key] = _exp_df[dens_key] / 1000
        # regress temperature vs density
        # linear fit to the data
        expX = _exp_df[temp_key].values.reshape(-1, 1)
        y = _exp_df[dens_key].values
        fit = LinearRegression().fit(expX, y)
        print(fit.coef_, fit.intercept_)
        plt.plot(X, fit.coef_ * X + fit.intercept_, "--", label="Linear Fit", color="g")

        plt.scatter(
            _exp_df[temp_key],
            _exp_df[dens_key],
            label="Experimental Data",
            color="g",
            zorder=10,
        )

    plt.show()
    return data, exp_df


def read_data(file):
    data = pd.read_csv(file)
    columns = list(data.columns)
    print(columns)
    xcol = columns[0]
    data = data.iloc[1000:]  # remove the first row
    return data, columns, xcol


def main(file, timestep=0.5, fitdf=None):
    data, columns, xcol = read_data(file)

    data["Time (ps)"] = data[xcol] * timestep * 1e-3  # convert to picoseconds
    xcol = "Time (ps)"
    plot_timeseries(data, columns, xcol)
    return plot_temperature_density(data, fitdf)


from dscribe.descriptors import MBTR

def get_descriptor(system, species, plot=True):
    MIN = 2.0
    MAX = 10.0
    N = 100
    DECAY = 0.5
    desc = MBTR(
        species=species,
        geometry={"function": "distance"},
        grid={"min": MIN, "max": MAX, "sigma": 0.1, "n": N},
        weighting={"function": "exp", "scale": DECAY, "threshold": 1e-3},
        periodic=False,
        sparse=False,
        normalization="l2",
    )

    # No weighting
    mbtr_output = desc.create(system)

    # chemical symbol
    n_elements = len(desc.species)
    x = np.linspace(0, MAX, N)

    species_list = []
    descriptor_list = []

    # Plot k=2
    if plot:
        fig, ax = plt.subplots()
    for i in range(n_elements):
        for j in range(n_elements):
            if j >= i:
                i_species = desc.species[i]
                j_species = desc.species[j]
                loc = desc.get_location((i_species, j_species))
                species_list.append((i_species, j_species))
                descriptor_list.append(mbtr_output[loc])
                if plot:
                    plt.plot(x, mbtr_output[loc], "o-", label="{}-{}".format(i_species, j_species))

    if plot:
        ax.set_xlabel("Distance (Angstrom)")
        ax.legend()
        plt.show()

    species_list
    descriptor = np.array(descriptor_list)
    
    if plot:
        plt.matshow(descriptor, vmin=0, vmax=.5, cmap="cubehelix_r")
        plt.show()
    
    return species_list, descriptor, system

def mass_to_atomic_number(mass):
    """Convert atomic mass to atomic number.
    
    Args:
        mass (float): Atomic mass
        
    Returns:
        int: Atomic number
        
    Raises:
        ValueError: If no matching atomic number is found
    """
    idxs = np.where(abs(ase_data_masses - mass) < 0.001)[0]
    if len(idxs) == 0:
        raise ValueError(f"No matching atomic number found for mass {mass}")
    return int(idxs[0])


def select_central_residue(residue_ids):
    """Select a random residue ID from the list."""
    random.shuffle(residue_ids)
    return residue_ids[0]


def get_nearby_residues(universe, central_resid):
    """Get residues within 5.5A of central residue."""
    initial_selection = f"byres sphzone 5.5 (resid {central_resid}) "
    sele = universe.select_atoms(initial_selection, periodic=False)
    return list(set([_.resid for _ in list(sele)]))


def calculate_residue_distances(universe, central_resid, nearby_resids):
    """Calculate mean distances between central residue and nearby residues."""
    dist_res = []
    for resi in nearby_resids:
        if resi != central_resid:
            sela = universe.select_atoms(f"(resid {central_resid}) ")
            selb = universe.select_atoms(f"(resid {resi}) ")
            mean_dist = dist(sela, selb)[-1, :].mean()
            dist_res.append((mean_dist, resi))
    dist_res.sort()
    return dist_res


def select_closest_residues(universe, central_resid, dist_res, n_closest=5):
    """Select central residue and n closest residues."""
    residue_selections = f"(resid {central_resid}) or " + " or ".join(
        [f"(resid {_[1]})" for _ in dist_res[:n_closest]]
    )
    return universe.select_atoms(residue_selections, periodic=False)


def process_selection(sele, output_path, ti, central_resid, ix):
    """Process selected atoms to get descriptors and save structures."""
    # Center positions
    R = sele.positions
    R = R - R.T.mean(axis=1)

    # Get atomic numbers and species
    Z = [mass_to_atomic_number(_.mass) for _ in sele.atoms]
    atoms = ase.Atoms(Z, R)
    species = list(set(atoms.get_chemical_symbols()))

    # Calculate descriptors
    a, b, c = get_descriptor(atoms, species, plot=False)

    # Save structures
    pdb_path = output_path / "pdb" / f"{ti}_{central_resid}_{ix}.pdb"
    sele.write(pdb_path)
    ase_io.write(output_path / "xyz" / f"{ti}_{central_resid}_{ix}.xyz", atoms)

    return a, b, c, pdb_path


def extract_molecular_descriptors(
    universe, output_path, samples_per_frame=10, stride=100, n_find=6
):
    """Extract molecular descriptors from trajectory frames.

    Args:
        universe: MDAnalysis Universe object containing trajectory
        output_path: Path object for output directory
        samples_per_frame: Number of random samples to take per frame
        stride: Number of frames to skip between samples
        n_find: Number of residues to find

    Returns:
        tuple: (all_descriptors_full, all_descriptors, all_pdb_filenames)
    """
    # Create output directories
    os.makedirs(output_path / "pdb", exist_ok=True)
    os.makedirs(output_path / "xyz", exist_ok=True)

    results = []

    for ti, _ in tqdm(enumerate(universe.trajectory[::stride])):
        residue_ids = list(range(1, len(universe.residues) + 1))

        for ix in range(samples_per_frame):
            # Select central residue
            central_resid = select_central_residue(residue_ids)
            residue_ids = [r for r in residue_ids if r != central_resid]

            # Get nearby residues and calculate distances
            nearby_resids = get_nearby_residues(universe, central_resid)
            dist_res = calculate_residue_distances(
                universe, central_resid, nearby_resids
            )

            # Select closest residues
            sele = select_closest_residues(universe, central_resid, dist_res)
            found = list(set([_.resid for _ in list(sele)]))

            if len(found) == n_find:
                # Process selection and save results
                result = process_selection(sele, output_path, ti, central_resid, ix)
                results.append(result)

    # Process results
    all_descriptors_full = [_[1] for _ in results]
    all_descriptors = np.array([_[1].flatten() for _ in results])
    all_pdb_filenames = [_[3] for _ in results]
    results_dict = {
        "all_descriptors_full": all_descriptors_full,
        "all_descriptors": all_descriptors,
        "all_pdb_filenames": all_pdb_filenames,
    }

    return results_dict


def find_simulation_files(sims_path, index):
    """Find simulation files given a base path and simulation index.

    Args:
        sims_path (Path): Base path containing simulation directories
        index (int): Index of simulation to process

    Returns:
        tuple: (logfile, psf_file, dcd_file, pdb_file, resid, sim_conds)
    """
    files = list(Path(sims_path).glob("*/*/log/equilibration_1_*"))
    logfile = files[index]

    resid = logfile.parents[2].stem
    sim_conds = str(logfile.parents[1]).split("/")[-1]

    psf_file = logfile.parents[1] / "system.psf"
    dcd_file = list((logfile.parents[1] / "dcd").glob("eq*_1_*dcd")).pop()
    pdb_file = logfile.parents[1] / "pdb" / "initial.pdb"

    return logfile, psf_file, dcd_file, pdb_file, resid, sim_conds


def setup_universe(psf_file, dcd_file, pdb_file, start=0, end=None, stride=1):
    """Set up MDAnalysis universe objects.

    Args:
        psf_file (Path): Path to PSF file
        dcd_file (Path): Path to DCD file
        pdb_file (Path): Path to PDB file

    Returns:
        tuple: (universe, labels, output_path)
    """
    u2 = mda.Universe(pdb_file)
    natoms = len(u2.atoms)
    u = mda.Universe(psf_file, dcd_file, start=start, end=end, stride=stride)
    # u = u[start:end:stride]
    labels = list(u.atoms[:natoms])
    return u, labels, natoms


def sim_to_data(u, labels, natoms, output_path):
    """Convert simulation data to numpy array.
    
    Args:
        u (mda.Universe): MDAnalysis universe object
        labels (list): List of atom labels
        natoms (int): Number of atoms
        output_path (Path): Path to output directory
        
    Returns:
        dict: Dictionary containing processed simulation data
    """
    # Extract positions and create trajectory array
    positions = []
    for ts in u.trajectory:
        positions.append(u.atoms.positions[:natoms])
    
    traj = np.array(positions)
    
    # Save trajectory data
    np.save(output_path / "trajectory.npy", traj)
    
    return {
        "trajectory": traj,
        "labels": labels,
        "n_atoms": natoms
    }


def process_simulation(args):
    """Process a single simulation.

    Args:
        sims_path (str): Path to simulation directory
        index (int): Index of simulation to process
    """

    # turn all logfiles into Paths
    logfile = Path(args.logfile)
    psf_file = Path(args.psf)
    dcd_file = Path(args.dcd)
    pdb_file = Path(args.pdb)
    resid = args.resid
    sim_conds = args.sim_conds

    print(f"Processing: {resid} {sim_conds}")

    u, labels, natoms = setup_universe(
        psf_file, dcd_file, pdb_file, start=args.start, end=args.end, stride=args.stride
    )
    output_path = logfile.parents[2] / "data" / logfile.parents[1].stem

    results = extract_molecular_descriptors(
        u,
        output_path,
        samples_per_frame=args.samples_per_frame,
        stride=args.stride,
        n_find=args.n_find,
    )

    return u, labels, natoms, output_path, results


def create_args(
    logfile=None,
    psf=None,
    dcd=None,
    pdb=None,
    start=0,
    end=None,
    resid=None,
    sim_conds=None,
    stride=1,
    samples_per_frame=10,
    n_find=6,
):
    """Create arguments for process_simulation.
    
    Args:
        logfile (Path): Path to logfile
        psf (Path): Path to PSF file
        dcd (Path): Path to DCD file
        pdb (Path): Path to PDB file
        start (int): Start frame
        end (int): End frame
        resid (str): Residue ID
        sim_conds (str): Simulation conditions
        stride (int): Frame stride
        samples_per_frame (int): Number of samples per frame
        n_find (int): Number of residues to find
        
    Returns:
        argparse.Namespace: Arguments namespace
        
    Raises:
        AssertionError: If required arguments are missing
    """
    assert logfile is not None, "logfile is required"
    assert psf is not None, "psf is required"
    assert dcd is not None, "dcd is required"
    assert pdb is not None, "pdb is required"
    assert start is not None, "start is required"
    assert end is not None, "end is required"
    assert stride is not None, "stride is required"
    assert samples_per_frame is not None, "samples_per_frame is required"
    assert n_find is not None, "n_find is required"
    
    namespace = argparse.Namespace(
        logfile=logfile,
        psf=psf,
        dcd=dcd,
        pdb=pdb,
        start=start,
        end=end,
        stride=stride,
        resid=resid,
        sim_conds=sim_conds,
        samples_per_frame=samples_per_frame,
        n_find=n_find,
    )
    return namespace


def main():
    """Main function to process molecular dynamics simulation data."""
    parser = argparse.ArgumentParser(
        description="Process molecular dynamics simulation data."
    )
    parser.add_argument(
        "--sims_path",
        type=str,
        default="/home/boittier/studix/ressim",
        help="Path to simulations directory",
    )
    parser.add_argument(
        "--start",
        type=int,
        default=49,
        help="Starting frame index for processing",
    )
    parser.add_argument(
        "--end",
        type=int,
        default=50,
        help="Ending frame index for processing",
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=100,
        help="Number of frames to skip between processing",
    )
    parser.add_argument(
        "--samples_per_frame",
        type=int,
        default=10,
        help="Number of random samples to take per frame",
    )
    parser.add_argument(
        "--n_find",
        type=int,
        default=6,
        help="Number of residues to find and analyze",
    )
    parser.add_argument("--psf", type=str, default="system.psf", help="Path to PSF file")
    parser.add_argument("--dcd", type=str, default="eq*_1_*dcd", help="Path to DCD file")
    parser.add_argument("--pdb", type=str, default="initial.pdb", help="Path to PDB file")
    parser.add_argument("--logfile", type=str, help="Path to log file")
    parser.add_argument("--output_path", type=str, default="data", help="Output directory path")

    args = parser.parse_args()
    
    u, labels, natoms, output_path, results = process_simulation(args)
    return results


if __name__ == "__main__":
    main()
