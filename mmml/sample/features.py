from pathlib import Path
import numpy as np
import numpy.linalg
import matplotlib.pyplot as plt
import MDAnalysis as mda
import rdkit
from rdkit.Chem.inchi import MolFromInchi
import dscribe
import sys
import pandas as pd
from sklearn.linear_model import LinearRegression
from rdkit import Chem
from rdkit.Chem import Draw, rdDepictor
from rdkit.Chem.Draw import rdMolDraw2D
import ase
from ase.visualize import view as viewmol
from dscribe.descriptors import MBTR
from tqdm import tqdm
from MDAnalysis.analysis import distances as mda_dist
from MDAnalysis.analysis.distances import dist
import os
import requests
from html.parser import HTMLParser
import ase
ase_data_masses = ase.data.atomic_masses
from MDAnalysis.analysis import distances as mda_dist
from MDAnalysis.analysis.distances import dist as dist
from tqdm import tqdm
import ase
from ase.visualize import view as viewmol


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
        exp_df = pd.read_json(fitdf) #"../../fitdata_meoh.json"
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
    


def mass_to_atomic_number(mass):
    idxs = np.where(abs(ase_data_masses - mass) < 0.001)
    atomic_number = int(idxs[0])
    return atomic_number



def extract_molecular_descriptors(universe, output_path, samples_per_frame=10, stride=100):
    """Extract molecular descriptors from trajectory frames.
    
    Args:
        universe: MDAnalysis Universe object containing trajectory
        output_path: Path object for output directory
        samples_per_frame: Number of random samples to take per frame
        stride: Number of frames to skip between samples
    
    Returns:
        tuple: (all_descriptors_full, all_descriptors, all_pdb_filenames)
    """
    # Create output directories
    os.makedirs(output_path / "pdb", exist_ok=True)
    os.makedirs(output_path / "xyz", exist_ok=True)
    
    results = []

    for ti, _ in tqdm(enumerate(universe.trajectory[::stride])):
        residue_ids = list(range(1, len(universe.residues)+1))
        for ix in range(samples_per_frame):
            random.shuffle(residue_ids)
            i = residue_ids[0]
            
            # Select atoms in sphere around central residue
            initial_selection = f"byres sphzone 5.5 (resid {i}) "
            sele = universe.select_atoms(initial_selection, periodic=False)
            resids = list(set([_.resid for _ in list(sele)]))
            
            # Calculate distances to nearby residues
            dist_res = []
            for resi in resids:
                if resi != i:
                    sela = universe.select_atoms(f"(resid {i}) ")
                    selb = universe.select_atoms(f"(resid {resi}) ")
                    mean_dist = dist(sela, selb)[-1,:].mean()
                    dist_res.append((mean_dist, resi))
            dist_res.sort()
            
            # Select 6 closest residues
            residue_selections = f"(resid {i}) or " + " or ".join([f"(resid {_[1]})" for _ in dist_res[:5]])
            sele = universe.select_atoms(residue_selections, periodic=False)
            found = list(set([_.resid for _ in list(sele)]))
            residue_ids = [_ for _ in residue_ids if _ != i]
            
            if len(found) == 6:
                # Center positions and calculate descriptors
                R = sele.positions
                R = R - R.T.mean(axis=1)
                Z = [mass_to_atomic_number(_.mass) for _ in sele.atoms]
                atoms = ase.Atoms(Z, R)
                species = list(set(atoms.get_chemical_symbols()))
                
                # Get descriptors and save structures
                a, b, c = get_descriptor(atoms, species, plot=False)
                d = output_path / "pdb" / f"{ti}_{i}_{ix}.pdb"
                sele.write(d)
                ase.io.write(output_path / "xyz" / f"{ti}_{i}_{ix}.xyz", atoms)
                results.append((a, b, c, d))

    # Process results
    all_descriptors_full = [_[1] for _ in results]
    all_descriptors = np.array([_[1].flatten() for _ in results])
    all_pdb_filenames = [_[3] for _ in results]
    
    return all_descriptors_full, all_descriptors, all_pdb_filenames



sims_path = Path("/home/boittier/studix/ressim")
files = list(sims_path.glob("*/*/log/equilibration_1_*"))


i = 49
logfile = files[i]
resid = logfile.parents[2].stem
sim_conds = str(logfile.parents[1]).split("/")[-1]
print(resid, sim_conds)

psf_file = logfile.parents[1] / "system.psf"
dcd_file = list((logfile.parents[1] / "dcd" ).glob("eq*_1_*dcd")).pop()
pdb_file = logfile.parents[1] / "pdb" / "initial.pdb"

u2 = mda.Universe(pdb_file)
natoms = len(u2.atoms)
u = mda.Universe(psf_file, dcd_file)
labels = list(u.atoms[:natoms])

output_path = logfile.parents[2] / "data" / logfile.parents[1].stem
os.makedirs(output_path / "pdb", exist_ok=True)
os.makedirs(output_path / "xyz", exist_ok=True)

