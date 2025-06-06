from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import MDAnalysis as mda
import ase
from ase import io as ase_io
from ase.data import atomic_masses as ase_data_masses
from dscribe.descriptors import MBTR
from MDAnalysis.analysis.distances import dist
import argparse
import random
from tqdm import tqdm
import os

from dscribe.descriptors import MBTR

from select_points import select_most_unique_samples


def concat_trajectory(files, output_path, selected=None):
    """Concatenate trajectory files.

    Args:
        files (list): List of file paths to concatenate
        selected (list): List of indices to select from each file

    Returns:
        Path: Path to concatenated trajectory
    """
    if selected is None:
        selected = list(range(len(files)))

    traj = []


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
                    plt.plot(
                        x,
                        mbtr_output[loc],
                        "o-",
                        label="{}-{}".format(i_species, j_species),
                    )

    if plot:
        ax.set_xlabel("Distance (Angstrom)")
        ax.legend()
        plt.show()

    species_list
    descriptor = np.array(descriptor_list)

    if plot:
        plt.matshow(descriptor, vmin=0, vmax=0.5, cmap="cubehelix_r")
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
    if residue_ids is None:
        return 1
    random.shuffle(residue_ids)
    return residue_ids[0]


def get_nearby_residues(universe, central_resid, distance=5.5):
    """Get residues within a sphere of radius distance of central residue."""
    initial_selection = f"byres sphzone {distance} (resid {central_resid}) "
    sele = universe.select_atoms(initial_selection, periodic=False)
    return list(set([_.resid for _ in list(sele)]))


def calculate_residue_distances(universe, central_resid, nearby_resids):
    """Calculate mean distances between central residue and nearby residues."""
    dist_res = []
    for resi in nearby_resids:
        if resi != central_resid:
            sela = universe.select_atoms(f"(resid {central_resid}) ")
            selb = universe.select_atoms(f"(resid {resi}) ")
            print(f"central_resid: {central_resid}, resi: {resi}")
            print(f"sela: {len(sela)}, selb: {len(selb)}")
            if len(sela) > 0 and len(selb) > 0 and len(sela) == len(selb):  
                mean_dist = dist(sela, selb)[-1, :].mean()
                dist_res.append((mean_dist, resi))
            else:
                com_dist = np.linalg.norm(sela.center_of_mass() - selb.center_of_mass())
                dist_res.append((com_dist, resi))
    dist_res.sort()
    return dist_res


def select_closest_residues(universe, central_resid, dist_res, n_closest=5):
    """Select central residue and n closest residues."""
    residue_selections = f"(resid {central_resid}) or " + " or ".join(
        [f"(resid {_[1]})" for _ in dist_res[:n_closest]]
    )
    return universe.select_atoms(residue_selections, periodic=False)


def process_selection(sele, output_path, ti, central_resid, ix, natoms, tag="", descriptors=False):
    """Process selected atoms to get descriptors and save structures."""
    # Center positions
    R = sele.positions
    R = R - R.T.mean(axis=1)

    # Get atomic numbers and species
    Z = [mass_to_atomic_number(_.mass) for _ in sele.atoms]
    atoms = ase.Atoms(Z, R)
    species = list(set(atoms.get_chemical_symbols()))



    # Save structures
    pdb_path = output_path / "pdb" / f"{tag}_{ti}_{central_resid}_{ix}{tag}.pdb"
    sele.write(pdb_path)
    ase_io.write(output_path / "xyz" / f"{tag}_{ti}_{central_resid}_{ix}{tag}.xyz", atoms)
    if descriptors:
        # Calculate descriptors
        a, b, c = get_descriptor(atoms, species, plot=False)
        return a, b, c, pdb_path
    else:
        return None, None, None, pdb_path


def extract_molecular_descriptors(
    universe,
    output_path,
    natoms,
    samples_per_frame=10,
    start=0,
    end=-1,
    stride=100,
    n_find=6,
    tag="",
    descriptors=False,
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
    n_find_minus_one = n_find - 1
    results = []

    trajectory_frames = universe.trajectory[start:end:stride]
    print("*" * 100)
    print(f"n_find: {n_find}")
    print("start", start)
    print("end", end)
    print("stride", stride)
    print("universe.trajectory", len(universe.trajectory))
    print("trajectory_frames", len(trajectory_frames))
    print("*" * 100)

    if len(trajectory_frames) == 0:
        raise ValueError("No trajectory frames found")
    if len(trajectory_frames) > 1000:
        raise ValueError("trajectory_frames is too long")

    for ti, _ in tqdm(enumerate(trajectory_frames)):
        print(f"ti: {ti}")
        residue_ids = list(range(1, len(universe.residues) + 1))

        for ix in range(samples_per_frame):
            # Select central residue
            if tag == "bulk":
                central_resid = select_central_residue(residue_ids)
            else:
                central_resid = 1
            residue_ids = [r for r in residue_ids if r != central_resid]

            # Get nearby residues and calculate distances
            nearby_resids = get_nearby_residues(universe, central_resid)
            dist_res = calculate_residue_distances(
                universe, central_resid, nearby_resids
            )

            # Select closest residues
            sele = select_closest_residues(
                universe, central_resid, dist_res, n_find_minus_one
            )
            found = list(set([_.resid for _ in list(sele)]))
            print(f"found: {found}, {len(found)}, n_find: {n_find}")
            if len(found) == n_find:
                # Process selection and save results
                result = process_selection(
                    sele, output_path, ti, central_resid, ix, natoms, tag, descriptors
                )
                results.append(result)

    # Process results
    all_descriptors_full = [_[1] for _ in results] if descriptors else None
    all_descriptors = np.array([_[1].flatten() for _ in results]) if descriptors else None
    all_pdb_filenames = [_[3] for _ in results]
    results_dict = {
        "all_pdb_filenames": all_pdb_filenames,
    }       
    if descriptors:
        results_dict["all_descriptors_full"] = all_descriptors_full
        results_dict["all_descriptors"] = all_descriptors

    return results_dict


# def find_simulation_files(sims_path, index):
#     """Find simulation files given a base path and simulation index.

#     Args:
#         sims_path (Path): Base path containing simulation directories
#         index (int): Index of simulation to process

#     Returns:
#         tuple: (logfile, psf_file, dcd_file, pdb_file, resid, sim_conds)
#     """
#     files = list(Path(sims_path).glob("*/*/log/equilibration_1_*"))
#     logfile = files[index]
#     logfile = Path(logfile).absolute()
#     print(logfile)
#     resid = logfile.parents[2].stem
#     sim_conds = str(logfile.parents[1]).split("/")[-1]

#     psf_file = logfile.parents[1] / "system.psf"
#     dcd_file = list((logfile.parents[1] / "dcd").glob("eq*_1_*dcd")).pop()
#     pdb_file = logfile.parents[1] / "pdb" / "initial.pdb"

#     return logfile, psf_file, dcd_file, pdb_file, resid, sim_conds


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

    return {"trajectory": traj, "labels": labels, "n_atoms": natoms}


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

    u, labels, natoms = setup_universe(psf_file, dcd_file, pdb_file)
    import os
    outfile_stem = logfile.stem + "_" + pdb_file.stem
    output_path = Path(args.sims_path) / "md_sampling" / outfile_stem
    os.makedirs(output_path, exist_ok=True)

    results = extract_molecular_descriptors(
        u,
        output_path,
        natoms,
        samples_per_frame=args.samples_per_frame,
        stride=args.stride,
        n_find=args.n_find,
        start=args.start,
        end=args.end,
        tag=args.tag,
        descriptors=args.descriptors,
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
    tag="",
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
        tag=tag,
    )
    return namespace


def load_ase_and_fix_atoms(pdb_fn):
    xyz_fn = str(pdb_fn).replace(".pdb", ".xyz").replace("/pdb/", "/xyz/")
    assert Path(xyz_fn).exists()
    a = ase_io.read(pdb_fn)
    a2 = ase_io.read(xyz_fn)
    a.set_atomic_numbers(a2.get_atomic_numbers())
    return a


def sample_and_save(results, output_path, key="test", tag="", descriptors=False):
    if descriptors:
        N = len(results["all_descriptors_full"])
        samples = select_most_unique_samples(results["all_descriptors"], int(N * 0.8))
    else:
        samples = None

    ase_atoms = [
        load_ase_and_fix_atoms(pdb_fn) for pdb_fn in results["all_pdb_filenames"]
    ]

    ase_dicts = [_.todict() for _ in ase_atoms]

    N = len(ase_dicts)

    for i, d in enumerate(ase_dicts):
        d["pdb_fn"] = np.array([str(results["all_pdb_filenames"][i])])
        if descriptors:
            d["desc_mbtr"] = results["all_descriptors_full"][i]
            d["train"] = np.int32(i in samples)
        # ase_atoms[i].info["set"] = "train" if i in samples else "test"
        # ase_atoms[i].info["desc_mbtr"] = results["all_descriptors"][i]
        ase_atoms[i].info["pdb_fn"] = np.array([str(results["all_pdb_filenames"][i])])

    ase_io.write(output_path / f"{key}.traj", ase_atoms)

    npz_collection = {}
    dict_keys = ase_dicts[0].keys()
    dtypes = [
        (k, ase_dicts[0][k].dtype, np.stack([ase_dicts[0][k] for _ in range(N)]).shape)
        for k in dict_keys
    ]
    data = {}
    for k in dict_keys:
        try:
            data[k] = np.stack([ase_dicts[i][k] for i in range(N)])
        except:
            print(f"Error stacking {k}")

    np.savez(output_path / f"{key}{tag}.npz", **data)

    return output_path / f"{key}{tag}.npz"


def parse_args():
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
    parser.add_argument(
        "--sim_conds", type=str, help="Simulation conditions", default="sim"
    )
    parser.add_argument("--resid", type=str, help="Residue ID", default="lig")
    parser.add_argument(
        "--psf", type=str, default="system.psf", help="Path to PSF file"
    )
    parser.add_argument(
        "--dcd", type=str, default="eq*_1_*dcd", help="Path to DCD file"
    )
    parser.add_argument(
        "--pdb", type=str, default="initial.pdb", help="Path to PDB file"
    )
    parser.add_argument("--logfile", type=str, help="Path to log file")
    parser.add_argument(
        "--output_path", type=str, default="data", help="Output directory path"
    )
    parser.add_argument("--tag", type=str, default="", help="Tag for output files")

    parser.add_argument("--descriptors", action="store_true", help="Use descriptors")

    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    if args.descriptors:
        print("Using descriptors")
    else:
        print("Not using descriptors")
    for k, v in vars(args).items():
        print(f"{k}: {v}")
    print("*" * 100)
    print("Starting sampling routine")
    print("*" * 100)
    u, labels, natoms, output_path, results = process_simulation(args)

    key = f"{args.resid}_{args.n_find}_{args.samples_per_frame}_{args.start}_{args.end}_{args.stride}_{args.tag}"
    print(f"key: {key}")
    npz_path = sample_and_save(results, output_path, key=key, tag=args.tag, descriptors=args.descriptors)
    print(f"Saved to {npz_path}")
    if args.descriptors:
        print(f"{len(results['all_descriptors'])} descriptors")
    return npz_path


if __name__ == "__main__":
    main()
