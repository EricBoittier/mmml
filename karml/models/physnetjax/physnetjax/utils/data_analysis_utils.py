import argparse

import matplotlib.pyplot as plt
import numpy as np

from physnetjax.analysis.cluster import cluster_ebc_soap
from physnetjax.utils.model_analysis_utils import load_data
from physnetjax.analysis.povray_tool import annotate_ebc

# Environment variables


def analysis(args):
    """
    Perform analysis on a specified dataset.

    :param args: Parsed arguments containing parameters for analysis.
    """
    data = load_data(
        NATOMS=args.natoms,
        PRNG=args.prng,
        files=args.files,
        num_train=args.num_train,
        num_valid=args.num_valid,
        batch_size=args.batch_size,
        load_test=args.load_test,
        load_train=args.load_train,
        load_validation=args.load_validation,
    )

    # Analyze model
    combined = data.get("test", {})
    if not combined:
        raise ValueError("No test data found for analysis.")

    # Prepare data for clustering
    energies = np.array([_["E"] for _ in combined])
    atomic_numbers = np.array([_["Z"] for _ in combined])
    coordinates = np.array([_["R"] for _ in combined])

    if len(energies) == 0 or len(atomic_numbers) == 0 or len(coordinates) == 0:
        raise ValueError("Insufficient data for clustering.")

    # Cluster data using EBC and SOAP
    ebc, soap, cluster_energies, ase_atoms = cluster_ebc_soap(
        coordinates, atomic_numbers, energies
    )

    # Annotate data
    annotate_ebc(ebc, cluster_energies, ase_atoms)

    # Optional: Visualize results here (e.g., using Matplotlib or Seaborn)
    # Example visualization of energies
    sns.histplot(energies, kde=True)
    plt.title("Energy Distribution")
    plt.xlabel("Energy")
    plt.ylabel("Frequency")
    plt.show()


def parse_args():
    """
    Parse command-line arguments for the script.

    :return: Parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Run data analysis utilities")
    parser.add_argument(
        "--files",
        type=str,
        required=True,
        nargs="+",
        help="List of file paths to load the data.",
    )
    parser.add_argument(
        "--prng", type=int, default=42, help="Pseudo-random number generator seed."
    )
    parser.add_argument(
        "--natoms", type=int, required=True, help="Number of atoms in the dataset."
    )
    parser.add_argument(
        "--num-train", type=int, default=1000, help="Number of training examples."
    )
    parser.add_argument(
        "--num-valid", type=int, default=200, help="Number of validation examples."
    )
    parser.add_argument(
        "--batch-size", type=int, default=32, help="Batch size for data loading."
    )
    parser.add_argument(
        "--load-test", action="store_true", help="Flag to load test data."
    )
    parser.add_argument(
        "--load-train", action="store_true", help="Flag to load training data."
    )
    parser.add_argument(
        "--load-validation", action="store_true", help="Flag to load validation data."
    )
    return parser.parse_args()


def main():
    """
    Entry point of the script.
    """
    args = parse_args()
    try:
        # Run the analysis with user-provided arguments
        analysis(args)
    except Exception as e:
        print(f"Error during analysis: {e}")


if __name__ == "__main__":
    main()
