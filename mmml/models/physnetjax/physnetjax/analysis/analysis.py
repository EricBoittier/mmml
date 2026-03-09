import ase
from scipy.stats import gaussian_kde, linregress
from tqdm import tqdm


def get_metrics(x, y):
    ERROR = x - y
    RMSE = np.mean(ERROR**2) ** 0.5
    MAE = np.mean(abs(ERROR))
    return RMSE, MAE


def count_params(params):
    from jax.flatten_util import ravel_pytree

    flattened_params, unravel_fn = ravel_pytree(params)
    total_params = flattened_params.size
    return total_params


def plot(x, y, ax, units="kcal/mol", _property="", kde=True, s=1, diag=True):
    x = x.flatten()
    y = y.flatten()
    try:
        slope, intercept, r_value, p_value, std_err = linregress(x, y)
        r_value = "$1 - r^2$: {:.3E}".format(1 - r_value)
    except:
        r_value = ""

    RMSE, MAE = get_metrics(x, y)

    ax.set_aspect("equal")
    ax.text(
        0.4,
        0.85,
        f"{RMSE:.2f}/{MAE:.2f} [{units}]\n{r_value}",
        transform=ax.transAxes,
        ha="center",
        va="center",
        fontsize=10,
    )
    if kde:
        NSTEP = min(y.shape[0], 300)
        if kde == "y":
            xy = np.vstack([y])
        else:
            # Calculate the point density
            xy = np.vstack([x, y])
        z = gaussian_kde(xy[:, ::NSTEP])(xy)
        # Sort the points by density (optional, for better visualization)
        idx = z.argsort()
        x, y, z = x[idx], y[idx], z[idx]
    else:
        z = "k"
    plt.set_cmap("plasma")
    ax.scatter(x, y, alpha=0.8, c=z, s=s)
    # plt.scatter(Fs, predFs, alpha=1, color="k")
    ax.set_aspect("equal")
    x_min, x_max = min(x), max(x)
    y_min, y_max = min(y), max(y)
    ax.set_ylim(min(x_min, y_min), max(x_max, y_max))
    ax.set_xlim(min(x_min, y_min), max(x_max, y_max))
    ax.set_title(_property)
    ax.set_xlabel(f"ref. [{units}]")
    ax.set_ylabel(f"pred. [{units}]")
    if diag:
        ax.plot([0, 1], [0, 1], transform=ax.transAxes, color="gray")
    else:
        ax.axhline(0, color="gray")
    return ax


def eval(batches, model, params, batch_size=500):
    Es, Eeles, predEs, Fs, predFs, Ds, predDs, charges, outputs = (
        [],
        [],
        [],
        [],
        [],
        [],
        [],
        [],
        [],
    )
    for i, batch in tqdm(enumerate(batches)):
        output = model.apply(
            params,
            atomic_numbers=batch["Z"],
            positions=batch["R"],
            dst_idx=batch["dst_idx"],
            src_idx=batch["src_idx"],
            batch_segments=batch["batch_segments"],
            batch_size=batch_size,
            batch_mask=batch["batch_mask"],
            atom_mask=batch["atom_mask"],
        )
        # nonzero = np.nonzero(batch["Z"])
        # print(nonzero)
        if "D" in batch.keys():
            Ds.append(batch["D"])
            D = output["dipoles"]
            predDs.append(D)
            charges.append(output["charges"])
        else:
            Ds.append(0)
            predDs.append(0)
            charges.append(0)

        # print(D,batch["D"])
        Es.append(batch["E"])
        predEs.append(output["energy"])
        _f = batch["F"].flatten()
        _predf = output["forces"].flatten()
        Fs.append(_f)
        predFs.append(_predf)

        Eeles.append(output["electrostatics"])
        # print("predF.shape", _predf.shape)
    Es = np.array(Es).flatten()
    Eeles = np.array(Eeles).flatten()
    predEs = np.array(predEs).flatten()
    Fs = np.concatenate(Fs).flatten()
    predFs = np.concatenate(predFs).flatten()
    Ds = np.array(Ds)  # .flatten()
    predDs = np.array(predDs)  # .flatten()
    outputs.append(output)
    return Es, Eeles, predEs, Fs, predFs, Ds, predDs, charges, outputs


def plot_stats(
    batches, model, params, _set="", do_kde=False, batch_size=500, do_plot=True
):
    Es, Eeles, predEs, Fs, predFs, Ds, predDs, charges, outputs = eval(
        batches, model, params, batch_size=batch_size
    )
    if model.charges:
        charges = np.concatenate(charges)
        Eeles = Eeles / (ase.units.kcal / ase.units.mol)
        summed_q = charges.reshape(len(batches) * batch_size, model.natoms).sum(axis=1)
    else:
        charges = np.zeros_like(Es)
        Eeles = np.zeros_like(Es)
        summed_q = np.zeros_like(Es)
    
    Es = Es / (ase.units.kcal / ase.units.mol)
    predEs = predEs / (ase.units.kcal / ase.units.mol)
    Fs = Fs / (ase.units.kcal / ase.units.mol)
    predFs = predFs / (ase.units.kcal / ase.units.mol)

    E_rmse, E_mae = get_metrics(Es, predEs)
    
    F_rmse, F_mae = get_metrics(Fs, predFs)
    
    if model.charges:
        D_rmse, D_mae = get_metrics(Ds, predDs)
    else:
        D_rmse, D_mae = None, None

    if do_plot:
        fig, axes = plt.subplots(2, 3, figsize=(13, 7.5))
        plot(Fs, predFs, axes[0, 1], _property="$F$", kde=do_kde)
        plot(Es, predEs, axes[0, 0], _property="$E$", s=10, kde=do_kde)

        if model.charges:
            plot(Ds, predDs, axes[0, 2], _property="$D$", units=r"$e \AA$", kde=do_kde)
            plot(
                Es - Es.mean(),
                Eeles - Eeles.mean(),
                axes[1, 0],
                _property="var($E$) vs var($E_{\\rm ele}$)",
                kde=do_kde,
            )
        else:
            axes[0, 2].axis("off")
            axes[1, 0].axis("off")

        plot(
            predFs,
            abs(predFs - Fs),
            axes[1, 1],
            _property="$F$",
            kde=do_kde,
            diag=False,
        )
        q_sum_kde = "y" if do_kde else False
        if model.charges:
            plot(
                np.zeros_like(summed_q),
                summed_q,
                axes[1, 2],
                _property="$Q$",
                units="$e$",
                kde=q_sum_kde,
            )
        else:
            axes[1, 2].axis("off")

        plt.subplots_adjust(hspace=0.55)
        plt.suptitle(_set + f" (n={len(predEs.flatten())})", fontsize=20)

    output = {
        "Es": Es,
        "Eeles": Eeles,
        "predEs": predEs,
        "Fs": Fs,
        "predFs": predFs,
        "Ds": Ds,
        "predDs": predDs,
        "charges": charges,
        "outputs": outputs,
        "batches": batches,
        "E_rmse": E_rmse,
        "E_mae": E_mae,
        "F_rmse": F_rmse,
        "F_mae": F_mae,
        "D_rmse": D_rmse,
        "D_mae": D_mae,
        "n_params": count_params(params),
    }
    model_kwargs = model.return_attributes()
    output.update(model_kwargs)

    return output


from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler


# Function to compute SOAP descriptors
def compute_soap_descriptors(
    positions, atomic_numbers, species, r_cut=5.0, n_max=8, l_max=6, sigma=0.5
):
    import numpy as np
    from ase import Atoms
    from dscribe.descriptors import SOAP

    """Compute SOAP descriptors for a list of structures."""
    soap = SOAP(
        species=species,
        r_cut=r_cut,
        n_max=n_max,
        l_max=l_max,
        sigma=sigma,
        periodic=False,
        sparse=False,
        average="inner",
    )
    descriptors = []
    ase_atoms = []
    for i, (pos, nums) in tqdm(enumerate(zip(positions, atomic_numbers))):
        atoms = Atoms(
            numbers=nums,
            positions=pos - pos.mean(axis=0),
        )
        ase_atoms.append(atoms)
        desc = soap.create(atoms)
        descriptors.append(desc)
    # Convert list of arrays to a single numpy array
    return np.array(descriptors), ase_atoms


# Function to flatten 3D descriptors to 2D
def flatten_descriptors(descriptors):
    """Flatten 3D descriptors (structures x centers x features) to 2D."""
    return descriptors.reshape(descriptors.shape[0], -1)


# Function to apply PCA
def apply_pca(data, n_components=2):
    """Apply PCA to the data."""
    pca = PCA(n_components=n_components)
    reduced_data = pca.fit_transform(data)
    return reduced_data, pca


# Function to apply t-SNE
def apply_tsne(data, n_components=2, perplexity=30, random_state=42):
    """Apply t-SNE to the data."""
    tsne = TSNE(
        n_components=n_components, perplexity=perplexity, random_state=random_state
    )
    reduced_data = tsne.fit_transform(data)
    return reduced_data, tsne


# Function to visualize the projection
def visualize_projection(data, labels=None, title="Projection", c=None):
    """Visualize 2D projection of the data."""
    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(data[:, 0], data[:, 1], c=c, cmap="jet", s=15)
    if labels is not None:
        plt.legend(*scatter.legend_elements(), title="Classes")
    plt.colorbar(scatter)
    plt.title(title)
    plt.xlabel("Component 1")
    plt.ylabel("Component 2")
    # plt.show()


def get_desc(
    positions,
    atomic_numbers,
    species,
):
    """Process data through SOAP and the chosen dimensionality reduction method."""
    print("Computing SOAP descriptors...")
    descriptors, ase_atoms = compute_soap_descriptors(
        positions, atomic_numbers, species
    )
    return descriptors, ase_atoms


# Main processing function
def process_data(descriptors, method, **kwargs):

    print("Flattening descriptors...")
    flattened_descriptors = flatten_descriptors(descriptors)
    print("Scaling data...")
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(flattened_descriptors)
    print(f"Applying {method.__name__}...")
    reduced_data, model = method(scaled_data, **kwargs)
    return (
        reduced_data,
        model,
    )


import re


def clean_and_cast_to_float(s):
    # Use regex to keep only digits, periods, and minus signs
    cleaned_str = re.sub(r"[^0-9.-]", "", s)
    return float(cleaned_str[:7]) if cleaned_str else None


import matplotlib.pyplot as plt
import numpy as np

# from MDAnalysis.analysis import dihedrals


# def make_bonds_ase():
## Create a NeighborList object
# cutoffs = [r + 0.5 for r in atomic_radii]  # Slightly increase radii for bonding cutoff
# nl = NeighborList(cutoffs=cutoffs, self_interaction=False, bothways=True)
# nl.update(molecule)
#
## Get bonds automatically
# bonds = []
# for atom in range(len(molecule)):
#    indices, offsets = nl.get_neighbors(atom)
#    for i in indices:
#        if (atom, i) not in bonds and (i, atom) not in bonds:
#            bonds.append((atom, i))
#
#


def kabsch_alignment(P, Q):
    """
    Calculate the optimal rotation matrix using the Kabsch algorithm.
    Aligns points P to points Q.

    Parameters:
    - P: Coordinates of the moving frame (Nx3 numpy array).
    - Q: Coordinates of the reference frame (Nx3 numpy array).

    Returns:
    - R: Optimal rotation matrix (3x3 numpy array).
    """
    # Compute the covariance matrix
    C = np.dot(P.T, Q)
    # Singular value decomposition
    V, S, W = np.linalg.svd(C)
    # Calculate the optimal rotation matrix
    d = (np.linalg.det(V) * np.linalg.det(W)) < 0.0
    if d:
        S[-1] = -S[-1]
        V[:, -1] = -V[:, -1]

    R = np.dot(V, W)
    return R


def do_alignment(coordinates):
    frames = coordinates[
        :, :, :
    ]  # Replace with your list of frames (arrays of Nx3 coordinates)
    reference_frame = frames[0]  # First frame as the reference

    aligned_frames = []
    for frame in frames:
        # Center the frames around their centroid
        centered_frame = frame - frame.mean(axis=0)
        centered_reference = reference_frame - reference_frame.mean(axis=0)

        # Calculate the optimal rotation matrix
        R = kabsch_alignment(centered_frame, centered_reference)

        # Apply the rotation to align to the reference frame
        aligned_frame = np.dot(centered_frame, R)
        aligned_frames.append(aligned_frame)

    aligned_frames = np.array(aligned_frames)  # Aligned list of frames
    return aligned_frames


def remove_mean_from_multimodal_distribution(data, n_modes=2):
    """ """
    print(data.shape)
    data = data.reshape(data.shape[0], 1)
    print(data.shape)
    # Calculate different clusters
    from sklearn.cluster import KMeans

    kmeans = KMeans(n_clusters=n_modes, random_state=0).fit(data)
    labels = kmeans.labels_
    centers = kmeans.cluster_centers_
    # Calculate the mean of each cluster
    means = []
    for i in range(n_modes):
        means.append(data[labels == i].mean(axis=0))
    # Remove the mean from each cluster
    for i in range(n_modes):
        data[labels == i] -= means[i]
    return data
