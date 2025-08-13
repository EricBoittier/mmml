from ebc.Clustering import EBC

from physnetjax.analysis.analysis import (
    compute_soap_descriptors,
    remove_mean_from_multimodal_distribution,
)


def get_soap_atoms(coordinates, atomic_numbers):
    species = np.unique(np.concatenate(atomic_numbers))
    # Apply PCA
    labels = None
    soap_descr, ase_atoms = compute_soap_descriptors(
        coordinates, atomic_numbers, species, r_cut=10.0, n_max=16, l_max=5, sigma=0.5
    )
    return soap_descr, ase_atoms


def get_ebc(soap_descr, energies_mean_removed, n=10):
    ebc = EBC(n_clusters=n)
    labels = ebc.fit_transform(soap_descr[:, :], energies_mean_removed)
    _ = ebc.show()
    plt.show()
    cluster_members_energy = {
        _: float(energies_mean_removed[_]) for _ in range(int(ebc._proto_2D.shape[0]))
    }

    # plt.scatter(ebc._proto_2D[:, 0], ebc._proto_2D[:, 1], alpha=0.1, c=errors, zorder=-1)
    s = plt.scatter(
        ebc._proto_2D[:, 0][::],
        ebc._proto_2D[:, 1][::],
        alpha=1,
        c=[cluster_members_energy[_] for _ in range(int(ebc._proto_2D.shape[0]))],
        vmin=energies.min(),
        vmax=energies.max(),
        zorder=-1,
        cmap="jet",
        s=10,
    )
    plt.colorbar(s)
    plt.show()
    return ebc


def cluster_ebc_soap(coordinates, atomic_numbers, energies, remove_mean=False):
    soap_descr, ase_atoms = get_soap_atoms(coordinates, atomic_numbers)
    if remove_mean:
        energies = remove_mean_from_multimodal_distribution(energies)
    ebc = get_ebc(soap_descr, energies)
    return ebc, soap, energies, ase_atoms
