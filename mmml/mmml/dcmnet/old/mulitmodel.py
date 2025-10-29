import functools
import os
from pathlib import Path
import pandas as pd
import e3x
import flax.linen as nn
import jax
import numpy as np
import optax
import ase

# Disable future warnings.
import warnings

warnings.simplefilter(action="ignore", category=FutureWarning)
import matplotlib.pyplot as plt

plt.set_cmap("bwr")
import jax.numpy as jnp
from jax import vmap
from multipoles import plot_3d

batch_size = 1
NATOMS = 60


def make_charge_xyz(mono, dipo, batch, batch_size, n_dcm):
    # n_dcm = mono.shape[1]
    i = 0
    b1_ = batch["atomic_numbers"].reshape(batch_size, 60)[i]
    c1_ = batch["mono"].reshape(batch_size, 60)[i]
    # print(b1_)
    nonzero = np.nonzero(c1_)
    dc = dipo.reshape(batch_size, 60, 3, n_dcm)
    dc = np.moveaxis(dc, -1, -2)
    dc = dc.reshape(batch_size, 60 * n_dcm, 3)
    dcq = mono.reshape(batch_size, 60 * n_dcm, 1)
    dcq = np.moveaxis(dcq, -1, -2)
    dcq = dcq.reshape(batch_size, 60 * n_dcm, 1)
    idx_nozero = len(nonzero[0]) * n_dcm

    n_atoms = idx_nozero // n_dcm

    atomwise_charge_array = np.zeros((n_atoms, n_dcm, 4))

    for count, (xyz, q) in enumerate(zip(dc[i][:idx_nozero], dcq[i][:idx_nozero])):
        # print(count, count // n_dcm, xyz, q)
        atomwise_charge_array[count // n_dcm, count % n_dcm, :3] = xyz
        atomwise_charge_array[count // n_dcm, count % n_dcm, 3] = q[0]

    # print(atomwise_charge_array.shape)

    return dc, dcq, atomwise_charge_array


def plot_3d_models(mono, dc, dcq, batch):
    n_dcm = mono.shape[1]
    i = 0
    b1_ = batch["atomic_numbers"].reshape(batch_size, 60)[i]
    c1_ = batch["mono"].reshape(batch_size, 60)[i]
    print(b1_)
    nonzero = np.nonzero(c1_)
    i = 0
    xyz = batch["positions"].reshape(batch_size, 60, 3)[i][nonzero]
    elem = batch["atomic_numbers"].reshape(batch_size, 60)[i][nonzero]

    from ase import Atoms
    from ase.visualize import view

    mol = Atoms(elem, xyz)
    V1 = view(mol, viewer="x3d")
    idx = len(nonzero[0]) * n_dcm
    dcmol = Atoms(["X" if _ > 0 else "He" for _ in dcq[i][:idx]], dc[i][:idx])
    V2 = view(dcmol, viewer="x3d")
    combined = dcmol + mol
    V3 = view(combined, viewer="x3d")
    return V1, V2, V3


def evaluate_dc(batch, dipo, mono, batch_size, ndcm, plot=False, rcut=10000):
    esp_dc_pred = esp_mono_loss_pots(
        dipo, mono, batch["vdw_surface"], batch["mono"], batch_size, ndcm
    )

    mono_pred = esp_loss_pots(
        batch["positions"],
        batch["mono"],
        batch["vdw_surface"],
        batch["mono"],
        batch_size,
    )

    non_zero = np.nonzero(batch["mono"])

    if plot:

        plt.scatter(batch["mono"][non_zero], mono.sum(axis=-1).squeeze()[non_zero])
        loss = esp_loss_eval(
            batch["mono"][non_zero], mono.sum(axis=-1).squeeze()[non_zero], None
        )
        plt.title(loss)
        plt.plot([-1, 1], [-1, 1], c="k", alpha=0.5)
        plt.xlim(-1, 1)
        plt.ylim(-1, 1)
        ax = plt.gca()
        ax.set_aspect("equal")
        plt.show()

        for id in range(batch_size):
            plt.scatter(
                esp_dc_pred[id][: batch["ngrid"][id]],
                batch["esp"][id][: batch["ngrid"][id]],
                alpha=0.1,
            )

        ax = plt.gca()
        plt.xlim(-0.1, 0.1)
        plt.ylim(-0.1, 0.1)
        plt.plot([-1, 1], [-1, 1], c="k", alpha=0.5)
        ax.set_aspect("equal")
        plt.show()

    esp_errors = []

    for mbID in range(batch_size):
        xyzs = batch["positions"].reshape(batch_size, 60, 3)
        vdws = batch["vdw_surface"][mbID][: batch["ngrid"][mbID]]
        diff = xyzs[mbID][:, None, :] - vdws[None, :, :]
        r = np.linalg.norm(diff, axis=-1)
        min_d = np.min(r, axis=-2)
        wheremind = np.where(min_d < rcut, min_d, 0)
        idx_cut = np.nonzero(wheremind)[0]
        loss1 = esp_loss_eval(
            esp_dc_pred[mbID][: batch["ngrid"][mbID]][idx_cut],
            batch["esp"][mbID][: batch["ngrid"][mbID]][idx_cut],
            batch["ngrid"][mbID],
        )
        loss2 = esp_loss_eval(
            mono_pred[mbID][: batch["ngrid"][mbID]][idx_cut],
            batch["esp"][mbID][: batch["ngrid"][mbID]][idx_cut],
            batch["ngrid"][mbID],
        )
        esp_errors.append([loss1, loss2])

        if plot:

            fig = plt.figure(figsize=(12, 6))

            ax1 = fig.add_subplot(151, projection="3d")
            s = ax1.scatter(
                *batch["vdw_surface"][mbID][: batch["ngrid"][mbID]][idx_cut].T,
                c=clip_colors(batch["esp"][mbID][: batch["ngrid"][mbID]][idx_cut]),
                vmin=-0.015,
                vmax=0.015,
            )
            ax1.set_title(f"GT {mbID}")

            ax2 = fig.add_subplot(152, projection="3d")
            s = ax2.scatter(
                *batch["vdw_surface"][mbID][: batch["ngrid"][mbID]][idx_cut].T,
                c=clip_colors(esp_dc_pred[mbID][: batch["ngrid"][mbID]][idx_cut]),
                vmin=-0.015,
                vmax=0.015,
            )
            ax2.set_title(loss1)

            ax4 = fig.add_subplot(153, projection="3d")
            s = ax4.scatter(
                *batch["vdw_surface"][mbID][: batch["ngrid"][mbID]][idx_cut].T,
                c=clip_colors(
                    esp_dc_pred[mbID][: batch["ngrid"][mbID]][idx_cut]
                    - batch["esp"][mbID][: batch["ngrid"][mbID]][idx_cut]
                ),
                vmin=-0.015,
                vmax=0.015,
            )

            ax3 = fig.add_subplot(154, projection="3d")
            s = ax3.scatter(
                *batch["vdw_surface"][mbID][: batch["ngrid"][mbID]][idx_cut].T,
                c=clip_colors(mono_pred[mbID][: batch["ngrid"][mbID]][idx_cut]),
                vmin=-0.015,
                vmax=0.015,
            )
            ax3.set_title(loss2)

            ax5 = fig.add_subplot(155, projection="3d")
            s = ax5.scatter(
                *batch["vdw_surface"][mbID][: batch["ngrid"][mbID]][idx_cut].T,
                c=clip_colors(
                    mono_pred[mbID][: batch["ngrid"][mbID]][idx_cut]
                    - batch["esp"][mbID][: batch["ngrid"][mbID]][idx_cut]
                ),
                vmin=-0.015,
                vmax=0.015,
            )

            for _ in [ax1, ax2, ax3]:
                _.set_xlim(-10, 10)
                _.set_ylim(-10, 10)
                _.set_zlim(-10, 10)
            plt.show()

    return esp_errors, mono_pred


def prepare_datasets(key, num_train, num_valid, filename="esp2000.npz"):
    # Load the dataset.
    dataset = np.load(filename)

    for k, v in dataset.items():
        print(k, v.shape)

    dataR = dataset["R"]
    dataZ = dataset["Z"]
    dataMono = dataset["mono"]
    dataEsp = dataset["esp"]
    dataVDW = dataset["vdw_surface"]
    dataNgrid = dataset["n_grid"]

    # Make sure that the dataset contains enough entries.
    num_data = len(dataR)
    print(num_data)
    num_draw = num_train + num_valid
    if num_draw > num_data:
        raise RuntimeError(
            f"datasets only contains {num_data} points, "
            f"requested num_train={num_train}, num_valid={num_valid}"
        )

    # Randomly draw train and validation sets from dataset.
    choice = np.asarray(
        jax.random.choice(key, num_data, shape=(num_draw,), replace=False)
    )
    train_choice = choice[:num_train]
    valid_choice = choice[num_train:]

    atomic_numbers = dataZ

    # Collect and return train and validation sets.
    train_data = dict(
        atomic_numbers=jnp.asarray(atomic_numbers[train_choice]),
        ngrid=jnp.array(dataNgrid[train_choice]),
        positions=jnp.asarray(dataR[train_choice]),
        mono=jnp.asarray(dataMono[train_choice]),
        esp=jnp.asarray(dataEsp[train_choice]),
        vdw_surface=jnp.asarray(dataVDW[train_choice]),
    )
    valid_data = dict(
        atomic_numbers=jnp.asarray(atomic_numbers[valid_choice]),
        positions=jnp.asarray(dataR[valid_choice]),
        mono=jnp.asarray(dataMono[valid_choice]),
        ngrid=jnp.array(dataNgrid[valid_choice]),
        esp=jnp.asarray(dataEsp[valid_choice]),
        vdw_surface=jnp.asarray(dataVDW[valid_choice]),
    )
    print("...")
    print("...")
    for k, v in train_data.items():
        print(k, v.shape)
    print("...")
    for k, v in valid_data.items():
        print(k, v.shape)

    return train_data, valid_data


def prepare_batches(key, data, batch_size):
    # Determine the number of training steps per epoch.
    data_size = len(data["mono"])
    steps_per_epoch = data_size // batch_size

    # Draw random permutations for fetching batches from the train data.
    perms = jax.random.permutation(key, data_size)
    perms = perms[
        : steps_per_epoch * batch_size
    ]  # Skip the last batch (if incomplete).
    perms = perms.reshape((steps_per_epoch, batch_size))

    # Prepare entries that are identical for each batch.
    num_atoms = len(data["atomic_numbers"][0])
    batch_segments = jnp.repeat(jnp.arange(batch_size), num_atoms)
    offsets = jnp.arange(batch_size) * num_atoms
    dst_idx, src_idx = e3x.ops.sparse_pairwise_indices(num_atoms)
    dst_idx = (dst_idx + offsets[:, None]).reshape(-1)
    src_idx = (src_idx + offsets[:, None]).reshape(-1)
    # Assemble and return batches.
    return [
        dict(
            mono=data["mono"][perm].reshape(-1),
            ngrid=data["ngrid"][perm].reshape(-1),
            esp=data["esp"][perm],  # .reshape(-1),
            vdw_surface=data["vdw_surface"][perm],  # .reshape(-1, 3),
            atomic_numbers=data["atomic_numbers"][perm].reshape(-1),
            positions=data["positions"][perm].reshape(-1, 3),
            dst_idx=dst_idx,
            src_idx=src_idx,
            batch_segments=batch_segments,
        )
        for perm in perms
    ]


def nan_safe_coulomb_potential(q, r):
    potential = jnp.where(jnp.isnan(r) | (r == 0.0), 0.0, q / (r * 1.88973))
    return potential


def calc_esp(charge_positions, charge_values, grid_positions, mono):
    chg_mask = jnp.where(mono != 0, 1.0, 0.0)
    # Expand the grid positions and charge positions to compute all pairwise differences
    diff = grid_positions[:, None, :] - charge_positions[None, :, :]
    # Compute the Euclidean distance between each grid point and each charge
    r = jnp.linalg.norm(diff, axis=-1)
    sum_chg = jnp.sum(chg_mask * charge_values)
    # jax.debug.print("x = {x}", x=sum_chg)
    avg_chg = jnp.sum(chg_mask * charge_values) / jnp.sum(chg_mask)
    new_charge_values = charge_values - avg_chg
    # sum_chg_new = jnp.sum(chg_mask * new_charge_values)
    # jax.debug.print("x2 = {x}", x=sum_chg_new)
    C = nan_safe_coulomb_potential((chg_mask * new_charge_values)[None, :], r)
    V = jnp.sum(C, axis=-1)
    return V


def esp_loss_eval(pred, target, ngrid):
    target = target.flatten()
    esp_non_zero = np.nonzero(target)
    l2_loss = optax.l2_loss(pred[esp_non_zero], target[esp_non_zero])
    esp_loss = np.mean(l2_loss)
    esp_loss = esp_loss * 1
    return esp_loss


batched_electrostatic_potential = vmap(calc_esp, in_axes=(0, 0, 0, 0), out_axes=0)


def clip_colors(c):
    return np.clip(c, -0.015, 0.015)


@functools.partial(jax.jit, static_argnames=("batch_size", "esp_w", "n_dcm"))
def esp_mono_loss(
    dipo_prediction,
    mono_prediction,
    esp_target,
    vdw_surface,
    mono,
    batch_size,
    esp_w,
    n_dcm,
):
    """ """
    nonzero = jnp.nonzero(mono, size=batch_size * 60)
    l2_loss_mono = optax.l2_loss(mono_prediction.sum(axis=-1), mono)
    mono_loss = jnp.mean(l2_loss_mono[nonzero])

    d = dipo_prediction.reshape(batch_size, NATOMS, 3, n_dcm)
    d = jnp.moveaxis(d, -1, -2)
    d = d.reshape(batch_size, NATOMS * n_dcm, 3)
    mono = jnp.repeat(mono.reshape(batch_size, NATOMS), n_dcm, axis=-1)
    m = mono_prediction.reshape(batch_size, NATOMS * n_dcm)

    batched_pred = batched_electrostatic_potential(d, m, vdw_surface, mono).flatten()
    esp_target = esp_target.flatten()
    esp_non_zero = jnp.nonzero(esp_target, size=batch_size * 3143)

    l2_loss = optax.l2_loss(batched_pred, esp_target)
    esp_loss = jnp.mean(l2_loss[esp_non_zero])
    esp_loss = esp_loss * esp_w
    return esp_loss + mono_loss


def esp_mono_loss_pots(
    dipo_prediction, mono_prediction, vdw_surface, mono, batch_size, n_dcm
):
    d = dipo_prediction.reshape(batch_size, NATOMS, 3, n_dcm)
    d = jnp.moveaxis(d, -1, -2)
    d = d.reshape(batch_size, NATOMS * n_dcm, 3)
    mono = jnp.repeat(mono.reshape(batch_size, NATOMS), n_dcm, axis=-1)
    m = mono_prediction.reshape(batch_size, NATOMS * n_dcm)

    batched_pred = batched_electrostatic_potential(d, m, vdw_surface, mono)

    return batched_pred


def esp_loss_pots(dipo_prediction, mono_prediction, vdw_surface, mono, batch_size):
    d = dipo_prediction.reshape(batch_size, NATOMS, 3)
    mono = mono.reshape(batch_size, NATOMS)
    m = mono_prediction.reshape(batch_size, NATOMS)
    batched_pred = batched_electrostatic_potential(d, m, vdw_surface, mono)

    return batched_pred


def mean_absolute_error(prediction, target, batch_size):
    nonzero = jnp.nonzero(target, size=batch_size * 60)
    return jnp.mean(jnp.abs(prediction[nonzero] - target[nonzero]))


class MessagePassingModel(nn.Module):
    features: int = 32
    max_degree: int = 2
    num_iterations: int = 3
    num_basis_functions: int = 8
    cutoff: float = 5.0
    max_atomic_number: int = 17
    n_dcm: int = 4

    def mono(
        self, atomic_numbers, positions, dst_idx, src_idx, batch_segments, batch_size
    ):
        # 1. Calculate displacement vectors.
        positions_dst = e3x.ops.gather_dst(positions, dst_idx=dst_idx)
        positions_src = e3x.ops.gather_src(positions, src_idx=src_idx)
        displacements = positions_src - positions_dst  # Shape (num_pairs, 3).

        # 2. Expand displacement vectors in basis functions.
        basis = e3x.nn.basis(
            # Shape (num_pairs, 1, (max_degree+1)**2, num_basis_functions).
            displacements,
            num=self.num_basis_functions,
            max_degree=self.max_degree,
            radial_fn=e3x.nn.reciprocal_bernstein,
            cutoff_fn=functools.partial(e3x.nn.smooth_cutoff, cutoff=self.cutoff),
        )

        x = e3x.nn.Embed(
            num_embeddings=self.max_atomic_number + 1, features=self.features
        )(atomic_numbers)

        # 4. Perform iterations (message-passing + atom-wise refinement).
        for i in range(self.num_iterations):
            # Message-pass.
            if i == self.num_iterations - 1:  # Final iteration.
                y = e3x.nn.MessagePass(
                    max_degree=max_degree, include_pseudotensors=False
                )(x, basis, dst_idx=dst_idx, src_idx=src_idx)
            else:
                y = e3x.nn.MessagePass()(x, basis, dst_idx=dst_idx, src_idx=src_idx)
            y = e3x.nn.add(x, y)
            # Atom-wise refinement MLP.
            y = e3x.nn.Dense(self.features)(y)
            y = e3x.nn.silu(y)
            y = e3x.nn.Dense(self.features)(y)
            # Residual connection.
            x = e3x.nn.add(x, y)

        x = e3x.nn.TensorDense(
            features=self.n_dcm,
            max_degree=1,
            include_pseudotensors=False,
        )(x)

        atomic_mono = e3x.nn.change_max_degree_or_type(
            x, max_degree=0, include_pseudotensors=False
        )
        element_bias = self.param(
            "element_bias",
            lambda rng, shape: jnp.zeros(shape),
            (self.max_atomic_number + 1),
        )
        atomic_mono = nn.Dense(
            self.n_dcm,
            use_bias=False,
        )(atomic_mono)
        atomic_mono = atomic_mono.squeeze(axis=1)
        atomic_mono = atomic_mono.squeeze(axis=1)
        atomic_mono += element_bias[atomic_numbers][:, None]

        atomic_dipo = x[:, 1, 1:4, :]
        atomic_dipo = e3x.nn.silu(atomic_dipo)
        atomic_dipo = jnp.clip(atomic_dipo, a_min=-0.3, a_max=0.3)
        atomic_dipo += positions[:, :, None]

        return atomic_mono, atomic_dipo

    @nn.compact
    def __call__(
        self,
        atomic_numbers,
        positions,
        dst_idx,
        src_idx,
        batch_segments=None,
        batch_size=None,
    ):
        if batch_segments is None:
            batch_segments = jnp.zeros_like(atomic_numbers)
            batch_size = 1

        return self.mono(
            atomic_numbers, positions, dst_idx, src_idx, batch_segments, batch_size
        )


def flatten(xss):
    return [x for xs in xss for x in xs]


def process_df(errors):
    h2kcal = 627.509
    df = pd.DataFrame(flatten(errors))
    df["model"] = df[0].apply(lambda x: np.sqrt(x) * h2kcal)
    df["mono"] = df[1].apply(lambda x: np.sqrt(x) * h2kcal)
    df["dif"] = df["model"] - df["mono"]
    return df


def get_predictions(mono_dc2, dipo_dc2, batch, nDCM):
    mono = mono_dc2
    dipo = dipo_dc2

    esp_dc_pred = esp_mono_loss_pots(
        dipo, mono, batch["vdw_surface"], batch["mono"], batch_size, nDCM
    )

    mono_pred = esp_loss_pots(
        batch["positions"],
        batch["mono"],
        batch["vdw_surface"],
        batch["mono"],
        batch_size,
    )
    return esp_dc_pred, mono_pred


def reshape_dipole(dipo, nDCM):
    d = dipo.reshape(1, 60, 3, nDCM)
    d = jnp.moveaxis(d, -1, -2)
    d = d.reshape(1, 60 * nDCM, 3)
    return d




def get_atoms_dcmol(batch, mono, d, nDCM):
    end = len(batch["atomic_numbers"].nonzero()[0])
    atoms = ase.Atoms(
        numbers=batch["atomic_numbers"][:end], positions=batch["positions"][:end, :]
    )
    dcmol = ase.Atoms(
        ["X" if _ > 0 else "He" for _ in mono.flatten()[: end * nDCM]], d[0][: end * nDCM]
    )
    return atoms, dcmol, end


import rdkit
from rdkit import Chem
from rdkit.Chem import MolFromXYZBlock
from rdkit.Chem import rdDetermineBonds
from rdkit.Chem import Draw


def getXYZblock(batch):
    end = len(batch["atomic_numbers"].nonzero()[0])
    atoms = ase.Atoms(
        numbers=batch["atomic_numbers"][:end], positions=batch["positions"][:end, :]
    )
    xyzBlock = f"{end}\n"
    for s, xyz in zip(atoms.symbols, atoms.positions):
        xyzBlock += f"\n{s} {xyz[0]:.3f} {xyz[1]:.3f} {xyz[2]:.3f}"
    return xyzBlock


def get_rdkit(batch):
    raw_mol = MolFromXYZBlock(getXYZblock(batch))
    conn_mol = Chem.Mol(raw_mol)
    rdDetermineBonds.DetermineConnectivity(conn_mol, charge=0)
    # DetermineBondOrders
    rdDetermineBonds.DetermineBondOrders(conn_mol, charge=0)
    Chem.SanitizeMol(conn_mol)
    Chem.Kekulize(conn_mol)
    bond_moll = Chem.Mol(conn_mol)
    smi = Chem.MolToSmiles(Chem.RemoveHs(bond_moll))
    print(smi)
    mol = Chem.MolFromSmiles(smi)
    img = Draw.MolToImage(mol)
    return img


from scipy.spatial.distance import cdist


def create_plots2(mono_dc2, dipo_dc2, batch, nDCM):
    esp_dc_pred, mono_pred = get_predictions(mono_dc2, dipo_dc2, batch, nDCM)
    dipo_dc2 = reshape_dipole(dipo_dc2, nDCM)
    atoms, dcmol, end = get_atoms_dcmol(batch, mono_dc2, dipo_dc2, nDCM)

    grid = batch["vdw_surface"][0]
    # esp = esp_dc_pred[0]
    # esp = batch["esp"][0]
    esp = esp_dc_pred[0] - batch["esp"][0]

    print(
        "rmse:",
        jnp.mean(optax.l2_loss(esp_dc_pred[0] * 627.503, batch["esp"][0] * 627.503)),
    )

    xyz = batch["positions"][:end]

    cull_min = 2.5
    cull_max = 4.0
    grid_idx = np.where(np.all(cdist(grid, xyz) >= (cull_min - 1e-10), axis=-1))[0]
    print(grid_idx)
    grid_idx = jnp.asarray(grid_idx)
    grid = grid[grid_idx]
    esp = esp[grid_idx]
    grid_idx = np.where(np.all(cdist(grid, xyz) >= (cull_max - 1e-10), axis=-1))[0]
    grid_idx = [_ for _ in range(grid.shape[0]) if _ not in grid_idx]
    grid_idx = jnp.asarray(grid_idx)
    grid = grid[grid_idx]
    esp = esp[grid_idx]
    # try:
    #     display(get_rdkit(batch))
    # except:
    #     pass
    plot_3d(grid, esp, atoms=atoms + dcmol)
    return atoms, dcmol, grid, esp, esp_dc_pred[0]


def get_lowest_loss(path, df=False):
    paths = []
    losses = []
    for _ in Path(path).glob("*.pkl"):
        loss = float((_.stem).split("-")[1])
        paths.append(_)
        losses.append(loss)
    if df:
        ans = pd.DataFrame([paths, losses]).T.sort_values(1)
        print(ans)
        return ans
    else:
        ans = pd.DataFrame([paths, losses]).T.sort_values(1).iloc[0][0]
        print(ans)
        return ans


def apply_model(model, params, batch, batch_size):
    mono_dc2, dipo_dc2 = model.apply(
        params,
        atomic_numbers=batch["atomic_numbers"],
        positions=batch["positions"],
        dst_idx=batch["dst_idx"],
        src_idx=batch["src_idx"],
        batch_segments=batch["batch_segments"],
        batch_size=batch_size,
    )
    return mono_dc2, dipo_dc2


def plot_model(DCM2, params, batch, batch_size, nDCM):
    mono_dc2, dipo_dc2 = apply_model(DCM2, params, batch, batch_size)
    
    esp_errors, mono_pred = evaluate_dc(batch, dipo_dc2, mono_dc2, batch_size, nDCM, plot=True)
    
    atoms, dcmol, grid, esp, esp_dc_pred = create_plots2(
        mono_dc2, dipo_dc2, batch, nDCM
    )
    outDict = {
        "mono": mono_dc2,
        "dipo": dipo_dc2,
        "esp_errors": esp_errors,
        "atoms": atoms,
        "dcmol": dcmol,
        "grid": grid,
        "esp": esp,
        "esp_dc_pred": esp_dc_pred,
        "esp_mono_pred": mono_pred,
    }
    return outDict


# Load model parameters.
dcm1_params = pd.read_pickle(get_lowest_loss("checkpoints/dcm1-10000.0/"))
dcm2_params = pd.read_pickle(get_lowest_loss("checkpoints/dcm2-10000.0/"))
dcm3_params = pd.read_pickle(get_lowest_loss("checkpoints/dcm3-10000.0/"))
dcm4_params = pd.read_pickle(get_lowest_loss("checkpoints/dcm4-10000.0/"))

# Model hyperparameters.
features = 32
max_degree = 2
num_iterations = 3
num_basis_functions = 16
cutoff = 4.0

# Create models
DCM1 = MessagePassingModel(
    features=features,
    max_degree=max_degree,
    num_iterations=num_iterations,
    num_basis_functions=num_basis_functions,
    cutoff=cutoff,
    n_dcm=1,
)
DCM2 = MessagePassingModel(
    features=features,
    max_degree=max_degree,
    num_iterations=num_iterations,
    num_basis_functions=num_basis_functions,
    cutoff=cutoff,
    n_dcm=2,
)
DCM3 = MessagePassingModel(
    features=features,
    max_degree=max_degree,
    num_iterations=num_iterations,
    num_basis_functions=num_basis_functions,
    cutoff=cutoff,
    n_dcm=3,
)
DCM4 = MessagePassingModel(
    features=features,
    max_degree=max_degree,
    num_iterations=num_iterations,
    num_basis_functions=num_basis_functions,
    cutoff=cutoff,
    n_dcm=4,
)


def get_esp_rmse(atomwise_charge_array2, batch):
    xyz2 = atomwise_charge_array2[:, :, :3].reshape(
        14 * atomwise_charge_array2.shape[1], 3
    )
    q2 = atomwise_charge_array2[:, :, 3].reshape(14 * atomwise_charge_array2.shape[1])
    res2 = calc_esp(xyz2, q2, batch["vdw_surface"][0], q2)
    true = batch["esp"][0]
    esp_non_zero = np.nonzero(true)
    l2_loss = optax.l2_loss(res2[esp_non_zero], true[esp_non_zero])
    print(np.mean(l2_loss))



def get_esp_rmse_from_combined(combined, batch, get_esp=False):
    xyz2 = combined[:, :3]
    q2 = combined[:, 3]
    res2 = calc_esp(xyz2, q2, batch["vdw_surface"][0], q2)
    true = batch["esp"][0]
    esp_non_zero = np.nonzero(true)
    l2_loss = optax.l2_loss(res2[esp_non_zero], true[esp_non_zero])
    mse = np.mean(l2_loss)
    if get_esp:
        return mse, res2
    return mse


def plot_3d_combined(combined, batch):
    xyz2 = combined[:, :3]
    q2 = combined[:, 3]
    i = 0
    nonzero = np.nonzero(batch["atomic_numbers"].reshape(batch_size, 60)[i])
    xyz = batch["positions"].reshape(batch_size, 60, 3)[i][nonzero]
    elem = batch["atomic_numbers"].reshape(batch_size, 60)[i][nonzero]

    from ase import Atoms
    from ase.visualize import view

    mol = Atoms(elem, xyz)
    V1 = view(mol, viewer="x3d")
    dcmol = Atoms(["X" if _ > 0 else "He" for _ in q2], xyz2)
    V2 = view(dcmol, viewer="x3d")
    combined = dcmol + mol
    V3 = view(combined, viewer="x3d")
    return V1, V2, V3



def combine_chg_arrays(batch, atomwise_charge_array1, atomwise_charge_array2):
    i = 0
    nonzero = np.nonzero(batch["atomic_numbers"].reshape(batch_size, 60)[i])
    xyz = batch["positions"].reshape(batch_size, 60, 3)[i][nonzero]
    elem = batch["atomic_numbers"].reshape(batch_size, 60)[i][nonzero]

    chg_qs = []

    for i, element in enumerate(elem):
        # print(i, element)
        if element == 1:
            chg_qs.append(atomwise_charge_array1[i])
        else:
            chg_qs.append(atomwise_charge_array2[i])
    result = np.concatenate(chg_qs)
    print(result.shape)
    return result

# def make_charge_xyz(mono, dipo, batch):
#     n_dcm = mono.shape[1]
#     i = 0
#     a1_ = mono.reshape(batch_size, 60, n_dcm)[i]
#     b1_ = batch['atomic_numbers'].reshape(batch_size, 60)[i]
#     c1_ = batch["mono"].reshape(batch_size, 60)[i]
#     print(b1_)
#     nonzero = np.nonzero(c1_)
#     dc = dipo.reshape(batch_size,60,3,n_dcm)
#     dc = np.moveaxis(dc, -1, -2)
#     dc = dc.reshape(batch_size, 60*n_dcm, 3)
#     dcq = mono.reshape(batch_size,60*n_dcm,1)
#     dcq = np.moveaxis(dcq, -1, -2)
#     dcq = dcq.reshape(batch_size, 60*n_dcm, 1)
#     idx_nozero = len(nonzero[0])*n_dcm

#     n_atoms = idx_nozero // n_dcm

#     atomwise_charge_array = np.zeros((n_atoms, n_dcm, 4))
    
#     for count, (xyz, q) in enumerate(zip(dc[i][:idx_nozero], dcq[i][:idx_nozero])):
#         print(count, count // n_dcm, xyz, q)
#         atomwise_charge_array[count // n_dcm, count % n_dcm, :3] = xyz
#         atomwise_charge_array[count // n_dcm, count % n_dcm, 3] = q[0]

#     print(atomwise_charge_array.shape)

#     return dc, dcq, atomwise_charge_array


def plot_3d_combined(combined, batch):
    xyz2 = combined[:,:3]
    q2 = combined[:,3]
    i = 0
    nonzero = np.nonzero(batch["atomic_numbers"].reshape(batch_size,60)[i])
    xyz = batch['positions'].reshape(batch_size,60,3)[i][nonzero]
    elem = batch["atomic_numbers"].reshape(batch_size,60)[i][nonzero]
    
    from ase import Atoms
    from ase.visualize import view
    
    mol = Atoms(elem, xyz)
    V1 = view(mol, viewer="x3d")
    dcmol = Atoms(["X" if _ > 0 else "He" for _ in q2], xyz2)
    V2 = view(dcmol, viewer="x3d")
    combined = dcmol + mol
    V3 = view(combined, viewer="x3d")
    return V1, V2, V3




def plot_esp(esp, batch, rcut=4.0):
        mbID = 0
        xyzs = batch['positions'].reshape(batch_size,60, 3)
        vdws = batch['vdw_surface'][mbID][:batch['ngrid'][mbID]]
        diff = xyzs[mbID][:, None, :] - vdws[None, :, :]
        r = np.linalg.norm(diff, axis=-1)
        min_d = np.min(r, axis=-2)
        wheremind = np.where( min_d < rcut, min_d, 0)
        idx_cut = np.nonzero(wheremind)[0]

        mono_pred =  esp_loss_pots(batch['positions'], batch['mono'],
                            batch['vdw_surface'], batch['mono'], batch_size)

        loss_mono = optax.l2_loss(mono_pred[0][idx_cut]* 627.509, batch['esp'][0][idx_cut] * 627.509)
        loss_mono = np.mean(loss_mono)
        loss_dc = optax.l2_loss(esp[idx_cut] * 627.509, batch['esp'][0][idx_cut] * 627.509)
        loss_dc = np.mean(loss_dc)

        fig = plt.figure(figsize=(12,6))

        # set white background
        fig.patch.set_facecolor('white')
        # whitebackground in 3d
        fig.patch.set_alpha(0.0)

        ax1 = fig.add_subplot(151, projection='3d')
        s = ax1.scatter(*batch['vdw_surface'][mbID][:batch['ngrid'][mbID]][idx_cut].T,
                       c=clip_colors(batch['esp'][mbID][:batch['ngrid'][mbID]][idx_cut]),
                       vmin=-0.015, vmax=0.015)
        ax1.set_title(f"GT {mbID}")

        ax2 = fig.add_subplot(152, projection='3d')
        s = ax2.scatter(*batch['vdw_surface'][mbID][:batch['ngrid'][mbID]][idx_cut].T,
                       c=clip_colors(esp[idx_cut]),
                       vmin=-0.015, vmax=0.015)
        ax2.set_title(loss_dc)

        ax4 = fig.add_subplot(153, projection='3d')
        s = ax4.scatter(*batch['vdw_surface'][mbID][:batch['ngrid'][mbID]][idx_cut].T,
                       c=clip_colors(esp[idx_cut]
                                    - batch['esp'][mbID][:batch['ngrid'][mbID]][idx_cut]
                                    ),
                       vmin=-0.015, vmax=0.015)


        ax3 = fig.add_subplot(154, projection='3d')
        s = ax3.scatter(*batch['vdw_surface'][mbID][:batch['ngrid'][mbID]][idx_cut].T,
                       c=clip_colors(mono_pred[mbID][:batch['ngrid'][mbID]][idx_cut]),
                       vmin=-0.015, vmax=0.015)
        ax3.set_title(loss_mono)

        ax5 = fig.add_subplot(155, projection='3d')
        s = ax5.scatter(*batch['vdw_surface'][mbID][:batch['ngrid'][mbID]][idx_cut].T,
                       c=clip_colors(mono_pred[mbID][:batch['ngrid'][mbID]][idx_cut]
                                    - batch['esp'][mbID][:batch['ngrid'][mbID]][idx_cut]
                                    ),
                       vmin=-0.015, vmax=0.015)

        for _ in [ax1,ax2,ax3]:
            _.set_xlim(-10,10)
            _.set_ylim(-10,10)
            _.set_zlim(-10,10)
        plt.show()


# optimize the fit to the esp
from scipy.optimize import minimize


class OptimizeCombined:
    def __init__(self, combined, batch):
        self.combined = combined
        self.batch = batch

    def loss_fn(self, x0):
        x0 = x0.reshape(self.combined.shape)
        new_combined = self.combined + x0
        loss = get_esp_rmse_from_combined(new_combined)
        # print(loss)
        return loss

    def esp_from_opt(self, x0, batch):
        x0 = x0.reshape(self.combined.shape)
        new_combined = self.combined + x0
        # do charge equilibration
        sum_of_charges = np.sum(new_combined[:,3])
        new_combined[:,3] = new_combined[:,3] - sum_of_charges / new_combined.shape[0]

        return get_esp_rmse_from_combined(new_combined, self.batch, get_esp=True)[1], new_combined

    def loss_fn_only_q(self, x0):
        new_combined = self.combined.copy()
        new_combined[:,3] += x0
        loss = get_esp_rmse_from_combined(new_combined, self.batch)
        # print(loss)
        return loss

    def esp_from_opt_only_q(self, x0, batch):
        # x0 = x0.reshape(combined[:,3].shape)
        new_combined = self.combined.copy()
        new_combined[:,3] += x0
        # do charge equilibration
        sum_of_charges = np.sum(new_combined[:,3])
        new_combined[:,3] = new_combined[:,3] - sum_of_charges / new_combined.shape[0]

        return get_esp_rmse_from_combined(new_combined, self.batch, get_esp=True)[1], new_combined

    def optimize(self):
        x0 = np.zeros(self.combined.shape[0])
        print(x0.shape)
        bounds = [(-0.1, 0.1)] * len(x0)
        res = minimize(self.loss_fn_only_q, x0,
                       # method='Nelder-Mead',
                       method='COBYLA',
                       bounds=bounds,
                       # callback=callback,
                       options={'xatol': 1e-8, 'disp': True,
                                'adaptive': True,  # default is False
                                'ftol': 1e-3,  # default is 1e-8
                                'eps': 1e-3,  # default is 1e-8
                                'maxiter': 4000, 'maxfev': 1000,
                                })
        print(res.message)
        combined_pred_esp, new_combined = self.esp_from_opt_only_q(res.x, self.batch)
        return res, combined_pred_esp, new_combined


