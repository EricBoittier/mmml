import numpy as np
import optax
import ase

# Disable future warnings.
import warnings

warnings.simplefilter(action="ignore", category=FutureWarning)
import matplotlib.pyplot as plt

plt.set_cmap("bwr")
import jax.numpy as jnp

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


def get_atoms_dcmol(batch, mono, d, nDCM):
    end = len(batch["atomic_numbers"].nonzero()[0])
    atoms = ase.Atoms(
        numbers=batch["atomic_numbers"][:end], positions=batch["positions"][:end, :]
    )
    dcmol = ase.Atoms(
        ["X" if _ > 0 else "He" for _ in mono.flatten()[: end * nDCM]],
        d[0][: end * nDCM],
    )
    return atoms, dcmol, end


from scipy.spatial.distance import cdist


def get_esp_rmse(atomwise_charge_array2, batch):
    xyz2 = atomwise_charge_array2[:, :, :3].reshape(
        14 * atomwise_charge_array2.shape[1], 3
    )
    q2 = atomwise_charge_array2[:, :, 3].reshape(14 * atomwise_charge_array2.shape[1])
    res2 = calc_esp(xyz2, q2, batch["vdw_surface"][0])
    true = batch["esp"][0]
    esp_non_zero = np.nonzero(true)
    l2_loss = optax.l2_loss(res2[esp_non_zero], true[esp_non_zero])
    print(np.mean(l2_loss))


def get_esp_rmse_from_combined(combined, batch, get_esp=False):
    xyz2 = combined[:, :3]
    q2 = combined[:, 3]
    res2 = calc_esp(xyz2, q2, batch["vdw_surface"][0])
    true = batch["esp"][0]
    esp_non_zero = np.nonzero(true)
    l2_loss = 2 * optax.l2_loss(
        res2[esp_non_zero] * 627.503, true[esp_non_zero] * 627.503
    )
    mse = np.mean(l2_loss) ** 0.5
    if get_esp:
        return mse, res2
    return mse


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


def combine_chg_arrays_indexed(
    batch, atomwise_charge_array1, atomwise_charge_array2, indices
):
    i = 0
    nonzero = np.nonzero(batch["atomic_numbers"].reshape(batch_size, 60)[i])
    xyz = batch["positions"].reshape(batch_size, 60, 3)[i][nonzero]
    elem = batch["atomic_numbers"].reshape(batch_size, 60)[i][nonzero]

    chg_qs = []

    for i, element in enumerate(elem):
        if i in indices:
            chg_qs.append(atomwise_charge_array2[i])
        else:
            chg_qs.append(atomwise_charge_array1[i])

    result = np.concatenate(chg_qs)
    print(result.shape)
    return result


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
        sum_of_charges = np.sum(new_combined[:, 3])
        new_combined[:, 3] = new_combined[:, 3] - sum_of_charges / new_combined.shape[0]

        return (
            get_esp_rmse_from_combined(new_combined, self.batch, get_esp=True)[1],
            new_combined,
        )

    def loss_fn_only_q(self, x0):
        new_combined = self.combined.copy()
        new_combined[:, 3] += x0
        loss = get_esp_rmse_from_combined(new_combined, self.batch)
        # print(loss)
        return loss

    def esp_from_opt_only_q(self, x0, batch):
        new_combined = self.combined.copy()
        new_combined[:, 3] += x0
        # do charge equilibration
        sum_of_charges = np.sum(new_combined[:, 3])
        new_combined[:, 3] = new_combined[:, 3] - sum_of_charges / new_combined.shape[0]

        return (
            get_esp_rmse_from_combined(new_combined, self.batch, get_esp=True)[1],
            new_combined,
        )

    def optimize(self):
        x0 = np.zeros(self.combined.shape[0])
        maxfev = int(200 * len(x0) + 2 * len(x0) ** 2)
        print(x0.shape, maxfev)
        bounds = [(-0.1, 0.1)] * len(x0)
        res = minimize(
            self.loss_fn_only_q,
            x0,
            method="Nelder-Mead",
            # method='COBYLA',
            bounds=bounds,
            # callback=callback,
            options={
                "xatol": 1e-8,
                "disp": True,
                "adaptive": True,  # default is False
                "ftol": 1e-3,  # default is 1e-8
                "eps": 1e-3,  # default is 1e-8
                # 'maxiter': 300*len(x0),
                "maxfev": maxfev,
            },
        )
        print(res.message)
        combined_pred_esp, new_combined = self.esp_from_opt_only_q(res.x, self.batch)
        return res, combined_pred_esp, new_combined
