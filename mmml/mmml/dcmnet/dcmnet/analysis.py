from pathlib import Path

import ase.data
import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
from jax import vmap
from scipy.spatial.distance import cdist
from tqdm import tqdm

from dcmnet.data import cut_vdw, prepare_batches
from dcmnet.loss import esp_mono_loss_pots, pred_dipole
from dcmnet.modules import MessagePassingModel, MessagePassingModelDEBUG
from dcmnet.multipoles import calc_esp_from_multipoles
from dcmnet.utils import apply_model

LOAD_DATA = False
if LOAD_DATA:
    data_path = Path("/pchem-data/meuwly/boittier/home/analysis")

    data = np.load(
        Path("/pchem-data/meuwly/boittier/home/jaxeq/") / "data/qm9-esp-dip-6907-3.npz"
    )
au_to_debye = 2.5417464519
au_to_kcal = 627.509


# Model hyperparameters.
features = 16
max_degree = 2
num_iterations = 2
num_basis_functions = 8
cutoff = 4.0


def create_model(
    n_dcm=2,
    features=16,
    max_degree=2,
    num_iterations=2,
    num_basis_functions=16,
    cutoff=4.0,
    include_pseudotensors=False,
    debug=False,
):
    model_type = MessagePassingModel if not debug else MessagePassingModelDEBUG
    return model_type(
        features=int(features),
        max_degree=int(max_degree),
        num_iterations=int(num_iterations),
        num_basis_functions=int(num_basis_functions),
        cutoff=float(cutoff),
        n_dcm=int(n_dcm),
        # include_pseudotensors=bool(include_pseudotensors),
        include_pseudotensors=True,
    )


def parm_dict_from_path(path):
    list_ = [
        _.split("=") for _ in open(Path(path).parents[0] / "manifest.txt").readlines()
    ]
    job_parms = {}
    for _ in list_:
        if len(_) == 2:
            try:
                job_parms[_[0].strip()] = float(_[1].strip())
            except:
                job_parms[_[0].strip()] = _[1].strip()

    if "include_pseudotensors" not in job_parms.keys():
        job_parms["include_pseudotensors"] = "False"
    if job_parms["include_pseudotensors"] == "False":
        job_parms["include_pseudotensors"] = False
    if job_parms["include_pseudotensors"] == "True":
        job_parms["include_pseudotensors"] = True

    return job_parms


def create_model_and_params(path, debug=False):
    """ """
    if type(path) == str:
        path = Path(path)
    n_dcm = str(path).find("dcm-")
    if n_dcm == -1:
        raise ValueError("DCM model not found")
    n_dcm = int(str(path)[n_dcm + 4])
    # raise an exception if dcm is not found

    params = pd.read_pickle(path)
    job_parms_ = parm_dict_from_path(path)
    job_args = [
        "n_dcm",
        "features",
        "max_degree",
        "num_iterations",
        "num_basis_functions",
        "cutoff",
        "include_pseudotensors",
    ]
    job_parms = {k: v for k, v in job_parms_.items() if k in job_args}
    job_parms["debug"] = debug
    print(job_parms)
    model = create_model(**job_parms)
    return model, params, job_parms_


def read_output(JOBNAME):
    lines = open(JOBNAME).readlines()
    mag = None
    dX, dY, dZ = None, None, None
    mag = None
    for _ in lines:
        if _.startswith(" Magnitude           :"):
            mag = float(_.split()[-1]) * au_to_debye
        if _.startswith(" Dipole X"):
            dX = float(_.split()[-1])
        if _.startswith(" Dipole Y"):
            dY = float(_.split()[-1])
        if _.startswith(" Dipole Z"):
            dZ = float(_.split()[-1])
    return mag, np.array([dX, dY, dZ])


def make_traceless(Q):
    trace_Q = jnp.trace(Q, axis1=0, axis2=1)  # Correct axis for trace
    correction = trace_Q / 3
    Q_traceless = Q - jnp.expand_dims(
        jnp.eye(3) * correction, axis=0
    )  # Correct shape broadcasting
    return Q_traceless.squeeze()


# Vectorized over each matrix in the 60 matrices group
vectorized_make_traceless = vmap(make_traceless, in_axes=(0), out_axes=(0))


def prepare_batch(path: Path):
    """ """
    _id = path.stem
    index = int(np.argwhere(data["id"] == _id))
    _dict = {k: np.array(v[[index]]) for k, v in data.items()}
    batch = prepare_batches(jax.random.PRNGKey(0), _dict, 1, True)[0]
    return batch


def prepare_mulitpoles_data(path: Path):
    """ """
    outdata = np.load(path / "multipoles.npz")
    coordinates = outdata["monomer_coords"]
    esp = np.fromfile(path / "grid_esp.dat", sep=" ")
    grid = outdata["surface_points"]
    mag, dipole = read_output(path / "output.dat")
    max_grid_points = 3200
    # pad the grid and esp
    pad_esp = np.pad(esp, ((0, max_grid_points - len(esp))))
    pad_esp = np.array(pad_esp)
    pad_grid = np.pad(
        grid,
        ((0, max_grid_points - len(grid)), (0, 0)),
        "constant",
        constant_values=(0, 10000),
    )

    outdata = {
        "monopoles": outdata["monopoles"],
        "dipoles": outdata["dipoles"],
        "quadrupoles": np.asarray(
            vectorized_make_traceless(jnp.array(outdata["quadrupoles"]))
        ),
        "elements": outdata["elements"],
        "xyz": coordinates,
        "esp": pad_esp,
        "grid": pad_grid,
        "D": mag,
        "D_xyz": dipole,
    }
    return outdata


def multipoles_analysis(outdata: dict):
    esp = outdata["esp"]
    grid = outdata["grid"]

    mono, dipo, quad = calc_esp_from_multipoles(outdata)
    dipo = mono + dipo
    quad = dipo + quad
    outdata["elements"] = np.array(
        [ase.data.atomic_numbers[i] for i in outdata["elements"]]
    )
    # print(outdata)
    mask, closest_atom_type, closest_atom = cut_vdw(
        grid, outdata["xyz"], outdata["elements"]
    )

    rmse_mono = rmse(esp, mono)
    rmse_dipo = rmse(esp, dipo)
    rmse_quad = rmse(esp, quad)
    rmse_mono_masked = rmse(esp[mask], mono[mask])
    rmse_dipo_masked = rmse(esp[mask], dipo[mask])
    rmse_quad_masked = rmse(esp[mask], quad[mask])

    masses = np.array([ase.data.atomic_masses[s] for s in outdata["elements"]])
    com = jnp.sum(outdata["xyz"] * masses[:, None], axis=0) / jnp.sum(masses)
    D_mono = pred_dipole(outdata["xyz"], com, outdata["monopoles"])

    D_dipo = D_mono + outdata["dipoles"].sum(axis=0)

    D_mae_mono = jnp.mean(jnp.abs(D_mono - outdata["D_xyz"])) * au_to_debye
    D_mae_dipo = jnp.mean(jnp.abs(D_dipo - outdata["D_xyz"])) * au_to_debye

    output_data = {
        "mono": mono,
        "dipo": dipo,
        "quad": quad,
        "esp": esp,
        "closest_atom_type": closest_atom_type,
        "closest_atom": closest_atom,
        "mask": mask,
        "rmse_mono": rmse_mono,
        "rmse_dipo": rmse_dipo,
        "rmse_quad": rmse_quad,
        "rmse_mono_masked": rmse_mono_masked,
        "rmse_dipo_masked": rmse_dipo_masked,
        "rmse_quad_masked": rmse_quad_masked,
        "D_mono": D_mono,
        "D_dipo": D_dipo,
        "D_mae_mono": D_mae_mono,
        "D_mae_dipo": D_mae_dipo,
        "D": outdata["D"],
        "D_xyz": outdata["D_xyz"],
    }
    return output_data


def rmse(y_true, y_pred):
    return jnp.sqrt(jnp.mean((y_true - y_pred) ** 2)) * au_to_kcal


def dcmnet_analysis(params, model, batch):
    """"""
    mono, dipo = apply_model(model, params, batch, 1)
    esp_dc_pred = esp_mono_loss_pots(
        dipo, mono, batch["vdw_surface"][0], batch["mono"], 1, model.n_dcm
    )
    D = pred_dipole(dipo, batch["com"], mono.reshape(60 * model.n_dcm))
    D_mae = jnp.mean(jnp.abs(D - batch["Dxyz"])) * au_to_debye
    mask, closest_atom_type, closest_atom = cut_vdw(
        batch["vdw_surface"][0], batch["R"], batch["Z"]
    )
    rmse_model = rmse(batch["esp"][0], esp_dc_pred[0])
    rmse_model_masked = rmse(batch["esp"][0][mask], esp_dc_pred[0][mask])
    output_data = {
        "mono": mono,
        "dipo": dipo,
        "D_xyz_pred": D,
        "D_mae": D_mae,
        "esp_pred": esp_dc_pred[0],
        "mask": mask,
        "closest_atom_type": closest_atom_type,
        "closest_atom": closest_atom,
        "rmse_model": rmse_model,
        "rmse_model_masked": rmse_model_masked,
    }
    return output_data


def multipoles(path: Path):
    """ """
    outdata = prepare_mulitpoles_data(path)
    output = multipoles_analysis(outdata)
    save_output(output, path)
    return output


def dcmnet(paths: list, params_path: Path):
    """ """
    model, params, _ = create_model_and_params(params_path)
    for path in tqdm(paths):
        batch = prepare_batch(path)
        output = dcmnet_analysis(params, model, batch)
        save_output(output, path, params_path)
    # return output


def save_output(output, path, params_path=None, data_path=Path(".")):
    """ """
    folder = path.stem
    if params_path is None:
        subfolder = "MBIS"
    else:
        subfolder = Path(params_path).parent.stem
    output_path = data_path / subfolder
    output_path.mkdir(parents=True, exist_ok=True)
    pd.to_pickle(output, output_path / f"{folder}.pkl")
