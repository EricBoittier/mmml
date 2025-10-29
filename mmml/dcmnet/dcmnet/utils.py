import os
from pathlib import Path

import numpy as np
import jax.numpy as jnp

import pandas as pd

NATOMS = 60


def apply_model(model, params, batch, batch_size) -> tuple:
    mono_prediction, dipo_prediction = model.apply(
        params,
        atomic_numbers=batch["Z"],
        positions=batch["R"],
        dst_idx=batch["dst_idx"],
        src_idx=batch["src_idx"],
        batch_segments=batch["batch_segments"],
        batch_size=batch_size,
    )
    n_dcm = model.n_dcm
    d = jnp.moveaxis(dipo_prediction, -1, -2).reshape(batch_size, NATOMS * n_dcm, 3)

    m = mono_prediction.reshape(batch_size, NATOMS * n_dcm)
    # 0 the charges for dummy atoms
    n_atoms = batch["N"][0]
    NDC = n_atoms * n_dcm
    valid_atoms = jnp.where(jnp.arange(60 * n_dcm) < NDC, 1, 0)
    d = d[0]
    m = m[0] * valid_atoms
    # constrain the net charge to 0.0
    avg_chg = m.sum() / NDC
    m = (m - avg_chg) * valid_atoms
    m = m.reshape(batch_size, NATOMS, n_dcm)
    return m, d


def flatten(xss):
    return [x for xs in xss for x in xs]


def clip_colors(c):
    return np.clip(c, -0.015, 0.015)


def reshape_dipole(dipo, nDCM):
    d = dipo.reshape(1, 60, 3, nDCM)
    d = np.moveaxis(d, -1, -2)
    d = d.reshape(1, 60 * nDCM, 3)
    return d


# def pred_dipole(dcm, com, q):
#     dipole_out = jnp.zeros(3)
#     for i, _ in enumerate(dcm):
#         dipole_out += q[i] * (_ - com) * 1.88873
#     # print(dipole_out)
#     return dipole_out


def process_df(errors):
    h2kcal = 627.509
    df = pd.DataFrame(flatten(errors))
    df["model"] = df[0].apply(lambda x: np.sqrt(x) * h2kcal)
    df["mono"] = df[1].apply(lambda x: np.sqrt(x) * h2kcal)
    df["dif"] = df["model"] - df["mono"]
    return df


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


def safe_mkdir(path):
    try:
        os.makedirs(path)
    except FileExistsError:
        pass
