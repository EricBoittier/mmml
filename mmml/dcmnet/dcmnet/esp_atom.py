import matplotlib.pyplot as plt
import numpy as np

h2kcal = 627.5096080306

top_cgenff_atom_types = [
    "CG1N1",
    "CG1T1",
    "CG1T2",
    "CG2O1",
    "CG2O4",
    "CG2R51",
    "CG2R52",
    "CG2R53",
    "CG2R61",
    "CG2R62",
    "CG2R63",
    "CG2R64",
    "CG2R66",
    "CG2R71",
    "CG2RC0",
    "CG302",
    "CG321",
    "CG331",
    "CG3C31",
    "CG3C41",
    "CG3C52",
    "FGA3",
    "FGR1",
    "HGA1",
    "HGA2",
    "HGA3",
    "HGP1",
    "HGP4",
    "HGPAM1",
    "HGPAM2",
    "HGR51",
    "HGR52",
    "HGR61",
    "HGR62",
    "NG1T1",
    "NG2D1",
    "NG2R50",
    "NG2R51",
    "NG2R60",
    "NG2R61",
    "NG2R62",
    "NG2S3",
    "NG311",
    "NG321",
    "OG2D1",
    "OG2D4",
    "OG2R50",
    "OG301",
    "OG311",
    "OG3R60",
]


def esp_rmse_atoms(data):
    """
    Calculate ESP RMSE by atom type.
    
    Computes RMSE between predicted and reference ESP values,
    grouped by the closest atom type to each grid point.
    
    Parameters
    ----------
    data : dict
        Dictionary containing 'esp', 'mono', 'dipo', 'quad', 
        'closest_atom_type', and 'mask' keys
        
    Returns
    -------
    tuple
        (Zs, mono_res, dipo_res, quad_res, esp_mav, nps) where:
        - Zs: Atomic numbers [1, 6, 7, 8, 9] for H, C, N, O, F
        - mono_res: RMSE for monopole predictions by atom type
        - dipo_res: RMSE for dipole predictions by atom type
        - quad_res: RMSE for quadrupole predictions by atom type
        - esp_mav: Mean absolute ESP values by atom type
        - nps: Number of grid points per atom type
    """
    esp = data["esp"]
    mono_esp = data["mono"]
    dipo_esp = data["dipo"]
    quad_esp = data["quad"]
    closest_atom_type = data["closest_atom_type"]
    mask = data["mask"]

    Zs = [1, 6, 7, 8, 9]
    atom_types = ["H", "C", "N", "O", "F"]

    mono_res = []
    dipo_res = []
    quad_res = []
    esp_mav = []
    nps = []
    for i, at in enumerate(atom_types):
        mask_at = closest_atom_type == Zs[i]
        mask_at = mask_at * mask
        if np.sum(mask_at) == 0:
            n_points = np.nan
        else:
            n_points = np.sum(mask_at)
        m_rmse = h2kcal * np.sqrt(
            np.sum((esp[mask_at] - mono_esp[mask_at]) ** 2) / n_points
        )
        d_rmse = h2kcal * np.sqrt(
            np.sum((esp[mask_at] - dipo_esp[mask_at]) ** 2) / n_points
        )
        q_rmse = h2kcal * np.sqrt(
            np.sum((esp[mask_at] - quad_esp[mask_at]) ** 2) / n_points
        )

        esp_mav.append(h2kcal * np.sum(abs(esp[mask_at])) / n_points)
        nps.append(n_points)
        mono_res.append(m_rmse)
        dipo_res.append(d_rmse)
        quad_res.append(q_rmse)

    return Zs, mono_res, dipo_res, quad_res, esp_mav, nps


def esp_rmse_cgenff(data, atom_types_df):
    """
    Calculate ESP RMSE by CGenFF atom type.
    
    Computes RMSE between predicted and reference ESP values,
    grouped by CGenFF atom types using a mapping dataframe.
    
    Parameters
    ----------
    data : dict
        Dictionary containing 'esp', 'mono', 'dipo', 'quad', 
        'closest_atom', and 'mask' keys
    atom_types_df : pandas.DataFrame
        DataFrame with 'atom_id' and 'idx' columns mapping atom indices
        to CGenFF atom types
        
    Returns
    -------
    tuple
        (top_cgenff_atom_types, mono_res, dipo_res, quad_res, esp_mav, nps)
        where each list contains values for each CGenFF atom type
    """
    esp = data["esp"]
    mono_esp = data["mono"]
    dipo_esp = data["dipo"]
    quad_esp = data["quad"]
    closest_atom = data["closest_atom"]
    mask = data["mask"]

    mono_res = []
    dipo_res = []
    quad_res = []
    esp_mav = []
    nps = []

    for i, at in enumerate(top_cgenff_atom_types):
        tmp_at_df = atom_types_df[atom_types_df["atom_id"] == at]
        atom_idxs = tmp_at_df["idx"].values
        # mask_at = np.array([1 if _ in atom_idxs else 0 for _ in closest_atom])
        mask_at = np.isin(closest_atom, atom_idxs).astype(int)

        if len(mask_at) == 0:
            n_points = np.nan
            mask_at = np.zeros_like(mask)
        mask_at = mask_at * mask
        if np.sum(mask_at) == 0:
            n_points = np.nan
        else:
            n_points = np.sum(mask_at)
        m_rmse = h2kcal * np.sqrt(
            np.sum((esp[mask_at] - mono_esp[mask_at]) ** 2) / n_points
        )
        d_rmse = h2kcal * np.sqrt(
            np.sum((esp[mask_at] - dipo_esp[mask_at]) ** 2) / n_points
        )
        q_rmse = h2kcal * np.sqrt(
            np.sum((esp[mask_at] - quad_esp[mask_at]) ** 2) / n_points
        )

        esp_mav.append(h2kcal * np.sum(abs(esp[mask_at])) / n_points)
        nps.append(n_points)
        mono_res.append(m_rmse)
        dipo_res.append(d_rmse)
        quad_res.append(q_rmse)

    return top_cgenff_atom_types, mono_res, dipo_res, quad_res, esp_mav, nps


def esp_rmse_atoms_model(data, esp):
    """
    Calculate ESP RMSE by atom type for model predictions.
    
    Similar to esp_rmse_atoms but for model predictions instead
    of multipole expansions.
    
    Parameters
    ----------
    data : dict
        Dictionary containing 'esp_pred' and 'closest_atom_type' keys
    esp : array_like
        Reference ESP values
        
    Returns
    -------
    tuple
        (Zs, mono_res, nps) where:
        - Zs: Atomic numbers [1, 6, 7, 8, 9]
        - mono_res: RMSE for model predictions by atom type
        - nps: Number of grid points per atom type
    """
    pred_esp = np.array(data["esp_pred"])
    closest_atom_type = np.array(data["closest_atom_type"])
    mask = data["mask"]

    Zs = [1, 6, 7, 8, 9]
    atom_types = ["H", "C", "N", "O", "F"]

    mono_res = []
    esp_mav = []
    nps = []
    for i, at in enumerate(atom_types):
        mask_at = closest_atom_type == Zs[i]
        mask_at = mask_at * mask
        if np.sum(mask_at) == 0:
            n_points = np.nan
        else:
            n_points = np.sum(mask_at)
        m_rmse = h2kcal * np.sqrt(
            np.sum((esp[mask_at] - pred_esp[mask_at]) ** 2) / n_points
        )
        nps.append(n_points)
        mono_res.append(m_rmse)

    return Zs, mono_res, nps


def esp_rmse_cgenff_model(data, esp, closest_atom, atom_types_df):
    """
    Calculate ESP RMSE by CGenFF atom type for model predictions.
    
    Similar to esp_rmse_cgenff but for model predictions instead
    of multipole expansions.
    
    Parameters
    ----------
    data : dict
        Dictionary containing 'esp_pred' and 'mask' keys
    esp : array_like
        Reference ESP values
    closest_atom : array_like
        Indices of closest atoms to each grid point
    atom_types_df : pandas.DataFrame
        DataFrame mapping atom indices to CGenFF atom types
        
    Returns
    -------
    tuple
        (top_cgenff_atom_types, mono_res, nps) where each list contains
        values for each CGenFF atom type
    """
    mono_esp = data["esp_pred"]
    mask = data["mask"]

    mono_res = []
    nps = []

    for i, at in enumerate(top_cgenff_atom_types):
        tmp_at_df = atom_types_df[atom_types_df["atom_id"] == at]
        atom_idxs = tmp_at_df["idx"].values
        mask_at = np.isin(closest_atom, atom_idxs).astype(int)

        if len(mask_at) == 0:
            n_points = np.nan
            mask_at = np.zeros_like(mask)
        mask_at = mask_at * mask
        if np.sum(mask_at) == 0:
            n_points = np.nan
        else:
            n_points = np.sum(mask_at)
        m_rmse = h2kcal * np.sqrt(
            np.sum((esp[mask_at] - mono_esp[mask_at]) ** 2) / n_points
        )

        nps.append(n_points)
        mono_res.append(m_rmse)

    return top_cgenff_atom_types, mono_res, nps
