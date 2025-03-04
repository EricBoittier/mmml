import polars as pl
import matplotlib.pyplot as plt
from pathlib import Path

pl_loaded = None

parquet_fn = Path()
def read_data_T_rho():
    if pl_loaded is None:
        pl_loaded = pl.read_parquet(parquet_fn)
    T_RHO_KEY = "[['Temperature, K'], ['Pressure, kPa']]"
    charmm_res_id = "meoh"
    residue_key = f"('{charmm_res_id.upper()}',)"
    T_RHO_DF = pl_loaded[pl_loaded["variable_names"] == T_RHO_KEY]
    T_RHO_DF = T_RHO_DF[T_RHO_DF["charmm_res_id"] == residue_key]
    K_KEY = "Temperature, K"
    P_KEY = "Pressure, kPa"
    RHO_KEY = "Mass density, kg/m3"
    


