import polars as pl
import matplotlib.pyplot as plt
from pathlib import Path

pl_loaded = None
main_dir = Path(__file__).parents[1]
parquet_fn = main_dir / "data" / "charmmthermoml.parquet"

K_KEY = "Temperature, K"
P_KEY = "Pressure, kPa"
RHO_KEY = "Mass density, kg/m3"


def read_data_T_rho(charmm_res_id: str, pl_loaded: pl.DataFrame | None = None, parquet_fn: Path | None = None) -> pl.DataFrame:
    if pl_loaded is None:
        pl_loaded = pl.read_parquet(parquet_fn)
    T_RHO_KEY = "[['Temperature, K'], ['Pressure, kPa']]"
    residue_key = f"('{charmm_res_id.upper()}',)"
    T_RHO_DF = pl_loaded[pl_loaded["variable_names"] == T_RHO_KEY]
    T_RHO_DF = T_RHO_DF[T_RHO_DF["charmm_res_id"] == residue_key]


    return T_RHO_DF



def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--residue", type=str, required=True)
    args = parser.parse_args()
    T_RHO_DF = read_data_T_rho(args.residue, pl_loaded, parquet_fn)
    print(T_RHO_DF)
    original = T_RHO_DF.clone()
    keys = list(T_RHO_DF.columns)
    print(keys)

    T_RHO_DF = T_RHO_DF.select(["charmm_res_id", RHO_KEY, K_KEY, P_KEY, ])
    print("charmm_res_id", RHO_KEY, K_KEY, P_KEY)
    # loop through the POLARS dataframe and print the data
    for i, row in enumerate(T_RHO_DF.rows()):
        print(i, row)


    # save data as csv
    original.to_csv(f"{args.residue}.csv")


if __name__ == "__main__":
    main()


