import os
import sys
from pathlib import Path

import toml

MAIN_PATH = Path(__file__).resolve().parents[1]
ANALYSIS_PATH = MAIN_PATH / "analysis"
DATA_PATH = MAIN_PATH / "data"
LOGS_PATH = MAIN_PATH / "logs"
BASE_CKPT_DIR = MAIN_PATH / "ckpts"
PYCHARMM_DIR = None

# check for paths.toml in main directory
if not MAIN_PATH.joinpath("paths.toml").exists():
    # raise FileNotFoundError(
    #     f"paths.toml not found in {MAIN_PATH}. Please create the file with the required paths."
    # )
    pass
else:
    # read the paths.toml file
    paths = toml.load(MAIN_PATH / "paths.toml")["paths"]

    if "data" in paths:
        DATA_PATH = Path(paths["data"])
    if "logs" in paths:
        LOGS_PATH = Path(paths["logs"])
    if "analysis" in paths:
        ANALYSIS_PATH = Path(paths["analysis"])
    if "main" in paths:
        MAIN_PATH = Path(paths["main"])
    if "pycharm" in paths:
        PYCHARMM_DIR = Path(paths["pycharm"])
    if "checkpoints" in paths:
        BASE_CKPT_DIR = Path(paths["checkpoints"])


def print_paths():
    from rich.console import Console

    console = Console()
    console.print(f"DATA_PATH: {DATA_PATH}")
    console.print(f"LOGS_PATH: {LOGS_PATH}")
    console.print(f"ANALYSIS_PATH: {ANALYSIS_PATH}")
    console.print(f"MAIN_PATH: {MAIN_PATH}")
    console.print(f"PYCHARMM_DIR: {PYCHARMM_DIR}")


if __name__ == "__main__":
    print_paths()
