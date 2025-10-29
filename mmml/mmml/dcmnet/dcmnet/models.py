import pandas as pd
from dcmnet.modules import MessagePassingModel


# Load model parameters.
dcm1_params = pd.read_pickle(
    "//pchem-data/meuwly/boittier/home/jaxeq/all_runs/runs/20240702-104005dcm-1-espw-0.0-restart-False/best_params.pkl"
)
dcm2_params = pd.read_pickle(
    "/pchem-data/meuwly/boittier/home/jaxeq/all_runs/runs/20240702-172544dcm-2-espw-10000.0-restart-True/best_60000.0_params.pkl"
)
dcm3_params = pd.read_pickle(
    "/pchem-data/meuwly/boittier/home/jaxeq/all_runs/runs/20240702-173904dcm-3-espw-10000.0-restart-True/best_60000.0_params.pkl"
)
dcm4_params = pd.read_pickle(
    "/pchem-data/meuwly/boittier/home/jaxeq/all_runs/runs/20240702-173929dcm-4-espw-10000.0-restart-True/best_60000.0_params.pkl"
)

# Model hyperparameters.
features = 16
max_degree = 2
num_iterations = 2
num_basis_functions = 8
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
