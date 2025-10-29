from dcmnet.analysis import *

params = "/pchem-data/meuwly/boittier/home/jaxeq/all_runs/runs7/20240830-165613dcm-3-espw-1000-qm9-esp40000-0.npz-re-False/best_1000_params.pkl"

path = Path("/pchem-data/meuwly/boittier/home/jaxeq/misc")
paths = list(path.glob("*"))
for path in paths:
    dcmnet(path, params)
