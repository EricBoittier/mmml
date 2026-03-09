import pandas as pd

from physnetjax.models.model import EF
from physnetjax.utils.utils import _process_model_attributes

pkl_path = "/pchem-data/meuwly/boittier/home/pycharmm_test/ckpts/cf3all-d069b2ca-0c5a-4fcd-b597-f8b28933693a/params.pkl"
model_path = "/pchem-data/meuwly/boittier/home/pycharmm_test/ckpts/cf3all-d069b2ca-0c5a-4fcd-b597-f8b28933693a/model_kwargs.pkl"

params = pd.read_pickle(pkl_path)
model_kwargs = pd.read_pickle(model_path)
print(model_kwargs)
model_kwargs = _process_model_attributes(model_kwargs)

model = EF(**model_kwargs)

print(model)
