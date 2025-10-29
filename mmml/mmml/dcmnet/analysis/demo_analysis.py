from dcmnet.analysis import *
from tqdm import tqdm

path = Path("/pchem-data/meuwly/boittier/home/jaxeq/misc")
paths = list(path.glob("*"))
print(paths[0])
for path in tqdm(paths):
    multipoles(path)
