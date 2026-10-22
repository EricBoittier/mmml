# Build-time gate for the mmml image (no GPU here): pinned versions and imports.
import importlib.metadata as m

want = {"jax": "0.10.2", "jaxlib": "0.10.2", "jax-cuda13-plugin": "0.10.2", "torch": "2.14.1",
        "metatrain": "2026.4.1", "numpy": "2.5.3"}
got = {k: m.version(k) for k in want}
assert got == want, f"resolver drifted: {got} != {want}"

import jax, flax, jax_md, e3x, cupy, vesin, pyscf, jaxpme  # noqa: E401,F401
import torch, metatrain, metatomic.torch, metatensor.torch, upet  # noqa: E401,F401
from mpi4py import MPI

assert torch.version.cuda and torch.version.cuda.startswith("13"), f"CPU or wrong-CUDA torch: {torch.version.cuda}"
print("imports ok:", got, "| torch cuda", torch.version.cuda, "|", MPI.Get_library_version().splitlines()[0])
