import numpy as np
from pathlib import Path

# Minimal two-frame water trajectory for cross-check tests
r = np.array(
    [
        [[0.0, 0.0, 0.1173], [0.0, 0.7572, -0.4692], [0.0, -0.7572, -0.4692]],
        [[0.0, 0.0, 0.1200], [0.0, 0.7600, -0.4700], [0.0, -0.7600, -0.4700]],
    ],
    dtype=np.float64,
)
z = np.array([8, 1, 1], dtype=np.int32)
e = np.array([-76.0, -76.01], dtype=np.float64)
f = np.zeros((2, 3, 3), dtype=np.float64)

out = Path(__file__).resolve().parent / "water_frames.npz"
np.savez(out, R=r, Z=z, N=np.array([3, 3], dtype=np.int32), E=e, F=f)
print(f"Wrote {out}")
