import numpy as np
import os


np.save("V.npy", V)   # (n,3)
np.save("F.npy", F)   # (m,3)  triangles, 0-based
np.save("S.npy", S)   # (n,)



# ---------- Inputs ----------
# Replace these with your own paths or arrays
V = np.load("V.npy")          # shape (n, 3), float
F = np.load("F.npy").astype(np.int32)  # shape (m, 3), int (0-based)
S = np.load("S.npy")          # shape (n,), float (scalar field)

assert V.ndim == 2 and V.shape[1] == 3, "V must be (n,3)"
assert F.ndim == 2 and F.shape[1] == 3, "F must be (m,3)"
assert S.shape[0] == V.shape[0], "S must be per-vertex (n,)"

# ---------- Helpers: IO (PLY) ----------
def save_ply(path, Vxyz, Ftri, scalars=None, uv=None):
    """
    Save a PLY with optional per-vertex scalar and/or uv properties.
    """
    n = Vxyz.shape[0]
    m = Ftri.shape[0]
    has_scalar = scalars is not None
    has_uv = uv is not None

    header = [
        "ply",
        "format ascii 1.0",
        f"element vertex {n}",
        "property float x",
        "property float y",
        "property float z",
    ]
    if has_scalar:
        header.append("property float scalar")
    if has_uv:
        header.append("property float u")
        header.append("property float v")
    header += [
        f"element face {m}",
        "property list uchar int vertex_indices",
        "end_header"
    ]

    with open(path, "w") as f:
        f.write("\n".join(header) + "\n")
        for i in range(n):
            row = [Vxyz[i,0], Vxyz[i,1], Vxyz[i,2]]
            if has_scalar:
                row.append(float(scalars[i]))
            if has_uv:
                row.extend([float(uv[i,0]), float(uv[i,1])])
            f.write(" ".join(map(str, row)) + "\n")
        for tri in Ftri:
            f.write(f"3 {int(tri[0])} {int(tri[1])} {int(tri[2])}\n")

# ---------- Topology checks ----------
def has_boundary(F):
    """Return True if mesh has a boundary."""
    from collections import Counter
    edges = []
    for a,b,c in F:
        edges += [(a,b),(b,c),(c,a)]
    # undirected edge count
    canon = tuple(sorted(e) for e in edges)
    cnt = Counter(canon)
    # boundary edges appear once
    return any(v == 1 for v in cnt.values())

boundary = has_boundary(F)

# ---------- Method A: LSCM via libigl (works best on meshes with a boundary) ----------
uv = None
lscm_ok = False
try:
    import igl
    if boundary:
        # Pick two boundary vertices to pin
        bnd = igl.boundary_loop(F)
        if len(bnd) >= 2:
            b = np.array([bnd[0], bnd[len(bnd)//2]], dtype=np.int32)  # two far-ish points
            bc = np.array([[0.0, 0.0],
                           [1.0, 0.0]], dtype=np.float64)             # pin them on x-axis
            uv = igl.lscm(V, F, b, bc)                                # (n,2)
            # Normalize UV to a nice box
            uvmin = uv.min(axis=0); uvmax = uv.max(axis=0)
            span = (uvmax - uvmin)
            span[span == 0] = 1.0
            uv = (uv - uvmin) / span
            lscm_ok = True
except Exception as e:
    # libigl not available or failed; will fall back
    lscm_ok = False

# ---------- Method B: Pure NumPy fallback (spherical -> Lambert azimuthal equal-area) ----------
def lambert_azimuthal_equal_area(xyz, center=None):
    """
    Map 3D points on (approx) sphere to 2D via Lambert azimuthal equal-area.
    xyz: (n,3). They don't need to be exactly unit length; we normalize.
    center: unit vector for map center. If None, use +Z.
    Returns (n,2)
    """
    X = xyz - xyz.mean(0, keepdims=True)
    r = np.linalg.norm(X, axis=1, keepdims=True) + 1e-12
    U = X / r

    if center is None:
        c = np.array([0.0, 0.0, 1.0])
    else:
        c = np.asarray(center, dtype=float)
        c /= (np.linalg.norm(c) + 1e-12)

    # Build local frame at center: c (north), e1, e2 (east/north-east)
    # Pick arbitrary up to build e1
    tmp = np.array([1.0, 0.0, 0.0]) if abs(c[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    e1 = np.cross(c, tmp); e1 /= (np.linalg.norm(e1) + 1e-12)
    e2 = np.cross(c, e1)

    # Cosine of angular distance
    cos_k = U @ c
    k = np.arccos(np.clip(cos_k, -1.0, 1.0))
    denom = 1.0 + cos_k
    # Lambert AEA radius factor
    R = np.sqrt(2.0 / np.maximum(denom, 1e-12))

    # Project onto local tangent basis
    u1 = U @ e1
    u2 = U @ e2
    x = R * u1
    y = R * u2
    return np.stack([x, y], axis=1)

if not lscm_ok:
    # Choose center as direction of largest variance (PCA first PC).
    X = V - V.mean(0, keepdims=True)
    cov = X.T @ X
    w, Ueig = np.linalg.eigh(cov)
    # principal axis = eigenvector with largest eigenvalue
    center = Ueig[:, np.argmax(w)]
    uv = lambert_azimuthal_equal_area(V, center=center)
    # normalize to 0..1 box
    uvmin = uv.min(axis=0); uvmax = uv.max(axis=0)
    span = (uvmax - uvmin)
    span[span == 0] = 1.0
    uv = (uv - uvmin) / span

# ---------- Exports ----------
os.makedirs("out", exist_ok=True)

# 1) Original 3D mesh with your scalar (good for checking)
save_ply("out/original_with_scalar.ply", V, F, scalars=S)

# 2) 2D “unfolded” mesh: put UV on the plane as XY, Z=0.
V2 = np.zeros_like(V)
V2[:,:2] = uv
save_ply("out/unfolded_2d.ply", V2, F, scalars=S, uv=uv)

print("Wrote:")
print(" - out/original_with_scalar.ply")
print(" - out/unfolded_2d.ply")
print("Open either PLY in Meshlab/ParaView. The unfolded file sits in Z=0 with per-vertex scalar.")
