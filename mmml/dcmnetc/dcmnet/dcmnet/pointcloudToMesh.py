
import numpy as np
import os
import sys

def save_ply_points(path, Pxyz, scalars=None, uv=None):
    # Save a PLY (ascii) point cloud with optional per-point scalar and/or uv.
    n = Pxyz.shape[0]
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
    header.append("end_header")

    with open(path, "w") as f:
        f.write("\n".join(header) + "\n")
        for i in range(n):
            row = [Pxyz[i,0], Pxyz[i,1], Pxyz[i,2]]
            if has_scalar:
                row.append(float(scalars[i]))
            if has_uv:
                row.extend([float(uv[i,0]), float(uv[i,1])])
            f.write(" ".join(map(str, row)) + "\n")


def save_ply_mesh(path, Vxyz, Ftri, scalars=None, uv=None):
    # Save a PLY (ascii) triangle mesh with optional per-vertex scalar and/or uv.
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


def lambert_azimuthal_equal_area(xyz, center=None):
    # Map 3D points on/near a sphere-like surface to 2D (Lambert azimuthal equal-area).
    X = xyz - xyz.mean(0, keepdims=True)
    r = np.linalg.norm(X, axis=1, keepdims=True) + 1e-12
    U = X / r

    if center is None:
        c = np.array([0.0, 0.0, 1.0])
    else:
        c = np.asarray(center, dtype=float)
        c /= (np.linalg.norm(c) + 1e-12)

    # Build local frame at center
    tmp = np.array([1.0, 0.0, 0.0]) if abs(c[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    e1 = np.cross(c, tmp); e1 /= (np.linalg.norm(e1) + 1e-12)
    e2 = np.cross(c, e1)

    cos_k = U @ c
    denom = 1.0 + cos_k
    R = np.sqrt(2.0 / np.maximum(denom, 1e-12))

    u1 = U @ e1
    u2 = U @ e2
    x = R * u1
    y = R * u2
    return np.stack([x, y], axis=1)


def principal_axis(points):
    X = points - points.mean(0, keepdims=True)
    cov = X.T @ X
    w, U = np.linalg.eigh(cov)
    return U[:, np.argmax(w)]  # eigenvector of largest eigenvalue


def reconstruct_mesh_open3d(P, k_normals=30, method="auto"):
    try:
        import open3d as o3d
    except Exception as e:
        print("[warn] open3d not available; skipping mesh reconstruction.")
        return None, None

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(P)

    # Estimate and orient normals
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamKNN(knn=k_normals))
    pcd.orient_normals_consistent_tangent_plane(k_normals)

    # Auto-select method
    if method == "auto":
        method = "bpa"  # Ball Pivoting is often good for surface samples

    mesh = None
    if method == "bpa":
        # Guess radii from average nn distance
        dists = pcd.compute_nearest_neighbor_distance()
        mean_dist = np.mean(dists) if len(dists) > 0 else 0.0
        if mean_dist <= 0:
            method = "poisson"
        else:
            radii = [mean_dist*1.2, mean_dist*2.0, mean_dist*3.0]
            try:
                mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
                    pcd, o3d.utility.DoubleVector(radii)
                )
                mesh.remove_duplicated_vertices()
                mesh.remove_degenerate_triangles()
                mesh.remove_duplicated_triangles()
                mesh.remove_non_manifold_edges()
            except Exception:
                mesh = None
            if mesh is None or len(mesh.triangles) == 0:
                method = "poisson"  # fallback

    if method == "poisson":
        try:
            mesh, density = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
                pcd, depth=9
            )
            # Crop by density to remove spurious components
            density = np.asarray(density)
            vertices_to_keep = density > np.quantile(density, 0.05)
            mesh = mesh.select_by_index(np.where(vertices_to_keep)[0])
            mesh.remove_duplicated_vertices()
            mesh.remove_degenerate_triangles()
            mesh.remove_duplicated_triangles()
            mesh.remove_non_manifold_edges()
        except Exception:
            mesh = None

    if mesh is None or len(mesh.triangles) == 0:
        print("[warn] mesh reconstruction failed.")
        return None, None

    V = np.asarray(mesh.vertices)
    F = np.asarray(mesh.triangles, dtype=np.int32)
    return V, F


def has_boundary(F):
    from collections import Counter
    edges = []
    for a, b, c in F:
        edges += [(a,b), (b,c), (c,a)]
    canon = [tuple(sorted(e)) for e in edges]
    cnt = Counter(canon)
    return any(v == 1 for v in cnt.values())


def compute_uv(V, F):
    # Try libigl LSCM if available & boundary present; else fallback to Lambert azimuthal.
    try:
        import igl
        if has_boundary(F):
            bnd = igl.boundary_loop(F)
            if len(bnd) >= 2:
                b = np.array([bnd[0], bnd[len(bnd)//2]], dtype=np.int32)
                bc = np.array([[0.0, 0.0],[1.0, 0.0]], dtype=np.float64)
                uv = igl.lscm(V, F, b, bc)
                uvmin = uv.min(axis=0); uvmax = uv.max(axis=0)
                span = (uvmax - uvmin); span[span==0] = 1.0
                return (uv - uvmin)/span
    except Exception:
        pass

    center = principal_axis(V)
    uv = lambert_azimuthal_equal_area(V, center=center)
    uvmin = uv.min(axis=0); uvmax = uv.max(axis=0)
    span = (uvmax - uvmin); span[span==0] = 1.0
    return (uv - uvmin)/span


def main(p="P.npy", s="S.npy"):
    if not os.path.exists(p):
        print("Error: expected P.npy (n,3) in current directory.")
        sys.exit(1)
    if not os.path.exists(s):
        print("Error: expected S.npy (n,) in current directory.")
        sys.exit(1)

    P = np.load(p)
    S = np.load(s)

    assert P.ndim == 2 and P.shape[1] == 3, "P must be (n,3)"
    assert S.shape[0] == P.shape[0], "S must be (n,) and match P"

    os.makedirs("out", exist_ok=True)

    # 0) Save the input points + scalar
    save_ply_points("out/points_with_scalar.ply", P, scalars=S)

    # 1) Try to reconstruct a mesh (if open3d is available)
    Vrec, Frec = reconstruct_mesh_open3d(P)

    # 2) Compute UVs (if mesh exists, on mesh vertices; else on points directly)
    if Vrec is not None and Frec is not None:
        uv_mesh = compute_uv(Vrec, Frec)
        V2 = np.zeros_like(Vrec)
        V2[:, :2] = uv_mesh

        # Transfer scalars from points to mesh vertices via nearest neighbor (simple brute force)
        diffs = Vrec[:, None, :] - P[None, :, :]
        d2 = np.sum(diffs*diffs, axis=2)
        nn = np.argmin(d2, axis=1)
        S_mesh = S[nn]

        save_ply_mesh("out/mesh_reconstructed.ply", Vrec, Frec, scalars=S_mesh)
        save_ply_mesh("out/unfolded_2d_mesh.ply", V2, Frec, scalars=S_mesh, uv=uv_mesh)

        # UV for original points (direct)
        center = principal_axis(P)
        uv_pts = lambert_azimuthal_equal_area(P, center=center)
        uvmin = uv_pts.min(axis=0); uvmax = uv_pts.max(axis=0)
        span = (uvmax - uvmin); span[span==0] = 1.0
        uv_pts = (uv_pts - uvmin)/span
        P2 = np.zeros_like(P); P2[:, :2] = uv_pts
        save_ply_points("out/unfolded_2d_points.ply", P2, scalars=S, uv=uv_pts)

        print("Wrote:")
        print(" - out/points_with_scalar.ply")
        print(" - out/mesh_reconstructed.ply")
        print(" - out/unfolded_2d_mesh.ply")
        print(" - out/unfolded_2d_points.ply")

    else:
        center = principal_axis(P)
        uv = lambert_azimuthal_equal_area(P, center=center)
        uvmin = uv.min(axis=0); uvmax = uv.max(axis=0)
        span = (uvmax - uvmin); span[span==0] = 1.0
        uv = (uv - uvmin)/span
        P2 = np.zeros_like(P); P2[:, :2] = uv
        save_ply_points("out/unfolded_2d_points.ply", P2, scalars=S, uv=uv)

        print("[warn] Mesh reconstruction not available; exported points-only files.")
        print("Wrote:")
        print(" - out/points_with_scalar.ply")
        print(" - out/unfolded_2d_points.ply")



