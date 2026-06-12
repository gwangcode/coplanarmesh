import numpy as np
from collections import defaultdict


def hash_plane_fuzzy(n, d, eps=1e-3, normal_eps=1e-5, border_eps=0.1):
    """
    Discretizes a 3D plane equation into discrete integer keys with a fuzzy boundary buffer.

    This prevents floating-point rounding errors from classifying the same physical
    coplanar interface into different buckets. If a plane falls close to a bin boundary,
    it is assigned to adjacent bins simultaneously to guarantee overlap detection.

    Args:
        n (np.ndarray): 3D unit normal vector of the plane [nx, ny, nz].
        d (float): The signed distance from the origin (intercept parameter).
        eps (float): Grid resolution scaling factor for voxel binning.
        normal_eps (float): Tolerance for detecting zero or near-zero normal components.
        border_eps (float): Sub-bin fractional threshold to trigger adjacent bucket duplication.

    Returns:
        list of tuple: A list of 4D integer keys `(nx_bin, ny_bin, nz_bin, d_bin)` representing the bins.
    """
    # Canonicalize the normal vector direction to handle anti-parallel faces (e.g., [0,0,1] vs [0,0,-1])
    flip = (n[0] < -normal_eps) if abs(n[0]) > normal_eps else (
        (n[1] < -normal_eps) if abs(n[1]) > normal_eps else (n[2] < -normal_eps))
    actual_n = -n if flip else n
    actual_d = -d if flip else d

    # Scale and discretize continuous space into integers
    val_nx = actual_n[0] / eps
    val_ny = actual_n[1] / eps
    val_nz = actual_n[2] / eps
    val_d = actual_d / eps

    keys = []
    base_d = int(np.floor(val_d))
    keys.append((int(np.floor(val_nx)), int(np.floor(val_ny)), int(np.floor(val_nz)), base_d))

    # Dual-grid mapping: Duplicate plane keys into adjacent bins if it sits on the boundary buffer
    frac_d = val_d - base_d
    if frac_d < border_eps:
        keys.append((int(np.floor(val_nx)), int(np.floor(val_ny)), int(np.floor(val_nz)), base_d - 1))
    elif frac_d > 1 - border_eps:
        keys.append((int(np.floor(val_nx)), int(np.floor(val_ny)), int(np.floor(val_nz)), base_d + 1))

    return keys


def precompute_plane_drawers(mesh, normal_eps=1e-5):
    """
    Pre-computes and groups all faces of a given mesh into fuzzy coplanar buckets.

    Args:
        mesh (trimesh.Trimesh): The target 3D triangular mesh.
        normal_eps (float): Tolerance threshold to discard degenerate zero-area faces.

    Returns:
        defaultdict(list): A mapping from 4D plane keys to lists of tuples `(face_index, normal, intercept)`.
    """
    drawer = defaultdict(list)
    for i, face in enumerate(mesh.faces):
        tri = mesh.vertices[face]
        # Compute face normal via cross product
        n = np.cross(tri[1] - tri[0], tri[2] - tri[0])
        norm = np.linalg.norm(n)
        if norm < normal_eps:
            continue  # Ignore degenerate triangles
        n /= norm
        d = -np.dot(n, tri[0])

        # Distribute the face ID to its corresponding fuzzy spatial bins
        keys = hash_plane_fuzzy(n, d)
        for key in keys:
            drawer[key].append((i, n, d))
    return drawer