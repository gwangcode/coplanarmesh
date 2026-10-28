import numpy as np
from collections import defaultdict


def _axis_bins(val, border_eps):
    """Return 1 or 2 candidate bins for a single axis.

    If the fractional part is within border_eps of 0 or 1, the axis is
    duplicated into the adjacent bin so that two faces straddling the
    boundary still share at least one bucket.
    """
    base = int(np.floor(val))
    frac = val - base
    if frac < border_eps:
        return (base, base - 1)
    if frac > 1.0 - border_eps:
        return (base, base + 1)
    return (base,)


def hash_plane_fuzzy(n, d, eps=1e-3, normal_eps=1e-5, border_eps=0.1):
    """Discretize a 3D plane into fuzzy integer keys.

    The fuzzy boundary buffer is now applied to all four dimensions
    (nx, ny, nz, d), not only to the offset d. This guarantees that two
    nearly identical planes straddling a bin boundary share at least one
    key even when their normals differ by less than ``border_eps * eps``.

    Args:
        n: 3D unit normal [nx, ny, nz].
        d: Signed distance from the origin.
        eps: Bin width in mesh units.
        normal_eps: Zero-component tolerance for canonicalization.
        border_eps: Fraction of a bin treated as a boundary buffer.

    Returns:
        list of 4-tuples (nx_bin, ny_bin, nz_bin, d_bin). Length is
        1, 2, 4, 8 or 16 depending on how many axes sit on a boundary.
    """
    # Canonicalize anti-parallel normals (unchanged)
    flip = (n[0] < -normal_eps) if abs(n[0]) > normal_eps else (
        (n[1] < -normal_eps) if abs(n[1]) > normal_eps else (n[2] < -normal_eps))
    actual_n = -n if flip else n
    actual_d = -d if flip else d

    val_nx = actual_n[0] / eps
    val_ny = actual_n[1] / eps
    val_nz = actual_n[2] / eps
    val_d = actual_d / eps

    nx_bins = _axis_bins(val_nx, border_eps)
    ny_bins = _axis_bins(val_ny, border_eps)
    nz_bins = _axis_bins(val_nz, border_eps)
    d_bins = _axis_bins(val_d, border_eps)

    # Cartesian product of candidate bins; at most 2^4 = 16 keys.
    keys = []
    for nx in nx_bins:
        for ny in ny_bins:
            for nz in nz_bins:
                for dd in d_bins:
                    keys.append((nx, ny, nz, dd))
    return keys


def precompute_plane_drawers(mesh, normal_eps=1e-5, eps=1e-3, border_eps=0.1):
    """Group all faces of a mesh into fuzzy coplanar buckets.

    Args:
        mesh: trimesh.Trimesh input.
        normal_eps: Zero-component tolerance used during canonicalization.
        eps: Bin width in mesh units.
        border_eps: Sub-bin boundary buffer fraction applied to all axes.
    """
    drawer = defaultdict(list)
    for i, face in enumerate(mesh.faces):
        tri = mesh.vertices[face]
        n = np.cross(tri[1] - tri[0], tri[2] - tri[0])
        norm = np.linalg.norm(n)
        if norm < normal_eps:
            continue
        n /= norm
        d = -np.dot(n, tri[0])
        keys = hash_plane_fuzzy(n, d, eps=eps, normal_eps=normal_eps, border_eps=border_eps)
        for key in keys:
            drawer[key].append((i, n, d))
    return drawer
