import numpy as np
import trimesh
import triangle as tr
from shapely.geometry import Polygon

def get_projection_axes(n):
    """
    Establishes stable 2D local projection axes (X and Y) orthogonal to a given 3D normal vector.

    Args:
        n (np.ndarray): 3D unit normal vector.

    Returns:
        tuple: Two orthogonal 3D unit vectors `(x_axis, y_axis)`.
    """
    up = np.array([0.0, 0.0, 1.0])
    # If the normal is nearly parallel to the Z-axis, swap the reference vector to the X-axis
    if abs(np.dot(up, n)) > 0.5:
        up = np.array([1.0, 0.0, 0.0])
    x = np.cross(up, n)
    x /= np.linalg.norm(x)
    y = np.cross(n, x)
    return x, y

def project_to_plane_with_axes(points, x, y):
    """
    Projects a set of 3D points onto a 2D local coordinate system using the provided axes.

    Args:
        points (np.ndarray): Nx3 array of 3D spatial coordinates.
        x (np.ndarray): 3D unit vector representing the local X axis.
        y (np.ndarray): 3D unit vector representing the local Y axis.

    Returns:
        np.ndarray: Nx2 array of projected 2D coordinates.
    """
    return np.dot(points, np.vstack([x, y]).T)

def remesh_shapely_polygon(polygon_2d, simplify_tolerance=1e-4, area_eps=1e-5):
    """
    Re-triangulates a complex 2D Shapely polygon (with optional holes) into a clean 2D mesh.

    This function attempts a dual-layer triangulation strategy: using Trimesh's native
    front-end first, and falling back to a raw C-based Triangle constraint solver if
    self-intersections or numerical singularities occur.

    Args:
        polygon_2d (shapely.geometry.Polygon): The 2D polygon region to be triangulated.
        simplify_tolerance (float): Geometric simplification factor used during fallback cleanup.
        area_eps (float): Minimum area threshold below which polygons are discarded as slivers.

    Returns:
        tuple: `(vertices_2d, triangles)` where vertices_2d is an Nx2 array and triangles is an Mx3 array.
               Returns `(None, None)` if triangulation fails or the polygon is too small.
    """
    if polygon_2d is None or polygon_2d.is_empty or polygon_2d.area < area_eps:
        return None, None

    # Resolve zero-width self-intersections using standard zero-buffer trick
    polygon_2d = polygon_2d.buffer(0)

    # Strategy 1: Trimesh high-level triangulation wrapper
    try:
        clean_poly = polygon_2d.simplify(tolerance=simplify_tolerance, preserve_topology=True)
        if clean_poly.is_empty or clean_poly.geom_type != "Polygon":
            clean_poly = polygon_2d
        verts2d, tris = trimesh.creation.triangulate_polygon(clean_poly)
        if verts2d is not None and len(tris) > 0:
            return np.array(verts2d), np.array(tris)
    except:
        pass  # Fall through to standard constraint solver on exception

    # Strategy 2: Fallback to robust C-based Triangle graph mesh generator
    try:
        if polygon_2d.geom_type == "Polygon":
            pts2d = np.array(polygon_2d.exterior.coords)[:-1]
            if len(pts2d) >= 3:
                poly = {'vertices': pts2d, 'segments': [[k, (k + 1) % len(pts2d)] for k in range(len(pts2d))]}
                t = tr.triangulate(poly, 'p')  # 'p' triggers constrained Delaunay triangulation
                return t['vertices'], t['triangles']
    except:
        pass

    return None, None