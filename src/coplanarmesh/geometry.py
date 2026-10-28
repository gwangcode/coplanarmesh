import numpy as np
import trimesh
import triangle as tr
from shapely.geometry import Polygon
import warnings

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
    if polygon_2d is None or polygon_2d.is_empty or polygon_2d.area < area_eps:
        return None, None

    polygon_2d = polygon_2d.buffer(0)

    # Strategy 1
    try:
        clean_poly = polygon_2d.simplify(tolerance=simplify_tolerance, preserve_topology=True)
        if clean_poly.is_empty or clean_poly.geom_type != "Polygon":
            clean_poly = polygon_2d
        verts2d, tris = trimesh.creation.triangulate_polygon(clean_poly)
        if verts2d is not None and len(tris) > 0:
            return np.array(verts2d), np.array(tris)
    except Exception as e:
        warnings.warn(
            f"trimesh triangulation failed ({type(e).__name__}: {e}); "
            f"falling back to Triangle.",
            RuntimeWarning,
        )

    # Strategy 2: 处理 holes
    try:
        if polygon_2d.geom_type == "Polygon":
            exterior = np.array(polygon_2d.exterior.coords)[:-1]
            vertices = exterior
            segments = [[k, (k + 1) % len(exterior)] for k in range(len(exterior))]
            holes = []
            for interior in polygon_2d.interiors:
                ring = np.array(interior.coords)[:-1]
                offset = len(vertices)
                vertices = np.vstack([vertices, ring])
                segments += [[offset + k, offset + (k + 1) % len(ring)] for k in range(len(ring))]
                holes.append([offset, offset + len(ring) // 2])  # 取内环上两点标记 hole

            poly = {'vertices': vertices, 'segments': segments}
            if holes:
                poly['holes'] = holes
            t = tr.triangulate(poly, 'p')
            return t['vertices'], t['triangles']
    except Exception as e:
        raise RuntimeError(
            f"Both triangulation strategies failed for polygon with "
            f"{len(polygon_2d.interiors)} hole(s)."
        ) from e

    return None, None
