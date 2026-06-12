import numpy as np
import trimesh
import pymeshfix

def snap_and_clean_mesh(mesh, grid_size=1e-5):
    """
    Collapses microscopic edge slivers and double-points by rounding vertices to a uniform 3D grid.

    This function acts as a critical topology purifier. It eliminates edge-shivering noise
    on macro planes and prevents multi-body cascades from generating degenerate triangles.

    Args:
        mesh (trimesh.Trimesh): The dirty input triangular mesh.
        grid_size (float): Spatial voxel resolution size for snapping coordinate vertices.

    Returns:
        trimesh.Trimesh: The sanitized, cleaned, and structurally cohesive output mesh.
    """
    if mesh is None:
        return None

    # Quantize continuous spatial coordinates onto discrete voxel markers
    v_rounded = np.round(mesh.vertices / grid_size) * grid_size
    unique_verts, inverse_indices = np.unique(v_rounded, axis=0, return_inverse=True)
    remapped_faces = inverse_indices[mesh.faces]

    # Clear identical or duplicate tri-face entries
    sorted_faces = np.sort(remapped_faces, axis=1)
    _, unique_face_idx = np.unique(sorted_faces, axis=0, return_index=True)
    unique_face_idx = sorted(list(unique_face_idx))

    clean_faces = remapped_faces[unique_face_idx]
    sanitized_mesh = trimesh.Trimesh(vertices=unique_verts, faces=clean_faces, process=False)

    # Erase flat degenerate face entries and clean up disconnected orphan nodes
    valid_mask = sanitized_mesh.nondegenerate_faces()
    sanitized_mesh.update_faces(valid_mask)
    sanitized_mesh.remove_unreferenced_vertices()
    sanitized_mesh.fix_normals()
    return sanitized_mesh

def watertight_mesh(mesh):
    """
    Applies an aggressive holistic repair algorithm to enforce watertight physical closure.

    Wraps the robust C++ MeshFix engine to seal complex spatial holes, resolve T-junctions,
    and fix flipped surface normals on structurally compromised shells.

    Args:
        mesh (trimesh.Trimesh): The non-watertight mesh profile.

    Returns:
        trimesh.Trimesh: A structurally sealed, closed manifold output mesh.
    """
    fixer = pymeshfix.MeshFix(mesh.vertices, mesh.faces)
    fixer.repair()
    return trimesh.Trimesh(vertices=fixer.points, faces=fixer.faces, process=False)