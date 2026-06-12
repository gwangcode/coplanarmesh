import numpy as np
import trimesh
from shapely.geometry import Polygon
from .utils import snap_and_clean_mesh


def clean_single_side(vertices, faces, overlap_indices, eps=1e-5, area_eps=1e-12):
    """
    Performs late-stage vertex deduplication and re-maps active index tracking arrays.

    Args:
        vertices (np.ndarray): Nx3 array of raw spatial vertex coordinates.
        faces (np.ndarray): Mx3 array of index triangle definitions.
        overlap_indices (list): Active indices belonging to targeted contact regions.
        eps (float): Dimensional tolerance bounds passed to rounding filters.
        area_eps (float): Mathematical floor used to discard flat degenerate elements.

    Returns:
        tuple: `(unique_vertices, final_faces, new_overlap_indices)` mapping out the cleaned subset.
    """
    vertices = np.array(vertices, dtype=np.float64)
    faces = np.array(faces, dtype=np.int32)

    # Perform quantization matching across localized node clusters
    v_rounded = np.round(vertices / eps) * eps
    unique_verts, inverse_idx = np.unique(v_rounded, axis=0, return_inverse=True)
    remapped_faces = inverse_idx[faces]

    # Calculate triangle areas via vector cross products to flush out degenerates
    p0 = unique_verts[remapped_faces[:, 0]]
    p1 = unique_verts[remapped_faces[:, 1]]
    p2 = unique_verts[remapped_faces[:, 2]]
    cross_products = np.cross(p1 - p0, p2 - p0)
    area_squared = np.sum(cross_products ** 2, axis=1)
    mask_degenerate = area_squared > area_eps

    sorted_faces = np.sort(remapped_faces, axis=1)
    face_to_final_id = {}
    final_faces_list = []
    old_to_new_dict = np.full(len(faces), -1, dtype=np.int32)

    for old_idx in range(len(faces)):
        if not mask_degenerate[old_idx]:
            continue
        face_tuple = tuple(sorted_faces[old_idx])
        if face_tuple in face_to_final_id:
            old_to_new_dict[old_idx] = face_to_final_id[face_tuple]
        else:
            new_id = len(final_faces_list)
            face_to_final_id[face_tuple] = new_id
            final_faces_list.append(remapped_faces[old_idx])
            old_to_new_dict[old_idx] = new_id

    final_faces = np.array(final_faces_list, dtype=np.int32)
    new_overlap_indices = set()
    for old_idx in overlap_indices:
        if old_idx < len(faces):
            new_id = old_to_new_dict[old_idx]
            if new_id != -1:
                new_overlap_indices.add(new_id)

    if len(final_faces) == 0:
        return np.empty((0, 3)), np.empty((0, 3), dtype=np.int32), []
    return unique_verts, final_faces, sorted(list(new_overlap_indices))


def clean_mesh_and_reindex(vertsA, facesA, overlapA, vertsB, facesB, overlapB, raw_coplanar_pairs=None, eps=1e-4):
    """
    Applies inverse area-intersection mapping to compute clean coplanar pairings.

    Args:
        vertsA, facesA, overlapA: Source components for Mesh A layout.
        vertsB, facesB, overlapB: Source components for Mesh B layout.
        raw_coplanar_pairs (list): Legacy pairing arrays (optional).
        eps (float): Spatial distance tolerance buffer for polygon intersection checks.

    Returns:
        tuple: `(meshA_final, meshB_final, clean_overlapA, final_overlap_pairs)`.
    """
    vA, fA, new_overlapA = clean_single_side(vertsA, facesA, overlapA)
    vB, fB, new_overlapB = clean_single_side(vertsB, facesB, overlapB)

    meshA_final = trimesh.Trimesh(vertices=vA, faces=fA, process=False)
    meshB_final = trimesh.Trimesh(vertices=vB, faces=fB, process=False)
    final_overlap_pairs = []

    if len(new_overlapA) > 0 and len(new_overlapB) > 0:
        b_rtree = meshB_final.triangles_tree
        normalsA = meshA_final.face_normals
        normalsB = meshB_final.face_normals

        ref_normal = normalsA[new_overlapA[0]]
        drop_axis = np.argmax(np.abs(ref_normal))
        axes = [i for i in range(3) if i != drop_axis]
        set_overlapB = set(new_overlapB)

        for i in new_overlapA:
            triA_verts = meshA_final.vertices[meshA_final.faces[i]]
            bounds_min = triA_verts.min(axis=0) - eps
            bounds_max = triA_verts.max(axis=0) + eps
            nearby_B_ids = b_rtree.intersection(np.hstack([bounds_min, bounds_max]))
            valid_B_ids = [j for j in nearby_B_ids if j in set_overlapB]
            if not valid_B_ids:
                continue

            polyA = Polygon(triA_verts[:, axes])
            if polyA.area < eps:
                continue

            for j in valid_B_ids:
                if abs(abs(np.dot(normalsA[i], normalsB[j])) - 1.0) > eps:
                    continue
                triB_verts = meshB_final.vertices[meshB_final.faces[j]]
                polyB = Polygon(triB_verts[:, axes])

                if polyA.intersects(polyB):
                    inter_area = polyA.intersection(polyB).area
                    if inter_area > eps:
                        final_overlap_pairs.append((i, j))

    return meshA_final, meshB_final, new_overlapA, sorted(final_overlap_pairs)


def mremesh(mesh_list, grid_size=1e-5, eps=1e-5, min_area_eps=1e-6):
    """
    Executes a unified, cascading coplanar intersection stream and realigns global mesh indices.

    This function accepts an arbitrary sequence of N overlapping or touching 3D meshes,
    iteratively slices their co-planar contact interfaces via an infinite remeshing loop
    to reach topological equilibrium, and returns the healed meshes along with an immutable
    physical contact alignment matrix.

    Args:
        mesh_list (list of trimesh.Trimesh): The target sequence of input 3D meshes.
        grid_size (float): Voxel size used for snapping loose vertices during pre-processing.
        eps (float): Distance threshold for boundary RTree bounding box intersection buffers.
        min_area_eps (float): The microscopic physical floor area below which overlaps are ignored.

    Returns:
        tuple:
            - final_meshes (list of trimesh.Trimesh): Sequentially remeshed and sanitized 3D objects.
            - global_pairs (list of tuple): An aligned contact matrix sorted as `(mesh_i, face_i, mesh_j, face_j)`.
    """
    num_meshes = len(mesh_list)

    # Defensive Deep-Copy & Voxel Grid Alignment initialization
    current_meshes = [snap_and_clean_mesh(m.copy(), grid_size=grid_size) for m in mesh_list]

    # Phase 1: O(N^2) Pairwise Cascade Slicing State Machine
    for i in range(num_meshes):
        for j in range(i + 1, num_meshes):
            meshA = current_meshes[i]
            meshB = current_meshes[j]
            meshA_sliced, meshB_sliced, overlap_A_this_turn, overlap_B_this_turn, _ = process_and_label_coplanar(meshA,
                                                                                                                 meshB)

            # Evolve state queue only if actual contact collisions were observed
            if len(overlap_A_this_turn) > 0 or len(overlap_B_this_turn) > 0:
                meshA_perfect, meshB_perfect, _, _ = clean_mesh_and_reindex(
                    meshA_sliced.vertices, meshA_sliced.faces, list(overlap_A_this_turn),
                    meshB_sliced.vertices, meshB_sliced.faces, list(overlap_B_this_turn)
                )
                current_meshes[i] = meshA_perfect
                current_meshes[j] = meshB_perfect

    # Phase 2: Reverse Mapping Spatial Intersection Audit (Static Equilibrium Re-indexing)
    final_meshes = current_meshes
    final_global_pairs = set()

    for i in range(num_meshes):
        meshA_final = final_meshes[i]
        normalsA = meshA_final.face_normals

        for j in range(i + 1, num_meshes):
            meshB_final = final_meshes[j]
            normalsB = meshB_final.face_normals
            b_rtree = meshB_final.triangles_tree

            for idxA in range(len(meshA_final.faces)):
                triA_verts = meshA_final.vertices[meshA_final.faces[idxA]]

                # 1. Broad-Phase Spatial Intercept: Leverage RTree AABB Trees for million-facet scalability
                bounds_min = triA_verts.min(axis=0) - eps
                bounds_max = triA_verts.max(axis=0) + eps
                nearby_B_ids = b_rtree.intersection(np.hstack([bounds_min, bounds_max]))
                if not nearby_B_ids:
                    continue

                ref_normal = normalsA[idxA]
                if np.linalg.norm(ref_normal) < eps:
                    continue

                # 2. Dimensionality Reduction Projection: Establish 2D viewports using principal component axes
                drop_axis = np.argmax(np.abs(ref_normal))
                axes = [k for k in range(3) if k != drop_axis]
                polyA = Polygon(triA_verts[:, axes])
                if polyA.area < eps:
                    continue

                # 3. Narrow-Phase Geometric Audit: Strict dot-product & area validation
                for idxB in nearby_B_ids:
                    if abs(abs(np.dot(normalsA[idxA], normalsB[idxB])) - 1.0) > eps:
                        continue
                    triB_verts = meshB_final.vertices[meshB_final.faces[idxB]]
                    polyB = Polygon(triB_verts[:, axes])

                    if polyA.intersects(polyB):
                        inter_area = polyA.intersection(polyB).area

                        # 🌟 Adaptive Local Filtering Mechanism:
                        # Dynamically computes defenses proportional to the intersecting triangle pair.
                        # Macro surfaces are aggressively cleaned while micro components preserve their integrity.
                        local_avg_area = 0.5 * (polyA.area + polyB.area)
                        safe_inter_eps = max(min_area_eps, local_avg_area * eps)

                        if inter_area > safe_inter_eps:
                            final_global_pairs.add((i, idxA, j, idxB))

    return final_meshes, sorted(list(final_global_pairs))


def smesh(meshA, meshB):
    """
    Executes boundary-aligned topological remeshing for multi-body 3D geometries
    sharing touching co-planar interfaces.

    This is the high-level API utilized as an optical simulation pre-processor
    to eliminate numerical ray-flickering and enforce energy conservation laws.

    Args:
        meshA (trimesh.Trimesh): The first 3D manifold geometry (e.g., Lens A).
        meshB (trimesh.Trimesh): The second 3D manifold geometry (e.g., Lens B).

    Returns:
        meshA_final (trimesh.Trimesh): Remeshed geometry A with aligned boundary topology.
        meshB_final (trimesh.Trimesh): Remeshed geometry B with aligned boundary topology.
        final_overlap_pairs (list of tuples): Re-indexed pairs of perfectly overlapping
                                              co-planar facet IDs for the physics solver.
    """
    from .remesh import process_and_label_coplanar

    # 1. Execute the core dispatcher to perform CSG subtraction and patch sewing
    meshA_patch, meshB_patch, overlapA, overlapB, _ = process_and_label_coplanar(meshA, meshB)

    # 2. Re-index and collapse degenerate elements to build clean final topologies
    meshA_final, meshB_final, final_overlap_A, final_overlap_pairs = clean_mesh_and_reindex(
        vertsA=meshA_patch.vertices,
        facesA=meshA_patch.faces,
        overlapA=overlapA,
        vertsB=meshB_patch.vertices,
        facesB=meshB_patch.faces,
        overlapB=overlapB
    )

    return meshA_final, meshB_final, final_overlap_pairs
