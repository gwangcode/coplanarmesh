# tests/test_clean_mesh_reindex.py
import numpy as np
import trimesh
from coplanarmesh.coplanarmesh import clean_mesh_and_reindex, clean_single_side


def test_clean_single_side_deduplicates_and_drops_degenerate():
    # two identical triangles + one zero-area triangle
    verts = np.array([
        [0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 0], [1, 0, 0], [0, 1, 0],
    ], dtype=float)
    faces = np.array([[0, 1, 2], [3, 4, 5], [0, 1, 1]], dtype=np.int32)
    v, f, overlap = clean_single_side(verts, faces, overlap_indices=[0, 2], eps=1e-6, area_eps=1e-12)
    assert len(f) == 1, "duplicate and degenerate faces should be removed"
    assert overlap == [0]


def test_clean_mesh_and_reindex_returns_pairs_for_overlapping_cubes():
    a = trimesh.creation.box(extents=(1, 1, 1))
    b = trimesh.creation.box(extents=(1, 1, 1))
    b.apply_translation((0.5, 0.0, 0.0))
    # use all faces as candidate overlap to exercise the RTree path
    overlapA = list(range(len(a.faces)))
    overlapB = list(range(len(b.faces)))
    mA, mB, newA, pairs = clean_mesh_and_reindex(
        a.vertices, a.faces, overlapA, b.vertices, b.faces, overlapB, eps=1e-4
    )
    assert isinstance(mA, trimesh.Trimesh) and isinstance(mB, trimesh.Trimesh)
    assert len(pairs) > 0, "touching cubes must produce at least one overlap pair"
