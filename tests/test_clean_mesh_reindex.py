# tests/test_clean_mesh_reindex.py
import numpy as np
import trimesh
from coplanarmesh.coplanarmesh import clean_mesh_and_reindex, clean_single_side


def test_clean_single_side_deduplicates_and_drops_degenerate():
    ...  # 保持不变


def test_clean_mesh_and_reindex_returns_pairs_for_coplanar_faces():
    """Two cubes stacked on a shared z-plane; only the touching faces are passed."""
    a = trimesh.creation.box(extents=(1, 1, 1))          # z ∈ [-0.5, 0.5]
    b = trimesh.creation.box(extents=(1, 1, 1))
    b.apply_translation((0.0, 0.0, 1.0))                 # z ∈ [0.5, 1.5]

    # 只选接触平面上的面：A 的 +Z 面、B 的 -Z 面
    A_top = [i for i, n in enumerate(a.face_normals) if np.dot(n, [0, 0, 1]) > 0.9]
    B_bottom = [i for i, n in enumerate(b.face_normals) if np.dot(n, [0, 0, -1]) > 0.9]
    assert len(A_top) == 2 and len(B_bottom) == 2

    mA, mB, newA, pairs = clean_mesh_and_reindex(
        a.vertices, a.faces, A_top,
        b.vertices, b.faces, B_bottom,
        eps=1e-4,
    )

    assert isinstance(mA, trimesh.Trimesh) and isinstance(mB, trimesh.Trimesh)
    assert len(newA) > 0
    assert len(pairs) > 0, "coplanar touching faces must produce at least one overlap pair"
    # pair 是 (face_i_in_A, face_j_in_B)
    assert all(0 <= i < len(mA.faces) and 0 <= j < len(mB.faces) for i, j in pairs)


def test_clean_mesh_and_reindex_returns_no_pairs_when_far_apart():
    """Sanity check: non-touching meshes produce zero pairs."""
    a = trimesh.creation.box(extents=(1, 1, 1))
    b = trimesh.creation.box(extents=(1, 1, 1))
    b.apply_translation((10.0, 0.0, 0.0))
    A_all = list(range(len(a.faces)))
    B_all = list(range(len(b.faces)))
    mA, mB, newA, pairs = clean_mesh_and_reindex(
        a.vertices, a.faces, A_all, b.vertices, b.faces, B_all, eps=1e-4
    )
    assert pairs == []
