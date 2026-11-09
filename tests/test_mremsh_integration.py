import itertools
import numpy as np
import pytest
import trimesh
from coplanarmesh.coplanarmesh import mremesh

def _base_and_two_bodies():
    base = trimesh.creation.box(extents=(2.0, 1.0, 1.0))
    base.apply_translation((0.0, 0.0, 0.5))
    b1 = trimesh.creation.box(extents=(1.0, 1.0, 1.0))
    b1.apply_translation((-0.4, 0.0, 1.5))
    b2 = trimesh.creation.box(extents=(1.0, 1.0, 1.0))
    b2.apply_translation((0.4, 0.0, 1.5))
    return base, b1, b2

def test_mremesh_returns_aligned_meshes_and_pairs():
    base, b1, b2 = _base_and_two_bodies()
    meshes, pairs = mremesh([base, b1, b2], grid_size=1e-5, eps=1e-5, min_area_eps=1e-6)
    assert len(meshes) == 3
    assert all(isinstance(m, trimesh.Trimesh) for m in meshes)
    assert all(len(m.faces) > 0 for m in meshes)
    assert all(len(p) == 4 for p in pairs)
    assert all(0 <= p[0] < p[2] < 3 for p in pairs)

@pytest.mark.slow
def test_mremesh_is_order_dependent():
    base, b1, b2 = _base_and_two_bodies()
    face_counts = {}
    for perm in itertools.permutations([base, b1, b2]):
        meshes, _ = mremesh(list(perm), grid_size=1e-5, eps=1e-5, min_area_eps=1e-6)
        face_counts[tuple(len(m.faces) for m in meshes)] = face_counts.get(
            tuple(len(m.faces) for m in meshes), 0
        ) + 1
    assert sum(face_counts.values()) == 6