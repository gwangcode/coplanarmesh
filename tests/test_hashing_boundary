# tests/test_hashing_boundary.py
import numpy as np
from coplanarmesh.hashing import hash_plane_fuzzy

def test_normal_boundary_shared_bucket():
    eps = 1e-3
    border_eps = 0.1
    n1 = np.array([0.99995, 0.0, 0.0])  # val_nx = 999.95
    n2 = np.array([1.0, 0.0, 0.0])      # val_nx = 1000.00
    k1 = set(hash_plane_fuzzy(n1, 0.0, eps=eps, border_eps=border_eps))
    k2 = set(hash_plane_fuzzy(n2, 0.0, eps=eps, border_eps=border_eps))
    assert k1 & k2, "near-identical normals must share at least one bucket"

def test_offset_and_normal_combined():
    eps = 1e-3
    n1 = np.array([0.0, 0.0, 1.0])
    n2 = np.array([1e-6, 0.0, 1.0]); n2 /= np.linalg.norm(n2)
    k1 = set(hash_plane_fuzzy(n1, 0.4999, eps=eps))
    k2 = set(hash_plane_fuzzy(n2, 0.5001, eps=eps))
    assert k1 & k2
