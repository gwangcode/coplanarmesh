import numpy as np
import pytest
from coplanarmesh.hashing import hash_planes, cluster_coplanar_faces


def test_plane_hashing_determinism():
    """测试平面方程在微小浮点误差下是否能映射到相同的哈希桶 (Deterministic Binning)"""
    # 构造两个具有极其微小扰动的共面平面方程 (nx, ny, nz, d)
    plane_a = np.array([0.0, 0.0, 1.0, 0.5000001])
    plane_b = np.array([0.0, 0.0, 1.0, 0.4999999])

    # 设定包容容差
    dist_tol = 1e-4
    angle_tol = 1e-3

    hash_a = hash_planes(plane_a[None, :], angle_tol=angle_tol, dist_tol=dist_tol)
    hash_b = hash_planes(plane_b[None, :], angle_tol=angle_tol, dist_tol=dist_tol)

    # 断言：微小扰动的平面必须落入相同的空间哈希桶
    assert hash_a[0] == hash_b[0]


def test_opposite_normal_flipping():
    """测试反向法向的共面平面（如 Top Mesh 底面与 Bottom Mesh 顶面）是否能正确聚类"""
    # 两个方向相反但位于同一平面的法线方程
    plane_top = np.array([0.0, 0.0, 1.0, -0.5])
    plane_bottom = np.array([0.0, 0.0, -1.0, 0.5])

    planes = np.vstack([plane_top, plane_bottom])
    clusters = cluster_coplanar_faces(planes, angle_tol=1e-3, dist_tol=1e-4)

    # 断言：相向/反向的两个共面接触面必须被归为同一个聚类组 (Cluster)
    assert len(clusters) == 1
    assert set(clusters[0]) == {0, 1}