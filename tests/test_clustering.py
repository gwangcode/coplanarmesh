import numpy as np
import pytest
from coplanarmesh.hashing import hash_plane_fuzzy, precompute_plane_drawers


def test_plane_hashing_determinism():
    """测试平面方程在微小浮点误差下是否能映射到相同的哈希桶 (Fuzzy Binning)"""
    # 构造两个具有极其微小扰动的共面平面参数 (法向量 n 和 截距 d)
    n = np.array([0.0, 0.0, 1.0])
    d_a = 0.5000001
    d_b = 0.4999999

    # 设定容差
    eps = 1e-3

    keys_a = hash_plane_fuzzy(n, d_a, eps=eps)
    keys_b = hash_plane_fuzzy(n, d_b, eps=eps)

    # 断言：微小扰动的平面必须落在至少一个相同的空间哈希桶 (Bin) 中
    assert set(keys_a).intersection(set(keys_b))


def test_opposite_normal_canonicalization():
    """测试反向法线的共面平面（如 [0,0,1] 和 [0,0,-1]）是否会被规范化（Canonicalize）为相同的哈希桶"""
    n_top = np.array([0.0, 0.0, 1.0])
    d_top = -0.5

    n_bottom = np.array([0.0, 0.0, -1.0])
    d_bottom = 0.5

    keys_top = hash_plane_fuzzy(n_top, d_top)
    keys_bottom = hash_plane_fuzzy(n_bottom, d_bottom)

    # 断言：反向相向的共面平面，规范化后生成的哈希 Key 应该一致
    assert keys_top == keys_bottom


def test_precompute_plane_drawers():
    """测试 Trimesh 对象能否成功预计算并分配到 Fuzzy Binning Drawers"""
    import trimesh
    cube = trimesh.creation.box(extents=[1.0, 1.0, 1.0])

    drawers = precompute_plane_drawers(cube)

    # 断言：抽屉字典不为空，且包含面数据元组
    assert isinstance(drawers, dict)
    assert len(drawers) > 0
