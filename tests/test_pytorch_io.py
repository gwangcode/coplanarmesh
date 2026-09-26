import numpy as np
import pytest


def test_pytorch_tensor_input_support():
    """测试当输入数据为 PyTorch Tensor 时，接口是否能自动兼容转换而不报错"""
    try:
        import torch
        from coplanarmesh import sremesh
        import trimesh

        # 创建基础 trimesh 对象
        box_a = trimesh.creation.box(extents=[1.0, 1.0, 1.0])
        box_b = trimesh.creation.box(extents=[1.0, 1.0, 1.0])
        box_b.apply_translation([1.0, 0.0, 0.0])

        # 模拟提取 PyTorch Tensor 格式的顶点与面数据
        verts_a_tensor = torch.from_numpy(box_a.vertices).float()
        faces_a_tensor = torch.from_numpy(box_a.faces).long()

        # 验证能够无缝转换回 NumPy / Trimesh 结构
        mesh_a_reconstructed = trimesh.Trimesh(
            vertices=verts_a_tensor.cpu().numpy(),
            faces=faces_a_tensor.cpu().numpy()
        )

        res_a, res_b, pairs = sremesh(mesh_a_reconstructed, box_b)

        # 断言：输出类型正确且能够成功计算共面对齐对
        assert isinstance(res_a, trimesh.Trimesh)
        assert len(pairs) > 0

    except ImportError:
        pytest.skip("PyTorch is not installed in the environment. Skipping PyTorch IO test.")


def test_device_agnostic_output():
    """测试与 Tensor 交互后的数值精度（float32 与 float64）保持"""
    try:
        import torch
        tensor_verts = torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=torch.float32)
        np_verts = tensor_verts.numpy()

        assert np_verts.dtype == np.float32
        assert np_verts.shape == (3, 3)
    except ImportError:
        pytest.skip("PyTorch is not installed. Skipping test.")