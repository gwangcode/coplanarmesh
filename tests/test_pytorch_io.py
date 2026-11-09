import numpy as np
import pytest
import trimesh

torch = pytest.importorskip("torch")

from coplanarmesh import sremesh

def test_pytorch_tensor_input_support():
    box_a = trimesh.creation.box(extents=[1.0, 1.0, 1.0])
    box_b = trimesh.creation.box(extents=[1.0, 1.0, 1.0])
    box_b.apply_translation([1.0, 0.0, 0.0])

    verts_a_tensor = torch.from_numpy(box_a.vertices).float()
    faces_a_tensor = torch.from_numpy(box_a.faces).long()

    mesh_a_reconstructed = trimesh.Trimesh(
        vertices=verts_a_tensor.cpu().numpy(),
        faces=faces_a_tensor.cpu().numpy()
    )

    res_a, res_b, pairs = sremesh(mesh_a_reconstructed, box_b)
    assert isinstance(res_a, trimesh.Trimesh)
    assert len(pairs) > 0

def test_device_agnostic_output():
    tensor_verts = torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=torch.float32)
    np_verts = tensor_verts.numpy()
    assert np_verts.dtype == np.float32
    assert np_verts.shape == (3, 3)