import os
import numpy as np
import trimesh
# 直接引入全新封装的高层统一接口 sremesh
from coplanarmesh import sremesh


def main():
    # 1. 配置文件路径（采用相对路径+防御性组装，确保跨平台鲁棒性）
    base_dir = '/content' if os.path.exists('/content') else 'mesh_files'

    cube_file = os.path.join(base_dir, 'central_cube-1.stl')
    pyramid_file = os.path.join(base_dir, 'pyramid-1.stl')

    # Defensive path check
    if not os.path.exists(cube_file) or not os.path.exists(pyramid_file):
        print("[ERROR] Target STL files not found in the specified directory!")
        print(f"Expected paths:\n  - {cube_file}\n  - {pyramid_file}")
        return

    print(">> Loading 3D geometries for optical simulation...")
    # 这里保持原代码中绑定的变量语义：Mesh A 为 Pyramid，Mesh B 为 Cube
    pyramid = trimesh.load(pyramid_file)
    central_cube = trimesh.load(cube_file)

    print(">> Invoking unified sremesh pipeline for boundary-aligned topological remeshing...")
    # 核心飞跃：一行代码替代低级调用，返回优化后的单体和交叠对元组列表 [(idxA, idxB), ...]
    meshA_final, meshB_final, final_overlap_pairs = sremesh(pyramid, central_cube)

    # 2. 打印工业级双体拓扑对齐全局账本
    print("\n" + "=" * 25 + " TOPOLOGICAL ALIGNMENT REPORT " + "=" * 25)
    print(
        f"Final Mesh A (Pyramid)     -> Vertices: {meshA_final.vertices.shape[0]:<6} | Faces: {meshA_final.faces.shape[0]}")
    print(
        f"Final Mesh B (Central Cube)  -> Vertices: {meshB_final.vertices.shape[0]:<6} | Faces: {meshB_final.faces.shape[0]}")
    print(f"Total Verified Coplanar Contact Pairs: {len(final_overlap_pairs)}")
    print("-" * 76)

    # =========================================================================
    # 3. 核心升级：遍历并解构交叠面的具体顶点和实数空间坐标
    # =========================================================================
    for idx, (face_idx_A, face_idx_B) in enumerate(final_overlap_pairs):
        print(f"  [{idx + 1:02d}] Interface Connection Verified:")

        # 提取 Mesh A (Pyramid) 对应的几何实体信息
        v_indices_A = meshA_final.faces[face_idx_A]
        v_coords_A = meshA_final.vertices[v_indices_A]
        print(f"       ▪ Mesh A (Face Index: {face_idx_A:<4}) -> Vertex IDs: {list(v_indices_A)}")
        print(f"         Coords:\n{np.round(v_coords_A, 5)}")

        # 提取 Mesh B (North Cube) 对应的几何实体信息
        v_indices_B = meshB_final.faces[face_idx_B]
        v_coords_B = meshB_final.vertices[v_indices_B]
        print(f"       ▪ Mesh B (Face Index: {face_idx_B:<4}) -> Vertex IDs: {list(v_indices_B)}")
        print(f"         Coords:\n{np.round(v_coords_B, 5)}")
        print(f"       " + "." * 64)

    print("=" * 76)
    print(">> Pipeline completed successfully. Interface is ready for optical ray-tracing.")

if __name__ == "__main__":
    main()