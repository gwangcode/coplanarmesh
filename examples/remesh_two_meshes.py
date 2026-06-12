import os
import trimesh
# 直接引入全新封装的高层统一接口 smesh
from coplanarmesh import sremesh


def main():
    # Define local target files (使用统一的斜杠以防跨平台路径转义错误)
    cube_file = os.path.join('mesh_files', 'central_cube-1.stl')
    pyramid_file = os.path.join('mesh_files', 'pyramid-1.stl')

    # Defensive path check
    if not os.path.exists(cube_file) or not os.path.exists(pyramid_file):
        print("[ERROR] Target STL files not found in the current working directory!")
        print("Please ensure 'central_cube-1.stl' and 'pyramid-1.stl' are present.")
        return

    print(">> Loading 3D geometries for optical simulation...")
    central_cube = trimesh.load(cube_file)
    pyramid = trimesh.load(pyramid_file)

    print(">> Invoking unified smesh pipeline for boundary-aligned topological remeshing...")
    # 核心飞跃：一行代码替代了原有的两步低级状态机调用与繁琐的参数传递
    meshA_final, meshB_final, final_overlap_pairs = sremesh(pyramid, central_cube)

    # 3. Print highly detailed industrial topology ledger
    print("\n" + "=" * 25 + " TOPOLOGICAL ALIGNMENT REPORT " + "=" * 25)
    print(
        f"Final Mesh A (Pyramid)      -> Vertices: {meshA_final.vertices.shape[0]:<6} | Faces: {meshA_final.faces.shape[0]}")
    print(
        f"Final Mesh B (Central Cube) -> Vertices: {meshB_final.vertices.shape[0]:<6} | Faces: {meshB_final.faces.shape[0]}")
    print(f"Total Verified Coplanar Contact Pairs: {len(final_overlap_pairs)}")
    print("-" * 76)

    for idx, pair in enumerate(final_overlap_pairs):
        print(f"  [{idx + 1:02d}] Mesh A (Face {pair[0]:<4}) <===> Mesh B (Face {pair[1]:<4}) is perfectly aligned.")
    print("=" * 76)
    print(">> Pipeline completed successfully. Interface is ready for optical ray-tracing.")


if __name__ == "__main__":
    main()