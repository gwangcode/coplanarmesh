import os
import trimesh
# Import lower-level state machine routines from the core package
from coplanarmesh import process_and_label_coplanar, clean_mesh_and_reindex


def main():
    # Define local target files
    cube_file = 'mesh_files\central_cube-1.stl'
    pyramid_file = 'mesh_files\pyramid-1.stl'

    # Defensive path check
    if not os.path.exists(cube_file) or not os.path.exists(pyramid_file):
        print("[ERROR] Target STL files not found in the current working directory!")
        print("Please ensure 'central_cube-1.stl' and 'pyramid-1.stl' are present.")
        return

    print(">> Initializing low-level coplanar slicing finite-state-machine (FSM)...")
    central_cube = trimesh.load(cube_file)
    pyramid = trimesh.load(pyramid_file)

    # 1. Execute the core dispatcher to perform CSG subtraction and patch sewing
    meshA_patch, meshB_patch, overlapA, overlapB, _ = process_and_label_coplanar(pyramid, central_cube)

    print(">> Slicing completed. Invoking post-processor for vertex deduplication and index alignment...")
    # 2. Re-index and collapse degenerate elements to build clean final topologies
    meshA_final, meshB_final, final_overlap_A, final_overlap_pairs = clean_mesh_and_reindex(
        vertsA=meshA_patch.vertices,
        facesA=meshA_patch.faces,
        overlapA=overlapA,
        vertsB=meshB_patch.vertices,
        facesB=meshB_patch.faces,
        overlapB=overlapB
    )

    # 3. Print highly detailed industrial topology ledger
    print("\n" + "=" * 25 + " TOPOLOGICAL ALIGNMENT REPORT " + "=" * 25)
    print(
        f"Final Mesh A (Pyramid)      -> Vertices: {meshA_final.vertices.shape[0]:<6} | Faces: {meshA_final.faces.shape[0]}")
    print(
        f"Final Mesh B (Central Cube) -> Vertices: {meshB_final.vertices.shape[0]:<6} | Faces: {meshB_final.faces.shape[0]}")
    print(f"Tracked Overlap Face IDs in Mesh A: {final_overlap_A}")
    print(f"Total Verified Coplanar Contact Pairs: {len(final_overlap_pairs)}")
    print("-" * 76)

    for idx, pair in enumerate(final_overlap_pairs):
        print(f"  [{idx + 1:02d}] Mesh A (Face {pair[0]:<4}) <===> Mesh B (Face {pair[1]:<4}) is perfectly aligned.")
    print("=" * 76)


if __name__ == "__main__":
    main()