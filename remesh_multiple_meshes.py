import os
import numpy as np
import trimesh
# Invoke the multi-body iterative interface under the new topological architecture
from coplanarmesh import mremesh


def main():
    # 1. Configure file paths using relative paths and defensive assembly for cross-platform robustness
    base_dir = '/content' if os.path.exists('/content') else 'mesh_files'

    mesh_names = [
        'central_cube-1.stl',
        'pyramid-1.stl',
        'north_cube-1.stl',
        'west_cube-1.stl',
        'big_cube-1.stl'
    ]

    mlist = []
    print(">> Scanning and loading multi-body 3D asset cluster...")
    for name in mesh_names:
        target_path = os.path.join(base_dir, name)
        if not os.path.exists(target_path):
            print(f"[ERROR] Required component missing: {target_path}")
            print("Please ensure all 5 industrial STL files are present.")
            return

        mlist.append(trimesh.load(target_path))
        print(f"   [LOADED] {name:<20} | Initial Faces: {mlist[-1].faces.shape[0]}")

    print(f"\n>> Pipeline ready. Initializing multi-body recursive slicing network for {len(mlist)} components...")
    print(">> Resolving n-way overlapping boundaries and stitching patches...")

    # 2. Invoke the core multi-body mesh remeshing pipeline
    fmeshes, fpairs = mremesh(mlist, eps=5e-3, min_area_eps=1e-4)

    # 3. Print the industrial-grade multi-body alignment master ledger
    print("\n" + "=" * 24 + " MULTI-BODY ALIGNMENT MASTER LEDGER " + "=" * 24)
    print(f"Total Processed Independent Bodies: {len(fmeshes)}")

    # Print the optimized final topological specifications for each component
    for idx, name in enumerate(mesh_names):
        orig_faces = mlist[idx].faces.shape[0]
        final_faces = fmeshes[idx].faces.shape[0]
        final_verts = fmeshes[idx].vertices.shape[0]
        print(
            f"  Component [{idx}] {name:<16} -> Verts: {final_verts:<5} | Faces: {final_faces:<5} (Orig: {orig_faces})")

    print("-" * 84)
    print(f"Total Discovered Coplanar Contact Interfaces: {len(fpairs)}")

    # =========================================================================
    # 4. High-fidelity parsing of multi-body contact pairs:
    #    Extract and print the intersecting geometric entities (Vertices & Faces)
    # =========================================================================
    pair_counter = 1

    # Convert to standard localized index pairs, compatible with flat (mesh_i, face_i, mesh_j, face_j) structure.
    # If mremesh returns flattened 4-tuples, aggregate and classify them by component pairs (i, j) first.
    structured_pairs = {}
    for item in fpairs:
        if len(item) == 4:
            mi, fi, mj, fj = item
            if (mi, mj) not in structured_pairs:
                structured_pairs[(mi, mj)] = []
            structured_pairs[(mi, mj)].append((fi, fj))
        elif len(item) == 3:
            mi, mj, face_pairs = item
            structured_pairs[(mi, mj)] = face_pairs
        else:
            print(f"  [WARNING] Unknown interface data format: {item}")

    # Traverse and highlight the spatial geometric attributes
    for (mesh_i, mesh_j), face_pairs in structured_pairs.items():
        name_A = mesh_names[mesh_i].split('-')[0]
        name_B = mesh_names[mesh_j].split('-')[0]
        num_contacts = len(face_pairs)

        print(f"\n  [{pair_counter:02d}] Interface Found: Mesh {mesh_i} ({name_A}) <===> Mesh {mesh_j} ({name_B})")
        print(f"       ↳ Validated Shared Topology: {num_contacts} perfectly aligned face pairs.")

        # Extract the two optimized mesh entities that are currently in contact
        fmesh_A = fmeshes[mesh_i]
        fmesh_B = fmeshes[mesh_j]

        # Traverse specific face index pairs to capture underlying 3D spatial data
        for sub_idx, (face_idx_A, face_idx_B) in enumerate(face_pairs):
            # Retrieve vertex indices for the triangular faces
            v_indices_A = fmesh_A.faces[face_idx_A]
            v_indices_B = fmesh_B.faces[face_idx_B]

            # Extract the corresponding 3D spatial coordinate matrices (3, 3)
            v_coords_A = fmesh_A.vertices[v_indices_A]
            v_coords_B = fmesh_B.vertices[v_indices_B]

            print(f"         [-] Contact Pair #{sub_idx + 1}:")
            print(f"             ▪ Mesh {mesh_i} (Face Index: {face_idx_A:<4}) -> Vertex IDs: {list(v_indices_A)}")
            print(f"               Coords:\n{np.round(v_coords_A, 5)}")
            print(f"             ▪ Mesh {mesh_j} (Face Index: {face_idx_B:<4}) -> Vertex IDs: {list(v_indices_B)}")
            print(f"               Coords:\n{np.round(v_coords_B, 5)}")

        pair_counter += 1

    print("\n" + "=" * 84)
    print(">> Multi-body topological serialization complete.")

if __name__ == "__main__":
    main()