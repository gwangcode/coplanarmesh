# CoplanarMesh 🚀

**CoplanarMesh** is a robust, industrial-grade 3D triangular mesh pipeline designed for **coplanar overlap detection, adaptive slicing, watertight stitching, and global index re-alignment**. 

It is engineered specifically to eliminate structural crashes (such as Shapely's `TopologyException` or facet intersecting slivers) caused by floating-point rounding errors, zero-thickness contact interfaces, and razor-sharp boundary anomalies typically found in CAD (STEP/IGES) or BIM (Revit/IFC) exported tessellated meshes.

---

## 🌟 Architecture & Technical Deep Dive

The framework abandons brittle historical dependency trees and instead adopts a strict spatial-computing state machine. The codebase is decoupled into five core dedicated pillars following production-level pipeline design:

### 1. Fuzzy Plane Binning & Hashing (`hashing.py`)
Instead of performing costly $O(N^2)$ all-to-all geometric plane checks, this module maps infinite 3D plane equations ($n_x \cdot x + n_y \cdot y + n_z \cdot z + d = 0$) into discrete voxel-like bucket tokens. By normal-alignment flipping and introducing a dual-grid overlapping zone tolerance (`border_eps`), it guarantees that faces with slight floating-point truncations fall into the same "spatial drawer" with absolute determinism.

### 2. Dimension Reduction & Constrained Triangulation (`geometry.py`)
To utilize topological computation engines safely, 3D coplanar faces are dynamically projected onto their principal 2D plane by discarding the axis of the largest normal component. When complex overlapping sub-regions are computed, this module invokes a dual-layer triangulation recovery fallback: integrating the high-speed `trimesh.creation.triangulate_polygon` with a raw `Triangle` library (`C-based Triangle wrapper`) constraint-graph solver to completely bypass degenerate face exceptions.

### 3. Pairwise Finite State Machine Slicing (`remesh.py`)
This is the "slicing workshop" of the project. It classifies interacting coplanar regions into distinct topological states:
- `PERFECT`: Absolute geometric identity matches (one face completely swallowed or perfectly aligned with another).
- `PARTIAL_SHARED`: Intersecting overlapping fragments that must be segmented out and tagged.
- `PARTIAL_REMAIN_A` / `PARTIAL_REMAIN_B`: Isolated boundary remnants representing independent standalone mesh spaces.

It uses boolean subtraction to hollow out the old interfaces and perfectly patches the newly bounded triangulation pieces back, generating exact boolean-remap tracking labels.

### 4. Grid Snapping & Deduplication (`utils.py`)
Microscopic slivers and overlapping double-points are the root cause of downstream mesh corruption. This utility clamps loose coordinates onto a standardized 3D discrete grid via mathematical quantization (`np.round(V / grid_size) * grid_size`), collapses degenerate triangles, merges coincidental borders, and leverages `pymeshfix` to ensure absolute watertight physical healing.

### 5. Multi-Mesh Cascade Stream Orchestration (`coplanarmesh.py`)
The top-level stream component. It accepts a sequence of $N$ arbitrary meshes and sets up a cascading pairwise processing loop. Because a mesh modified in an early combination (e.g., Mesh A hit by Mesh B) will affect later collisions (e.g., Mesh A hitting Mesh C), the pipeline performs **"Infinite Iterative Remeshing"**. Once the physical bodies reach static topological equilibrium, it runs a reverse global RTree AABB tracking pass to map the definitive contact interface matrix.

---

## 🔥 Key Competitive Advantages

### ⚡ Adaptive Local Tolerances (`safe_inter_eps`)
Global rigid tolerances ruin complex assemblies. If set too small, jagged floating-point boundaries create sliver-face artifacts; if set too large, micro-scale features are ignored. **CoplanarMesh** uses an adaptive threshold mechanism:
```math
\text{local_avg_area} = \frac{\text{Area}(\text{polyA}) + \text{Area}(\text{polyB})}{2}
```
This allows massive structural wall sheets to flush out micro-meter edge noise with aggressive filtering while automatically tightening down the defense barrier to preserve tiny mechanical pin connectors.

### 🏷️ Lossless Reverse Index Alignment
Tracking face indices through deep multi-body boolean operations is historically a nightmare. **CoplanarMesh** solves this by separating the geometric cutting from index logging. It cuts meshes freely, stabilizes their topologies, and then uses a highly optimized 2D spatial overlapping area index-reflection pass at the very end to deliver an immutable, perfectly clean global contact ledger.

---

## 🛠️ Installation

Clone this repository, navigate to the folder containing `pyproject.toml`, and choose one of the installation strategies:

### 1. Developer Mode (Highly Recommended)
Any structural or algorithmic modifications made to the source `.py` files take effect immediately across your python environment without re-running scripts.
```bash
pip install -e .

```

### 2. Standard Distribution Installation

```bash
pip install .

```

---

## 🚀 Quick Start Guide

The entire multi-body workflow is wrapped into a clean, unified `remesh` interface. Here is a production-level usage example creating a close-contact coplanar scene:

```python
import trimesh
from coplanarmesh import remesh

# 1. Instantiate two standard primitives with a perfect contact interface
boxA = trimesh.creation.box(extents=[1.0, 1.0, 1.0])
boxB = trimesh.creation.box(extents=[1.0, 1.0, 1.0])

# Translate boxB along X axis so its Left face perfectly touches boxA's Right face
boxB.apply_translation([1.0, 0.0, 0.0])

mesh_list = [boxA, boxB]

# 2. Invoke the unified cascading remesh pipeline
# It handles slicing, collapses sliver nodes, and aligns global indices
final_meshes, global_pairs = remesh(mesh_list, grid_size=1e-5, eps=1e-5)

print(f"Remeshing process completed.")
print(f"Total structured meshes generated: {len(final_meshes)}")
print(f"\n--- Aligned Contact Interface Matrix (Total Intersections: {len(global_pairs)}) ---")

for pair in global_pairs:
    mesh_i, face_i, mesh_j, face_j = pair
    print(f" -> Mesh index [{mesh_i}] Face [{face_i}] aligns perfectly in 3D space against Mesh index [{mesh_j}] Face [{face_j}].")

```

---

## 📂 Package Repository Layout

The localized architecture inside `src/coplanarmesh/` is laid out as follows:

```text
coplanarmesh/
├── pyproject.toml              # Modern setuptools pep517 build backend
├── README.md                   # Technical documentation and layout manual
└── src/
    └── coplanarmesh/           # Main package namespace
        ├── __init__.py         # Clean semantic exposure interface
        ├── hashing.py          # Fuzzy discretization plane mapping logic
        ├── geometry.py         # Coordinate reductions & triangulation backends
        ├── remesh.py           # Pairwise FSM patch-generation framework
        ├── utils.py            # Vertex rounding, deduplication & meshfix repairs
        └── coplanarmesh.py     # High-level 'remesh()' pipeline orchestration

```

---

## 📝 License & Contributions

Distributed under the MIT License. Contributions focused on accelerating the 2D polygon intersection speed via customized C-bindings or vectorized spatial-trees are highly welcome.

```

```
