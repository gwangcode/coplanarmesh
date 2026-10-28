#!/usr/bin/env python3
"""
reproduce_paper_artifacts.py
Generate benchmark_results.csv, conformance.csv, and environment.txt
for the CoplanarMesh manuscript.

Usage:
    python examples/reproduce_paper_artifacts.py
    python examples/reproduce_paper_artifacts.py --quick
    python examples/reproduce_paper_artifacts.py --output-dir paper_artifacts
"""

import argparse
import csv
import platform
import sys
import time
from pathlib import Path

import numpy as np
import trimesh

from coplanarmesh.coplanarmesh import mremesh


# ---------------------------------------------------------------------------
# Process-level peak RSS (R1 Major 4: tracemalloc alone is insufficient)
# ---------------------------------------------------------------------------
def peak_rss_bytes():
    """Return the process's peak resident set size in bytes.

    ru_maxrss is KiB on Linux and bytes on macOS. On Windows the resource
    module is unavailable, so we return None and the CSV column is left empty.
    This is a cumulative peak for the whole process, not per-case.
    """
    try:
        import resource
    except ImportError:
        return None
    raw = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(raw) if sys.platform == "darwin" else int(raw) * 1024


# ---------------------------------------------------------------------------
# Benchmark cases (names match Table 2 of the manuscript)
# ---------------------------------------------------------------------------
def make_two_body():
    """Two unit cubes stacked on a shared z-plane."""
    a = trimesh.creation.box(extents=(1, 1, 1))
    b = trimesh.creation.box(extents=(1, 1, 1))
    b.apply_translation((0, 0, 1))
    return [a, b]


def make_five_body():
    """Central cube + four neighbours touching on distinct faces."""
    meshes = [trimesh.creation.box(extents=(1, 1, 1))]           # central

    west = trimesh.creation.box(extents=(1, 1, 1))
    west.apply_translation((-1, 0, 0))
    meshes.append(west)

    north = trimesh.creation.box(extents=(1, 1, 1))
    north.apply_translation((0, 1, 0))
    meshes.append(north)

    top = trimesh.creation.box(extents=(1, 1, 1))
    top.apply_translation((0, 0, 1))
    meshes.append(top)

    big = trimesh.creation.box(extents=(2, 1, 1))
    big.apply_translation((1.5, 0, 0))
    meshes.append(big)
    return meshes


def make_dense_pair():
    """Two subdivided cubes; many triangle pairs on the shared interface."""
    a = trimesh.creation.box(extents=(1, 1, 1)).subdivide().subdivide()
    b = trimesh.creation.box(extents=(1, 1, 1)).subdivide().subdivide()
    b.apply_translation((0, 0, 1))
    return [a, b]


def make_wall_grid(n=3, gap=1e-3):
    """3x3 array of unit cubes with a small gap so no interfaces are detected."""
    meshes = []
    for i in range(n):
        for j in range(n):
            c = trimesh.creation.box(extents=(1, 1, 1))
            c.apply_translation((i * (1 + gap), j * (1 + gap), 0))
            meshes.append(c)
    return meshes


def make_cylindrical_pair():
    """Two cylinders touching along their flat end-cap plane."""
    a = trimesh.creation.cylinder(radius=0.5, height=1.0, sections=32)
    b = trimesh.creation.cylinder(radius=0.5, height=1.0, sections=32)
    b.apply_translation((0, 0, 1))
    return [a, b]


BENCHMARKS = [
    ("Two-Body Basic", make_two_body),
    ("5-Body Stack", make_five_body),
    ("High-Density Pair", make_dense_pair),
    ("3x3 Wall Grid", make_wall_grid),
    ("Cylindrical Pair", make_cylindrical_pair),
]


# ---------------------------------------------------------------------------
# Benchmark runner
# ---------------------------------------------------------------------------
def run_case(name, factory, repeats, grid_size, eps, min_area_eps):
    meshes = factory()
    in_v = sum(len(m.vertices) for m in meshes)
    in_f = sum(len(m.faces) for m in meshes)

    runtimes, rss = [], []
    out_f, n_pairs, status = None, None, "Passed"

    for _ in range(repeats):
        copies = [m.copy() for m in meshes]
        try:
            t0 = time.perf_counter()
            aligned, pairs = mremesh(
                copies,
                grid_size=grid_size,
                eps=eps,
                min_area_eps=min_area_eps,
            )
            t1 = time.perf_counter()
            runtimes.append(t1 - t0)
            rss.append(peak_rss_bytes())
            out_f = sum(len(m.faces) for m in aligned)
            n_pairs = len(pairs)
        except Exception as exc:  # noqa: BLE001 - we want to record failures
            status = f"Failed: {type(exc).__name__}: {exc}"
            break

    row = {
        "case": name,
        "contact_pairs": n_pairs,
        "input_vertices": in_v,
        "input_faces": in_f,
        "output_faces": out_f,
        "runtime_mean_s": round(float(np.mean(runtimes)), 4) if runtimes else None,
        "runtime_std_s": round(float(np.std(runtimes)), 4) if runtimes else None,
        "rss_peak_mb": (
            round(max(rss) / (1024 * 1024), 2) if rss and rss[0] is not None else None
        ),
        "n_repeats": repeats,
        "status": status,
    }
    print(f"[benchmark] {row}")
    return row


# ---------------------------------------------------------------------------
# Hanging-node conformity
# ---------------------------------------------------------------------------
def _interface_vertices(mesh, plane_z, tol):
    return mesh.vertices[np.abs(mesh.vertices[:, 2] - plane_z) < tol]


def count_hanging(mesh_a, mesh_b, plane_z, tol=1e-5):
    """Count interface vertices that do not appear in both meshes.

    A vertex is considered part of the interface if it lies within tol of the
    shared plane. It is considered matched if its quantized coordinates appear
    in both meshes' interface vertex sets. The denominator is the number of
    unique interface vertices from both meshes combined.
    """
    va = _interface_vertices(mesh_a, plane_z, tol)
    vb = _interface_vertices(mesh_b, plane_z, tol)

    keys_a = set(map(tuple, np.round(va / tol).astype(np.int64)))
    keys_b = set(map(tuple, np.round(vb / tol).astype(np.int64)))

    unique = keys_a | keys_b
    hanging = sum(1 for k in unique if not (k in keys_a and k in keys_b))
    return hanging, len(unique)


def run_conformance():
    """Mixed-resolution stack: coarse cube below, subdivided cube above."""
    a = trimesh.creation.box(extents=(1, 1, 1))                       # 2 tri on top
    b = trimesh.creation.box(extents=(1, 1, 1)).subdivide().subdivide()
    b.apply_translation((0, 0, 1))                                    # many verts on bottom
    plane_z = 0.5

    h0, n0 = count_hanging(a, b, plane_z)
    aligned, _ = mremesh(
        [a.copy(), b.copy()], grid_size=1e-5, eps=1e-5, min_area_eps=1e-6
    )
    h1, n1 = count_hanging(aligned[0], aligned[1], plane_z)

    rows = [
        {
            "case": "Two-Body Stack (mixed resolution)",
            "stage": "unaligned",
            "hanging_nodes": h0,
            "interface_vertices": n0,
            "mismatch_rate": round(h0 / n0, 4) if n0 else 0.0,
        },
        {
            "case": "Two-Body Stack (mixed resolution)",
            "stage": "after CoplanarMesh",
            "hanging_nodes": h1,
            "interface_vertices": n1,
            "mismatch_rate": round(h1 / n1, 4) if n1 else 0.0,
        },
    ]
    for r in rows:
        print(f"[conformance] {r}")
    return rows


# ---------------------------------------------------------------------------
# Environment report
# ---------------------------------------------------------------------------
def collect_environment():
    env = {
        "python": sys.version.replace("\n", " "),
        "platform": platform.platform(),
        "numpy": np.__version__,
        "trimesh": trimesh.__version__,
    }
    try:
        import scipy
        env["scipy"] = scipy.__version__
    except ImportError:
        pass
    try:
        import shapely
        env["shapely"] = shapely.__version__
    except ImportError:
        pass
    try:
        import triangle
        env["triangle"] = getattr(triangle, "__version__", "unknown")
    except ImportError:
        pass
    return env


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="paper_artifacts")
    parser.add_argument(
        "--quick",
        action="store_true",
        help="single repeat and skip the two slowest cases",
    )
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    repeats = 1 if args.quick else 5
    skip = {"High-Density Pair", "3x3 Wall Grid"} if args.quick else set()

    rows = []
    for name, factory in BENCHMARKS:
        if name in skip:
            print(f"[benchmark] skipping {name} (--quick)")
            continue
        rows.append(run_case(name, factory, repeats, 1e-5, 1e-5, 1e-6))

    bench_csv = out_dir / "benchmark_results.csv"
    with bench_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {bench_csv}")

    conf_rows = run_conformance()
    conf_csv = out_dir / "conformance.csv"
    with conf_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(conf_rows[0].keys()))
        writer.writeheader()
        writer.writerows(conf_rows)
    print(f"Wrote {conf_csv}")

    env = collect_environment()
    env_txt = out_dir / "environment.txt"
    with env_txt.open("w") as f:
        for key, value in env.items():
            f.write(f"{key}: {value}\n")
    print(f"Wrote {env_txt}")


if __name__ == "__main__":
    main()
