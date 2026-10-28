import numpy as np
import pytest
from shapely.geometry import Polygon

from coplanarmesh.geometry import remesh_shapely_polygon


def _poly_with_hole():
    outer = [(0, 0), (4, 0), (4, 4), (0, 4)]
    hole = [(1, 1), (1, 2), (2, 2), (2, 1)]
    return Polygon(outer, [hole])


def test_triangulates_polygon_with_hole():
    v, t = remesh_shapely_polygon(_poly_with_hole(), simplify_tolerance=1e-6, area_eps=1e-12)
    assert v is not None and t is not None
    assert len(t) > 0
    centroids = v[t].mean(axis=1)
    inside_hole = (
        (centroids[:, 0] > 1.0) & (centroids[:, 0] < 2.0)
        & (centroids[:, 1] > 1.0) & (centroids[:, 1] < 2.0)
    )
    assert not inside_hole.any(), "triangulation filled the hole"


def test_fallback_path_is_exercised(monkeypatch):
    """Force Strategy 1 to raise so Strategy 2 runs, and assert the warning."""
    import trimesh

    def _boom(*args, **kwargs):
        raise RuntimeError("forced failure")

    monkeypatch.setattr(trimesh.creation, "triangulate_polygon", _boom)

    # The call MUST be inside pytest.warns, otherwise the warning is emitted
    # before the context manager is entered and cannot be captured.
    with pytest.warns(RuntimeWarning, match="falling back to Triangle"):
        v, t = remesh_shapely_polygon(
            _poly_with_hole(), simplify_tolerance=1e-6, area_eps=1e-12
        )

    assert v is not None and t is not None, (
        "fallback must succeed or raise, not silently return None"
    )
