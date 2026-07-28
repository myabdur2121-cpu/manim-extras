"""Tests for SmoothPolygon.

Pure geometry checks -- nothing is rendered, so the whole file runs in well
under a second. The important one is `test_seam_is_smooth`, which verifies the
actual claim of the class: the tangent is continuous across the seam.
"""

from __future__ import annotations

import numpy as np
import pytest

from manim_extras import SmoothPolygon

SQUARE = [(-1, 2), (1, 1), (2, -1), (-2, -1)]


def tangent(mob, alpha: float, eps: float = 1e-4) -> np.ndarray:
    """Unit tangent of the curve at proportion ``alpha``."""
    a, b = max(0.0, alpha - eps), min(1.0, alpha + eps)
    d = mob.point_from_proportion(b) - mob.point_from_proportion(a)
    return d / np.linalg.norm(d)


# --------------------------------------------------------------------------- #
# construction
# --------------------------------------------------------------------------- #
def test_default_vertices():
    """No arguments -> the default unit square, four cubic segments."""
    mob = SmoothPolygon()
    assert len(mob.get_vertices()) == 4
    assert len(mob.points) == 16  # 4 segments * 4 points per cubic


def test_no_padding_anchors():
    """Exactly one segment per edge -- no leftover padding anchors."""
    mob = SmoothPolygon(SQUARE)
    assert len(mob.get_vertices()) == 4
    assert len(mob.points) == 4 * 4


def test_curve_passes_through_vertices():
    """The spline interpolates: each vertex is a segment anchor, in order."""
    mob = SmoothPolygon(SQUARE)
    assert np.allclose(mob.points[0::4], mob.get_vertices())


def test_closed_curve_is_closed():
    mob = SmoothPolygon(SQUARE)
    assert np.allclose(mob.get_start(), mob.get_end())


def test_open_curve_is_not_closed():
    mob = SmoothPolygon(SQUARE, is_closed=False)
    assert not np.allclose(mob.get_start(), mob.get_end())
    assert len(mob.points) == 3 * 4  # n-1 segments


# --------------------------------------------------------------------------- #
# the actual point of the class
# --------------------------------------------------------------------------- #
def test_seam_is_smooth():
    """Tangents entering and leaving the start point must agree.

    This is what the padded `set_points_smoothly` workaround gets wrong.
    """
    mob = SmoothPolygon(SQUARE)
    incoming = tangent(mob, 1.0 - 1e-3)
    outgoing = tangent(mob, 1e-3)
    assert np.dot(incoming, outgoing) > 0.999


def test_smoother_than_padded_workaround():
    """Beat the old approach on seam continuity, not just pass a threshold."""
    from manim import VMobject

    pts = [np.array([x, y, 0.0]) for x, y in SQUARE]
    old = VMobject()
    padded = [pts[-1], *pts, pts[0], pts[1]]
    old.set_points_smoothly(padded)
    old.set_points(old.points[4 : 4 * len(padded) - 8])

    new = SmoothPolygon(SQUARE)

    def seam_error(m):
        return 1.0 - float(np.dot(tangent(m, 1.0 - 1e-3), tangent(m, 1e-3)))

    assert seam_error(new) < seam_error(old)


def test_interior_joints_are_smooth():
    """Not only the seam: every joint is smooth."""
    mob = SmoothPolygon(SQUARE)
    n = len(mob.get_vertices())
    for i in range(1, n):
        alpha = i / n
        assert np.dot(tangent(mob, alpha - 2e-3), tangent(mob, alpha + 2e-3)) > 0.99


# --------------------------------------------------------------------------- #
# input handling
# --------------------------------------------------------------------------- #
def test_accepts_2d_points():
    mob = SmoothPolygon([(0, 0), (1, 0), (1, 1)])
    assert mob.get_vertices().shape == (3, 3)
    assert np.allclose(mob.get_vertices()[:, 2], 0)


def test_accepts_manim_direction_constants():
    from manim import DL, DR, UL, UR

    assert SmoothPolygon([UR, UL, DL, DR]).get_vertices().shape == (4, 3)


def test_drops_consecutive_duplicates_and_closing_vertex():
    mob = SmoothPolygon([(0, 0), (0, 0), (1, 0), (1, 1), (0, 0)])
    assert mob.get_vertices().shape == (3, 3)


def test_degenerate_input_yields_empty_mobject():
    """A single point must not raise -- a render should never die halfway."""
    assert len(SmoothPolygon([(0, 0)]).points) == 0


def test_rejects_bad_dimensions():
    with pytest.raises(ValueError):
        SmoothPolygon([(0, 0, 0, 0), (1, 1, 1, 1)])
    with pytest.raises(ValueError):
        SmoothPolygon([])


# --------------------------------------------------------------------------- #
# mutation API
# --------------------------------------------------------------------------- #
def test_add_vertex_appends_and_prepends():
    mob = SmoothPolygon([(0, 0), (1, 0), (1, 1)])
    mob.add_vertex((0, 1))
    assert np.allclose(mob.get_vertices()[-1], [0, 1, 0])
    mob.add_vertex((-1, -1), index=0)
    assert np.allclose(mob.get_vertices()[0], [-1, -1, 0])
    assert len(mob.points) == 5 * 4


def test_set_vertices_rebuilds():
    mob = SmoothPolygon(SQUARE)
    mob.set_vertices([(0, 0), (2, 0), (1, 2)])
    assert len(mob.get_vertices()) == 3
    assert len(mob.points) == 3 * 4


def test_get_vertices_returns_a_copy():
    mob = SmoothPolygon(SQUARE)
    mob.get_vertices()[0] = [99, 99, 99]
    assert not np.allclose(mob.get_vertices()[0], [99, 99, 99])


# --------------------------------------------------------------------------- #
# integration with Manim
# --------------------------------------------------------------------------- #
def test_style_kwargs_are_applied():
    from manim import BLUE

    mob = SmoothPolygon(SQUARE, color=BLUE, fill_opacity=0.5, stroke_width=6)
    assert mob.get_fill_opacity() == pytest.approx(0.5)
    assert mob.get_stroke_width() == pytest.approx(6)


def test_supports_standard_vmobject_transforms():
    mob = SmoothPolygon(SQUARE)
    before = mob.get_center().copy()
    mob.scale(2).shift(np.array([1.0, 0.0, 0.0]))
    assert not np.allclose(mob.get_center(), before)
