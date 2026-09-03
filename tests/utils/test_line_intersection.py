"""Tests for line intersection geometry.

Pure math on plain point lists -- nothing is rendered, so the whole file runs
in a few milliseconds. The important ones:

* ``test_intersection_point`` -- the canonical [[0,0],[2,2]] × [[0,2],[2,0]]
  case must meet at exactly [1.0, 1.0].
* ``test_segments_intersect_false`` -- two infinite lines that cross, but whose
  bounded segments miss each other, correctly report ``False``.
* ``test_parallel_lines_raise`` -- a zero determinantee must raise, not divide
  by zero or silently return a bogus point.
"""

from __future__ import annotations

import pytest

from manim_extras import GeometryOperations


def test_intersection_point():
    """The two crossing diagonals of the unit square meet at [1, 1]."""
    pt = GeometryOperations.intersection_point([[0, 0], [2, 2]], [[0, 2], [2, 0]])
    assert pt == pytest.approx([1.0, 1.0])


def test_intersection_point_integer_input():
    """Integer inputs come back as plain floats, not numpy scalars."""
    pt = GeometryOperations.intersection_point([[0, 0], [2, 2]], [[0, 2], [2, 0]])
    assert all(isinstance(c, float) for c in pt)


def test_segments_intersect_true():
    """The same lines as bounded segments intersect."""
    assert GeometryOperations.segments_intersect([[0, 0], [2, 2]], [[0, 2], [2, 0]])


def test_segments_intersect_false():
    """Lines cross, but the bounded segments don't touch."""
    assert not GeometryOperations.segments_intersect([[0, 0], [2, 0]], [[0, 2], [0, 1]])


def test_segment_intersection_point_none():
    """When segments miss, `segment_intersection_point` returns None."""
    result = GeometryOperations.segment_intersection_point([[0, 0], [2, 0]], [[0, 2], [0, 1]])
    assert result is None


def test_segment_intersection_point_value():
    """When segments meet, the point is returned."""
    result = GeometryOperations.segment_intersection_point([[0, 0], [2, 2]], [[0, 2], [2, 0]])
    assert result == pytest.approx([1.0, 1.0])


def test_parallel_lines_raise():
    """Parallel lines have a zero determinant and must raise ValueError."""
    with pytest.raises(ValueError):
        GeometryOperations.get_proportional_value([[0, 0], [1, 1]], [[2, 2], [3, 3]], 1)


def test_coincident_lines_raise():
    """Coincident lines are also degenerate."""
    with pytest.raises(ValueError):
        GeometryOperations.get_proportional_value([[0, 0], [1, 1]], [[0, 0], [1, 1]], 1)


def test_get_point_by_proportion():
    """A t of 0.5 halfway along [[0,0],[2,2]] is [1, 1]."""
    pt = GeometryOperations.get_point_by_proportion([[0, 0], [2, 2]], 0.5)
    assert pt == [1.0, 1.0]


def test_get_point_by_proportion_returns_float():
    """The interpolated point is a plain list of floats."""
    pt = GeometryOperations.get_point_by_proportion([[0, 0], [2, 2]], 0.5)
    assert all(isinstance(c, float) for c in pt)


def test_displacement():
    """Displacement from A to B is B - A."""
    assert GeometryOperations.displacement([1, 2], [4, 6]) == [3, 4]
