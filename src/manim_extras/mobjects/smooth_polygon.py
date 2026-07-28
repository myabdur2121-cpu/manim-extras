"""
smooth_polygon.py
-----------------
A Manim Mobject that draws a *truly* smooth closed curve through a list of
vertices, using Manim's periodic (cyclic) cubic-Bezier spline solver.

The problem
===========
The usual workaround for "smooth closed shape through points" is::

    self.vertices = [v[-1], *v, v[0], v[1]]
    self.set_points_smoothly(self.vertices)
    self.set_points(self.points[4:-8])

``set_points_smoothly`` solves an *open* spline with natural (free-end)
boundary conditions, so padding the ends only approximates periodicity: the
tangent arriving at the first vertex never exactly matches the tangent leaving
it. That mismatch is the visible kink at the start point. It also allocates
extra anchors that must then be removed with fragile magic slice indices.

The fix
=======
Manim already ships a solver for the periodic case,
``get_smooth_closed_cubic_bezier_handle_points``, which solves the cyclic
tridiagonal system. The result is C2 continuous all the way around the loop,
including across the seam. No padding, no slicing, no kink.

Usage
=====
::

    from manim import *
    import numpy as np

    from manim_extras import SmoothPolygon


    class AnimationScene2(Scene):
        def construct(self):
            points = [
                np.array([-1, 2, 0]),
                np.array([1, 1, 0]),
                np.array([2, -1, 0]),
                np.array([-2, -1, 0]),
            ]
            shape = SmoothPolygon(points)
            self.add(shape)

Styling and open curves::

    SmoothPolygon(points, color=BLUE, fill_opacity=0.25, stroke_width=6)
    SmoothPolygon(points, is_closed=False)          # open smooth spline

Requires manim (Community Edition) >= 0.18; tested on v0.20.1. No LaTeX needed.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence

import numpy as np
from manim import DL, DR, UL, UR, VMobject
from manim.utils.bezier import (
    get_smooth_closed_cubic_bezier_handle_points,
    get_smooth_open_cubic_bezier_handle_points,
)

__all__ = ["SmoothPolygon"]


class SmoothPolygon(VMobject):
    """A smooth closed (or open) curve interpolating a list of vertices.

    Unlike :meth:`VMobject.set_points_smoothly`, the closed version uses a
    *periodic* spline solve, so the curve is smooth across the seam as well --
    there is no discontinuity at the starting point.

    Parameters
    ----------
    vertices
        Points the curve passes through, in order. Each may be any array-like
        of length 3 (``np.array([1, 2, 0])``, ``(1, 2, 0)``, ``UR``, ...).
        2D points are accepted and padded with ``z = 0``.
    is_closed
        ``True`` (default) loops back to the first vertex with full periodic
        smoothness. ``False`` builds an open smooth spline.
    kwargs
        Forwarded to :class:`VMobject` (``color``, ``stroke_width``,
        ``fill_color``, ``fill_opacity``, ...).

    Notes
    -----
    * Consecutive duplicate points, and a repeated closing vertex, are dropped
      automatically -- they would make the cyclic system singular.
    * Fewer than two distinct vertices yields an empty Mobject instead of
      raising, so a render never dies mid-animation.
    * This is a plain :class:`VMobject`: ``Create``, ``Transform``,
      ``MoveAlongPath``, ``.animate``, ``point_from_proportion`` all work.
    """

    DEFAULT_POINTS: list = [DR, DL, UL, UR]
    # Alias kept for backwards compatibility with the original misspelling.
    DEFULT_POINTS: list = DEFAULT_POINTS

    def __init__(
        self,
        vertices: Iterable[Sequence[float]] | None = None,
        is_closed: bool = True,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.is_closed = is_closed
        self.set_vertices(self.DEFAULT_POINTS if vertices is None else vertices)

    # ------------------------------------------------------------------ #
    # public API
    # ------------------------------------------------------------------ #
    def set_vertices(self, vertices: Iterable[Sequence[float]]) -> SmoothPolygon:
        """Replace the vertices and rebuild the curve. Returns ``self``."""
        self.vertices = self._clean(vertices)
        self._rebuild()
        return self

    def get_vertices(self) -> np.ndarray:
        """The anchor points the curve passes through, as an ``(n, 3)`` array."""
        return self.vertices.copy()

    def add_vertex(self, point: Sequence[float], index: int | None = None) -> SmoothPolygon:
        """Insert a vertex (append by default) and rebuild. Returns ``self``."""
        point = self._clean([point])
        index = len(self.vertices) if index is None else index
        return self.set_vertices(np.insert(self.vertices, index, point, axis=0))

    # ------------------------------------------------------------------ #
    # internals
    # ------------------------------------------------------------------ #
    @staticmethod
    def _clean(vertices: Iterable[Sequence[float]]) -> np.ndarray:
        """Normalise any array-like of 2D/3D points into an ``(n, 3)`` float array."""
        pts = np.asarray([np.asarray(v, dtype=float) for v in vertices], dtype=float)
        if pts.ndim != 2 or pts.shape[0] == 0:
            raise ValueError("`vertices` must be a non-empty sequence of 2D/3D points.")
        if pts.shape[1] == 2:  # allow (x, y) -> (x, y, 0)
            pts = np.hstack([pts, np.zeros((len(pts), 1))])
        if pts.shape[1] != 3:
            raise ValueError(f"Each vertex must have 2 or 3 coordinates, got {pts.shape[1]}.")

        # Drop consecutive duplicates and a duplicated closing vertex: they make
        # the spline system singular / produce zero-length segments.
        keep = np.append(True, ~np.all(np.isclose(pts[1:], pts[:-1]), axis=1))
        pts = pts[keep]
        if len(pts) > 1 and np.allclose(pts[0], pts[-1]):
            pts = pts[:-1]
        return pts

    def _rebuild(self) -> None:
        """Solve for the Bezier handles and write the point list."""
        anchors = self.vertices

        self.clear_points()
        if len(anchors) < 2:
            return

        if self.is_closed:
            # Periodic solve -> smooth across the seam. Close the ring so the
            # last segment runs from anchors[-1] back to anchors[0].
            ring = np.vstack([anchors, anchors[0]])
            h1, h2 = get_smooth_closed_cubic_bezier_handle_points(ring)
            start, end = ring[:-1], ring[1:]
        else:
            h1, h2 = get_smooth_open_cubic_bezier_handle_points(anchors)
            start, end = anchors[:-1], anchors[1:]

        # Interleave into Manim's flat [a0, h0, h1, a1, a1, h0, h1, a2, ...] layout.
        quads = np.empty((len(start) * 4, 3))
        quads[0::4], quads[1::4], quads[2::4], quads[3::4] = start, h1, h2, end
        self.set_points(quads)
