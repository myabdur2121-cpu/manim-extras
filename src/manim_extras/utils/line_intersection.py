"""
line_intersection.py
--------------------
2D line intersection helpers: a pure-math core plus a thin bridge to Manim
Mobjects.

The package keeps geometry separate from rendering, so this module has two
layers:

``GeometryOperations``
    Pure 2D math on plain ``[[x, y], [x, y]]`` lists (a "Line2D"). No Manim
    involved, so it can be unit-tested and reused anywhere.

``ManimGeometryAdapter``
    Converts Manim ``Mobject``\\ s (``Line``, dots) to the same plain lists and
    back, so the geometry functions can be called on real scene objects without
    ever exposing the raw points.

Example (plain maths)::

    from manim_extras import GeometryOperations

    pt = GeometryOperations.intersection_point([[0, 0], [2, 2]], [[0, 2], [2, 0]])
    pt == [1.0, 1.0]

Example (inside a scene)::

    from manim_extras import ManimGeometryAdapter

    dot = Dot(ManimGeometryAdapter.intersection_point(l1, l2), color=RED)
"""

from __future__ import annotations

from typing import Literal, TypeAlias

import numpy as np
from manim import Mobject

__all__ = ["GeometryOperations", "ManimGeometryAdapter"]

Point2D: TypeAlias = list[float]
Line2D: TypeAlias = list[Point2D]
Point3D: TypeAlias = np.ndarray
Line3D: TypeAlias = list[Point3D]


class GeometryOperations:
    """2D line/segment geometry on plain ``[[x, y], [x, y]]`` point lists.

    Every method is a ``@staticmethod`` and takes only lists/numbers, so the
    whole class is usable without Manim installed.
    """

    @staticmethod
    def displacement(point1: Point2D, point2: Point2D) -> Point2D:
        """``point2 - point1`` as a ``[dx, dy]`` pair."""
        x1, y1 = point1
        x2, y2 = point2
        return [x2 - x1, y2 - y1]

    @staticmethod
    def get_proportional_value(line1: Line2D, line2: Line2D, proportion_to: Literal[1, 2]) -> float:
        """The ``t`` (or ``u``) parameter of the intersection.

        Solves ``A + t * AB = C + u * CD``. ``proportion_to`` selects which of
        ``t`` (``1``) or ``u`` (``2``) is returned. Raises ``ValueError`` when
        the lines are parallel or coincident.
        """
        A, B = line1
        C, D = line2
        AB = GeometryOperations.displacement(A, B)
        CD = GeometryOperations.displacement(C, D)
        AC = GeometryOperations.displacement(A, C)
        denominator = np.linalg.det(np.array([AB, CD]))
        if np.isclose(denominator, 0):
            raise ValueError("Lines are parallel or coincident.")
        if proportion_to == 1:
            numerator = np.linalg.det(np.array([AC, CD]))
        elif proportion_to == 2:
            numerator = np.linalg.det(np.array([AC, AB]))
        else:
            raise ValueError("proportion_to must be 1 or 2.")
        return numerator / denominator

    @staticmethod
    def get_point_by_proportion(line: Line2D, proportional_t: float) -> Point2D:
        """``A + t * (B - A)`` -- a point ``t`` of the way along ``line``."""
        A, B = line
        return [
            float(A[0] + proportional_t * (B[0] - A[0])),
            float(A[1] + proportional_t * (B[1] - A[1])),
        ]

    @staticmethod
    def lines_intersect(line1: Line2D, line2: Line2D) -> bool:
        """Whether the (infinite) lines are not parallel/coincident."""
        GeometryOperations.get_proportional_value(line1, line2, 1)
        return True

    @staticmethod
    def segments_intersect(line1: Line2D, line2: Line2D) -> bool:
        """Whether the two bounded segments intersect."""
        t = GeometryOperations.get_proportional_value(line1, line2, 1)
        u = GeometryOperations.get_proportional_value(line1, line2, 2)
        return 0 <= t <= 1 and 0 <= u <= 1

    @staticmethod
    def intersection_point(line1: Line2D, line2: Line2D) -> Point2D:
        """The intersection point of two (infinite) lines."""
        t = GeometryOperations.get_proportional_value(line1, line2, 1)
        return GeometryOperations.get_point_by_proportion(line1, proportional_t=t)

    @staticmethod
    def segment_intersection_point(line1: Line2D, line2: Line2D) -> Point2D | None:
        """The intersection point, or ``None`` if the segments don't meet."""
        if not GeometryOperations.segments_intersect(line1, line2):
            return None
        return GeometryOperations.intersection_point(line1, line2)


class ManimGeometryAdapter:
    """Bridge between Manim ``Mobject``\\ s and the plain 2D point lists used by
    ``GeometryOperations``.

    Manim points are 3D ``np.ndarray`` s; these helpers drop the ``z`` and turn
    the result back into a list (or a full 3D array) as needed.
    """

    @staticmethod
    def convert_line_3d_to_2d(line: Mobject) -> Line2D:
        """Start/end points of a Manim line as a 2D ``Line2D``."""
        return [line.get_start()[:2].tolist(), line.get_end()[:2].tolist()]

    @staticmethod
    def convert_point_3d_to_2d(point: Mobject) -> Point2D:
        """A Mobject's centred point as a ``[x, y]`` list."""
        return point.get_center()[:2].tolist()

    @staticmethod
    def convert_point_2d_to_3d(point: Point2D) -> Point3D:
        """A ``[x, y]`` list back to a 3D ``np.ndarray`` at ``z = 0``."""
        return np.array([point[0], point[1], 0.0])

    @staticmethod
    def convert_line_2d_to_3d(line: Line2D) -> Line3D:
        """A 2D ``Line2D`` to a 3D point-pair."""
        return [
            ManimGeometryAdapter.convert_point_2d_to_3d(line[0]),
            ManimGeometryAdapter.convert_point_2d_to_3d(line[1]),
        ]

    @staticmethod
    def displacement_2d(point1: Mobject, point2: Mobject) -> Point2D:
        """Displacement between two Manim mobjects, in 2D."""
        point1 = ManimGeometryAdapter.convert_point_3d_to_2d(point1)
        point2 = ManimGeometryAdapter.convert_point_3d_to_2d(point2)
        return GeometryOperations.displacement(point1, point2)

    @staticmethod
    def displacement_3d(point1: Mobject, point2: Mobject) -> Point3D:
        """Displacement between two Manim mobjects, as a 3D array."""
        displacement = ManimGeometryAdapter.displacement_2d(point1, point2)
        return ManimGeometryAdapter.convert_point_2d_to_3d(displacement)

    @staticmethod
    def get_proportional_value(
        line1: Mobject, line2: Mobject, proportion_to: Literal[1, 2]
    ) -> float:
        """``GeometryOperations.get_proportional_value`` on Manim lines."""
        line1 = ManimGeometryAdapter.convert_line_3d_to_2d(line1)
        line2 = ManimGeometryAdapter.convert_line_3d_to_2d(line2)
        return GeometryOperations.get_proportional_value(line1, line2, proportion_to)

    @staticmethod
    def get_point_by_proportion(line: Mobject, proportional_t: float) -> Point3D:
        """A point ``t`` of the way along a Manim line, as a 3D array."""
        line = ManimGeometryAdapter.convert_line_3d_to_2d(line)
        point = GeometryOperations.get_point_by_proportion(line, proportional_t)
        return ManimGeometryAdapter.convert_point_2d_to_3d(point)

    @staticmethod
    def lines_intersect(line1: Mobject, line2: Mobject) -> bool:
        """Whether two Manim lines are not parallel/coincident."""
        line1 = ManimGeometryAdapter.convert_line_3d_to_2d(line1)
        line2 = ManimGeometryAdapter.convert_line_3d_to_2d(line2)
        return GeometryOperations.lines_intersect(line1, line2)

    @staticmethod
    def segments_intersect(line1: Mobject, line2: Mobject) -> bool:
        """Whether two Manim segments intersect."""
        line1 = ManimGeometryAdapter.convert_line_3d_to_2d(line1)
        line2 = ManimGeometryAdapter.convert_line_3d_to_2d(line2)
        return GeometryOperations.segments_intersect(line1, line2)

    @staticmethod
    def intersection_point(line1: Mobject, line2: Mobject) -> Point3D:
        """Intersection point of two Manim lines, as a 3D array."""
        t = ManimGeometryAdapter.get_proportional_value(line1, line2, 1)
        return ManimGeometryAdapter.get_point_by_proportion(line1, proportional_t=t)

    @staticmethod
    def segment_intersection_point(line1: Mobject, line2: Mobject) -> Point3D | None:
        """Intersection point of two Manim segments, or ``None``."""
        if not ManimGeometryAdapter.segments_intersect(line1, line2):
            return None
        return ManimGeometryAdapter.intersection_point(line1, line2)
