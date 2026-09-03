# `GeometryOperations` and `ManimGeometryAdapter`

[← back to README](../../README.md)

2D line-intersection helpers. The package keeps geometry separate from
rendering, so there are two layers:

| Class | What it is |
| --- | --- |
| `GeometryOperations` | Pure 2D maths on plain `[[x, y], [x, y]]` lists. No Manim needed. |
| `ManimGeometryAdapter` | Converts Manim `Mobject`s to those lists and back, so the maths works on scene objects. |

```python
from manim import *

from manim_extras import GeometryOperations, ManimGeometryAdapter


class IntersectionScene(Scene):
    def construct(self):
        l1 = Line([0, 0, 0], [4, 4, 0])
        l2 = Line([0, 4, 0], [4, 0, 0])
        self.add(l1, l2)
        self.add(Dot(ManimGeometryAdapter.intersection_point(l1, l2), color=RED))
```

---

## `GeometryOperations` — the pure-math core

Takes a "line" as two points: `line = [[x1, y1], [x2, y2]]`. Every method is a
`@staticmethod`, and none of them touch Manim, so they work in any plain Python
script or test.

```python
from manim_extras import GeometryOperations

# the crossing diagonals of a square meet at [1, 1]
GeometryOperations.intersection_point([[0, 0], [2, 2]], [[0, 2], [2, 0]])   # -> [1.0, 1.0]

# were they bounded segments, do they touch?
GeometryOperations.segments_intersect([[0, 0], [2, 2]], [[0, 2], [2, 0]])  # -> True
```

| Method | Returns | Notes |
| --- | --- | --- |
| `displacement(p1, p2)` | `[dx, dy]` | `p2 - p1`. |
| `get_proportional_value(l1, l2, proportion_to)` | `float` | `t` (`1`) or `u` (`2`) of the intersection. Raises `ValueError` for parallel/coincident lines. |
| `get_point_by_proportion(line, t)` | `[x, y]` | `A + t·(B - A)`. |
| `lines_intersect(l1, l2)` | `bool` | Not parallel/coincident. |
| `segments_intersect(l1, l2)` | `bool` | Both `t` and `u` in `[0, 1]`. |
| `intersection_point(l1, l2)` | `[x, y]` | Point of the two infinite lines. |
| `segment_intersection_point(l1, l2)` | `[x, y] \| None` | `None` when the segments don't meet. |

Lines are stored as two points; the "segments" versions simply also require the
parameter to fall inside `[0, 1]`.

---

## `ManimGeometryAdapter` — the Manim bridge

Manim points are 3D `np.ndarray`s. These helpers read a `Mobject`'s endpoints
(or centre), drop the `z`, run the maths, and hand back a 3D array you can pass
to a `Dot` or `Line`.

```python
l1, l2 = Line([0, 0, 0], [2, 2, 0]), Line([0, 2, 0], [2, 0, 0])

pt = ManimGeometryAdapter.intersection_point(l1, l2)   # np.array([1., 1., 0.])
Dot(pt, color=RED)

ManimGeometryAdapter.lines_intersect(l1, l2)           # True
ManimGeometryAdapter.segments_intersect(l1, l2)        # True
```

| Method | Input | Returns | Notes |
| --- | --- | --- | --- |
| `convert_line_3d_to_2d(line)` | `Mobject` | `Line2D` | Start/end → 2D list. |
| `convert_point_3d_to_2d(point)` | `Mobject` | `[x, y]` | Centre → 2D list. |
| `convert_point_2d_to_3d(point)` | `[x, y]` | `np.ndarray` | `z = 0`. |
| `convert_line_2d_to_3d(line)` | `Line2D` | `Line3D` | Both endpoints to 3D. |
| `displacement_2d(p1, p2)` | `Mobject`s | `[dx, dy]` | |
| `displacement_3d(p1, p2)` | `Mobject`s | `np.ndarray` | |
| `get_proportional_value(l1, l2, proportion_to)` | `Mobject`s | `float` | |
| `get_point_by_proportion(line, t)` | `Mobject` | `np.ndarray` | |
| `lines_intersect(l1, l2)` | `Mobject`s | `bool` | |
| `segments_intersect(l1, l2)` | `Mobject`s | `bool` | |
| `intersection_point(l1, l2)` | `Mobject`s | `np.ndarray` | |
| `segment_intersection_point(l1, l2)` | `Mobject`s | `np.ndarray \| None` | |

Each `ManimGeometryAdapter` method is a thin wrapper over the matching
`GeometryOperations` call, so the two layers can never disagree.

---

## Errors

`get_proportional_value` (and everything that calls it) raises `ValueError`
when the determinant is (near) zero — i.e. the lines are parallel or
coincident. `np.isclose` is used, so nearly-parallel lines also raise rather
than returning a point millions of units away.

---

## Run the demos

```bash
manim -ql examples/utils/line_intersection_demo.py LineIntersectionDemo
manim -ql examples/utils/line_intersection_demo.py SegmentIntersectionDemo
```
