# `SmoothPolygon`

[← back to README](../../README.md)

A smooth closed curve through a list of vertices — **without the kink at the
start point**.

```python
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
```

```python
SmoothPolygon(points, color=BLUE, fill_opacity=0.25, stroke_width=6)
SmoothPolygon(points, is_closed=False)   # open smooth spline
```

The common workaround for this problem pads the vertex list and slices the
result:

```python
self.vertices = [v[-1], *v, v[0], v[1]]
self.set_points_smoothly(self.vertices)
self.set_points(self.points[4:-8])
```

`set_points_smoothly` solves an *open* spline with natural boundary conditions,
so padding only approximates periodicity — the tangent arriving at the first
vertex never quite matches the one leaving it. That mismatch is the visible
kink. `SmoothPolygon` uses Manim's periodic solver,
`get_smooth_closed_cubic_bezier_handle_points`, which solves the cyclic
tridiagonal system: C² continuous all the way around, seam included. No
padding, no magic slice indices, no kink.

Vertices may be `np.ndarray`, `list` or `tuple`; Manim constants like `UR`
work, and 2D points `(x, y)` are padded to `z = 0`. Methods: `set_vertices()`,
`get_vertices()`, `add_vertex()`.

## Reference

| Signature | Notes |
| --- | --- |
| `SmoothPolygon(points, is_closed=True, **style)` | `points` may be `np.ndarray`, `list` or `tuple`; 2D points are padded to `z = 0` |
| `set_vertices(points)` | Replace every vertex and rebuild the curve |
| `get_vertices()` | Return a copy of the vertex array |
| `add_vertex(point, index=None)` | Append, or insert at `index` |

Standard `VMobject` styling applies: `color`, `fill_opacity`, `stroke_width`
and friends all work as usual.

## Run the demo

```bash
manim -ql examples/mobjects/smooth_polygon_demo.py SmoothPolygonDemo
```
