# manim-extras

A collection of reusable [Manim](https://www.manim.community/) components —
custom Mobjects, animations, scenes and helpers — kept in one installable
package instead of being copy-pasted between projects.

## Install

```bash
git clone https://github.com/myabdur2121-cpu/manim-extras
cd manim-extras
```

Editable install: edits to the source take effect immediately, no reinstall.

```bash
pip install -e ".[dev]"   # plus pytest and ruff
```

## Usage

Everything public is re-exported at the top level, so imports stay short and
stable even if files move around internally:

```python
from manim_extras import SmoothPolygon
```

## What's inside

### `SmoothPolygon`

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

## Layout

```
src/manim_extras/
├── __init__.py       public API — re-export everything here
├── mobjects/         custom Mobjects
├── animations/       custom Animation classes
├── scenes/           reusable Scene base classes
└── utils/            helper functions
examples/             runnable demo scenes, one file per component
tests/                pytest suite (pure geometry, nothing is rendered)
```

## Adding a component

1. Drop the module in the matching sub-package, e.g.
   `src/manim_extras/mobjects/my_thing.py`.
2. Re-export it from that sub-package's `__init__.py`, then from
   `src/manim_extras/__init__.py`, so `from manim_extras import MyThing` works.
3. Add `tests/test_my_thing.py` — assert on geometry, don't render.
4. Add `examples/my_thing_demo.py` for the visual check.

## Development

```bash
pytest                                    # fast, renders nothing
ruff check . && ruff format .
manim -ql examples/smooth_polygon_demo.py SmoothPolygonDemo
```

## Requirements

Manim Community ≥ 0.18 (tested on 0.20.1), NumPy ≥ 1.22, Python ≥ 3.9, and
ffmpeg on `PATH` for rendering. No LaTeX needed.

## License

MIT — see [LICENSE](LICENSE).
