# manim-extras

A collection of reusable [Manim](https://www.manim.community/) components —
custom Mobjects, animations, scenes and helpers — kept in one installable
package instead of being copy-pasted between projects.

## Install

```bash
git clone https://github.com/myabdur2121-cpu/manim-extras
cd manim-extras
```
or , 
```bash 
!git clone https://github.com/myabdur2121-cpu/manim-extras.git
!pip install /content/manim-extras
```
here -e is not used 
------------+

Editable install: edits to the source take effect immediately, no reinstall.

```bash
pip install -e ".[dev]"   # plus pytest and ruff
```

## Usage

Everything public is re-exported at the top level, so imports stay short and
stable even if files move around internally:

```python
from manim_extras import Blur, SmoothPolygon
from manim_extras import GeometryOperations, ManimGeometryAdapter
```

## What's inside

Each component has its own page; this table is the index.

| Component | What it does |
| --- | --- |
| [`SmoothPolygon`](docs/mobjects/smooth_polygon.md) | A smooth closed curve through a list of vertices — without the kink at the start point. |
| [`Blur` and friends](docs/mobjects/blur.md) | Frosted-glass layers that blur whatever is drawn beneath them: `Blur`, `IMGBlur`, `BlurCard`, `IMGBlurCard`, `CameraBlur`. |
| [`StreamAlongPath` and `ParticleStream`](docs/animations/stream_along_path.md) | Streams of particles flowing along any path: density, path window, orientation, plus colour, jitter and a variable emission rate. |
| [`GlowDot` and friends](docs/mobjects/glow_dot.md) | 3Blue1Brown's glowing dots for Manim Community: `GlowDot`, `GlowDots`, `TrueDot`, `DotCloud`, with a custom falloff and a hot core. |
| [`GeometryOperations` and `ManimGeometryAdapter`](docs/utils/line_intersection.md) | 2D line-intersection helpers: a pure-math core, plus a bridge to Manim `Mobject`s. |

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
docs/                 one page per component
```

## Adding a component

1. Drop the module in the matching sub-package, e.g.
   `src/manim_extras/mobjects/my_thing.py`.
2. Re-export it from that sub-package's `__init__.py`, then from
   `src/manim_extras/__init__.py`, so `from manim_extras import MyThing` works.
3. Add `tests/<sub-package>/test_my_thing.py` — assert on geometry, don't render.
4. Add `examples/<sub-package>/my_thing_demo.py` for the visual check.
5. Add `docs/<sub-package>/my_thing.md` and link it from the table above — keep
   the README itself short.

## Development

```bash
pytest                                    # fast, renders nothing
ruff check . && ruff format .
manim -ql examples/mobjects/smooth_polygon_demo.py SmoothPolygonDemo
```

## Requirements

Manim Community ≥ 0.18 (tested on 0.20.1), NumPy ≥ 1.22, SciPy ≥ 1.8, Python ≥
3.9, and ffmpeg on `PATH` for rendering. No LaTeX needed.

SciPy is only used for the blur layers' separable Gaussian; without it they
fall back to Pillow and still work, just slightly softer.

## License

MIT — see [LICENSE](LICENSE).
