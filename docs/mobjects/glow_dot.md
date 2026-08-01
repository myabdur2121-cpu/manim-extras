# `GlowDot` and friends

[← back to README](../../README.md)

Glowing dots for Manim Community, ported from 3Blue1Brown's ManimGL:
`GlowDot`, `GlowDots`, `TrueDot` and the `DotCloud` they are built on.

```python
from manim import *

from manim_extras import GlowDot


class AnimationScene1(Scene):
    def construct(self):
        self.add(GlowDot())
        self.add(GlowDot(LEFT * 2, color=BLUE, radius=0.6))
```

---

## Where the falloff comes from

ManimGL draws glow dots on the GPU. Its `true_dot` fragment shader is
essentially two lines:

```glsl
float r = length(uv_coords.xy);
if (r > 1.0) discard;
if (glow_factor > 0) frag_color.a *= pow(1 - r, glow_factor);
```

So a dot's alpha is **`(1 - r) ** glow_factor`** inside the unit disc and zero
outside it, where `r` is 0 at the centre and 1 at the rim:

| `r` | `glow_factor=0.5` | `glow_factor=2` | `glow_factor=4` |
| --- | --- | --- | --- |
| 0.00 | 1.000 | 1.000 | 1.000 |
| 0.25 | 0.866 | 0.563 | 0.316 |
| 0.50 | 0.707 | 0.250 | 0.063 |
| 1.00 | 0 | 0 | 0 |

Manim Community's default renderer is Cairo, which has no shaders, so the same
formula is evaluated on the CPU instead. `test_matches_manimgl_shader` checks
the two agree exactly.

---

## The four classes

| Class | What it is |
| --- | --- |
| `GlowDot(center, ...)` | One glowing dot. Yellow, `radius=0.2`, `glow_factor=2`. |
| `GlowDots(points, ...)` | Many glowing dots in a single mobject. |
| `TrueDot(center, ...)` | A crisp dot: `glow_factor=0`, `radius=0.05`. |
| `DotCloud(points, ...)` | The base class. Grey, `glow_factor=0`. |

```python
GlowDot()                                    # a single dot at the origin
GlowDots(np.array([[-2, 0, 0], [2, 0, 0]]))  # two dots, one object
TrueDot(radius=0.1, color=WHITE)             # no glow at all
```

---

## Sharpness, size and colour

```python
GlowDot(glow_factor=0)     # a solid disc
GlowDot(glow_factor=2)     # the default
GlowDot(glow_factor=8)     # a tight, bright core

GlowDot(radius=1.2)
GlowDot(color=BLUE, opacity=0.6)
```

`glow_factor` controls the *shape* of the light and `radius` its *size*; the
two are independent. `opacity` scales the whole profile, so the centre never
exceeds it.

A cloud can take one colour per dot, cycling if the list is shorter:

```python
GlowDots(points, color=[RED, GREEN, BLUE])
```

---

## A hot centre

Not in ManimGL, and off by default. `core_color` blends from the centre out to
`color` at the rim, which is what makes a light look hot rather than tinted:

```python
GlowDot(color=BLUE, core_color=WHITE)               # white-hot middle
GlowDot(color=RED, core_color=YELLOW, core_size=0.4)
```

`core_size` is how far the core reaches, 0–1 of the radius.

---

## Your own light distribution

`falloff` replaces the `(1 - r) ** glow_factor` formula entirely. It receives
`r` (0 at the centre, 1 at the rim) and the current `glow_factor`, and returns
the alpha:

```python
GlowDot(falloff=lambda r, gf: np.exp(-5 * r**2))          # gaussian
GlowDot(falloff=lambda r, gf: 1 - r**2)                    # parabolic
GlowDot(falloff=lambda r, gf: np.clip(1 - r, 0, 1) ** 0.5) # broad halo
GlowDot(falloff=lambda r, gf: np.clip(1 - r, 0, 1) ** 2
        * (0.55 + 0.45 * np.cos(9 * np.pi * r)))           # rings
```

Two guarantees hold whatever you pass: the result is clipped to `[0, 1]`, and
anything past `r = 1` is still discarded, so the dot stays inside its radius.

The falloff depends only on `r`, so the glow is always radially symmetric.
An elliptical or star-shaped glow would need the raw `(x, y)`, which is not
exposed.

---

## Two backends

```python
GlowDot(render_mode="raster")   # default
GlowDot(render_mode="vector")
```

| | `"raster"` | `"vector"` |
| --- | --- | --- |
| How | one RGBA image, alpha per pixel | stacked filled circles |
| Accuracy | matches the shader, including overlap | quantised into `num_layers` steps |
| Zooming | can pixelate | stays sharp, exports to SVG |
| Tuning | `canvas_size` | `num_layers` |

Both evaluate the same `falloff`, so they agree to within a step of the
quantisation. In vector mode the layer alphas are *solved* rather than set
directly: painting each circle with its target alpha would accumulate towards
1 in the middle and produce a solid disc instead of a glow.

---

## The ManimGL API

Every method from `manimlib`'s `DotCloud` is present:

```python
cloud = GlowDots(points)

cloud.set_radius(0.4)          # one radius for all
cloud.set_radii([0.1, 0.3])    # one per dot
cloud.get_radius()             # the largest, as ManimGL does
cloud.get_radii()
cloud.scale_radii(2)

cloud.set_glow_factor(4)
cloud.get_glow_factor()

cloud.set_points(new_points)
cloud.get_points()

cloud.to_grid(5, 9, height=4)  # arrange in a grid
cloud.make_3d()                # recorded, but Cairo has no lighting
```

`scale` scales the radii too, matching ManimGL:

```python
dot.scale(2)                    # radius 0.2 -> 0.4
dot.scale(2, scale_radii=False) # radius stays 0.2
```

> `to_grid` sizes the grid by the spread of the dot *centres*, not the outer
> edge of the glow, which is what ManimGL measures.

---

## Animation

Glow dots interpolate like any other mobject:

```python
self.play(FadeIn(dot, scale=0.5))
self.play(dot.animate.shift(RIGHT * 3))
self.play(dot.animate.set_color(RED))
```

Updaters work too, which is how a cloud is driven by an equation:

```python
cloud.add_updater(lambda m: m.set_points(orbit(tracker.get_value())))
```

> **Why the canvas is a fixed size.** In raster mode the RGBA canvas is always
> `canvas_size` on its longest side, independent of the radius, and the image
> is stretched afterwards. Manim's `ImageMobject.interpolate_color` asserts
> that two pixel arrays share a shape, so a canvas that grew with the radius
> would make `FadeIn(scale=...)` and `Transform` fail with a shape assertion.
> Keeping it fixed also stops a large dot from allocating a huge array.

---

## Parameters

| Parameter | Default | Meaning |
| --- | --- | --- |
| `points` | one dot at the origin | Dot centres, shape `(n, 2)` or `(n, 3)`. |
| `color` | `GREY_C` | One colour, or one per dot. |
| `opacity` | `1.0` | Peak alpha at the centre. |
| `radius` | `0.05` | Dot radius in scene units. |
| `glow_factor` | `0.0` | Falloff sharpness. `0` is a solid disc. |
| `render_mode` | `"raster"` | `"raster"` or `"vector"`. |
| `falloff` | `None` | `f(r, glow_factor) -> alpha`, replacing the formula. |
| `core_color` | `None` | Colour at the very centre. |
| `core_size` | `0.35` | How far the core reaches, 0–1. |
| `canvas_size` | `512` | Raster resolution on the longest side. |
| `num_layers` | `60` | Circles per dot in vector mode. |
| `anti_alias_width` | `2.0` | Edge fade in pixels, as in ManimGL. |

`GlowDot` and `GlowDots` override the defaults to `color=YELLOW`,
`radius=0.2`, `glow_factor=2.0`; `TrueDot` uses `radius=0.05` and no glow.

---

## Errors

Invalid input raises rather than producing a silently wrong image: an unknown
`render_mode`, a negative `radius` or `glow_factor`, an `opacity` or
`core_size` outside 0–1, a `canvas_size` below one step, `num_layers` under 1,
an empty colour list, or points that are not `(n, 2)` or `(n, 3)`.

---

## Demos

```bash
manim -ql examples/mobjects/glow_dot_demo.py GlowDotDemo
manim -ql examples/mobjects/glow_dot_demo.py GlowFactorDemo
manim -ql examples/mobjects/glow_dot_demo.py CoreColorDemo
manim -ql examples/mobjects/glow_dot_demo.py FalloffDemo
manim -ql examples/mobjects/glow_dot_demo.py RenderModeDemo
manim -ql examples/mobjects/glow_dot_demo.py DotCloudDemo
manim -ql examples/mobjects/glow_dot_demo.py StarfieldDemo
manim -ql examples/mobjects/glow_dot_demo.py OrbitDemo
```
