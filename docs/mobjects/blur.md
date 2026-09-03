# `Blur`, `IMGBlur`, `BlurCard`, `IMGBlurCard`, `CameraBlur`

[← back to README](../../README.md)

Frosted-glass layers. A blur layer is a real `VMobject` whose **interior**
blurs whatever has already been drawn beneath it.

```python
from manim import *
from manim_extras import Blur


class GlassDoor(Scene):
    def construct(self):
        rec = Rectangle(width=5, height=5, stroke_color=WHITE, stroke_width=4)
        glass = Blur(rec, 30)
        glass.set_z_index(1)
        self.add(glass)

        person = Circle(radius=0.6, fill_opacity=1, color=RED)
        person.move_to(LEFT * 6).set_z_index(0)
        self.add(person)

        self.play(person.animate.move_to(RIGHT * 6), run_time=3)
```

The person is drawn *below* the glass, so it goes past blurred.

## The one rule

Ordering is plain `z_index`: anything **below** the blur is blurred, anything
**above** stays sharp. Equal values fall back to insertion order, so this does
the obvious thing:

```python
self.add(background)   # blurred
self.add(Blur(25))     # <- the blur happens here
self.add(title)        # sharp
```

`Blur(rec)` only copies `rec`'s geometry and stroke. `rec` keeps its own
`z_index`, is never mutated, and the two are independent mobjects from then on.

## The five classes

| Class | Behaviour |
| --- | --- |
| `Blur(shape, amount)` | Re-blurs every frame — live frosted glass |
| `IMGBlur(shape, amount)` | Blurs once, then keeps that snapshot |
| `BlurCard(...)` | Floating glass card, live |
| `IMGBlurCard(...)` | The same card, static |
| `CameraBlur(amount)` | The whole camera frame; always live |

## `Blur` versus `IMGBlur`

This is the one distinction worth internalising. `Blur` re-reads the frame
buffer every frame; `IMGBlur` stores it the first time it renders and reuses it
forever. Send something past both and only the `Blur` notices.

```python
Blur(rec, 28)      # you see the shape drift past, blurred
IMGBlur(rec, 28)   # the pane has no idea anything moved
```

The stored image stays fixed relative to the screen — like sliding a window
across a painted wall. To carry it along with the pane instead:

```python
IMGBlur(rec, 28, frozen_content=True)
glass.recapture()   # grab a fresh snapshot on the next frame
```

`IMGBlur` also exposes `has_snapshot()`, which is occasionally handy in tests.

## Any outline works

The mask is built by replaying the shape's Bézier path into a Cairo
`FORMAT_A8` surface, so the edge comes out properly anti-aliased and shapes
with holes behave:

```python
Blur(Circle(radius=2), 28)
Blur(Star(7), 28)
Blur(RegularPolygon(6).scale(1.7), 28)
Blur(Annulus(inner_radius=1, outer_radius=2), 28)   # the hole stays sharp

blob = VMobject()
blob.set_points_as_corners([UP * 2, RIGHT * 3, DOWN * 2, LEFT * 3, UP * 2])
blob.make_smooth()
Blur(blob, 28)
```

Omit the shape entirely and the layer covers the whole frame: `Blur(25)`.

## Cards

Cards default to a rounded rectangle but accept any outline, and every colour
is a keyword argument:

```python
BlurCard(width=8, height=4.5)

BlurCard(Star(7), 40,
         tint=GREEN, tint_opacity=0.35,
         border_color=GREEN, border_width=5)

BlurCard(Star(7), match_size=True, width=6, height=3)   # stretch to fit
BlurCard(35)                                            # bare number = amount
```

Colours can be changed after construction with `set_tint(color, opacity)` and
`set_border(color=..., width=..., opacity=...)`, and both animate.

## Parameters

```python
Blur(
    shape,                # any VMobject; omit for the whole frame
    30,                   # blur strength (also c=30 or amount=30)
    intensity=0.8,        # 0-1 mix of blur over the original pixels
    feather=2,            # mask edge softness, in pixels
    tint=WHITE,           # colour wash over the blur
    tint_opacity=0.2,
    copy_style=True,      # copy the input shape's stroke
    quality="fast",       # "high" (scipy) or "fast" (downscaled)
    even_odd=True,        # keep holes open
)
```

Blur strength is normalised against a 1920px-wide frame, so `-ql` previews and
`-qh` renders look the same.

Card-only extras: `width`, `height`, `corner_radius`, `border_color`,
`border_width`, `border_opacity`, `match_size`.

> The input shape's **stroke** is copied but its **fill** is not — an opaque
> interior would simply hide the blur. Pass `fill_opacity=...` explicitly if
> you want a tinted fill.

## Presets

Available on both `Blur` and `IMGBlur`:

```python
Blur.glass()    # soft edge, faint white wash
Blur.dark()     # dark overlay, a good bed for light text
Blur.subtle()   # amount 8
Blur.heavy()    # amount 55

Blur.glass(30, shape=Star(7), tint=GREEN, tint_opacity=0.4)   # override anything
```

## Animation

`FadeIn` and `FadeOut` dissolve the blur itself, because opacity doubles as a
strength multiplier:

```python
self.play(glass.fade_in())          # 0 -> current amount
self.play(glass.to(55))
self.play(glass.fade_out())
self.play(FadeOut(glass))
self.play(glass.animate.shift(LEFT).set_blur(40))
self.play(Rotate(glass, PI / 2))
```

## Performance

```python
from manim_extras import blur_config

blur_config.fast()            # quick previews, 3-10x faster
blur_config.high()            # final renders
blur_config.enabled = False   # switch every blur off at once
blur_config.default_blur = 30
```

Only the shape's bounding box is processed, so small shapes are cheap. On a
1920x1080 frame the mask costs roughly 8 ms, plus the blur itself.

## How it works

`Camera.capture_mobjects` sorts mobjects by `z_index` and paints them one after
another into a single RGBA buffer. This module patches that method once, so it
applies to `Scene`, `MovingCameraScene`, `ThreeDScene` and anything else
deriving from `Camera`. When a blur layer's turn comes, six things happen:

1. the shape's Bézier path is rasterised into an anti-aliased mask
2. the bounding box is taken, plus 3σ of padding
3. source pixels are read — live buffer, or the stored snapshot
4. a Gaussian blur runs on premultiplied alpha
5. the result is masked back into place
6. the shape's own stroke is drawn on top

Because `capture_mobjects` runs per frame, stills and videos share one code
path — there is no separate video mode.

A few details that keep the output clean: premultiplied alpha stops transparent
pixels bleeding dark fringes into visible ones; the 3σ padding prevents a halo
at the region edge; the even-odd fill rule keeps holes open; and the mask is
cached while the shape stays put.

`scipy` is used for the separable Gaussian when available. Without it the
module falls back to Pillow and still works, just slightly softer.

## Run the demos

```bash
manim -ql examples/mobjects/blur_demo.py GlassDoorDemo
manim -ql examples/mobjects/blur_demo.py LiveVsStaticDemo
manim -ql examples/mobjects/blur_demo.py AnyShapeDemo
manim -ql examples/mobjects/blur_demo.py CardDemo
manim -ql examples/mobjects/blur_demo.py CameraBlurDemo
```
