# `StreamAlongPath`

[← back to README](../../README.md) · [colour, randomness and emission rate →](particle_stream.md)

A continuous stream of copies of a mobject flowing along any path.

```python
from manim import *

from manim_extras import StreamAlongPath


class AnimationScene1(Scene):
    def construct(self):
        path = Circle(radius=2)
        self.play(StreamAlongPath(Dot(), path, run_time=6, travel_time=3))
```

One mobject is used as a template. Copies of it are born at one end of the
path, travel along it, and die at the other end — so the stream keeps flowing
for as long as you ask.

For colour gradients, jitter and a variable emission rate, see
[`ParticleStream`](particle_stream.md), which is a subclass and accepts
everything on this page.

---

## Density

Two ways to decide how many particles the stream contains. They form a clear
hierarchy: **an explicit count always wins.**

```python
# 1. relaxed -- spacing as a fraction of travel_time, count follows from timing
StreamAlongPath(Dot(), path, run_time=10, travel_time=5, spawn_lag_ratio=0.04)

# 2. exact -- overrides the lag ratio entirely
StreamAlongPath(Dot(), path, run_time=10, travel_time=5, particle_count=26)
```

Smaller `spawn_lag_ratio` means a denser stream. After construction you can
always read back what actually happened:

```python
stream = StreamAlongPath(Dot(), path, run_time=10, travel_time=5)
stream.particle_count   # how many copies exist
stream.spawn_interval   # mean seconds between two births
stream.count_source     # "spawn_lag_ratio" or "particle_count"
stream.birth_times      # birth time of every particle
```

`run_time` is how long the whole effect plays; `travel_time` is how long **one**
particle takes to cross the path. A single particle cannot fill a longer
`run_time`, so if the numbers only allow one, the stream shortens itself to
`travel_time` and warns you.

---

## Which part of the path

`start_proportion` and `end_proportion` are positions along the path measured
as a fraction of its length — the same convention as Manim's
`path.point_from_proportion`. They default to the whole path.

```python
StreamAlongPath(Dot(), path, start_proportion=0.25, end_proportion=0.75)  # middle half
StreamAlongPath(Dot(), path, start_proportion=0.9, end_proportion=0.1)    # backwards
```

Setting `start_proportion` **greater** than `end_proportion` simply reverses the
flow; there is no separate "reverse" flag.

---

## Birth and death

`spawn_style` decides how a particle appears and disappears:

| value | size | opacity |
| --- | --- | --- |
| `"scale_fade"` *(default)* | grows and shrinks | fades in and out |
| `"scale"` | grows and shrinks | constant |
| `"fade"` | constant | fades in and out |
| `"none"` | pops in and out | pops in and out |

```python
StreamAlongPath(
    Dot(), path,
    spawn_style="scale_fade",
    spawn_duration=0.7,     # seconds spent growing in
    despawn_duration=0.7,   # seconds spent shrinking out
    taper_func=smooth,      # shape of that ramp; linear by default
    min_scale=0.3,          # never shrink below 30% -- 0.0 vanishes completely
)
```

If `spawn_duration + despawn_duration` exceeds `travel_time` the two are scaled
down proportionally, so a particle can never be born and die at the same time.

`particle_rate_func` is separate: it eases each particle's journey **along the
path**, while `taper_func` only shapes the grow/shrink ramp.

---

## Which way particles face

Three options, checked in this order — **the first one you set wins**:

1. `face_point` — keep looking at a target
2. `face_direction` — keep one fixed heading
3. `align_to_path` — follow the path's tangent

```python
StreamAlongPath(arrow, path, align_to_path=True, reference_direction=UP)
StreamAlongPath(arrow, path, face_direction=UP, reference_direction=UP)
StreamAlongPath(arrow, path, face_point=ORIGIN, reference_direction=UP)
StreamAlongPath(arrow, path, face_point=ORIGIN, face_away=True, reference_direction=UP)
```

`face_direction` accepts a vector (`UP`, `[1, 0.5, 0]`) **or** an angle in
radians (`PI / 4`).

`face_point` accepts a point (`ORIGIN`, `[2, 1, 0]`) **or a Mobject**. Passing a
Mobject reads its centre every frame, so the particles keep tracking it even
while it moves:

```python
target = Dot().move_to(LEFT * 3)
self.play(
    StreamAlongPath(arrow, path, face_point=target, reference_direction=UP),
    target.animate.move_to(RIGHT * 3),
)
```

### `reference_direction`

This is the one that trips people up. It declares which way your mobject
**already points** when it is unrotated, so the class knows how much to turn it.

Manim's convention is `RIGHT`, which is the default. But a `Triangle()` points
**up**, so it needs:

```python
StreamAlongPath(Triangle(), path, align_to_path=True, reference_direction=UP)
```

Use this instead of pre-rotating the mobject yourself.

---

## Any path works

Anything with a `point_from_proportion` method — `Line`, `Circle`, `Square`,
`Star`, `FunctionGraph`, `ParametricFunction`, even the outline of a letter:

```python
word = Text("manim")
self.play(*[StreamAlongPath(Dot(radius=0.05), letter) for letter in word])
```

> **Performance note.** `point_from_proportion` walks the path's points on every
> frame for every particle, so the cost scales with how finely the path is
> sampled. A default `ParametricFunction` spiral has ~7500 points and will crawl.
> Pass a coarser sampling step — `t_range=[0, 6 * PI, 0.2]` — for a large speed-up
> at no visible cost. Text outlines are heavy for the same reason.

---

## Parameters

| Parameter | Default | Meaning |
| --- | --- | --- |
| `particle` | — | The mobject that gets copied. |
| `path` | — | The curve to follow. |
| `run_time` | `5.0` | How long the whole stream plays, in seconds. |
| `travel_time` | `2.0` | How long **one** particle takes to cross, in seconds. |
| `start_proportion` | `0.0` | Where a particle is born, as a 0–1 proportion of the path. |
| `end_proportion` | `1.0` | Where it dies. Less than `start_proportion` flows backwards. |
| `spawn_lag_ratio` | `0.05` | Spacing between births, as a fraction of `travel_time`. |
| `particle_count` | `None` | Exact count. Overrides `spawn_lag_ratio`. |
| `spawn_style` | `"scale_fade"` | `"scale_fade"`, `"scale"`, `"fade"` or `"none"`. |
| `spawn_duration` | `0.5` | Seconds spent being born. |
| `despawn_duration` | `0.5` | Seconds spent dying. |
| `taper_func` | `linear` | Shape of the birth/death ramp. |
| `min_scale` | `0.0` | Smallest size at birth and death, 0–1. |
| `align_to_path` | `False` | Rotate to follow the tangent. |
| `face_direction` | `None` | Fixed heading: a vector or an angle in radians. |
| `face_point` | `None` | Aim at a point or a Mobject, updated every frame. |
| `face_away` | `False` | Invert `face_point`. |
| `reference_direction` | `RIGHT` | Which way the mobject already points. |
| `particle_rate_func` | `linear` | Easing of each particle's journey. |
| `max_particles` | `400` | Safety cap on how many copies may be created. |

### Read-back attributes

| Attribute | Meaning |
| --- | --- |
| `particle_count` | How many copies were created. |
| `count_source` | Which parameter decided the count. |
| `spawn_interval` | Mean seconds between births. |
| `birth_times` | Birth time of every particle. |

---

## Errors and warnings

Invalid input raises immediately rather than producing a silently wrong
animation — an out-of-range proportion, a zero-length direction vector, a
negative duration, `start_proportion == end_proportion`, and so on.

Warnings are used where the animation can still run but the result is not quite
what was asked for: a single particle that cannot fill `run_time`, a particle
count hitting `max_particles`, more than one orientation option set at once, or
`face_away` without `face_point`.

---

## Demos

```bash
manim -ql examples/animations/stream_along_path_demo.py StreamAlongPathDemo
manim -ql examples/animations/stream_along_path_demo.py DensityDemo
manim -ql examples/animations/stream_along_path_demo.py PathWindowDemo
manim -ql examples/animations/stream_along_path_demo.py SpawnStyleDemo
manim -ql examples/animations/stream_along_path_demo.py OrientationDemo
manim -ql examples/animations/stream_along_path_demo.py FacePointDemo
manim -ql examples/animations/stream_along_path_demo.py AnyPathDemo
```
