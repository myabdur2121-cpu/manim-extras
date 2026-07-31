# `ParticleStream`

[← back to README](../../README.md) · [← the base class](stream_along_path.md)

[`StreamAlongPath`](stream_along_path.md) with **colour**, **randomness** and a
**variable emission rate**.

```python
from manim import *

from manim_extras import ParticleStream


class AnimationScene1(Scene):
    def construct(self):
        path = Circle(radius=2)
        self.play(
            ParticleStream(
                Dot(), path,
                run_time=8,
                travel_time=4,
                path_colors=[BLUE, TEAL, GREEN],
                size_jitter=0.4,
                emission_rate_func=lambda t: 0.2 + t,
            )
        )
```

`ParticleStream` is a subclass, so **every parameter on the base-class page
works here too**: density, path window, spawn style, orientation, all of it.
Everything below is off by default, which means a `ParticleStream` with no extra
arguments behaves exactly like a `StreamAlongPath`.

---

## Colour

Two different ideas, so two parameters:

```python
# a gradient along the journey -- one particle changes colour as it travels
ParticleStream(Dot(), path, path_colors=[BLUE, YELLOW, RED])

# confetti -- each particle keeps one colour, cycling through the list
ParticleStream(Dot(), path, particle_colors=[RED, GREEN, BLUE])
```

`path_colors` interpolates through the list as a function of **position**:
a particle starts blue, passes through yellow and dies red.

`particle_colors` assigns **per particle** and never changes during the journey,
which is what you want for variety rather than a positional gradient.

If both are given, `path_colors` wins and a warning is issued.

---

## Randomness

A perfectly even stream reads as mechanical. These four knobs break that up, and
they are independent — mix as needed.

```python
ParticleStream(
    Dot(), path,
    position_jitter=0.15,   # scene units of random offset per particle
    size_jitter=0.3,        # each particle is 70%-130% of the original size
    birth_jitter=0.5,       # nudges *when* each particle is born
    wobble_amplitude=0.2,   # smooth side-to-side drift while travelling
    wobble_frequency=2.5,   # oscillations per trip
    seed=7,                 # same seed -> identical render, every time
)
```

| Parameter | What it randomises |
| --- | --- |
| `position_jitter` | Where the particle sits, in scene units. |
| `size_jitter` | How big it is, 0–1 as a fraction of its size. |
| `birth_jitter` | When it is born, 0–1 as a fraction of the spawn interval. |
| `wobble_amplitude` | A smooth sideways drift, not a one-off offset. |

`birth_jitter` is the subtle one: it attacks the *rhythm* rather than the
positions, which is often what makes a stream look artificial. The timeline
stays anchored, so the stream still fills `run_time` exactly.

`wobble_amplitude` differs from `position_jitter` — the offset is frozen per
particle, while the wobble is a continuous sine drift perpendicular to the
direction of travel, so particles appear to swim.

### `seed`

Without a seed, every render is different. With one, the same stream comes back
every time — worth setting once you like what you see, so a re-render doesn't
change the shot.

---

## Emission rate

`emission_rate_func` controls **when** particles are born, spread across the
animation. It receives `0 → 1` (progress through the emission window) and
returns a relative rate — only the shape matters, not the units.

```python
ParticleStream(Dot(), path, particle_count=30, emission_rate_func=lambda t: 1 - t)
```

| Curve | Effect |
| --- | --- |
| `None` *(default)* | Steady, evenly spaced births. |
| `lambda t: t` | Starts sparse, builds to a torrent. |
| `lambda t: 1 - t` | Bursts at the start, then thins out. |
| `lambda t: 0.1 + t**3` | A long calm, then a sudden surge. |
| `lambda t: abs(np.sin(3 * t)) + 0.05` | Pulses. |

**The particle count never changes** — only the timing of the births does. With
`particle_count=30` you always get 30 particles; a rising rate just packs more
of them into the later part of the animation.

Rates are clamped to be non-negative. A curve that is zero everywhere falls back
to a steady rate and warns, rather than producing an empty animation.

```python
stream = ParticleStream(Dot(), path, particle_count=30, emission_rate_func=lambda t: t)
stream.birth_times   # inspect exactly when each one is born
```

---

## Everything together

```python
ParticleStream(
    Triangle().scale(0.15), path,
    run_time=12,
    travel_time=5,
    particle_count=44,
    # from the base class
    face_point=ORIGIN,
    face_away=True,
    reference_direction=UP,
    taper_func=smooth,
    # from this class
    path_colors=[BLUE, TEAL, GREEN, YELLOW, RED],
    position_jitter=0.1,
    size_jitter=0.45,
    birth_jitter=0.5,
    wobble_amplitude=0.09,
    emission_rate_func=lambda t: 0.25 + t,
    seed=11,
)
```

---

## Parameters

Base-class parameters are listed on
[the `StreamAlongPath` page](stream_along_path.md#parameters). These are the
additions:

| Parameter | Default | Meaning |
| --- | --- | --- |
| `path_colors` | `None` | Gradient applied along the journey. |
| `particle_colors` | `None` | One fixed colour per particle, cycling. |
| `position_jitter` | `0.0` | Random offset in scene units. |
| `size_jitter` | `0.0` | Random size variation, 0–1. |
| `birth_jitter` | `0.0` | Random nudge to birth times, 0–1. |
| `wobble_amplitude` | `0.0` | Sideways drift in scene units. |
| `wobble_frequency` | `1.0` | Wobble oscillations per trip. |
| `seed` | `None` | Fix the randomness for a reproducible render. |
| `emission_rate_func` | `None` | Curve shaping when particles are born. |

---

## Demos

```bash
manim -ql examples/animations/stream_along_path_demo.py ParticleStreamDemo
manim -ql examples/animations/stream_along_path_demo.py EmissionRateDemo
manim -ql examples/animations/stream_along_path_demo.py AnyPathDemo
```
