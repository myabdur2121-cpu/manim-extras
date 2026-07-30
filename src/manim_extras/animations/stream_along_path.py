"""Streams of particles flowing along a path.

Two classes live here:

* :class:`StreamAlongPath` -- the core stream. Controls density, which slice of
  the path is used, how particles are born and die, and which way they face.
* :class:`ParticleStream` -- a subclass adding colour, randomness and a
  variable emission rate.

``ParticleStream`` inherits every feature of ``StreamAlongPath`` and extends it
through hooks, so there is a single implementation of the shared behaviour.

See ``docs/animations/stream_along_path.md`` and
``docs/animations/particle_stream.md`` for the full guides.
"""

from __future__ import annotations

import warnings
from collections.abc import Callable, Sequence

import numpy as np
from manim import (
    OUT,
    RIGHT,
    AnimationGroup,
    Mobject,
    UpdateFromAlphaFunc,
    VMobject,
    angle_of_vector,
    color_gradient,
    linear,
    normalize,
)

__all__ = [
    "DEFAULT_MAX_PARTICLES",
    "SPAWN_STYLES",
    "ParticleStream",
    "StreamAlongPath",
    "as_direction_vector",
    "as_target_point",
    "clamp",
]

#: Accepted values for ``spawn_style``.
SPAWN_STYLES = ("scale_fade", "scale", "fade", "none")

#: Default ceiling on how many particle copies may be created at once.
DEFAULT_MAX_PARTICLES = 400

# Scaling by exactly 0 collapses the transform matrix. This is a numerical
# guard, not an artistic knob -- the artistic floor is ``min_scale``.
_SCALE_EPSILON = 1e-6

# Step used to sample the path tangent.
_TANGENT_DELTA = 1e-4

# Resolution of the lookup table used to invert an emission-rate curve.
_RATE_SAMPLES = 2000

# Number of stops in a ``path_colors`` gradient.
_COLOR_STOPS = 256


def clamp(value: float, lower: float = 0.0, upper: float = 1.0) -> float:
    """Constrain ``value`` to the closed interval [``lower``, ``upper``]."""
    return max(lower, min(upper, value))


def as_direction_vector(value) -> np.ndarray:
    """Turn a direction-ish input into a 3D unit vector.

    Accepts a vector (``RIGHT``, ``[1, 0.5, 0]``) or an angle in radians
    (``PI / 4``).
    """
    if isinstance(value, (int, float, np.floating, np.integer)):
        angle = float(value)
        return np.array([np.cos(angle), np.sin(angle), 0.0])

    vector = np.asarray(value, dtype=float).flatten()
    if vector.size == 2:
        vector = np.array([vector[0], vector[1], 0.0])
    if vector.size != 3:
        raise ValueError(f"Direction must be a 2D/3D vector or an angle in radians, got {value!r}.")
    norm = np.linalg.norm(vector)
    if norm == 0:
        raise ValueError("Direction vector cannot be the zero vector.")
    return vector / norm


def as_target_point(value) -> np.ndarray:
    """Resolve a target into a point, reading a Mobject's centre live."""
    if isinstance(value, Mobject):
        return value.get_center()
    point = np.asarray(value, dtype=float).flatten()
    if point.size == 2:
        point = np.array([point[0], point[1], 0.0])
    if point.size != 3:
        raise ValueError(f"A target point must be a 2D/3D point or a Mobject, got {value!r}.")
    return point


class StreamAlongPath(AnimationGroup):
    """Emit a steady stream of ``particle`` copies travelling along ``path``.

    Parameters
    ----------
    particle
        The template that gets copied. Each copy travels the path on its own.
    path
        The curve the particles follow.
    run_time
        How long the whole stream effect plays, in seconds.
    travel_time
        How long ONE particle takes to go from ``start_proportion`` to
        ``end_proportion``, in seconds.
    start_proportion, end_proportion
        Where along the path a particle is born and where it dies, as a
        proportion of the path's length (the same convention as
        ``path.point_from_proportion``). Defaults to the whole path.
        ``start_proportion > end_proportion`` makes the stream flow backwards.
    spawn_lag_ratio
        Spacing between births as a fraction of ``travel_time``; smaller means
        denser. Ignored when ``particle_count`` is given.
    particle_count
        Exact number of particles. Overrides ``spawn_lag_ratio`` when set.
    spawn_style
        How a particle appears and disappears: ``"scale_fade"``, ``"scale"``,
        ``"fade"`` or ``"none"``.
    spawn_duration, despawn_duration
        Length of the birth / death transition in seconds. Scaled down
        proportionally if together they exceed ``travel_time``.
    taper_func
        Shape of the birth / death ramp; receives 0->1 and returns 0->1.
    min_scale
        Artistic floor for the shrink. ``0.0`` shrinks to nothing, ``0.3``
        stops at 30% of the original size.
    align_to_path
        Rotate each particle to follow the path's tangent.
    face_direction
        Lock every particle to one heading: a vector or an angle in radians.
    face_point
        Keep every particle aiming at a target: a point or a Mobject. Passing
        a Mobject tracks its centre live.
    face_away
        Invert ``face_point`` so particles point away from the target.
    reference_direction
        Which way the mobject already points when unrotated. ``RIGHT`` by
        default; a ``Triangle()`` points up, so pass ``UP``.
    particle_rate_func
        Easing applied to each particle's journey along the path.
    max_particles
        Safety cap so a tiny ``spawn_lag_ratio`` cannot stall the render.

    Attributes
    ----------
    particle_count
        How many copies were actually created.
    count_source
        Which parameter decided the count.
    spawn_interval
        Mean gap between births, in seconds.
    birth_times
        Birth time of every particle, in seconds.

    Notes
    -----
    Orientation is decided by the first option that is set, in this order:
    ``face_point``, ``face_direction``, ``align_to_path``.
    """

    def __init__(
        self,
        particle: Mobject,
        path: VMobject,
        run_time: float = 5.0,
        travel_time: float = 2.0,
        start_proportion: float = 0.0,
        end_proportion: float = 1.0,
        spawn_lag_ratio: float = 0.05,
        particle_count: int | None = None,
        spawn_style: str = "scale_fade",
        spawn_duration: float = 0.5,
        despawn_duration: float = 0.5,
        taper_func: Callable[[float], float] = linear,
        min_scale: float = 0.0,
        align_to_path: bool = False,
        face_direction=None,
        face_point=None,
        face_away: bool = False,
        reference_direction=RIGHT,
        particle_rate_func: Callable[[float], float] = linear,
        max_particles: int = DEFAULT_MAX_PARTICLES,
        **kwargs,
    ) -> None:
        # -- validation ------------------------------------------------
        if spawn_style not in SPAWN_STYLES:
            raise ValueError(f"spawn_style must be one of {SPAWN_STYLES}, got {spawn_style!r}.")
        if not 0.0 <= start_proportion <= 1.0:
            raise ValueError(f"start_proportion must lie in [0, 1], got {start_proportion}.")
        if not 0.0 <= end_proportion <= 1.0:
            raise ValueError(f"end_proportion must lie in [0, 1], got {end_proportion}.")
        if start_proportion == end_proportion:
            raise ValueError(
                "start_proportion and end_proportion must differ, otherwise particles never move."
            )
        if run_time <= 0:
            raise ValueError(f"run_time must be positive, got {run_time}.")
        if travel_time <= 0:
            raise ValueError(f"travel_time must be positive, got {travel_time}.")
        if not 0.0 <= spawn_lag_ratio <= 1.0:
            raise ValueError(f"spawn_lag_ratio must lie in [0, 1], got {spawn_lag_ratio}.")
        if spawn_duration < 0 or despawn_duration < 0:
            raise ValueError("spawn_duration and despawn_duration must be non-negative.")
        if not 0.0 <= min_scale <= 1.0:
            raise ValueError(f"min_scale must lie in [0, 1], got {min_scale}.")
        if max_particles < 1:
            raise ValueError(f"max_particles must be at least 1, got {max_particles}.")
        if particle_count is not None:
            if not isinstance(particle_count, (int, np.integer)):
                raise TypeError(
                    f"particle_count must be an int or None, got {type(particle_count).__name__}."
                )
            if particle_count < 1:
                raise ValueError(f"particle_count must be at least 1, got {particle_count}.")

        # -- orientation: resolve now so bad input fails immediately ----
        reference_angle = angle_of_vector(as_direction_vector(reference_direction))
        fixed_heading = (
            angle_of_vector(as_direction_vector(face_direction))
            if face_direction is not None
            else None
        )
        if face_point is not None:
            as_target_point(face_point)  # validate eagerly, resolve per frame

        if (
            sum(
                (
                    face_point is not None,
                    face_direction is not None,
                    bool(align_to_path),
                )
            )
            > 1
        ):
            chosen = (
                "face_point"
                if face_point is not None
                else "face_direction"
                if face_direction is not None
                else "align_to_path"
            )
            warnings.warn(
                "More than one orientation option was given "
                "(face_point / face_direction / align_to_path); "
                f"using {chosen} and ignoring the others.",
                stacklevel=2,
            )
        if face_away and face_point is None:
            warnings.warn("face_away has no effect unless face_point is set.", stacklevel=2)

        # A single particle cannot outlive the whole animation.
        travel_time = min(travel_time, run_time)

        # Birth and death may not overlap; squeeze them if they would.
        taper_total = spawn_duration + despawn_duration
        if taper_total > travel_time:
            squeeze = travel_time / taper_total
            spawn_duration *= squeeze
            despawn_duration *= squeeze

        # -- how many particles? count wins over lag ratio --------------
        count, source = self._resolve_particle_count(
            run_time=run_time,
            travel_time=travel_time,
            spawn_lag_ratio=spawn_lag_ratio,
            particle_count=particle_count,
            max_particles=max_particles,
        )

        if count == 1 and run_time > travel_time:
            warnings.warn(
                f"A single particle cannot fill run_time={run_time}; the stream "
                f"will last travel_time={travel_time} instead. Increase "
                f"particle_count or lower spawn_lag_ratio for a longer stream.",
                stacklevel=2,
            )
            run_time = travel_time

        # Order matters: the birth-time jitter draws from the RNG before the
        # per-particle randomness does, so keep these two calls in this order.
        stream_span = run_time - travel_time
        birth_times = np.asarray(self._compute_birth_times(count, stream_span), dtype=float)

        # Hook: subclasses draw their per-particle state here.
        self._on_count_resolved(count)

        # -- shared state the updater and hooks read --------------------
        self._path = path
        self._travel_time = travel_time
        self._start_proportion = start_proportion
        self._end_proportion = end_proportion
        self._proportion_span = end_proportion - start_proportion
        self._applies_scale = spawn_style in ("scale_fade", "scale")
        self._applies_fade = spawn_style in ("scale_fade", "fade")
        self._min_scale = min_scale
        self._birth_times = birth_times

        # -- public read-back of what was actually built ----------------
        self.particle_count = count
        self.count_source = source
        self.birth_times = birth_times
        self.spawn_interval = float(np.mean(np.diff(birth_times))) if count > 1 else 0.0
        self.travel_time = travel_time
        self.start_proportion = start_proportion
        self.end_proportion = end_proportion
        self.spawn_style = spawn_style

        # -- helpers ----------------------------------------------------
        def taper_factor(elapsed: float) -> float:
            """0 at birth, 1 in the middle, 0 at death."""
            if spawn_style == "none":
                return 1.0
            if spawn_duration > 0 and elapsed < spawn_duration:
                raw = elapsed / spawn_duration
            elif despawn_duration > 0 and elapsed > travel_time - despawn_duration:
                raw = (travel_time - elapsed) / despawn_duration
            else:
                return 1.0
            return clamp(taper_func(clamp(raw)))

        def heading_angle(position: np.ndarray, proportion: float) -> float | None:
            """Absolute angle to face, or None to leave the particle alone."""
            if face_point is not None:
                delta = as_target_point(face_point) - position
                if face_away:
                    delta = -delta
                if not np.any(delta):
                    return None
                return angle_of_vector(delta)
            if fixed_heading is not None:
                return fixed_heading
            if align_to_path:
                return angle_of_vector(self._tangent_vector(proportion))
            return None

        def make_updater(index: int) -> Callable[[Mobject, float], None]:
            def update_particle(copy: Mobject, alpha: float) -> None:
                # Reset to the pristine state so nothing compounds.
                copy.restore()

                factor = taper_factor(alpha * travel_time)
                self._apply_appearance(copy, index, alpha, factor)

                proportion = clamp(start_proportion + alpha * self._proportion_span)
                position = self._adjust_position(
                    path.point_from_proportion(proportion), index, alpha, proportion
                )

                # Rotate once the final position is known, so face_point aims
                # from where the particle actually ends up.
                angle = heading_angle(position, proportion)
                if angle is not None:
                    copy.rotate(angle - reference_angle)
                copy.move_to(position)

            return update_particle

        animations = [
            UpdateFromAlphaFunc(
                self._make_copy(particle, index),
                make_updater(index),
                run_time=travel_time,
                rate_func=particle_rate_func,
            )
            for index in range(count)
        ]

        kwargs.setdefault("remover", True)
        super().__init__(*animations, lag_ratio=0.0, run_time=run_time, **kwargs)

    # ------------------------------------------------------------------
    # timing
    # ------------------------------------------------------------------
    def build_animations_with_timings(self) -> None:
        """Lay out sub-animations using explicit birth times.

        ``AnimationGroup`` can only space births evenly via a single
        ``lag_ratio``. Driving the timings directly keeps ``travel_time``
        truthful and lets subclasses place births wherever they like.
        """
        births = getattr(self, "_birth_times", None)
        if births is None:
            super().build_animations_with_timings()
            return

        run_times = np.array([anim.run_time for anim in self.animations])
        dtype = [("anim", "O"), ("start", "f8"), ("end", "f8")]
        self.anims_with_timings = np.zeros(len(self.animations), dtype=dtype)
        self.anims_begun = np.zeros(len(self.animations), dtype=bool)
        self.anims_finished = np.zeros(len(self.animations), dtype=bool)
        if not len(self.animations):
            return

        self.anims_with_timings["anim"] = self.animations
        self.anims_with_timings["start"] = births
        self.anims_with_timings["end"] = births + run_times

    @staticmethod
    def _resolve_particle_count(
        run_time: float,
        travel_time: float,
        spawn_lag_ratio: float,
        particle_count: int | None,
        max_particles: int,
    ) -> tuple[int, str]:
        """Decide the particle count. An explicit count overrides the timing."""
        if particle_count is not None:
            if particle_count > max_particles:
                warnings.warn(
                    f"particle_count={particle_count} exceeds max_particles="
                    f"{max_particles}; capping. Raise max_particles to allow more.",
                    stacklevel=3,
                )
                return max_particles, "particle_count (capped)"
            return int(particle_count), "particle_count"

        spawn_interval = spawn_lag_ratio * travel_time
        stream_span = run_time - travel_time
        if spawn_interval <= 0 or stream_span <= 0:
            # spawn_lag_ratio == 0 starts everything together -> one suffices.
            return 1, "spawn_lag_ratio"

        count = max(2, round(stream_span / spawn_interval) + 1)
        if count > max_particles:
            warnings.warn(
                f"spawn_lag_ratio={spawn_lag_ratio} implies {count} particles, "
                f"above max_particles={max_particles}; capping. The stream will "
                f"be sparser than requested.",
                stacklevel=3,
            )
            return max_particles, "spawn_lag_ratio (capped)"
        return count, "spawn_lag_ratio"

    # ------------------------------------------------------------------
    # geometry
    # ------------------------------------------------------------------
    def _tangent_vector(self, proportion: float) -> np.ndarray:
        """Unit direction of travel at ``proportion``."""
        step = _TANGENT_DELTA if self._proportion_span > 0 else -_TANGENT_DELTA
        ahead = clamp(proportion + step)
        behind = clamp(proportion - step)
        if ahead == behind:
            return np.array([1.0, 0.0, 0.0])
        delta = self._path.point_from_proportion(ahead) - self._path.point_from_proportion(behind)
        if not np.any(delta):
            return np.array([1.0, 0.0, 0.0])
        return normalize(delta)

    # ------------------------------------------------------------------
    # hooks -- subclasses override these
    # ------------------------------------------------------------------
    def _on_count_resolved(self, count: int) -> None:
        """Called once the particle count is known, before anything is built."""

    def _compute_birth_times(self, count: int, stream_span: float) -> np.ndarray:
        """Return the birth time of every particle. Evenly spaced by default."""
        if count == 1 or stream_span <= 0:
            return np.zeros(count)
        return np.linspace(0.0, stream_span, count)

    def _make_copy(self, particle: Mobject, index: int) -> Mobject:
        """Create one particle copy and freeze its pristine state."""
        copy = particle.copy()
        copy.save_state()
        return copy

    def _apply_appearance(self, copy: Mobject, index: int, alpha: float, factor: float) -> None:
        """Apply size and opacity for the current taper ``factor``."""
        if self._applies_scale:
            scale = self._min_scale + (1.0 - self._min_scale) * factor
            copy.scale(max(_SCALE_EPSILON, scale))
        if self._applies_fade:
            copy.fade(1.0 - factor)

    def _adjust_position(
        self, position: np.ndarray, index: int, alpha: float, proportion: float
    ) -> np.ndarray:
        """Final say on where the particle sits. Unchanged by default."""
        return position


class ParticleStream(StreamAlongPath):
    """A :class:`StreamAlongPath` with colour, randomness and a variable rate.

    Every ``StreamAlongPath`` parameter works here too; the extras below are
    all off by default, so a ``ParticleStream`` with no extra arguments behaves
    exactly like a ``StreamAlongPath``.

    Parameters
    ----------
    path_colors
        Recolour each particle continuously as it travels, interpolating
        through the list. ``[BLUE, YELLOW, RED]`` starts blue and dies red.
    particle_colors
        Give each particle one fixed colour, cycling through the list. Use
        this for confetti-like variety rather than a positional gradient.
        Ignored when ``path_colors`` is also given.
    position_jitter
        Random offset per particle, in scene units, so the stream stops
        looking like a ruler.
    size_jitter
        Random size variation per particle. ``0.3`` means 70%-130%.
    birth_jitter
        Random nudge to each birth time, as a fraction of the spawn interval.
        Breaks up the metronome rhythm.
    wobble_amplitude, wobble_frequency
        Smooth side-to-side drift while travelling. Amplitude is in scene
        units, frequency is oscillations per trip.
    seed
        Fix the randomness so a render is reproducible.
    emission_rate_func
        Spread births across the animation following a curve. Receives 0->1
        and returns a relative rate; only the shape matters. The particle
        count is unchanged -- only *when* they are born.

            ``lambda t: 1``          steady (the default)
            ``lambda t: t``          starts empty, builds up
            ``lambda t: 1 - t``      bursts, then thins out
            ``lambda t: 0.1 + t**3``  a long calm, then a surge
    """

    def __init__(
        self,
        particle: Mobject,
        path: VMobject,
        *args,
        path_colors: Sequence | None = None,
        particle_colors: Sequence | None = None,
        position_jitter: float = 0.0,
        size_jitter: float = 0.0,
        birth_jitter: float = 0.0,
        wobble_amplitude: float = 0.0,
        wobble_frequency: float = 1.0,
        seed: int | None = None,
        emission_rate_func: Callable[[float], float] | None = None,
        **kwargs,
    ) -> None:
        # -- validation -------------------------------------------------
        if position_jitter < 0:
            raise ValueError(f"position_jitter must be non-negative, got {position_jitter}.")
        if not 0.0 <= size_jitter <= 1.0:
            raise ValueError(f"size_jitter must lie in [0, 1], got {size_jitter}.")
        if not 0.0 <= birth_jitter <= 1.0:
            raise ValueError(f"birth_jitter must lie in [0, 1], got {birth_jitter}.")
        if wobble_amplitude < 0:
            raise ValueError(f"wobble_amplitude must be non-negative, got {wobble_amplitude}.")
        if path_colors is not None and not len(path_colors):
            raise ValueError("path_colors cannot be an empty sequence.")
        if particle_colors is not None and not len(particle_colors):
            raise ValueError("particle_colors cannot be an empty sequence.")
        if path_colors is not None and particle_colors is not None:
            warnings.warn(
                "Both path_colors and particle_colors were given; "
                "path_colors takes priority and particle_colors is ignored.",
                stacklevel=2,
            )

        self._position_jitter = position_jitter
        self._size_jitter = size_jitter
        self._birth_jitter = birth_jitter
        self._wobble_amplitude = wobble_amplitude
        self._wobble_frequency = wobble_frequency
        self._emission_rate_func = emission_rate_func
        self._rng = np.random.default_rng(seed)
        self.seed = seed

        # Colour lookups, resolved once.
        self._path_ramp = None
        if path_colors is not None:
            stops = list(path_colors)
            self._path_ramp = (
                color_gradient(stops, _COLOR_STOPS) if len(stops) > 1 else [stops[0]] * _COLOR_STOPS
            )
        self._fixed_colors = list(particle_colors) if particle_colors is not None else None

        # Filled in by _on_count_resolved once the count is known.
        self._offsets: np.ndarray | None = None
        self._size_factors: np.ndarray | None = None
        self._wobble_phases: np.ndarray | None = None

        super().__init__(particle, path, *args, **kwargs)

    # ------------------------------------------------------------------
    # hooks
    # ------------------------------------------------------------------
    def _on_count_resolved(self, count: int) -> None:
        """Draw the frozen per-particle randomness.

        Values are drawn one particle at a time, in the order
        offset -> size -> wobble phase, so a given ``seed`` always produces
        the same stream.
        """
        rng = self._rng
        offsets = np.zeros((count, 3))
        sizes = np.ones(count)
        phases = np.zeros(count)

        for index in range(count):
            if self._position_jitter > 0:
                offsets[index, :2] = rng.uniform(-1.0, 1.0, 2) * self._position_jitter
            if self._size_jitter > 0:
                sizes[index] = 1.0 + float(rng.uniform(-1.0, 1.0)) * self._size_jitter
            if self._wobble_amplitude > 0:
                phases[index] = float(rng.uniform(0.0, 2 * np.pi))

        self._offsets = offsets if self._position_jitter > 0 else None
        self._size_factors = sizes if self._size_jitter > 0 else None
        self._wobble_phases = phases if self._wobble_amplitude > 0 else None

    def _compute_birth_times(self, count: int, stream_span: float) -> np.ndarray:
        """Spread births following the emission curve, then jitter them."""
        births = self._emission_birth_times(count, stream_span)

        if self._birth_jitter > 0 and count > 1 and stream_span > 0:
            nominal = stream_span / (count - 1)
            wiggle = self._rng.uniform(-1.0, 1.0, count) * self._birth_jitter * nominal
            births = np.clip(births + wiggle, 0.0, stream_span)
            # Keep the timeline anchored so the group still fills run_time.
            births[0] = 0.0
            births[-1] = stream_span
        return births

    def _apply_appearance(self, copy: Mobject, index: int, alpha: float, factor: float) -> None:
        """Colour first, then size (with jitter) and opacity."""
        if self._path_ramp is not None:
            copy.set_color(self._path_ramp[int(alpha * (_COLOR_STOPS - 1))])
        elif self._fixed_colors is not None:
            copy.set_color(self._fixed_colors[index % len(self._fixed_colors)])

        size_factor = float(self._size_factors[index]) if self._size_factors is not None else 1.0
        if self._applies_scale:
            scale = self._min_scale + (1.0 - self._min_scale) * factor
            copy.scale(max(_SCALE_EPSILON, scale * size_factor))
        elif size_factor != 1.0:
            copy.scale(max(_SCALE_EPSILON, size_factor))

        if self._applies_fade:
            copy.fade(1.0 - factor)

    def _adjust_position(
        self, position: np.ndarray, index: int, alpha: float, proportion: float
    ) -> np.ndarray:
        """Apply the frozen offset and the travelling wobble."""
        if self._offsets is not None:
            position = position + self._offsets[index]

        if self._wobble_amplitude > 0:
            # Cross with the screen normal to get a perpendicular that stays
            # in the xy-plane, i.e. visibly sideways.
            sideways = np.cross(self._tangent_vector(proportion), OUT)
            if np.any(sideways):
                swing = np.sin(
                    2 * np.pi * self._wobble_frequency * alpha + float(self._wobble_phases[index])
                )
                position = position + normalize(sideways) * (self._wobble_amplitude * swing)
        return position

    # ------------------------------------------------------------------
    def _emission_birth_times(self, count: int, stream_span: float) -> np.ndarray:
        """Place births so their density follows ``emission_rate_func``."""
        if count == 1 or stream_span <= 0:
            return np.zeros(count)
        if self._emission_rate_func is None:
            return np.linspace(0.0, stream_span, count)

        # Sample the rate, integrate it, then invert: births land where equal
        # slices of "emitted quantity" fall, so a high rate packs them closer.
        samples = np.linspace(0.0, 1.0, _RATE_SAMPLES)
        rates = np.array([float(self._emission_rate_func(t)) for t in samples])
        if not np.all(np.isfinite(rates)):
            raise ValueError("emission_rate_func returned a non-finite value.")
        rates = np.clip(rates, 0.0, None)

        cumulative = np.concatenate([[0.0], np.cumsum(rates[:-1] + rates[1:]) / 2.0])
        total = cumulative[-1]
        if total <= 0:
            warnings.warn(
                "emission_rate_func is zero everywhere; falling back to a steady emission rate.",
                stacklevel=3,
            )
            return np.linspace(0.0, stream_span, count)

        births = np.interp(np.linspace(0.0, 1.0, count), cumulative / total, samples)
        births *= stream_span
        births[0] = 0.0
        births[-1] = stream_span
        return births
