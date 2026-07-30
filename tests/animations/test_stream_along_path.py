"""Tests for StreamAlongPath and ParticleStream.

Pure geometry and timing checks -- nothing is rendered, so the whole file runs
in about a second. Particles are inspected by driving a sub-animation directly
with ``begin()`` / ``interpolate(alpha)``.

Two invariants matter most:

* ``test_timeline_fills_run_time`` -- if the group's own end time drifts away
  from ``run_time``, AnimationGroup silently rescales the whole timeline and
  ``travel_time`` stops meaning what it says.
* ``test_seed_is_reproducible`` -- a fixed ``seed`` must always produce the
  same stream.
"""

from __future__ import annotations

import itertools

import numpy as np
import pytest
from manim import (
    BLUE,
    GREEN,
    ORIGIN,
    PI,
    RED,
    RIGHT,
    UP,
    YELLOW,
    Circle,
    Dot,
    Line,
    angle_of_vector,
    rush_into,
    smooth,
)

from manim_extras import ParticleStream, StreamAlongPath

CIRCLE_R = 2.0


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def circle() -> Circle:
    return Circle().scale(CIRCLE_R)


def probe() -> Line:
    """A particle whose heading can be measured exactly.

    A rotated ``Triangle``'s ``get_center()`` is its *bounding box* centre,
    which shifts as it turns, so vertex-minus-centre is not a valid heading.
    A ``Line`` has an exact direction: ``get_end() - get_start()``.
    """
    return Line(ORIGIN, RIGHT * 0.4)


def frame(anim, alpha: float):
    """Drive one sub-animation to ``alpha`` and return its mobject."""
    anim.begin()
    anim.interpolate(alpha)
    return anim.mobject


def first_frame(stream, alpha: float):
    return frame(stream.animations[0], alpha)


def heading(mob) -> float:
    """Direction of a ``probe()`` particle, in degrees."""
    return float(np.degrees(angle_of_vector(mob.get_end() - mob.get_start())))


def angle_gap(a: float, b: float) -> float:
    """Smallest absolute difference between two angles, in degrees."""
    return abs((a - b + 180) % 360 - 180)


# --------------------------------------------------------------------------- #
# density: spawn_lag_ratio vs particle_count
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    ("lag", "expected"),
    [(0.2, 6), (0.1, 11), (0.04, 26)],
)
def test_lag_ratio_sets_the_count(lag, expected):
    """With no explicit count, the timing decides how many particles exist."""
    stream = StreamAlongPath(Dot(), circle(), run_time=10, travel_time=5, spawn_lag_ratio=lag)
    assert stream.particle_count == expected
    assert stream.count_source == "spawn_lag_ratio"


def test_particle_count_overrides_lag_ratio():
    """An explicit count wins, however dense the lag ratio implies."""
    stream = StreamAlongPath(
        Dot(),
        circle(),
        run_time=10,
        travel_time=5,
        spawn_lag_ratio=0.5,
        particle_count=37,
    )
    assert stream.particle_count == 37
    assert stream.count_source == "particle_count"


def test_zero_lag_ratio_gives_one_particle():
    """lag_ratio 0 starts everything together, so one particle is enough."""
    with pytest.warns(UserWarning, match="single particle"):
        stream = StreamAlongPath(Dot(), circle(), run_time=10, travel_time=5, spawn_lag_ratio=0.0)
    assert stream.particle_count == 1


def test_count_is_capped():
    with pytest.warns(UserWarning, match="max_particles"):
        stream = StreamAlongPath(Dot(), circle(), run_time=10, travel_time=5, particle_count=9999)
    assert stream.particle_count == 400
    assert "capped" in stream.count_source


def test_single_particle_shortens_run_time():
    """One particle cannot fill the run time, so the stream says so."""
    with pytest.warns(UserWarning, match="single particle"):
        stream = StreamAlongPath(Dot(), circle(), run_time=10, travel_time=5, particle_count=1)
    assert stream.run_time == 5


# --------------------------------------------------------------------------- #
# timing
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "kwargs",
    [
        {"run_time": 10, "travel_time": 5},
        {"run_time": 10, "travel_time": 5, "spawn_lag_ratio": 0.04},
        {"run_time": 10, "travel_time": 5, "particle_count": 7},
        {"run_time": 3, "travel_time": 5},
        {
            "run_time": 8,
            "travel_time": 4,
            "start_proportion": 0.9,
            "end_proportion": 0.1,
        },
        {"run_time": 10, "travel_time": 2, "spawn_duration": 3, "despawn_duration": 3},
    ],
)
def test_timeline_fills_run_time(kwargs):
    """The group must end exactly on run_time, or the timeline gets rescaled."""
    stream = StreamAlongPath(Dot(), circle(), **kwargs)
    assert stream.max_end_time == pytest.approx(stream.run_time)


def test_births_span_the_stream():
    stream = StreamAlongPath(Dot(), circle(), run_time=10, travel_time=4, particle_count=9)
    assert stream.birth_times[0] == pytest.approx(0.0)
    assert stream.birth_times[-1] == pytest.approx(6.0)
    assert np.all(np.diff(stream.birth_times) > 0)


def test_travel_time_cannot_exceed_run_time():
    stream = StreamAlongPath(Dot(), circle(), run_time=3, travel_time=5)
    assert stream.travel_time == 3


# --------------------------------------------------------------------------- #
# path window
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    ("start", "end"),
    [(0.0, 1.0), (0.25, 0.75), (0.0, 0.35), (0.9, 0.1)],
)
def test_particles_span_the_requested_window(start, end):
    """Birth lands on start_proportion, death on end_proportion."""
    path = circle()
    stream = StreamAlongPath(
        Dot(),
        path,
        run_time=8,
        travel_time=4,
        start_proportion=start,
        end_proportion=end,
    )
    anim = stream.animations[0]
    assert np.allclose(frame(anim, 0.0).get_center(), path.point_from_proportion(start), atol=1e-6)
    assert np.allclose(frame(anim, 1.0).get_center(), path.point_from_proportion(end), atol=1e-6)


def test_reversed_window_travels_backwards():
    """start > end means the tangent points the other way round the path."""
    path = circle()
    forward = StreamAlongPath(probe(), path, run_time=8, travel_time=4, align_to_path=True)
    backward = StreamAlongPath(
        probe(),
        path,
        run_time=8,
        travel_time=4,
        start_proportion=1.0,
        end_proportion=0.0,
        align_to_path=True,
    )
    at_quarter = heading(first_frame(forward, 0.25))
    # Same point on the path, opposite direction of travel.
    reversed_at_same_point = heading(first_frame(backward, 0.75))
    assert angle_gap(at_quarter, reversed_at_same_point) == pytest.approx(180, abs=1)


# --------------------------------------------------------------------------- #
# spawn style, taper and scale
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    ("style", "shrinks", "fades"),
    [
        ("scale_fade", True, True),
        ("scale", True, False),
        ("fade", False, True),
        ("none", False, False),
    ],
)
def test_spawn_style_controls_size_and_opacity(style, shrinks, fades):
    stream = StreamAlongPath(
        Dot(radius=0.2),
        circle(),
        run_time=8,
        travel_time=4,
        spawn_duration=1,
        despawn_duration=1,
        spawn_style=style,
    )
    anim = stream.animations[0]
    born = frame(anim, 0.0)
    born_width, born_opacity = born.width, born.get_fill_opacity()
    middle = frame(anim, 0.5)

    assert middle.width == pytest.approx(0.4)
    assert middle.get_fill_opacity() == pytest.approx(1.0)
    assert bool(born_width < 0.4 - 1e-9) is shrinks
    assert bool(born_opacity < 1.0 - 1e-9) is fades


@pytest.mark.parametrize("min_scale", [0.0, 0.3, 1.0])
def test_min_scale_is_the_floor_at_birth(min_scale):
    stream = StreamAlongPath(
        Dot(radius=0.2),
        circle(),
        run_time=8,
        travel_time=4,
        spawn_duration=1,
        despawn_duration=1,
        min_scale=min_scale,
    )
    born = first_frame(stream, 0.0)
    assert born.width == pytest.approx(0.4 * min_scale, abs=1e-6)


def test_taper_func_reshapes_the_ramp():
    """A non-linear taper differs from linear halfway up the ramp."""
    common = {
        "run_time": 8,
        "travel_time": 4,
        "spawn_duration": 2,
        "despawn_duration": 2,
    }
    straight = StreamAlongPath(Dot(radius=0.2), circle(), **common)
    eased = StreamAlongPath(Dot(radius=0.2), circle(), taper_func=rush_into, **common)
    # A quarter of the way in, i.e. halfway through the 2s spawn ramp.
    assert first_frame(eased, 0.25).width < first_frame(straight, 0.25).width


def test_overlapping_taper_is_squeezed():
    """spawn + despawn longer than travel_time must not produce a negative size."""
    stream = StreamAlongPath(
        Dot(radius=0.2),
        circle(),
        run_time=10,
        travel_time=2,
        spawn_duration=3,
        despawn_duration=3,
    )
    for alpha in np.linspace(0, 1, 9):
        assert first_frame(stream, alpha).width >= 0


def test_particle_rate_func_changes_progress():
    """Easing moves a particle to a different point at the same alpha."""
    straight = StreamAlongPath(Dot(), circle(), run_time=8, travel_time=4)
    eased = StreamAlongPath(
        Dot(), circle(), run_time=8, travel_time=4, particle_rate_func=rush_into
    )
    assert not np.allclose(
        first_frame(straight, 0.5).get_center(), first_frame(eased, 0.5).get_center()
    )


# --------------------------------------------------------------------------- #
# orientation
# --------------------------------------------------------------------------- #
def test_unoriented_particles_are_not_rotated():
    stream = StreamAlongPath(probe(), circle(), run_time=8, travel_time=4)
    assert angle_gap(heading(first_frame(stream, 0.5)), 0.0) < 0.5


def test_align_to_path_follows_the_tangent():
    """On a circle the heading advances a quarter turn per quarter travelled."""
    stream = StreamAlongPath(probe(), circle(), run_time=8, travel_time=4, align_to_path=True)
    anim = stream.animations[0]
    headings = [heading(frame(anim, a)) for a in (0.0, 0.25, 0.5, 0.75)]
    for before, after in itertools.pairwise(headings):
        assert angle_gap(before, after) == pytest.approx(90, abs=0.5)


@pytest.mark.parametrize(
    ("direction", "expected"),
    [(RIGHT, 0.0), (UP, 90.0), (PI / 4, 45.0), ([1, 1, 0], 45.0), ([0, 1], 90.0)],
)
def test_face_direction_is_fixed(direction, expected):
    """A vector, a 2D vector or an angle in radians all lock the heading."""
    stream = StreamAlongPath(probe(), circle(), run_time=8, travel_time=4, face_direction=direction)
    anim = stream.animations[0]
    for alpha in (0.0, 0.5, 1.0):
        assert angle_gap(heading(frame(anim, alpha)), expected) < 0.5


def test_face_point_aims_at_the_target():
    """The heading is recomputed from wherever the particle currently is."""
    stream = StreamAlongPath(probe(), circle(), run_time=8, travel_time=4, face_point=ORIGIN)
    anim = stream.animations[0]
    for alpha in (0.0, 0.2, 0.4, 0.6, 0.8):
        mob = frame(anim, alpha)
        wanted = np.degrees(angle_of_vector(ORIGIN - mob.get_center()))
        assert angle_gap(heading(mob), wanted) < 0.5


def test_face_away_points_outwards():
    stream = StreamAlongPath(
        probe(),
        circle(),
        run_time=8,
        travel_time=4,
        face_point=ORIGIN,
        face_away=True,
    )
    anim = stream.animations[0]
    for alpha in (0.0, 0.3, 0.6):
        mob = frame(anim, alpha)
        wanted = np.degrees(angle_of_vector(mob.get_center() - ORIGIN))
        assert angle_gap(heading(mob), wanted) < 0.5


def test_face_point_tracks_a_moving_mobject():
    """Passing a Mobject reads its centre every frame, not once at build time."""
    target = Dot().move_to([3, 0, 0])
    stream = StreamAlongPath(probe(), circle(), run_time=8, travel_time=4, face_point=target)
    anim = stream.animations[0]
    anim.begin()
    anim.interpolate(0.5)
    before = heading(anim.mobject)

    target.move_to([-3, 2, 0])
    anim.interpolate(0.5)
    after = heading(anim.mobject)

    assert angle_gap(before, after) > 1
    wanted = np.degrees(angle_of_vector(np.array([-3.0, 2.0, 0.0]) - anim.mobject.get_center()))
    assert angle_gap(after, wanted) < 0.5


def test_reference_direction_offsets_the_rotation():
    """Declaring which way the mobject already points avoids pre-rotating it."""
    stream = StreamAlongPath(
        probe(),
        circle(),
        run_time=8,
        travel_time=4,
        face_direction=RIGHT,
        reference_direction=UP,
    )
    assert angle_gap(heading(first_frame(stream, 0.5)), -90.0) < 0.5


def test_face_point_beats_the_other_orientation_modes():
    with pytest.warns(UserWarning, match="More than one orientation"):
        stream = StreamAlongPath(
            probe(),
            circle(),
            run_time=8,
            travel_time=4,
            face_point=ORIGIN,
            face_direction=UP,
            align_to_path=True,
        )
    mob = first_frame(stream, 0.3)
    wanted = np.degrees(angle_of_vector(ORIGIN - mob.get_center()))
    assert angle_gap(heading(mob), wanted) < 0.5


def test_face_direction_beats_align_to_path():
    with pytest.warns(UserWarning, match="More than one orientation"):
        stream = StreamAlongPath(
            probe(),
            circle(),
            run_time=8,
            travel_time=4,
            face_direction=UP,
            align_to_path=True,
        )
    assert angle_gap(heading(first_frame(stream, 0.3)), 90.0) < 0.5


def test_face_away_alone_warns():
    with pytest.warns(UserWarning, match="face_away has no effect"):
        StreamAlongPath(Dot(), circle(), run_time=8, travel_time=4, face_away=True)


# --------------------------------------------------------------------------- #
# ParticleStream: colour
# --------------------------------------------------------------------------- #
def test_particle_stream_defaults_match_the_base_class():
    """With no extras, a ParticleStream is a StreamAlongPath."""
    kwargs = {"run_time": 8, "travel_time": 4, "spawn_lag_ratio": 0.05}
    base = StreamAlongPath(Dot(radius=0.2), circle(), **kwargs)
    extended = ParticleStream(Dot(radius=0.2), circle(), **kwargs)

    assert extended.particle_count == base.particle_count
    assert np.allclose(extended.birth_times, base.birth_times)
    for alpha in (0.0, 0.5, 1.0):
        assert np.allclose(
            first_frame(extended, alpha).get_center(),
            first_frame(base, alpha).get_center(),
        )


def test_path_colors_change_along_the_journey():
    stream = ParticleStream(
        Dot(), circle(), run_time=8, travel_time=4, path_colors=[BLUE, YELLOW, RED]
    )
    anim = stream.animations[0]
    colours = [frame(anim, a).get_color().to_hex() for a in (0.0, 0.5, 1.0)]
    assert colours[0] == BLUE.to_hex()
    assert colours[-1] == RED.to_hex()
    assert len(set(colours)) == 3


def test_particle_colors_cycle_and_stay_fixed():
    palette = [BLUE, GREEN, RED]
    stream = ParticleStream(
        Dot(),
        circle(),
        run_time=8,
        travel_time=4,
        particle_count=6,
        particle_colors=palette,
    )
    got = [frame(anim, 0.5).get_color().to_hex() for anim in stream.animations]
    assert got == [c.to_hex() for c in palette] * 2

    # A single particle keeps one colour for its whole journey.
    anim = stream.animations[0]
    assert frame(anim, 0.1).get_color().to_hex() == frame(anim, 0.9).get_color().to_hex()


def test_path_colors_take_priority_over_particle_colors():
    with pytest.warns(UserWarning, match="path_colors takes priority"):
        stream = ParticleStream(
            Dot(),
            circle(),
            run_time=8,
            travel_time=4,
            path_colors=[BLUE],
            particle_colors=[RED],
        )
    assert first_frame(stream, 0.5).get_color().to_hex() == BLUE.to_hex()


# --------------------------------------------------------------------------- #
# ParticleStream: randomness
# --------------------------------------------------------------------------- #
def test_position_jitter_scatters_particles():
    path = circle()
    stream = ParticleStream(
        Dot(),
        path,
        run_time=8,
        travel_time=4,
        particle_count=8,
        position_jitter=0.3,
        seed=1,
    )
    offsets = [
        np.linalg.norm(frame(anim, 0.5).get_center() - path.point_from_proportion(0.5))
        for anim in stream.animations
    ]
    assert max(offsets) > 0
    assert max(offsets) <= 0.3 * np.sqrt(2) + 1e-9


def test_size_jitter_varies_sizes():
    stream = ParticleStream(
        Dot(radius=0.2),
        circle(),
        run_time=8,
        travel_time=4,
        particle_count=8,
        size_jitter=0.5,
        spawn_style="none",
        seed=1,
    )
    widths = [frame(anim, 0.5).width for anim in stream.animations]
    assert min(widths) < 0.4 < max(widths)
    assert min(widths) >= 0.4 * 0.5 - 1e-9
    assert max(widths) <= 0.4 * 1.5 + 1e-9


def test_birth_jitter_breaks_the_rhythm():
    even = ParticleStream(Dot(), circle(), run_time=10, travel_time=4, particle_count=12)
    jittered = ParticleStream(
        Dot(),
        circle(),
        run_time=10,
        travel_time=4,
        particle_count=12,
        birth_jitter=0.8,
        seed=3,
    )
    assert np.std(np.diff(even.birth_times)) == pytest.approx(0)
    assert np.std(np.diff(jittered.birth_times)) > 0
    # The timeline must still be anchored at both ends.
    assert jittered.max_end_time == pytest.approx(jittered.run_time)


def test_wobble_pushes_particles_off_the_path():
    path = Line(4 * np.array([-1.0, 0, 0]), 4 * np.array([1.0, 0, 0]))
    stream = ParticleStream(
        Dot(),
        path,
        run_time=8,
        travel_time=4,
        particle_count=5,
        wobble_amplitude=0.4,
        wobble_frequency=2,
        spawn_style="none",
        seed=5,
    )
    anim = stream.animations[0]
    deviations = [
        np.linalg.norm(frame(anim, a).get_center() - path.point_from_proportion(a))
        for a in np.linspace(0, 1, 17)
    ]
    assert max(deviations) == pytest.approx(0.4, abs=0.05)


def test_seed_is_reproducible():
    def sample(seed):
        stream = ParticleStream(
            Dot(radius=0.2),
            circle(),
            run_time=8,
            travel_time=4,
            particle_count=6,
            position_jitter=0.3,
            size_jitter=0.4,
            birth_jitter=0.5,
            seed=seed,
        )
        return [
            (frame(anim, 0.5).get_center().tolist(), frame(anim, 0.5).width)
            for anim in stream.animations
        ]

    assert sample(42) == sample(42)
    assert sample(42) != sample(7)


# --------------------------------------------------------------------------- #
# ParticleStream: emission rate
# --------------------------------------------------------------------------- #
def test_emission_rate_keeps_the_count():
    """Only *when* particles are born changes, never how many."""
    stream = ParticleStream(
        Dot(),
        circle(),
        run_time=10,
        travel_time=4,
        particle_count=13,
        emission_rate_func=lambda t: t,
    )
    assert stream.particle_count == 13
    assert stream.max_end_time == pytest.approx(stream.run_time)


def test_rising_rate_back_loads_the_births():
    steady = ParticleStream(Dot(), circle(), run_time=10, travel_time=4, particle_count=13)
    rising = ParticleStream(
        Dot(),
        circle(),
        run_time=10,
        travel_time=4,
        particle_count=13,
        emission_rate_func=lambda t: t,
    )
    half = (steady.run_time - steady.travel_time) / 2
    assert np.sum(rising.birth_times < half) < np.sum(steady.birth_times < half)


def test_falling_rate_front_loads_the_births():
    steady = ParticleStream(Dot(), circle(), run_time=10, travel_time=4, particle_count=13)
    falling = ParticleStream(
        Dot(),
        circle(),
        run_time=10,
        travel_time=4,
        particle_count=13,
        emission_rate_func=lambda t: 1 - t,
    )
    half = (steady.run_time - steady.travel_time) / 2
    assert np.sum(falling.birth_times < half) > np.sum(steady.birth_times < half)


def test_zero_rate_falls_back_to_steady():
    with pytest.warns(UserWarning, match="zero everywhere"):
        stream = ParticleStream(
            Dot(),
            circle(),
            run_time=8,
            travel_time=3,
            particle_count=6,
            emission_rate_func=lambda t: 0,
        )
    assert np.allclose(stream.birth_times, np.linspace(0, 5, 6))


# --------------------------------------------------------------------------- #
# rejected input
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"spawn_style": "glow"}, "spawn_style must be one of"),
        ({"start_proportion": 1.5}, "start_proportion must lie"),
        ({"end_proportion": -0.2}, "end_proportion must lie"),
        ({"start_proportion": 0.3, "end_proportion": 0.3}, "must differ"),
        ({"travel_time": 0}, "travel_time must be positive"),
        ({"spawn_lag_ratio": 2}, "spawn_lag_ratio must lie"),
        ({"spawn_duration": -1}, "must be non-negative"),
        ({"min_scale": 1.2}, "min_scale must lie"),
        ({"particle_count": 0}, "particle_count must be at least 1"),
        ({"max_particles": 0}, "max_particles must be at least 1"),
        ({"face_direction": [0, 0, 0]}, "cannot be the zero vector"),
        ({"face_direction": [1, 2, 3, 4]}, "Direction must be"),
        ({"face_point": [1, 2, 3, 4]}, "target point must be"),
    ],
)
def test_invalid_arguments_are_rejected(kwargs, message):
    with pytest.raises(ValueError, match=message):
        StreamAlongPath(Dot(), circle(), run_time=5, **kwargs)


def test_non_integer_particle_count_is_rejected():
    with pytest.raises(TypeError, match="must be an int or None"):
        StreamAlongPath(Dot(), circle(), run_time=5, particle_count=2.5)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"position_jitter": -1}, "position_jitter must be non-negative"),
        ({"size_jitter": 2}, "size_jitter must lie"),
        ({"birth_jitter": 1.5}, "birth_jitter must lie"),
        ({"wobble_amplitude": -2}, "wobble_amplitude must be non-negative"),
        ({"path_colors": []}, "path_colors cannot be an empty sequence"),
        ({"particle_colors": []}, "particle_colors cannot be an empty sequence"),
    ],
)
def test_invalid_particle_stream_arguments_are_rejected(kwargs, message):
    with pytest.raises(ValueError, match=message):
        ParticleStream(Dot(), circle(), run_time=5, **kwargs)


# --------------------------------------------------------------------------- #
# works on any path
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("smoothing", [None, smooth])
def test_runs_on_a_straight_line(smoothing):
    path = Line(np.array([-3.0, 0, 0]), np.array([3.0, 0, 0]))
    kwargs = {"taper_func": smoothing} if smoothing else {}
    stream = StreamAlongPath(Dot(), path, run_time=6, travel_time=3, **kwargs)
    assert np.allclose(first_frame(stream, 0.5).get_center(), ORIGIN, atol=1e-6)
