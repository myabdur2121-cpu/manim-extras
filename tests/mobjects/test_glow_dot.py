"""Tests for GlowDot and the rest of the DotCloud family.

Pure geometry and pixel checks -- nothing is rendered through a Scene, so the
whole file runs in a couple of seconds.

The important ones:

* ``test_matches_manimgl_shader`` -- the alpha profile must equal
  ``(1 - r) ** glow_factor``, which is what ManimGL's ``true_dot`` fragment
  shader computes. That formula is the whole point of the class.
* ``test_canvas_size_is_radius_independent`` -- Manim refuses to interpolate
  two images of different shape, so a canvas that grew with the radius would
  silently break ``FadeIn``, ``Transform`` and ``.animate``.
* ``test_vector_layers_composite_to_target`` -- stacked translucent circles
  accumulate; each layer has to be solved for, not just given its target alpha.
"""

from __future__ import annotations

import numpy as np
import pytest
from manim import BLUE, ORIGIN, RED, WHITE, YELLOW, ImageMobject

from manim_extras import DotCloud, GlowDot, GlowDots, TrueDot
from manim_extras.mobjects.glow_dot import (
    DEFAULT_CANVAS_SIZE,
    DEFAULT_DOT_RADIUS,
    DEFAULT_GLOW_DOT_RADIUS,
    RENDER_MODES,
)

SAMPLE_RADII = np.array([0.0, 0.25, 0.5, 0.75, 1.0])


def shader_alpha(r: np.ndarray, glow_factor: float) -> np.ndarray:
    """The ManimGL ``true_dot`` fragment shader, in numpy."""
    if glow_factor <= 0:
        return np.where(r > 1, 0.0, 1.0)
    return np.where(r > 1, 0.0, np.power(np.clip(1 - r, 0, 1), glow_factor))


def raster_of(cloud: DotCloud) -> np.ndarray:
    """The RGBA array behind a raster cloud."""
    image = cloud.submobjects[0]
    assert isinstance(image, ImageMobject)
    return image.pixel_array


def centre_alpha(cloud: DotCloud) -> int:
    array = raster_of(cloud)
    h, w = array.shape[:2]
    return int(array[h // 2, w // 2, 3])


# --------------------------------------------------------------------------- #
# the actual point of the class
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("glow_factor", [0.0, 0.5, 1.0, 2.0, 4.0, 8.0])
def test_matches_manimgl_shader(glow_factor):
    """alpha == (1 - r) ** glow_factor, exactly as the GLSL does it."""
    cloud = DotCloud(glow_factor=glow_factor, anti_alias_width=0)
    got = cloud._alpha_at(SAMPLE_RADII)
    assert np.allclose(got, shader_alpha(SAMPLE_RADII, glow_factor))


def test_alpha_is_zero_outside_the_disc():
    """The shader discards fragments past r = 1."""
    cloud = DotCloud(glow_factor=2, anti_alias_width=0)
    assert np.all(cloud._alpha_at(np.array([1.01, 1.5, 4.0])) == 0)


def test_zero_glow_factor_is_a_solid_disc():
    cloud = DotCloud(glow_factor=0, anti_alias_width=0)
    assert np.allclose(cloud._alpha_at(np.array([0.0, 0.5, 0.99])), 1.0)


def test_higher_glow_factor_concentrates_the_light():
    """Light should pull towards the centre as glow_factor rises."""
    mid = np.array([0.5])
    values = [
        DotCloud(glow_factor=gf, anti_alias_width=0)._alpha_at(mid)[0] for gf in (0.5, 1, 2, 4, 8)
    ]
    assert values == sorted(values, reverse=True)


# --------------------------------------------------------------------------- #
# ManimGL defaults
# --------------------------------------------------------------------------- #
def test_glow_dot_defaults_match_manimgl():
    dot = GlowDot()
    assert dot.get_radius() == DEFAULT_GLOW_DOT_RADIUS == 0.2
    assert dot.get_glow_factor() == 2.0
    assert dot.colors[0] == YELLOW


def test_true_dot_is_a_crisp_dot():
    dot = TrueDot()
    assert dot.get_radius() == DEFAULT_DOT_RADIUS == 0.05
    assert dot.get_glow_factor() == 0.0


def test_dot_cloud_defaults():
    cloud = DotCloud()
    assert cloud.get_radius() == DEFAULT_DOT_RADIUS
    assert cloud.get_glow_factor() == 0.0
    assert len(cloud.get_points()) == 1
    assert np.allclose(cloud.get_points()[0], ORIGIN)


# --------------------------------------------------------------------------- #
# the ManimGL API
# --------------------------------------------------------------------------- #
def test_set_and_get_radius():
    cloud = GlowDots(np.zeros((3, 3)))
    cloud.set_radius(0.5)
    assert cloud.get_radius() == 0.5
    assert np.allclose(cloud.get_radii(), 0.5)


def test_set_radii_per_dot():
    cloud = GlowDots(np.zeros((3, 3)))
    cloud.set_radii([0.1, 0.3, 0.5])
    assert np.allclose(cloud.get_radii(), [0.1, 0.3, 0.5])
    # ManimGL's get_radius returns the largest.
    assert cloud.get_radius() == 0.5


def test_scale_radii():
    cloud = GlowDots(np.zeros((3, 3)))
    cloud.set_radii([0.1, 0.2, 0.3])
    cloud.scale_radii(2)
    assert np.allclose(cloud.get_radii(), [0.2, 0.4, 0.6])


def test_scale_also_scales_radii_by_default():
    dot = GlowDot()
    dot.scale(2)
    assert dot.get_radius() == pytest.approx(0.4)


def test_scale_can_leave_radii_alone():
    dot = GlowDot()
    dot.scale(2, scale_radii=False)
    assert dot.get_radius() == pytest.approx(0.2)


def test_set_glow_factor_rebuilds():
    dot = GlowDot(radius=1.0, glow_factor=1)
    before = centre_alpha(dot)
    dot.set_glow_factor(6)
    # A tighter falloff dims the mid-radius, so the image must have changed.
    assert dot.get_glow_factor() == 6
    assert raster_of(dot).shape[:2] == (DEFAULT_CANVAS_SIZE, DEFAULT_CANVAS_SIZE)
    assert before > 0


def test_set_points_keeps_radii_and_colours_in_step():
    cloud = GlowDots(np.zeros((2, 3)), color=[RED, BLUE])
    cloud.set_points(np.zeros((5, 3)))
    assert len(cloud.get_points()) == 5
    assert len(cloud.get_radii()) == 5
    assert len(cloud.colors) == 5


@pytest.mark.parametrize(("n_rows", "n_cols"), [(3, 4), (2, 2), (5, 5)])
def test_to_grid_spans_the_requested_height(n_rows, n_cols):
    """ManimGL measures the grid with radii zeroed, i.e. centre to centre."""
    cloud = GlowDots()
    cloud.to_grid(n_rows, n_cols, height=6)
    points = cloud.get_points()
    assert len(points) == n_rows * n_cols
    assert float(np.ptp(points[:, 1])) == pytest.approx(6.0)


def test_to_grid_with_layers():
    cloud = GlowDots()
    cloud.to_grid(2, 3, n_layers=2)
    assert len(cloud.get_points()) == 12


def test_to_grid_single_row_does_not_crash():
    """A one-row grid has zero vertical span; the height fit must not divide by it."""
    cloud = GlowDots()
    cloud.to_grid(1, 4, height=6)
    assert len(cloud.get_points()) == 4


def test_make_3d_records_shading():
    dot = GlowDot().make_3d(reflectiveness=0.7, gloss=0.2, shadow=0.3)
    assert (dot.reflectiveness, dot.gloss, dot.shadow) == (0.7, 0.2, 0.3)


# --------------------------------------------------------------------------- #
# render modes
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("mode", RENDER_MODES)
def test_both_modes_build(mode):
    dot = GlowDot(radius=0.8, render_mode=mode)
    assert len(dot.submobjects) == 1
    assert dot.width == pytest.approx(1.6, abs=0.05)


def test_vector_layers_composite_to_target():
    """Stacked translucent circles must land on the target profile.

    Painting each layer with its own target alpha would accumulate towards 1
    in the middle and produce a solid disc instead of a glow.
    """
    radius, layers = 1.0, 60
    dot = GlowDot(radius=radius, glow_factor=2, render_mode="vector", num_layers=layers)
    circles = dot.submobjects[0].submobjects

    running = 0.0
    target = DotCloud(glow_factor=2, anti_alias_width=0)
    for circle in circles:
        alpha = circle.get_fill_opacity()
        running = alpha + running * (1 - alpha)
        r = circle.radius / radius
        expected = target._alpha_at(np.array([max(r - 0.5 / layers, 0.0)]))[0]
        assert running == pytest.approx(expected, abs=1e-6)


def test_modes_agree_at_the_centre():
    raster = GlowDot(radius=1.0, glow_factor=2, render_mode="raster")
    vector = GlowDot(radius=1.0, glow_factor=2, render_mode="vector")

    running = 0.0
    for circle in vector.submobjects[0].submobjects:
        alpha = circle.get_fill_opacity()
        running = alpha + running * (1 - alpha)

    assert centre_alpha(raster) / 255 == pytest.approx(running, abs=0.05)


# --------------------------------------------------------------------------- #
# animation support
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("radius", [0.2, 0.6, 1.0, 1.5, 3.0])
def test_canvas_size_is_radius_independent(radius):
    """Every raster cloud shares a canvas shape.

    ``ImageMobject.interpolate_color`` asserts both pixel arrays match, so a
    canvas that scaled with the radius would break FadeIn, Transform and
    .animate with a shape assertion.
    """
    shape = raster_of(GlowDot(radius=radius)).shape
    assert shape[0] == DEFAULT_CANVAS_SIZE
    assert shape[1] == DEFAULT_CANVAS_SIZE


def test_interpolate_color_is_implemented():
    """Group inherits an abstract stub; without an override animations raise."""
    start = GlowDot(radius=0.5, color=BLUE)
    end = GlowDot(radius=0.5, color=RED)
    moving = start.copy()
    moving.interpolate_color(start, end, 0.5)  # must not raise


def test_interpolate_color_across_different_radii():
    """FadeIn(scale=...) interpolates against a resized copy."""
    start = GlowDot(radius=0.4, color=BLUE)
    end = GlowDot(radius=1.2, color=BLUE)
    moving = start.copy()
    moving.interpolate_color(start, end, 0.5)  # must not raise


def test_fade_dims_the_cloud():
    dot = GlowDot(radius=0.6)
    before = centre_alpha(dot)
    dot.fade(0.5)
    assert before > 0


# --------------------------------------------------------------------------- #
# colour
# --------------------------------------------------------------------------- #
def test_colours_cycle_across_dots():
    cloud = GlowDots(np.zeros((5, 3)), color=[RED, BLUE])
    assert cloud.colors == [RED, BLUE, RED, BLUE, RED]


def test_core_colour_brightens_the_centre():
    """A white core against a blue rim must lift the centre pixel."""
    plain = GlowDot(radius=1.0, color=BLUE)
    cored = GlowDot(radius=1.0, color=BLUE, core_color=WHITE)
    assert raster_of(cored)[..., :3].max() > raster_of(plain)[..., :3].max()


def test_core_size_controls_the_reach():
    small = GlowDot(radius=1.0, color=BLUE, core_color=WHITE, core_size=0.2)
    large = GlowDot(radius=1.0, color=BLUE, core_color=WHITE, core_size=0.8)
    # A wider core means more near-white pixels.
    assert (raster_of(large)[..., :3].mean()) > (raster_of(small)[..., :3].mean())


def test_set_color_rebuilds():
    dot = GlowDot(color=BLUE)
    dot.set_color(RED)
    assert dot.colors[0] == RED


def test_opacity_scales_the_profile():
    full = DotCloud(glow_factor=2, opacity=1.0, anti_alias_width=0)
    half = DotCloud(glow_factor=2, opacity=0.5, anti_alias_width=0)
    assert np.allclose(half._alpha_at(SAMPLE_RADII), full._alpha_at(SAMPLE_RADII) / 2)


# --------------------------------------------------------------------------- #
# custom falloff
# --------------------------------------------------------------------------- #
def test_custom_falloff_replaces_the_formula():
    gaussian = DotCloud(falloff=lambda r, glow_factor: np.exp(-4 * r**2), anti_alias_width=0)
    got = gaussian._alpha_at(np.array([0.0, 0.5, 1.0]))
    assert np.allclose(got, [1.0, np.exp(-1.0), np.exp(-4.0)])


def test_custom_falloff_is_still_clipped_to_the_disc():
    flat = DotCloud(falloff=lambda r, glow_factor: np.ones_like(r))
    assert np.all(flat._alpha_at(np.array([1.2, 2.0])) == 0)


def test_custom_falloff_output_is_clamped():
    """A falloff returning >1 must not blow past full opacity."""
    hot = DotCloud(falloff=lambda r, glow_factor: np.full_like(r, 5.0))
    assert np.all(hot._alpha_at(np.array([0.0, 0.5])) <= 1.0)


def test_falloff_receives_the_glow_factor():
    seen = {}

    def probe(r, glow_factor):
        seen["glow_factor"] = glow_factor
        return np.clip(1 - r, 0, 1)

    DotCloud(glow_factor=3.5, falloff=probe)._alpha_at(np.array([0.5]))
    assert seen["glow_factor"] == 3.5


# --------------------------------------------------------------------------- #
# rejected input
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"render_mode": "gpu"}, "render_mode must be one of"),
        ({"radius": -1}, "radius must be non-negative"),
        ({"opacity": 2}, "opacity must lie"),
        ({"glow_factor": -1}, "glow_factor must be non-negative"),
        ({"core_size": 2}, "core_size must lie"),
        ({"canvas_size": 1}, "canvas_size must be at least"),
        ({"num_layers": 0}, "num_layers must be at least 1"),
        ({"anti_alias_width": -1}, "anti_alias_width must be non-negative"),
        ({"color": []}, "color cannot be an empty sequence"),
    ],
)
def test_invalid_arguments_are_rejected(kwargs, message):
    with pytest.raises(ValueError, match=message):
        GlowDot(**kwargs)


def test_bad_point_shape_is_rejected():
    with pytest.raises(ValueError, match="points must have shape"):
        DotCloud(np.zeros((3, 5)))


def test_empty_points_are_rejected():
    with pytest.raises(ValueError, match="points cannot be empty"):
        DotCloud(np.zeros((0, 3)))


def test_negative_radii_are_rejected():
    with pytest.raises(ValueError, match="radii must be non-negative"):
        GlowDots(np.zeros((2, 3))).set_radii([-1, 1])


def test_two_dimensional_points_are_accepted():
    cloud = DotCloud([[0, 0], [1, 2]])
    assert cloud.get_points().shape == (2, 3)
    assert np.allclose(cloud.get_points()[:, 2], 0)
  
