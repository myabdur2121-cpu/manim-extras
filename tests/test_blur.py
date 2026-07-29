"""Tests for the blur layers.

Nothing is rendered here. The pixel maths is exercised directly on small
numpy arrays and the mobject behaviour on plain geometry, so the whole
file runs in well under a second.

The interesting ones are ``test_blur_uses_live_pixels`` and
``test_imgblur_keeps_first_snapshot``, which pin down the single
difference between the two classes.
"""

from __future__ import annotations

import numpy as np
import pytest
from manim import BLACK, BLUE, WHITE, Annulus, Camera, Circle, Rectangle, Square, Star

from manim_extras import (
    Blur,
    BlurCard,
    CameraBlur,
    IMGBlur,
    IMGBlurCard,
    blur_config,
    gaussian_blur_rgba,
)


def solid(w: int = 64, h: int = 48, value: int = 200) -> np.ndarray:
    """An opaque RGBA block with a single bright square in the middle."""
    arr = np.zeros((h, w, 4), dtype=np.uint8)
    arr[..., 3] = 255
    arr[h // 4 : 3 * h // 4, w // 4 : 3 * w // 4, :3] = value
    return arr


# --------------------------------------------------------------------------- #
# pixel maths
# --------------------------------------------------------------------------- #
def test_zero_sigma_is_a_no_op():
    arr = solid()
    assert np.array_equal(gaussian_blur_rgba(arr, 0), arr)


def test_blur_preserves_shape_and_dtype():
    arr = solid()
    out = gaussian_blur_rgba(arr, 3.0)
    assert out.shape == arr.shape
    assert out.dtype == arr.dtype


def test_blur_softens_edges():
    """Blurring must reduce the sharpest gradient in the image."""
    arr = solid()

    def max_step(a):
        return np.abs(np.diff(a[..., 0].astype(float), axis=1)).max()

    assert max_step(gaussian_blur_rgba(arr, 4.0)) < max_step(arr)


def test_blur_conserves_overall_brightness():
    arr = solid()
    before = arr[..., :3].astype(float).mean()
    after = gaussian_blur_rgba(arr, 3.0)[..., :3].astype(float).mean()
    assert after == pytest.approx(before, rel=0.05)


def test_fast_and_high_quality_agree_roughly():
    arr = solid()
    high = gaussian_blur_rgba(arr, 4.0, quality="high").astype(float)
    fast = gaussian_blur_rgba(arr, 4.0, quality="fast").astype(float)
    assert np.abs(high - fast).mean() < 20


def test_transparent_pixels_do_not_darken_neighbours():
    """The premultiplied path must not bleed black out of empty pixels."""
    arr = np.zeros((32, 32, 4), dtype=np.uint8)
    arr[12:20, 12:20] = (255, 255, 255, 255)  # opaque white patch
    out = gaussian_blur_rgba(arr, 2.0)
    lit = out[..., 3] > 40
    assert out[lit][..., :3].min() > 100


# --------------------------------------------------------------------------- #
# construction
# --------------------------------------------------------------------------- #
def test_amount_spellings_are_equivalent():
    rec = Rectangle()
    assert Blur(rec, 30).get_blur() == 30
    assert Blur(rec, c=30).get_blur() == 30
    assert Blur(rec, amount=30).get_blur() == 30


def test_bare_number_means_full_frame():
    """``Blur(25)`` with no shape covers the frame."""
    mob = Blur(25)
    assert mob.get_blur() == 25
    assert len(mob.get_all_points()) > 0


def test_input_shape_is_left_alone():
    rec = Rectangle(width=4, height=2)
    rec.set_z_index(7)
    before = rec.points.copy()

    glass = Blur(rec, 20)
    glass.set_z_index(3)
    glass.shift(np.array([2.0, 0.0, 0.0]))

    assert np.allclose(rec.points, before)
    assert rec.z_index == 7
    assert glass.z_index == 3


def test_geometry_is_copied_from_the_shape():
    star = Star(7, outer_radius=2)
    assert np.allclose(Blur(star, 20).points, star.points)


def test_stroke_is_copied_but_fill_is_not():
    """An opaque fill would hide the blur, so it is dropped on purpose."""
    rec = Rectangle(stroke_color=BLUE, stroke_width=6, fill_opacity=1.0)
    glass = Blur(rec, 20)
    assert glass.get_stroke_width() == pytest.approx(6)
    assert glass.get_fill_opacity() == pytest.approx(0.0)


def test_explicit_fill_is_respected():
    glass = Blur(Circle(), 20, fill_color=BLUE, fill_opacity=0.4)
    assert glass.get_fill_opacity() == pytest.approx(0.4)


def test_standard_transforms_apply():
    glass = Blur(Circle(radius=1), 20)
    glass.scale(2).shift(np.array([1.0, 0.0, 0.0]))
    assert glass.get_center()[0] == pytest.approx(1.0)
    assert glass.width == pytest.approx(4.0)


# --------------------------------------------------------------------------- #
# the actual difference between Blur and IMGBlur
# --------------------------------------------------------------------------- #
def test_blur_uses_live_pixels():
    glass = Blur(Rectangle(), 20)
    first, second = solid(value=50), solid(value=250)
    assert glass._source_pixels(Camera(), first) is first
    assert glass._source_pixels(Camera(), second) is second


def test_imgblur_keeps_first_snapshot():
    glass = IMGBlur(Rectangle(), 20)
    camera = Camera()
    first, second = solid(value=50), solid(value=250)

    kept = glass._source_pixels(camera, first)
    assert np.array_equal(kept, first)

    # a later, different frame must not replace what was stored
    assert np.array_equal(glass._source_pixels(camera, second), first)


def test_recapture_takes_a_fresh_snapshot():
    glass = IMGBlur(Rectangle(), 20)
    camera = Camera()
    first, second = solid(value=50), solid(value=250)

    glass._source_pixels(camera, first)
    assert glass.has_snapshot()

    glass.recapture()
    assert not glass.has_snapshot()
    assert np.array_equal(glass._source_pixels(camera, second), second)


# --------------------------------------------------------------------------- #
# masking
# --------------------------------------------------------------------------- #
def test_mask_is_confined_to_the_shape():
    """Corners of the bounding box fall outside a circle, so stay unmasked."""
    camera = Camera()
    glass = Blur(Circle(radius=2), 20)
    h, w = camera.pixel_height, camera.pixel_width
    box = glass._pixel_box(camera, w, h)
    mask = glass._get_mask(camera, box)

    assert mask.max() == pytest.approx(1.0, abs=0.02)
    assert mask[mask.shape[0] // 2, mask.shape[1] // 2, 0] == pytest.approx(1.0, abs=0.02)
    assert mask[0, 0, 0] == pytest.approx(0.0, abs=0.02)


def test_mask_keeps_holes_open():
    """An Annulus must leave its centre untouched."""
    camera = Camera()
    glass = Blur(Annulus(inner_radius=1.0, outer_radius=2.0), 20)
    box = glass._pixel_box(camera, camera.pixel_width, camera.pixel_height)
    mask = glass._get_mask(camera, box)

    centre = mask[mask.shape[0] // 2, mask.shape[1] // 2, 0]
    ring = mask[mask.shape[0] // 2, mask.shape[1] // 8, 0]
    assert centre == pytest.approx(0.0, abs=0.05)
    assert ring > 0.9


def test_larger_shapes_cover_more_pixels():
    camera = Camera()
    w, h = camera.pixel_width, camera.pixel_height

    def covered(radius):
        glass = Blur(Circle(radius=radius), 20)
        return glass._get_mask(glass and camera, glass._pixel_box(camera, w, h)).sum()

    assert covered(2.0) > covered(1.0)


def test_camera_blur_covers_the_whole_frame():
    camera = Camera()
    veil = CameraBlur(20)
    w, h = camera.pixel_width, camera.pixel_height
    assert veil._pixel_box(camera, w, h) == (0, 0, w, h)
    assert veil._get_mask(camera, (0, 0, w, h)).min() == pytest.approx(1.0)


# --------------------------------------------------------------------------- #
# applying to a buffer
# --------------------------------------------------------------------------- #
def test_apply_changes_pixels_inside_only():
    camera = Camera()
    buf = np.zeros((camera.pixel_height, camera.pixel_width, 4), dtype=np.uint8)
    buf[..., :3] = 255
    buf[..., 3] = 255
    buf[:, camera.pixel_width // 2 :, :3] = 0  # hard vertical edge

    before = buf.copy()
    Blur(Rectangle(width=4, height=4), 30).apply_to_pixel_array(camera, buf)

    assert not np.array_equal(buf, before)
    assert np.array_equal(buf[0, 0], before[0, 0])  # far corner untouched


def test_disabled_blur_does_nothing():
    camera = Camera()
    buf = solid(camera.pixel_width, camera.pixel_height)
    before = buf.copy()
    Blur(Rectangle(width=4, height=4), 30).disable().apply_to_pixel_array(camera, buf)
    assert np.array_equal(buf, before)


def test_global_switch_disables_every_blur():
    camera = Camera()
    buf = solid(camera.pixel_width, camera.pixel_height)
    before = buf.copy()
    blur_config.enabled = False
    try:
        Blur(Rectangle(width=4, height=4), 30).apply_to_pixel_array(camera, buf)
        assert np.array_equal(buf, before)
    finally:
        blur_config.enabled = True


def test_fading_out_weakens_the_blur():
    """Opacity drives blur strength, which is what makes FadeIn work."""
    camera = Camera()

    def spread(opacity):
        buf = np.zeros((camera.pixel_height, camera.pixel_width, 4), dtype=np.uint8)
        buf[..., 3] = 255
        buf[:, camera.pixel_width // 2 :, :3] = 255
        glass = Blur(Rectangle(width=6, height=6, stroke_width=4), 40)
        glass.set_opacity(opacity)
        glass.apply_to_pixel_array(camera, buf)
        row = buf[camera.pixel_height // 2, :, 0].astype(float)
        return float(np.count_nonzero((row > 20) & (row < 235)))

    assert spread(1.0) > spread(0.25)
    assert spread(0.0) == 0


# --------------------------------------------------------------------------- #
# cards
# --------------------------------------------------------------------------- #
def test_card_defaults_to_a_rounded_rectangle():
    card = BlurCard(width=8, height=4.5)
    assert card.width == pytest.approx(8, abs=0.01)
    assert card.height == pytest.approx(4.5, abs=0.01)


def test_card_accepts_any_outline():
    star = Star(7, outer_radius=2)
    card = BlurCard(star)
    assert card.width == pytest.approx(star.width, abs=0.01)
    assert IMGBlurCard(Square(3)).width == pytest.approx(3, abs=0.01)


def test_card_leading_number_is_the_blur_amount():
    assert BlurCard(35).get_blur() == pytest.approx(35)


def test_card_match_size_stretches_the_outline():
    card = BlurCard(Star(7), match_size=True, width=6, height=3)
    assert card.width == pytest.approx(6, abs=0.01)
    assert card.height == pytest.approx(3, abs=0.01)


def test_card_border_style_overrides_the_shape():
    card = BlurCard(
        Star(7, stroke_color=BLACK, stroke_width=1),
        border_color=WHITE,
        border_width=5,
    )
    assert card.get_stroke_width() == pytest.approx(5)


def test_card_colours_are_settable_after_creation():
    card = BlurCard(width=6, height=3)
    card.set_tint(BLUE, 0.4)
    card.set_border(color=WHITE, width=8)
    assert card.tint_opacity == pytest.approx(0.4)
    assert card.get_stroke_width() == pytest.approx(8)


def test_input_card_shape_is_not_mutated():
    star = Star(7, stroke_width=1)
    BlurCard(star, border_width=9)
    assert star.get_stroke_width() == pytest.approx(1)


# --------------------------------------------------------------------------- #
# presets
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("cls", [Blur, IMGBlur])
def test_presets_exist_on_both_classes(cls):
    assert cls.glass().tint_opacity > 0
    assert cls.dark().tint_opacity > 0
    assert cls.subtle().get_blur() < cls.heavy().get_blur()


def test_preset_defaults_can_be_overridden():
    assert Blur.glass(30, tint_opacity=0.5).tint_opacity == pytest.approx(0.5)


def test_presets_accept_a_shape():
    assert len(Blur.glass(30, shape=Star(7)).points) > 0


# --------------------------------------------------------------------------- #
# animation helpers
# --------------------------------------------------------------------------- #
def test_fade_in_starts_from_zero():
    glass = Blur(Rectangle(), 30)
    glass.fade_in()
    assert glass.get_blur() == pytest.approx(0)


def test_interpolate_blends_the_blur_amount():
    a, b = Blur(Rectangle(), 0), Blur(Rectangle(), 40)
    mid = Blur(Rectangle(), 0)
    mid.interpolate_color(a, b, 0.5)
    assert mid.get_blur() == pytest.approx(20)
  
