"""Gaussian blur layers -- ``Blur``, ``IMGBlur`` and friends.

A blur layer is a real ``VMobject`` whose *interior* blurs whatever has
already been drawn beneath it, exactly like a pane of frosted glass::

    from manim import *
    from manim_extras import Blur

    class GlassDoor(Scene):
        def construct(self):
            rec = Rectangle(width=5, height=5, stroke_color=WHITE)
            glass = Blur(rec, 30)
            glass.set_z_index(1)
            self.add(glass)

            person = Circle(radius=0.6, fill_opacity=1, color=RED)
            person.move_to(LEFT * 6).set_z_index(0)
            self.add(person)
            self.play(person.animate.move_to(RIGHT * 6), run_time=3)

The person is drawn *below* the glass, so it goes past blurred.

Ordering is plain ``z_index``: anything with a lower ``z_index`` than the
blur is blurred, anything above stays sharp. Equal values fall back to
insertion order, so ``self.add(bg); self.add(Blur(...)); self.add(title)``
does the obvious thing.

Classes
-------
``Blur``          re-blurs every frame -- live frosted glass
``IMGBlur``       blurs once, then keeps that snapshot -- static
``BlurCard``      website-style floating glass card (live)
``IMGBlurCard``   the same card, static
``CameraBlur``    the whole camera frame; always live

How it works
------------
``Camera.capture_mobjects`` sorts mobjects by ``z_index`` and paints them
one after another into a single RGBA buffer. This module patches that
method: when a blur layer's turn comes it blurs the pixels drawn so far,
masks the result to the shape's own outline and writes it back.

The mask is produced by replaying the shape's Bézier path into a Cairo
``FORMAT_A8`` surface, so any ``VMobject`` works -- circles, stars,
hand-built splines, even shapes with holes such as ``Annulus``. Because
``capture_mobjects`` runs per frame, stills and videos need no separate
code path.

``scipy`` is used for the separable Gaussian when available; without it
the module falls back to Pillow and still works, just slightly softer.
"""

from __future__ import annotations

__all__ = [
    "Blur",
    "BlurCard",
    "CameraBlur",
    "IMGBlur",
    "IMGBlurCard",
    "blur_config",
    "gaussian_blur_rgba",
]

import itertools as it
from collections.abc import Iterable
from typing import Any

import cairo
import numpy as np
from manim import (
    BLACK,
    WHITE,
    Animation,
    Camera,
    Mobject,
    Rectangle,
    RoundedRectangle,
    VMobject,
    config,
)
from manim.utils.color import ManimColor
from PIL import Image, ImageFilter

try:
    from scipy.ndimage import gaussian_filter as _scipy_gaussian_filter

    _HAS_SCIPY = True
except ImportError:  # pragma: no cover
    _HAS_SCIPY = False


# =============================================================================
# GLOBAL CONFIG
# =============================================================================
class _BlurConfig:
    """Global defaults shared by every blur layer.

    >>> blur_config.fast()           # quick previews
    >>> blur_config.enabled = False  # switch every blur off
    """

    def __init__(self) -> None:
        self.default_blur: float = 20.0
        self.quality: str = "high"  # "high" | "fast"
        self.fast_downscale: int = 4
        self.padding_factor: float = 3.0
        self.enabled: bool = True

    def fast(self) -> _BlurConfig:
        self.quality = "fast"
        return self

    def high(self) -> _BlurConfig:
        self.quality = "high"
        return self

    # legacy name
    @property
    def default_c(self) -> float:
        return self.default_blur

    @default_c.setter
    def default_c(self, v: float) -> None:
        self.default_blur = float(v)


blur_config = _BlurConfig()


# =============================================================================
# PIXEL MATH
# =============================================================================
def _gaussian_high(arr: np.ndarray, sigma: float) -> np.ndarray:
    """Exact separable Gaussian on premultiplied alpha.

    Without premultiplying, the meaningless colour stored in fully
    transparent pixels bleeds into neighbouring visible ones and leaves a
    dark fringe around the edges.
    """
    f = arr.astype(np.float32)
    rgb, alpha = f[..., :3], f[..., 3:4] / 255.0
    pre_b = _scipy_gaussian_filter(rgb * alpha, sigma=(sigma, sigma, 0), mode="nearest")
    a_b = _scipy_gaussian_filter(alpha, sigma=(sigma, sigma, 0), mode="nearest")
    out_rgb = np.where(a_b > 1e-6, pre_b / np.maximum(a_b, 1e-6), 0.0)
    out = np.concatenate([out_rgb, a_b * 255.0], axis=-1)
    return np.clip(out, 0, 255).astype(arr.dtype)


def _gaussian_fast(arr: np.ndarray, sigma: float, downscale: int) -> np.ndarray:
    """Downscale, blur, upscale. Nearly identical, considerably faster."""
    h, w = arr.shape[:2]
    d = max(1, int(downscale))
    sw, sh = max(1, w // d), max(1, h // d)
    small = Image.fromarray(arr, mode="RGBA").resize((sw, sh), Image.BILINEAR)
    small = small.filter(ImageFilter.GaussianBlur(radius=max(sigma / d, 0.1)))
    return np.asarray(small.resize((w, h), Image.BILINEAR), dtype=arr.dtype)


def gaussian_blur_rgba(
    arr: np.ndarray,
    sigma: float,
    quality: str | None = None,
    downscale: int | None = None,
) -> np.ndarray:
    """Gaussian-blur an ``(H, W, 4)`` uint8 RGBA array."""
    if sigma is None or sigma <= 0 or arr.size == 0:
        return arr
    quality = quality or blur_config.quality
    downscale = blur_config.fast_downscale if downscale is None else downscale
    if quality == "high" and _HAS_SCIPY:
        return _gaussian_high(arr, sigma)
    return _gaussian_fast(arr, sigma, downscale)


# =============================================================================
# SHAPE -> PIXEL MASK  (the foundation everything else rests on)
# =============================================================================
def _shape_mask(
    vmob: VMobject,
    camera: Camera,
    box: tuple[int, int, int, int],
    even_odd: bool = True,
) -> np.ndarray:
    """Build an ``(h, w, 1)`` float mask of the shape's interior.

    The shape's Bézier path is replayed into a Cairo surface, so any
    outline works -- circles, stars, hand-built splines, even shapes with
    holes such as ``Annulus`` or the counter of an "O" -- and the edge
    comes out properly anti-aliased.

    Parameters
    ----------
    box
        ``(x0, y0, x1, y1)`` pixel bounding box; only this is rasterised.
    even_odd
        Use the even-odd fill rule so holes stay holes.
    """
    x0, y0, x1, y1 = box
    w, h = x1 - x0, y1 - y0
    if w <= 0 or h <= 0:
        return np.zeros((max(h, 0), max(w, 0), 1), dtype=np.float32)

    surface = cairo.ImageSurface(cairo.FORMAT_A8, w, h)
    ctx = cairo.Context(surface)

    drew = False
    for sub in vmob.family_members_with_points():
        if not isinstance(sub, VMobject):
            continue
        for subpath in sub.get_subpaths():
            quads = sub.gen_cubic_bezier_tuples_from_points(subpath)
            if len(quads) == 0:
                continue
            start = camera.points_to_subpixel_coords(sub, quads[0][0].reshape(1, 3))[0]
            ctx.new_sub_path()
            ctx.move_to(start[0] - x0, start[1] - y0)
            for _a, b, c, d in quads:
                p = camera.points_to_subpixel_coords(sub, np.array([b, c, d]))
                ctx.curve_to(
                    p[0][0] - x0,
                    p[0][1] - y0,
                    p[1][0] - x0,
                    p[1][1] - y0,
                    p[2][0] - x0,
                    p[2][1] - y0,
                )
            ctx.close_path()
            drew = True

    if not drew:
        return np.zeros((h, w, 1), dtype=np.float32)

    ctx.set_fill_rule(cairo.FILL_RULE_EVEN_ODD if even_odd else cairo.FILL_RULE_WINDING)
    ctx.fill()
    surface.flush()

    buf = np.ndarray(
        shape=(h, surface.get_stride()), dtype=np.uint8, buffer=surface.get_data()
    )
    return (buf[:, :w].astype(np.float32) / 255.0)[..., None]


def _soften(mask: np.ndarray, feather: float) -> np.ndarray:
    """Optionally soften the mask edge."""
    if feather <= 0:
        return mask
    if _HAS_SCIPY:
        return _scipy_gaussian_filter(mask, sigma=(feather, feather, 0), mode="nearest")
    img = Image.fromarray((mask[..., 0] * 255).astype(np.uint8), mode="L")
    img = img.filter(ImageFilter.GaussianBlur(radius=feather))
    return (np.asarray(img, dtype=np.float32) / 255.0)[..., None]


# =============================================================================
# BASE CLASS
# =============================================================================
class _BlurBase(VMobject):
    """Shared base for :class:`Blur` and :class:`IMGBlur`.

    This is a genuine ``VMobject``: the input shape's stroke and outline
    are copied and drawn as usual. The only extra step is that the pixels
    inside the outline get blurred first.

    Subclasses differ in one method, :meth:`_source_pixels`:

    * :class:`Blur` -- the buffer as it stands in the current frame
    * :class:`IMGBlur` -- a snapshot taken the first time it rendered
    """

    def __init__(
        self,
        shape: VMobject | float | None = None,
        blur: float | None = None,
        *,
        c: float | None = None,
        amount: float | None = None,
        intensity: float = 1.0,
        feather: float = 0.0,
        tint: Any = None,
        tint_opacity: float = 0.0,
        copy_style: bool = True,
        quality: str | None = None,
        downscale: int | None = None,
        even_odd: bool = True,
        z_index: float | None = None,
        **kwargs: Any,
    ) -> None:
        # Legacy form: Blur(25) -- a number in the shape slot is the amount
        if isinstance(shape, (int, float)) and shape is not None:
            blur = float(shape) if blur is None else blur
            shape = None

        # blur / c / amount are accepted spellings of the same thing
        for alt in (c, amount):
            if blur is None and alt is not None:
                blur = alt
        self.blur_amount = blur if blur is not None else blur_config.default_blur

        self.intensity = float(intensity)
        self.mask_feather = float(feather)
        self.tint = ManimColor(tint) if tint is not None else None
        self.tint_opacity = float(tint_opacity)
        self.quality = quality
        self.downscale = downscale
        self.even_odd = bool(even_odd)
        self.enabled = True
        self._mask_cache: tuple[Any, np.ndarray] | None = None

        # ---- shape ----
        style_kwargs: dict[str, Any] = {}
        if shape is None:
            shape = self._default_shape(**kwargs)
        elif copy_style:
            # Copy the stroke. Fill is deliberately dropped: an opaque
            # interior would simply hide the blur. Pass fill_opacity=...
            # explicitly if a tinted fill is wanted.
            src = shape
            style_kwargs = {
                "stroke_color": src.get_stroke_color(),
                "stroke_width": src.get_stroke_width(),
                "stroke_opacity": src.get_stroke_opacity(),
                "fill_color": src.get_fill_color(),
                "fill_opacity": 0.0,
            }

        vm_kwargs = {k: v for k, v in kwargs.items() if k not in _SHAPE_ONLY_KEYS}
        super().__init__(**{**style_kwargs, **vm_kwargs})

        # Copy the geometry; the caller's mobject is left untouched
        self.match_shape(shape)

        # Remember the original opacity so fades can be measured
        try:
            self._base_stroke_opacity = float(self.get_stroke_opacity())
            self._base_fill_opacity = float(self.get_fill_opacity())
        except (AttributeError, TypeError, ValueError):
            self._base_stroke_opacity = None
            self._base_fill_opacity = None

        if z_index is not None:
            self.set_z_index(z_index)

    # ------------------------------------------------------------------
    def _default_shape(self, **kwargs: Any) -> VMobject:
        """Fallback shape when none is supplied; subclasses may override."""
        return Rectangle(
            width=kwargs.pop("width", config.frame_width),
            height=kwargs.pop("height", config.frame_height),
            stroke_width=0,
        )

    def match_shape(self, shape: VMobject) -> _BlurBase:
        """Adopt another shape's geometry, leaving styling alone."""
        self.set_points(shape.points.copy())
        self.submobjects = [s.copy() for s in shape.submobjects]
        for s in self.submobjects:
            s.set_opacity(0)  # submobjects feed the mask only, never drawn
        self._mask_cache = None
        return self

    # =====================================================================
    # parameters
    # =====================================================================
    def set_blur(self, v: float) -> _BlurBase:
        self.blur_amount = float(v)
        return self

    def get_blur(self) -> float:
        return self.blur_amount

    set_c = set_blur
    get_c = get_blur

    def set_intensity(self, v: float) -> _BlurBase:
        self.intensity = float(np.clip(v, 0.0, 1.0))
        return self

    def set_tint(self, color: Any, opacity: float | None = None) -> _BlurBase:
        self.tint = ManimColor(color) if color is not None else None
        if opacity is not None:
            self.tint_opacity = float(opacity)
        return self

    def set_tint_opacity(self, v: float) -> _BlurBase:
        self.tint_opacity = float(v)
        return self

    def set_feather(self, v: float) -> _BlurBase:
        self.mask_feather = float(v)
        return self

    def enable(self) -> _BlurBase:
        self.enabled = True
        return self

    def disable(self) -> _BlurBase:
        self.enabled = False
        return self

    # ------------------------------------------------- animation helpers
    def fade_in(self, run_time: float = 1.2, **kw: Any) -> Animation:
        """Ramp the blur up from 0 to its current amount."""
        target = self.blur_amount
        self.set_blur(0)
        return self.animate(run_time=run_time, **kw).set_blur(target).build()

    def fade_out(self, run_time: float = 1.2, **kw: Any) -> Animation:
        return self.animate(run_time=run_time, **kw).set_blur(0).build()

    def to(self, v: float, run_time: float = 1.2, **kw: Any) -> Animation:
        return self.animate(run_time=run_time, **kw).set_blur(v).build()

    # ------------------------------------------- Transform / .animate
    def interpolate_color(self, m1: _BlurBase, m2: _BlurBase, alpha: float):
        super().interpolate_color(m1, m2, alpha)
        lerp = lambda a, b: float(a + (b - a) * alpha)
        self.blur_amount = lerp(m1.blur_amount, m2.blur_amount)
        self.intensity = lerp(m1.intensity, m2.intensity)
        self.mask_feather = lerp(m1.mask_feather, m2.mask_feather)
        self.tint_opacity = lerp(m1.tint_opacity, m2.tint_opacity)
        self.tint = m2.tint if alpha > 0.5 else m1.tint
        self.enabled = m2.enabled if alpha > 0.5 else m1.enabled
        return self

    # =====================================================================
    # rendering
    # =====================================================================
    def _source_pixels(self, camera: Camera, pixel_array: np.ndarray) -> np.ndarray:
        """Which pixels to blur; decided by the subclass."""
        raise NotImplementedError

    def apply_to_pixel_array(self, camera: Camera, pixel_array: np.ndarray) -> None:
        """Called by the camera at this mobject's place in the z-order."""
        if not (self.enabled and blur_config.enabled):
            return

        # Make FadeIn/FadeOut behave: as the mobject's opacity drops the
        # blur weakens in step. A fade factor of 1 changes nothing.
        fade = self._fade_alpha()
        if fade <= 0.0:
            return

        sigma = max(0.0, self.blur_amount * camera.pixel_width / 1920.0)
        if sigma <= 0 and self.tint_opacity <= 0:
            return

        H, W = pixel_array.shape[:2]
        box = self._pixel_box(camera, W, H)
        if box is None:
            return
        x0, y0, x1, y1 = box

        source = self._source_pixels(camera, pixel_array)
        if source is None or source.shape[:2] != pixel_array.shape[:2]:
            source = pixel_array

        # Borrow pixels from outside the box, else the edge shows a halo
        pad = int(np.ceil(sigma * blur_config.padding_factor)) + 1
        px0, py0 = max(0, x0 - pad), max(0, y0 - pad)
        px1, py1 = min(W, x1 + pad), min(H, y1 + pad)

        blurred = gaussian_blur_rgba(
            source[py0:py1, px0:px1], sigma, self.quality, self.downscale
        )

        if self.tint is not None and self.tint_opacity > 0:
            t = np.array(self.tint.to_rgb(), dtype=np.float32) * 255.0
            k = float(np.clip(self.tint_opacity, 0.0, 1.0))
            b = blurred.astype(np.float32)
            b[..., :3] = b[..., :3] * (1 - k) + t[None, None, :] * k
            blurred = np.clip(b, 0, 255).astype(pixel_array.dtype)

        ox0, oy0 = x0 - px0, y0 - py0
        src = blurred[oy0 : oy0 + (y1 - y0), ox0 : ox0 + (x1 - x0)].astype(np.float32)
        dst = pixel_array[y0:y1, x0:x1].astype(np.float32)
        if src.shape != dst.shape:
            return

        mask = self._get_mask(camera, (x0, y0, x1, y1))
        if mask.shape[:2] != dst.shape[:2]:
            return
        mask = mask * float(np.clip(self.intensity, 0.0, 1.0)) * fade

        pixel_array[y0:y1, x0:x1] = np.clip(
            dst * (1.0 - mask) + src * mask, 0, 255
        ).astype(pixel_array.dtype)

    # ------------------------------------------------------------ helpers
    def _fade_alpha(self) -> float:
        """Let ``FadeIn``/``FadeOut`` dissolve the blur as well.

        Manim's fades animate stroke and fill opacity, so their ratio to
        the original value doubles as a blur strength multiplier. A shape
        with neither stroke nor fill gives nothing to measure, so the
        factor falls back to 1.0.
        """
        a = getattr(self, "_blur_fade", None)
        if a is not None:
            return float(np.clip(a, 0.0, 1.0))
        try:
            so = float(self.get_stroke_opacity())
            fo = float(self.get_fill_opacity())
        except (AttributeError, TypeError, ValueError):
            return 1.0
        base_s = getattr(self, "_base_stroke_opacity", None)
        base_f = getattr(self, "_base_fill_opacity", None)
        if base_s is None and base_f is None:
            return 1.0
        vals = []
        if base_s:
            vals.append(so / base_s)
        if base_f:
            vals.append(fo / base_f)
        return float(np.clip(max(vals) if vals else 1.0, 0.0, 1.0))

    def set_blur_fade(self, a: float) -> _BlurBase:
        """Set blur visibility directly, from 0 to 1."""
        self._blur_fade = float(np.clip(a, 0.0, 1.0))
        return self

    def _get_mask(self, camera: Camera, box: tuple[int, int, int, int]) -> np.ndarray:
        """The shape mask, reused while the shape stays put."""
        key = (
            box,
            self.mask_feather,
            self.even_odd,
            round(float(np.sum(self.points)), 4),
        )
        if self._mask_cache is not None and self._mask_cache[0] == key:
            return self._mask_cache[1]
        mask = _shape_mask(self, camera, box, self.even_odd)
        mask = _soften(mask, self.mask_feather)
        self._mask_cache = (key, mask)
        return mask

    def _pixel_box(
        self, camera: Camera, W: int, H: int
    ) -> tuple[int, int, int, int] | None:
        pts = self.get_all_points()
        if len(pts) == 0:
            return None
        p = camera.points_to_pixel_coords(self, pts)
        x0 = max(0, int(np.floor(p[:, 0].min())) - 1)
        y0 = max(0, int(np.floor(p[:, 1].min())) - 1)
        x1 = min(W, int(np.ceil(p[:, 0].max())) + 1)
        y1 = min(H, int(np.ceil(p[:, 1].max())) + 1)
        if x1 <= x0 or y1 <= y0:
            return None
        return x0, y0, x1, y1


_SHAPE_ONLY_KEYS = {"width", "height", "corner_radius", "side_length", "radius"}


# =============================================================================
# 1. Blur -- re-blurred every frame (true frosted glass)
# =============================================================================
class Blur(_BlurBase):
    """Blur the shape's interior on every frame -- true frosted glass.

    Parameters
    ----------
    shape
        Any ``VMobject`` -- Circle, Square, Rectangle, Star, Polygon, a
        hand-built Bézier, even a shape with holes. Only the geometry and
        styling are copied: **the input mobject is left untouched and keeps
        its own z_index**. Omit it to cover the whole frame.
    blur
        Blur strength; also accepted as ``c=`` or ``amount=``. Normalised
        against a 1920px-wide frame, so ``-ql`` previews and ``-qh``
        renders look the same.
    intensity
        0 to 1: how far to mix the blur over the original pixels.
    feather
        How many pixels of softening to apply to the mask edge.
    tint, tint_opacity
        A colour wash laid over the blur.
    copy_style
        Whether to copy the input shape's stroke (default ``True``).

    Examples
    --------
    ::

        rec = Rectangle(width=6, height=3)
        glass = Blur(rec)
        glass.set_z_index(1)
        self.add(glass)

        circle = Circle().set_z_index(0)
        self.play(circle.animate.shift(LEFT * 4))   # passes by blurred
        self.play(glass.animate.shift(UP))          # the pane moves too
    """

    def _source_pixels(self, camera: Camera, pixel_array: np.ndarray) -> np.ndarray:
        return pixel_array  # the current frame, hence live


# =============================================================================
# 2. IMGBlur -- snapshot (static)
# =============================================================================
class IMGBlur(_BlurBase):
    """Blur once, then keep that image no matter what moves underneath.

    The only difference from :class:`Blur` is that the blur is not
    recomputed each frame. The first time it renders, the screen is stored
    internally and reused from then on.

    Whatever happens below therefore leaves the interior unchanged -- the
    pane simply has no idea anything is going on.

    The stored image stays **fixed relative to the screen**, like sliding a
    window across a painted wall. Pass ``frozen_content=True`` to carry the
    image along with the pane instead.

    Parameters
    ----------
    frozen_content
        ``False`` (default) keeps the image fixed to the screen.
        ``True`` moves it with the pane, like a photo in a frame.

    Examples
    --------
    ::

        glass = IMGBlur(rec)
        self.add(glass)
        self.play(glass.animate.shift(LEFT * 3))   # interior stays put
        glass.recapture()                          # grab a fresh snapshot
    """

    def __init__(self, *args: Any, frozen_content: bool = False, **kwargs: Any) -> None:
        self._snapshot: np.ndarray | None = None
        self._snapshot_center: np.ndarray | None = None
        self.frozen_content = bool(frozen_content)
        super().__init__(*args, **kwargs)

    # ------------------------------------------------------------------
    def recapture(self) -> IMGBlur:
        """Take a fresh snapshot on the next frame."""
        self._snapshot = None
        self._snapshot_center = None
        return self

    capture = recapture

    def has_snapshot(self) -> bool:
        return self._snapshot is not None

    # ------------------------------------------------------------------
    def _source_pixels(self, camera: Camera, pixel_array: np.ndarray) -> np.ndarray:
        if self._snapshot is None or self._snapshot.shape != pixel_array.shape:
            # First time through: keep what is on screen right now
            self._snapshot = pixel_array.copy()
            self._snapshot_center = self.get_center().copy()

        if not self.frozen_content:
            return self._snapshot  # fixed to the screen

        # frozen_content: shift the stored image along with the pane
        shift = self.get_center() - self._snapshot_center
        if np.allclose(shift, 0):
            return self._snapshot
        dx = round(shift[0] * camera.pixel_width / camera.frame_width)
        dy = round(-shift[1] * camera.pixel_height / camera.frame_height)
        return np.roll(self._snapshot, (dy, dx), axis=(0, 1))


# =============================================================================
# 3. CameraBlur -- the whole frame (live only)
# =============================================================================
class CameraBlur(Blur):
    """Blur exactly as much as the camera can see.

    Adapts on its own when the camera zooms or pans. There is deliberately
    no ``IMG`` counterpart: this one is always live.

    Examples
    --------
    ::

        self.add(CameraBlur(30))
        self.add(title)              # stays sharp
    """

    def __init__(self, blur: float | None = None, **kwargs: Any) -> None:
        kwargs.setdefault("stroke_width", 0)
        kwargs.setdefault("fill_opacity", 0)
        super().__init__(None, blur, **kwargs)
        self._follow_camera = True

    def _pixel_box(self, camera: Camera, W: int, H: int):
        return 0, 0, W, H

    def _get_mask(self, camera: Camera, box: tuple[int, int, int, int]) -> np.ndarray:
        # Whole frame, so there is no outline to rasterise
        h, w = box[3] - box[1], box[2] - box[0]
        return np.ones((h, w, 1), dtype=np.float32)


# =============================================================================
# 4 and 5. Cards -- the floating glass board look
# =============================================================================
def _card_shape(
    width: float, height: float, corner_radius: float, **style: Any
) -> VMobject:
    if corner_radius and corner_radius > 0:
        return RoundedRectangle(
            corner_radius=corner_radius, width=width, height=height, **style
        )
    return Rectangle(width=width, height=height, **style)


class _CardMixin:
    """Shared appearance for :class:`BlurCard` and :class:`IMGBlurCard`."""

    def __init__(
        self,
        shape: VMobject | float | None = None,
        blur: float | None = None,
        *,
        width: float = 8.0,
        height: float = 4.5,
        corner_radius: float = 0.45,
        tint: Any = WHITE,
        tint_opacity: float = 0.18,
        border_color: Any = WHITE,
        border_width: float = 2.5,
        border_opacity: float = 0.55,
        match_size: bool = False,
        **kwargs: Any,
    ) -> None:
        """A card: a rounded rectangle by default, any shape on request.

        Parameters
        ----------
        shape
            Optional ``VMobject`` -- Star, Circle, Polygon, Bézier, as you
            like. Omit it and a rounded rectangle is built from ``width``,
            ``height`` and ``corner_radius``. A number here is read as
            ``blur``, so ``BlurCard(35)`` works.
        match_size
            ``True`` stretches the supplied shape to ``width`` x
            ``height``. Default ``False`` keeps the shape's own size.
        """
        # BlurCard(35): a number in the shape slot means the blur amount
        if isinstance(shape, (int, float)):
            blur = float(shape) if blur is None else blur
            shape = None

        if shape is None:
            shape = _card_shape(width, height, corner_radius)
        else:
            shape = shape.copy()
            if match_size:
                shape.stretch_to_fit_width(width).stretch_to_fit_height(height)

        # The card's own border styling always wins
        shape.set_stroke(
            color=ManimColor(border_color),
            width=border_width,
            opacity=border_opacity,
        )
        shape.set_fill(opacity=0)

        kwargs.setdefault("feather", 1.2)  # take the hard edge off
        super().__init__(
            shape,
            blur if blur is not None else 30.0,
            tint=tint,
            tint_opacity=tint_opacity,
            **kwargs,
        )

    # convenience
    def set_border(
        self,
        color: Any = None,
        width: float | None = None,
        opacity: float | None = None,
    ):
        if color is not None:
            self.set_stroke(color=ManimColor(color))
        if width is not None:
            self.set_stroke(width=width)
        if opacity is not None:
            self.set_stroke(opacity=opacity)
        return self


class BlurCard(_CardMixin, Blur):
    """Website-style floating glass card, blurred **live**.

    Examples
    --------
    ::

        card = BlurCard(width=8, height=4.5)
        card.set_z_index(1)
        self.add(card)
        self.add(Text("Hello").set_z_index(2))
    """


class IMGBlurCard(_CardMixin, IMGBlur):
    """The same card, blurred **once** and held as a snapshot."""


# =============================================================================
# CAMERA HOOK -- patched once, works in every Scene
# =============================================================================
def _install() -> None:
    """Teach ``Camera.capture_mobjects`` about blur layers.

    Every camera subclass ends up calling ``Camera.capture_mobjects``, so
    patching it once covers ``Scene``, ``MovingCameraScene``,
    ``ThreeDScene`` and the rest.
    """
    if getattr(Camera, "_blur_installed", False):
        return

    original = Camera.capture_mobjects

    def capture_mobjects(self, mobjects: Iterable[Mobject], **kwargs: Any) -> None:
        mobs = self.get_mobjects_to_display(mobjects, **kwargs)
        if not any(isinstance(m, _BlurBase) for m in mobs):
            return original(self, mobjects, **kwargs)

        def kind(m: Mobject) -> Any:
            return "blur" if isinstance(m, _BlurBase) else self.type_or_raise(m)

        for group_type, group in it.groupby(mobs, kind):
            batch = list(group)
            if group_type == "blur":
                for b in batch:
                    # 1) blur the interior
                    b.apply_to_pixel_array(self, self.pixel_array)
                    # 2) then draw the shape's own stroke and fill
                    if b.get_stroke_width() > 0 or b.get_fill_opacity() > 0:
                        self.display_multiple_vectorized_mobjects([b], self.pixel_array)
            else:
                self.display_funcs[group_type](batch, self.pixel_array)

    Camera.capture_mobjects = capture_mobjects
    Camera._blur_installed = True


_install()


# =============================================================================
# PRESETS
# =============================================================================
def _install_presets(cls: type) -> type:
    """Attach the shared look-presets to a blur class."""

    @classmethod
    def glass(c, blur_amount: float = 30, **kw: Any):
        """Frosted glass: soft edge, faint white wash."""
        kw.setdefault("tint", WHITE)
        kw.setdefault("tint_opacity", 0.22)
        kw.setdefault("feather", 1.2)
        return c(kw.pop("shape", None), blur_amount, **kw)

    @classmethod
    def dark(c, blur_amount: float = 25, **kw: Any):
        """Dark overlay -- a good bed for light text."""
        kw.setdefault("tint", BLACK)
        kw.setdefault("tint_opacity", 0.45)
        return c(kw.pop("shape", None), blur_amount, **kw)

    @classmethod
    def subtle(c, blur_amount: float = 8, **kw: Any):
        """A gentle blur, enough to suggest depth."""
        return c(kw.pop("shape", None), blur_amount, **kw)

    @classmethod
    def heavy(c, blur_amount: float = 55, **kw: Any):
        """A heavy blur; the content behind is barely readable."""
        return c(kw.pop("shape", None), blur_amount, **kw)

    cls.glass = glass
    cls.dark = dark
    cls.subtle = subtle
    cls.heavy = heavy
    return cls


_install_presets(Blur)
_install_presets(IMGBlur)
