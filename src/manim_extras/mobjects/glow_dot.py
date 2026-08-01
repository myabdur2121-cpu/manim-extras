"""Glowing dots for Manim Community, ported from 3Blue1Brown's ManimGL.

ManimGL renders glow dots on the GPU. Its ``true_dot`` fragment shader reduces
to two lines::

    float r = length(uv_coords.xy);
    if (r > 1.0) discard;
    if (glow_factor > 0) frag_color.a *= pow(1 - r, glow_factor);

So the alpha of a glow dot is ``(1 - r) ** glow_factor`` inside the unit disc
and zero outside it. Manim Community's default renderer is Cairo, which has no
shaders, so that falloff is reproduced on the CPU instead.

Two backends are available, chosen with ``render_mode``:

``"raster"`` (default)
    Evaluates the falloff per pixel into an RGBA image. This matches the
    shader exactly, including how overlapping dots blend.
``"vector"``
    Stacks concentric filled circles. Stays sharp at any zoom and exports
    cleanly to SVG, but the gradient is quantised into ``num_layers`` steps.

The ManimGL API is kept: ``DotCloud``, ``GlowDots``, ``GlowDot`` and
``TrueDot``, with ``set_radius``, ``set_radii``, ``scale_radii``,
``set_glow_factor``, ``to_grid``, ``make_3d`` and a ``scale`` that also scales
the radii.

Two things ManimGL does not offer are added here, both off by default:
a custom ``falloff`` function, and a ``core_color`` for a hot centre.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np
from manim import (
    GREY_C,
    ORIGIN,
    YELLOW,
    Circle,
    Group,
    ImageMobject,
    Mobject,
    VGroup,
)
from manim.utils.color import ManimColor

__all__ = [
    "DEFAULT_DOT_RADIUS",
    "DEFAULT_GLOW_DOT_RADIUS",
    "DEFAULT_CANVAS_SIZE",
    "DEFAULT_GRID_HEIGHT",
    "RENDER_MODES",
    "DotCloud",
    "GlowDot",
    "GlowDots",
    "TrueDot",
]

# Matching manimlib/mobject/types/dot_cloud.py
DEFAULT_DOT_RADIUS = 0.05
DEFAULT_GLOW_DOT_RADIUS = 0.2
DEFAULT_GRID_HEIGHT = 6
DEFAULT_BUFF_RATIO = 0.5

RENDER_MODES = ("raster", "vector")

# Resolution of the raster canvas along its longest side. Fixed rather than
# derived from the radius, so a scaled cloud keeps the same pixel dimensions
# and remains interpolatable by Manim's animations.
DEFAULT_CANVAS_SIZE = 512

# A raster canvas is never allowed to exceed this on either side, so a huge
# radius cannot allocate a gigabyte of pixels.
MAX_CANVAS_PIXELS = 2048

# Raster canvases are rounded up to a multiple of this, so that a scaled cloud
# keeps the same pixel dimensions and stays interpolatable.
RASTER_SIZE_STEP = 128

# Layers used by the vector backend.
DEFAULT_NUM_LAYERS = 60


def _as_rgb(color) -> np.ndarray:
    """Colour to a float RGB triple in 0-1."""
    return np.array(ManimColor(color).to_rgb(), dtype=float)


def _resize_rgba(array: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    """Nearest-neighbour resize of an RGBA array, for blending two canvases."""
    rows = np.linspace(0, array.shape[0] - 1, shape[0]).round().astype(int)
    cols = np.linspace(0, array.shape[1] - 1, shape[1]).round().astype(int)
    return array[np.ix_(rows, cols)]


def _default_falloff(r: np.ndarray, glow_factor: float) -> np.ndarray:
    """The ManimGL shader falloff: ``(1 - r) ** glow_factor``.

    ``glow_factor=0`` skips the branch in the shader, giving a solid disc.
    """
    if glow_factor <= 0:
        return np.ones_like(r)
    return np.power(np.clip(1.0 - r, 0.0, 1.0), glow_factor)


class DotCloud(Group):
    """A cloud of dots, each with its own position, radius and colour.

    Parameters
    ----------
    points
        Dot centres, shape ``(n, 3)``. Defaults to a single dot at the origin.
    color
        Dot colour. A single colour, or one per dot.
    opacity
        Peak opacity at the centre of each dot.
    radius
        Dot radius in scene units. Use ``set_radii`` for per-dot radii.
    glow_factor
        Sharpness of the falloff. ``0`` is a solid disc; larger values
        concentrate the light in the middle. ManimGL's ``GlowDot`` uses ``2``.
    render_mode
        ``"raster"`` for a per-pixel image, ``"vector"`` for stacked circles.
    falloff
        Optional ``falloff(r, glow_factor) -> alpha`` where ``r`` is 0 at the
        centre and 1 at the rim. Overrides the default formula, which lets you
        use a Gaussian or any other profile.
    core_color
        Optional colour at the very centre, blending out to ``color`` at the
        rim. ``None`` keeps a single flat colour, as ManimGL does.
    core_size
        How far the core colour reaches, 0-1 of the radius.
    canvas_size
        Raster resolution along the longest side. Ignored in vector mode.
    num_layers
        Circles per dot in vector mode. Ignored in raster mode.
    anti_alias_width
        Width in pixels of the edge fade, matching ManimGL's uniform of the
        same name. Raster mode only.
    """

    def __init__(
        self,
        points: Sequence | np.ndarray | None = None,
        color=GREY_C,
        opacity: float = 1.0,
        radius: float = DEFAULT_DOT_RADIUS,
        glow_factor: float = 0.0,
        render_mode: str = "raster",
        falloff: Callable[[np.ndarray, float], np.ndarray] | None = None,
        core_color=None,
        core_size: float = 0.35,
        canvas_size: int = DEFAULT_CANVAS_SIZE,
        num_layers: int = DEFAULT_NUM_LAYERS,
        anti_alias_width: float = 2.0,
        **kwargs,
    ) -> None:
        if render_mode not in RENDER_MODES:
            raise ValueError(f"render_mode must be one of {RENDER_MODES}, got {render_mode!r}.")
        if radius < 0:
            raise ValueError(f"radius must be non-negative, got {radius}.")
        if not 0.0 <= opacity <= 1.0:
            raise ValueError(f"opacity must lie in [0, 1], got {opacity}.")
        if glow_factor < 0:
            raise ValueError(f"glow_factor must be non-negative, got {glow_factor}.")
        if not 0.0 <= core_size <= 1.0:
            raise ValueError(f"core_size must lie in [0, 1], got {core_size}.")
        if canvas_size < RASTER_SIZE_STEP:
            raise ValueError(f"canvas_size must be at least {RASTER_SIZE_STEP}, got {canvas_size}.")
        if num_layers < 1:
            raise ValueError(f"num_layers must be at least 1, got {num_layers}.")
        if anti_alias_width < 0:
            raise ValueError(f"anti_alias_width must be non-negative, got {anti_alias_width}.")

        super().__init__(**kwargs)

        self.render_mode = render_mode
        self.falloff = falloff
        self.core_color = core_color
        self.core_size = core_size
        self.canvas_size = canvas_size
        self.num_layers = num_layers
        self.anti_alias_width = anti_alias_width
        self.glow_factor = glow_factor
        self.opacity = opacity

        self.points = self._clean_points(points)
        n = len(self.points)
        self.radii = np.full(n, float(radius))
        self.colors = self._clean_colors(color, n)

        self._rebuild()

    # ------------------------------------------------------------------
    # input handling
    # ------------------------------------------------------------------
    @staticmethod
    def _clean_points(points) -> np.ndarray:
        if points is None:
            return np.array([ORIGIN], dtype=float)
        array = np.asarray(points, dtype=float)
        if array.ndim == 1:
            array = array.reshape(1, -1)
        if array.ndim != 2 or array.shape[1] not in (2, 3):
            raise ValueError(f"points must have shape (n, 2) or (n, 3), got {array.shape}.")
        if array.shape[1] == 2:
            array = np.column_stack([array, np.zeros(len(array))])
        if len(array) == 0:
            raise ValueError("points cannot be empty.")
        return array

    @staticmethod
    def _clean_colors(color, n: int) -> list:
        if isinstance(color, (list, tuple, np.ndarray)) and not isinstance(color, str):
            colors = list(color)
            if not colors:
                raise ValueError("color cannot be an empty sequence.")
            return [colors[i % len(colors)] for i in range(n)]
        return [color] * n

    # ------------------------------------------------------------------
    # building
    # ------------------------------------------------------------------
    def _alpha_at(self, r: np.ndarray) -> np.ndarray:
        """Alpha profile over normalised radius ``r``, clipped to the disc."""
        if self.falloff is not None:
            alpha = np.asarray(self.falloff(r, self.glow_factor), dtype=float)
        else:
            alpha = _default_falloff(r, self.glow_factor)
        return np.clip(np.where(r > 1.0, 0.0, alpha), 0.0, 1.0) * self.opacity

    def _rgb_at(self, r: np.ndarray, color) -> np.ndarray:
        """RGB over normalised radius, blending the core colour if set."""
        rim = _as_rgb(color)
        if self.core_color is None:
            return np.broadcast_to(rim, (*r.shape, 3)).copy()
        core = _as_rgb(self.core_color)
        if self.core_size <= 0:
            mix = np.ones_like(r)
        else:
            mix = np.clip(r / self.core_size, 0.0, 1.0)
        return core + (rim - core) * mix[..., None]

    def _rebuild(self) -> None:
        """Regenerate every submobject from the current data."""
        self.remove(*self.submobjects)
        if self.render_mode == "raster":
            built = self._build_raster()
        else:
            built = self._build_vector()
        if built is not None:
            self.add(built)

    def _build_raster(self) -> Mobject | None:
        """One RGBA image covering every dot, alpha-composited."""
        radii = np.maximum(self.radii, 1e-9)
        low = (self.points[:, :2] - radii[:, None]).min(axis=0)
        high = (self.points[:, :2] + radii[:, None]).max(axis=0)
        span = np.maximum(high - low, 1e-6)

        # The canvas resolution is fixed by the cloud's aspect ratio, not by
        # its size in scene units. Manim's ImageMobject.interpolate_color
        # asserts that two pixel arrays share a shape, so a canvas that grew
        # with the radius would break FadeIn(scale=...), Transform and
        # .animate. Keeping it size-independent means a scaled copy still
        # interpolates cleanly; the image is stretched to the right dimensions
        # afterwards.
        aspect = span[0] / span[1]
        if aspect >= 1:
            width_px = self.canvas_size
            height_px = max(RASTER_SIZE_STEP, int(round(self.canvas_size / aspect)))
        else:
            height_px = self.canvas_size
            width_px = max(RASTER_SIZE_STEP, int(round(self.canvas_size * aspect)))
        width_px = int(np.clip(width_px, RASTER_SIZE_STEP, MAX_CANVAS_PIXELS))
        height_px = int(np.clip(height_px, RASTER_SIZE_STEP, MAX_CANVAS_PIXELS))

        # Effective sampling density, used for the anti-aliased edge.
        px_per_unit = width_px / span[0]

        # Pixel centres in scene coordinates. Rows run top to bottom.
        xs = low[0] + (np.arange(width_px) + 0.5) * span[0] / width_px
        ys = high[1] - (np.arange(height_px) + 0.5) * span[1] / height_px
        grid_x, grid_y = np.meshgrid(xs, ys)

        accum_rgb = np.zeros((height_px, width_px, 3))
        accum_a = np.zeros((height_px, width_px))

        for centre, radius, color in zip(self.points, self.radii, self.colors):
            if radius <= 0:
                continue
            r = np.sqrt((grid_x - centre[0]) ** 2 + (grid_y - centre[1]) ** 2) / radius
            alpha = self._alpha_at(r)

            # Match the shader's final smoothstep edge fade.
            if self.anti_alias_width > 0:
                edge = self.anti_alias_width / max(radius * px_per_unit, 1e-9)
                if edge > 0:
                    t = np.clip((1.0 - r) / max(edge, 1e-9), 0.0, 1.0)
                    alpha = alpha * (t * t * (3.0 - 2.0 * t))

            if not alpha.any():
                continue

            rgb = self._rgb_at(r, color)
            # "Over" compositing, so overlapping dots blend like the shader.
            out_a = alpha + accum_a * (1.0 - alpha)
            safe = np.where(out_a > 0, out_a, 1.0)
            accum_rgb = (
                rgb * alpha[..., None] + accum_rgb * accum_a[..., None] * (1.0 - alpha[..., None])
            ) / safe[..., None]
            accum_a = out_a

        rgba = np.zeros((height_px, width_px, 4), dtype=np.uint8)
        rgba[..., :3] = np.clip(accum_rgb * 255, 0, 255).astype(np.uint8)
        rgba[..., 3] = np.clip(accum_a * 255, 0, 255).astype(np.uint8)

        image = ImageMobject(rgba)
        image.stretch_to_fit_width(span[0])
        image.stretch_to_fit_height(span[1])
        image.move_to(np.array([*(low + span / 2), 0.0]))
        return image

    def _build_vector(self) -> Mobject | None:
        """Concentric filled circles per dot, outermost first."""
        group = VGroup()
        for centre, radius, color in zip(self.points, self.radii, self.colors):
            if radius <= 0:
                continue
            # Circles are drawn outermost first and painted over each other,
            # so the visible alpha at any point is the "over" composite of
            # every disc covering it. Giving each layer its target alpha would
            # accumulate to 1 in the middle. Instead solve for the per-layer
            # alpha that makes the running composite equal the target:
            #     target = layer + running * (1 - layer)
            # =>  layer  = (target - running) / (1 - running)
            edges = np.linspace(1.0, 0.0, self.num_layers + 1)[:-1]
            mids = np.clip(edges - 0.5 / self.num_layers, 0.0, 1.0)
            targets = self._alpha_at(mids)
            rgbs = self._rgb_at(mids, color)

            running = 0.0
            for edge, target, rgb in zip(edges, targets, rgbs):
                if edge <= 0:
                    continue
                if running >= 1.0:
                    break
                layer_alpha = (target - running) / (1.0 - running)
                layer_alpha = float(np.clip(layer_alpha, 0.0, 1.0))
                if layer_alpha <= 1e-6:
                    continue
                group.add(
                    Circle(
                        radius=edge * radius,
                        stroke_width=0,
                        fill_color=ManimColor.from_rgb(rgb),
                        fill_opacity=layer_alpha,
                    ).move_to(centre)
                )
                running = target
        return group if len(group) else None

    # ------------------------------------------------------------------
    # ManimGL-compatible API
    # ------------------------------------------------------------------
    def set_points(self, points) -> DotCloud:
        """Replace the dot centres, keeping radii and colours in step."""
        new_points = self._clean_points(points)
        n = len(new_points)
        old = len(self.points)
        if n != old:
            self.radii = np.resize(self.radii, n)
            self.colors = [self.colors[i % len(self.colors)] for i in range(n)]
        self.points = new_points
        self._rebuild()
        return self

    def get_points(self) -> np.ndarray:
        return self.points

    def set_radius(self, radius: float) -> DotCloud:
        """Give every dot the same radius."""
        if radius < 0:
            raise ValueError(f"radius must be non-negative, got {radius}.")
        self.radii = np.full(len(self.points), float(radius))
        self._rebuild()
        return self

    def get_radius(self) -> float:
        """The largest radius, matching ManimGL."""
        return float(self.radii.max())

    def set_radii(self, radii) -> DotCloud:
        """Set a radius per dot."""
        values = np.asarray(radii, dtype=float).flatten()
        if values.size == 0:
            raise ValueError("radii cannot be empty.")
        if (values < 0).any():
            raise ValueError("radii must be non-negative.")
        self.radii = np.resize(values, len(self.points))
        self._rebuild()
        return self

    def get_radii(self) -> np.ndarray:
        return self.radii

    def scale_radii(self, scale_factor: float) -> DotCloud:
        return self.set_radii(self.radii * scale_factor)

    def set_glow_factor(self, glow_factor: float) -> DotCloud:
        if glow_factor < 0:
            raise ValueError(f"glow_factor must be non-negative, got {glow_factor}.")
        self.glow_factor = glow_factor
        self._rebuild()
        return self

    def get_glow_factor(self) -> float:
        return self.glow_factor

    def set_color(self, color, **kwargs) -> DotCloud:
        """Recolour every dot, or give one colour per dot."""
        self.colors = self._clean_colors(color, len(self.points))
        self._rebuild()
        return self

    def set_opacity(self, opacity: float, **kwargs) -> DotCloud:
        if not 0.0 <= opacity <= 1.0:
            raise ValueError(f"opacity must lie in [0, 1], got {opacity}.")
        self.opacity = opacity
        self._rebuild()
        return self

    def scale(self, scale_factor, scale_radii: bool = True, **kwargs) -> DotCloud:
        """Scale positions, and by default the radii too."""
        about_point = kwargs.get("about_point")
        if about_point is None:
            about_point = self.get_center()
        self.points = about_point + (self.points - about_point) * scale_factor
        if scale_radii:
            self.radii = self.radii * scale_factor
        self._rebuild()
        return self

    def to_grid(
        self,
        n_rows: int,
        n_cols: int,
        n_layers: int = 1,
        buff_ratio: float | None = None,
        h_buff_ratio: float = 1.0,
        v_buff_ratio: float = 1.0,
        d_buff_ratio: float = 1.0,
        height: float | None = DEFAULT_GRID_HEIGHT,
    ) -> DotCloud:
        """Arrange the dots in a grid, as ManimGL's ``to_grid`` does."""
        if min(n_rows, n_cols, n_layers) < 1:
            raise ValueError("n_rows, n_cols and n_layers must all be at least 1.")

        n_points = n_rows * n_cols * n_layers
        idx = np.arange(n_points)
        points = np.zeros((n_points, 3))
        points[:, 0] = idx % n_cols
        points[:, 1] = (idx // n_cols) % n_rows
        points[:, 2] = idx // (n_rows * n_cols)

        if buff_ratio is not None:
            h_buff_ratio = v_buff_ratio = d_buff_ratio = buff_ratio

        radius = self.get_radius()
        spacing = [2 * radius * (1 + br) for br in (h_buff_ratio, v_buff_ratio, d_buff_ratio)]
        points *= np.array(spacing)
        points -= points.mean(axis=0)

        self.set_points(points)
        self.set_radius(radius)
        if height is not None:
            # ManimGL measures the grid with the radii zeroed, so the target
            # height describes the spread of the dot centres, not the outer
            # edge of the glow.
            span = points[:, 1].max() - points[:, 1].min()
            if span > 0:
                factor = height / span
                self.points = self.points * factor
                self._rebuild()
        return self

    def interpolate_color(self, mobject1, mobject2, alpha) -> DotCloud:
        """Blend colours between two clouds.

        ``Group`` inherits ``Mobject.interpolate_color``, which is an abstract
        stub, so without this any animation that interpolates -- ``FadeIn``,
        ``Transform``, ``.animate`` -- raises ``NotImplementedError``.
        Delegating to the submobjects makes those work.
        """
        for child, m1, m2 in zip(self.submobjects, mobject1.submobjects, mobject2.submobjects):
            if not hasattr(child, "interpolate_color"):
                continue
            if isinstance(child, ImageMobject):
                # Manim asserts both pixel arrays share a shape. Two clouds of
                # different radius produce different canvases, so blend the
                # arrays directly against a common size instead.
                a1 = m1.pixel_array.astype(float)
                a2 = m2.pixel_array.astype(float)
                if a1.shape != a2.shape:
                    a2 = _resize_rgba(a2, a1.shape[:2])
                blended = (1 - alpha) * a1 + alpha * a2
                child.pixel_array = np.clip(blended, 0, 255).astype(np.uint8)
            else:
                child.interpolate_color(m1, m2, alpha)
        return self

    def fade(self, darkness: float = 0.5, family: bool = True) -> DotCloud:
        """Fade the whole cloud, keeping ``Group`` animations working."""
        for child in self.submobjects:
            child.fade(darkness, family=family)
        return self

    def make_3d(
        self, reflectiveness: float = 0.5, gloss: float = 0.1, shadow: float = 0.2
    ) -> DotCloud:
        """Present for API parity with ManimGL.

        The Cairo renderer has no lighting model, so this is a no-op beyond
        recording the values.
        """
        self.reflectiveness = reflectiveness
        self.gloss = gloss
        self.shadow = shadow
        return self


class GlowDots(DotCloud):
    """Several glowing dots. ManimGL defaults: yellow, radius 0.2, glow 2."""

    def __init__(
        self,
        points=None,
        color=YELLOW,
        radius: float = DEFAULT_GLOW_DOT_RADIUS,
        glow_factor: float = 2.0,
        **kwargs,
    ) -> None:
        super().__init__(points, color=color, radius=radius, glow_factor=glow_factor, **kwargs)


class GlowDot(GlowDots):
    """A single glowing dot.

    ::

        GlowDot()                                  # yellow, radius 0.2
        GlowDot(LEFT * 2, color=BLUE, radius=0.5)
        GlowDot(glow_factor=4)                     # tighter, brighter core
        GlowDot(core_color=WHITE)                  # hot white centre
    """

    def __init__(self, center=ORIGIN, **kwargs) -> None:
        super().__init__(points=np.array([center], dtype=float), **kwargs)


class TrueDot(DotCloud):
    """A crisp, solid dot: ``glow_factor=0``, the ManimGL default radius."""

    def __init__(self, center=ORIGIN, radius: float = DEFAULT_DOT_RADIUS, **kwargs) -> None:
        super().__init__(points=np.array([center], dtype=float), radius=radius, **kwargs)
