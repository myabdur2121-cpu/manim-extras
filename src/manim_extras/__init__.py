"""manim_extras -- a collection of reusable Manim components."""

from __future__ import annotations

from .animations import ParticleStream, StreamAlongPath
from .mobjects import (
    Blur,
    BlurCard,
    CameraBlur,
    IMGBlur,
    IMGBlurCard,
    SmoothPolygon,
    blur_config,
    gaussian_blur_rgba,
)

__version__ = "0.1.0"

__all__ = [
    "Blur",
    "BlurCard",
    "CameraBlur",
    "IMGBlur",
    "IMGBlurCard",
    "ParticleStream",
    "SmoothPolygon",
    "StreamAlongPath",
    "__version__",
    "blur_config",
    "gaussian_blur_rgba",
]
