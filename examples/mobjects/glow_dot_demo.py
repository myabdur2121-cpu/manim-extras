"""Visual demos for GlowDot and the DotCloud family.

manim -ql examples/mobjects/glow_dot_demo.py GlowDotDemo
manim -ql examples/mobjects/glow_dot_demo.py GlowFactorDemo
manim -ql examples/mobjects/glow_dot_demo.py CoreColorDemo
manim -ql examples/mobjects/glow_dot_demo.py FalloffDemo
manim -ql examples/mobjects/glow_dot_demo.py RenderModeDemo
manim -ql examples/mobjects/glow_dot_demo.py DotCloudDemo
manim -ql examples/mobjects/glow_dot_demo.py StarfieldDemo
manim -ql examples/mobjects/glow_dot_demo.py OrbitDemo
"""

import numpy as np
from manim import *

from manim_extras import GlowDot, GlowDots, TrueDot


def lanes(labels, top=1.35, gap=0.95, font_size=18):
    """A row of captions under evenly spaced slots."""
    tags = VGroup()
    for i, (text, colour) in enumerate(labels):
        tag = Text(text, font_size=font_size, color=colour)
        tag.move_to([-5.6 + i * (11.2 / max(len(labels) - 1, 1)), -top - gap, 0])
        tags.add(tag)
    return tags


class GlowDotDemo(Scene):
    """The short version: a glow, a crisp dot, and a small cloud."""

    def construct(self):
        self.camera.background_color = BLACK
        title = Text("GlowDot", font_size=40).to_edge(UP)
        self.play(Write(title))

        glow = GlowDot(LEFT * 3.5, radius=1.0, color=YELLOW)
        crisp = TrueDot(ORIGIN, radius=0.12, color=WHITE)
        cored = GlowDot(RIGHT * 3.5, radius=1.0, color=BLUE, core_color=WHITE)

        labels = VGroup(
            Text("GlowDot", font_size=18).move_to([-3.5, -1.9, 0]),
            Text("TrueDot", font_size=18).move_to([0, -1.9, 0]),
            Text("core_color=WHITE", font_size=18).move_to([3.5, -1.9, 0]),
        )

        self.play(
            LaggedStart(
                FadeIn(glow, scale=0.6),
                FadeIn(crisp, scale=0.6),
                FadeIn(cored, scale=0.6),
                lag_ratio=0.25,
            ),
            FadeIn(labels),
            run_time=2.4,
        )
        self.wait(1.0)
        self.play(glow.animate.shift(UP * 0.8), cored.animate.shift(UP * 0.8), run_time=1.2)
        self.wait(0.8)


class GlowFactorDemo(Scene):
    """alpha = (1 - r) ** glow_factor, the ManimGL shader formula."""

    def construct(self):
        self.camera.background_color = BLACK
        self.add(Text("glow_factor", font_size=34).to_edge(UP))
        self.add(Text("(1 - r) ** glow_factor", font_size=20, color=GREY_B).shift(UP * 2.1))

        factors = [0, 0.5, 1, 2, 4, 8]
        dots = Group()
        for i, gf in enumerate(factors):
            x = -5.6 + i * 2.24
            dots.add(GlowDot([x, 0.35, 0], glow_factor=gf, radius=0.82, color=YELLOW))
        self.add(dots, lanes([(str(gf), GREY_A) for gf in factors], top=1.0, gap=0.7))
        self.wait(0.5)


class CoreColorDemo(Scene):
    """A hot centre blending out to the rim colour."""

    def construct(self):
        self.camera.background_color = BLACK
        self.add(Text("core_color / core_size", font_size=32).to_edge(UP))

        specs = [
            ("plain", {}, BLUE),
            ("core_color=WHITE", {"core_color": WHITE}, BLUE),
            ("core_size=0.6", {"core_color": WHITE, "core_size": 0.6}, BLUE),
            ("yellow core", {"core_color": YELLOW, "core_size": 0.4}, RED),
        ]
        dots = Group()
        for i, (_, kwargs, colour) in enumerate(specs):
            x = -4.9 + i * 3.3
            dots.add(GlowDot([x, 0.4, 0], radius=1.0, color=colour, **kwargs))
        self.add(dots, lanes([(n, GREY_A) for n, _, _ in specs], top=1.2, gap=0.85, font_size=17))
        self.wait(0.5)


class FalloffDemo(Scene):
    """Any f(r, glow_factor) -> alpha can shape the light."""

    def construct(self):
        self.camera.background_color = BLACK
        self.add(Text("falloff", font_size=34).to_edge(UP))
        self.add(
            Text("any f(r, glow_factor) -> alpha", font_size=19, color=GREY_B).shift(UP * 2.15)
        )

        specs = [
            ("default", None, YELLOW),
            ("gaussian", lambda r, gf: np.exp(-5 * r**2), GREEN),
            ("parabolic", lambda r, gf: 1 - r**2, TEAL),
            ("sqrt", lambda r, gf: np.clip(1 - r, 0, 1) ** 0.5, BLUE),
            (
                "ringed",
                lambda r, gf: np.clip(1 - r, 0, 1) ** 2 * (0.55 + 0.45 * np.cos(9 * np.pi * r)),
                PURPLE,
            ),
        ]
        dots = Group()
        for i, (_, fn, colour) in enumerate(specs):
            x = -5.3 + i * 2.65
            kwargs = {"falloff": fn} if fn else {}
            dots.add(GlowDot([x, 0.5, 0], radius=0.95, color=colour, **kwargs))
        self.add(dots, lanes([(n, GREY_A) for n, _, _ in specs], top=1.15, gap=0.8, font_size=17))
        self.wait(0.5)


class RenderModeDemo(Scene):
    """The same maths through two backends."""

    def construct(self):
        self.camera.background_color = BLACK
        self.add(Text("render_mode", font_size=34).to_edge(UP))

        self.add(GlowDot([-3.2, 0.3, 0], radius=1.25, color=BLUE, render_mode="raster"))
        self.add(GlowDot([3.2, 0.3, 0], radius=1.25, color=BLUE, render_mode="vector"))
        self.add(
            Text("raster", font_size=21).move_to([-3.2, -1.7, 0]),
            Text("per-pixel, matches the shader", font_size=15, color=GREY_B).move_to(
                [-3.2, -2.2, 0]
            ),
            Text("vector", font_size=21).move_to([3.2, -1.7, 0]),
            Text("stacked circles, sharp at any zoom", font_size=15, color=GREY_B).move_to(
                [3.2, -2.2, 0]
            ),
        )
        self.wait(0.5)


class DotCloudDemo(Scene):
    """Many dots in one object, arranged with to_grid."""

    def construct(self):
        self.camera.background_color = BLACK
        self.add(Text("GlowDots + to_grid", font_size=32).to_edge(UP))

        grid = GlowDots(color=[BLUE, TEAL, GREEN, YELLOW, RED], radius=0.1)
        grid.to_grid(5, 9, height=3.4)
        grid.shift(DOWN * 0.3)
        self.play(FadeIn(grid, scale=0.7), run_time=1.6)
        self.add(
            Text(
                "GlowDots(color=[...]).to_grid(5, 9, height=3.4)", font_size=17, color=GREY_B
            ).shift(DOWN * 2.6)
        )
        self.wait(0.8)


class StarfieldDemo(Scene):
    """Per-dot radius and colour across a whole cloud."""

    def construct(self):
        self.camera.background_color = BLACK
        rng = np.random.default_rng(7)
        n = 130
        points = np.column_stack(
            [rng.uniform(-6.8, 6.8, n), rng.uniform(-3.6, 3.6, n), np.zeros(n)]
        )
        radii = rng.uniform(0.03, 0.13, n) * (1 + rng.random(n) ** 3 * 2.5)
        palette = ["#FFFFFF", "#CFE3FF", "#FFE9B0", "#FFD0D0", "#D9C7FF"]

        stars = GlowDots(
            points,
            color=[palette[i % len(palette)] for i in range(n)],
            glow_factor=2.6,
            core_color=WHITE,
            core_size=0.22,
        )
        stars.set_radii(radii)
        moon = GlowDot(
            [4.6, 2.1, 0],
            radius=1.5,
            color="#BFD4FF",
            core_color=WHITE,
            core_size=0.45,
            glow_factor=2.2,
        )

        self.play(FadeIn(stars, scale=0.9), run_time=2.0)
        self.play(FadeIn(moon, scale=0.7), run_time=1.2)
        self.wait(1.0)


class OrbitDemo(Scene):
    """Glow dots driven by updaters."""

    def construct(self):
        self.camera.background_color = BLACK
        sun = GlowDot(
            ORIGIN, radius=1.15, color="#FFB020", core_color=WHITE, core_size=0.4, glow_factor=2.2
        )
        self.play(FadeIn(sun, scale=0.5), run_time=0.9)

        specs = [
            (1.75, "#7FD4FF", 0.17, 1.0),
            (2.60, "#FF8A6B", 0.20, 0.66),
            (3.45, "#B79BFF", 0.24, 0.45),
        ]
        rings, planets = VGroup(), Group()
        for radius, colour, size, _ in specs:
            rings.add(Circle(radius=radius).set_stroke(GREY_D, 1, opacity=0.35))
            planets.add(
                GlowDot([radius, 0, 0], radius=size, color=colour, core_color=WHITE, core_size=0.3)
            )

        self.play(
            LaggedStart(*[Create(r) for r in rings], lag_ratio=0.2),
            LaggedStart(*[FadeIn(p, scale=0.6) for p in planets], lag_ratio=0.2),
            run_time=1.6,
        )

        tracker = ValueTracker(0.0)
        for planet, (radius, _, _, speed) in zip(planets, specs):

            def follow(mob, rad=radius, spd=speed):
                angle = tracker.get_value() * spd * TAU
                mob.move_to([rad * np.cos(angle), rad * np.sin(angle), 0])

            planet.add_updater(follow)

        self.play(tracker.animate.set_value(1.5), run_time=8, rate_func=linear)
        for planet in planets:
            planet.clear_updaters()
        self.wait(0.4)
      
