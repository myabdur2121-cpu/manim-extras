"""Visual demo for SmoothPolygon.

    manim -ql examples/smooth_polygon_demo.py SmoothPolygonDemo
    manim -ql examples/smooth_polygon_demo.py SeamComparison
"""

import numpy as np
from manim import *

from manim_extras import SmoothPolygon

POINTS = [
    np.array([-1, 2, 0]),
    np.array([1, 1, 0]),
    np.array([2, -1, 0]),
    np.array([-2, -1, 0]),
]


class SmoothPolygonDemo(Scene):
    def construct(self):
        title = Text("SmoothPolygon", font_size=40).to_edge(UP)
        self.play(Write(title))

        # polygon -> smooth version through the same vertices
        poly = Polygon(*POINTS, color=GREY_B, stroke_width=3)
        dots = VGroup(*[Dot(p, color=YELLOW, radius=0.07) for p in POINTS])
        shape = SmoothPolygon(POINTS, color=BLUE, stroke_width=6, fill_opacity=0.25)

        self.play(Create(poly), FadeIn(dots, scale=0.5))
        self.play(ReplacementTransform(poly, shape), run_time=1.5)
        self.wait(0.3)

        # sweep the loop: constant speed through the start point == no kink
        tracer = Dot(color=RED, radius=0.1).move_to(shape.get_start())
        label = Text("start point", font_size=22, color=RED).next_to(tracer, UR, buff=0.15)
        self.play(FadeIn(tracer), FadeIn(label))
        self.play(MoveAlongPath(tracer, shape), run_time=3, rate_func=linear)
        self.play(FadeOut(label), FadeOut(tracer), FadeOut(dots))

        # it is a normal VMobject
        self.play(shape.animate.set_fill(TEAL, 0.4).set_stroke(TEAL).scale(1.15))
        self.play(Rotate(shape, PI / 2), run_time=1.5)

        # more vertices, same class
        star_pts = [
            np.array([np.cos(a) * r, np.sin(a) * r, 0])
            for i, a in enumerate(np.linspace(PI / 2, PI / 2 + TAU, 10, endpoint=False))
            for r in [2.2 if i % 2 == 0 else 1.0]
        ]
        star = SmoothPolygon(star_pts, color=YELLOW, stroke_width=6, fill_opacity=0.3)
        self.play(ReplacementTransform(shape, star), run_time=1.5)
        self.wait(0.4)

        # open curve
        wave_pts = [np.array([x, np.sin(x * 1.4) * 1.2, 0]) for x in np.linspace(-4, 4, 7)]
        wave = SmoothPolygon(wave_pts, is_closed=False, color=GREEN, stroke_width=6)
        wave_dots = VGroup(*[Dot(p, color=WHITE, radius=0.06) for p in wave_pts])
        self.play(FadeOut(star))
        self.play(Create(wave), FadeIn(wave_dots), run_time=1.5)
        self.wait(1)
        self.play(FadeOut(wave), FadeOut(wave_dots), FadeOut(title))


class SeamComparison(Scene):
    """Old padded workaround (left) vs periodic solve (right)."""

    def construct(self):
        old = VMobject(color=RED, stroke_width=6)
        padded = [POINTS[-1], *POINTS, POINTS[0], POINTS[1]]
        old.set_points_smoothly(padded)
        old.set_points(old.points[4 : 4 * len(padded) - 8])

        new = SmoothPolygon(POINTS, color=GREEN, stroke_width=6)

        for mob, dx in ((old, -3.2), (new, 3.2)):
            mob.scale(0.55).shift(RIGHT * dx + DOWN * 0.3)

        lbl_old = Text("padded + sliced\n(kink at start)", font_size=24, color=RED)
        lbl_new = Text("periodic solve\n(smooth seam)", font_size=24, color=GREEN)
        lbl_old.next_to(old, UP, buff=0.5)
        lbl_new.next_to(new, UP, buff=0.5)

        self.play(Create(old), Create(new), run_time=2)
        self.play(
            FadeIn(lbl_old),
            FadeIn(lbl_new),
            FadeIn(Dot(old.get_start(), color=WHITE, radius=0.06)),
            FadeIn(Dot(new.get_start(), color=WHITE, radius=0.06)),
        )
        self.wait(2)
