"""Runnable demos for the blur layers.

manim -ql examples/blur_demo.py GlassDoorDemo
manim -ql examples/blur_demo.py LiveVsStaticDemo
manim -ql examples/blur_demo.py AnyShapeDemo
manim -ql examples/blur_demo.py CardDemo
manim -ql examples/blur_demo.py CameraBlurDemo
"""

from __future__ import annotations

from manim import (
    BLUE,
    DOWN,
    GREEN,
    LEFT,
    ORANGE,
    PURPLE,
    RED,
    RIGHT,
    UP,
    WHITE,
    YELLOW,
    Annulus,
    Circle,
    Rectangle,
    RegularPolygon,
    RoundedRectangle,
    Scene,
    Star,
    Text,
    Triangle,
    VGroup,
    linear,
)

from manim_extras import Blur, BlurCard, CameraBlur, IMGBlur


def scenery() -> VGroup:
    """A row of coloured discs, so the blur is easy to judge."""
    return VGroup(
        *[
            Circle(radius=1.4, fill_opacity=1, stroke_width=0, color=c).shift(RIGHT * x)
            for x, c in zip([-5, -2.5, 0, 2.5, 5], [RED, GREEN, BLUE, YELLOW, PURPLE])
        ]
    )


def walker(color=RED) -> VGroup:
    """A crude figure to send past the glass."""
    return VGroup(
        Circle(radius=0.42, fill_opacity=1, stroke_width=0, color=color),
        RoundedRectangle(
            corner_radius=0.25,
            width=1.1,
            height=2.0,
            fill_opacity=1,
            stroke_width=0,
            color=color,
        ),
    ).arrange(DOWN, buff=0.08)


class GlassDoorDemo(Scene):
    """Someone walks past a frosted glass door and blurs as they pass."""

    def construct(self):
        door = Rectangle(width=5, height=5.4, stroke_color=WHITE, stroke_width=4)
        glass = Blur(door, 30)
        glass.set_z_index(1)
        self.add(glass)

        person = walker().move_to(LEFT * 7).set_z_index(0)
        self.add(person)

        self.play(person.animate.move_to(RIGHT * 7), run_time=3.5, rate_func=linear)


class LiveVsStaticDemo(Scene):
    """Blur reacts to what passes beneath; IMGBlur never notices."""

    def construct(self):
        top = Rectangle(width=4, height=2.6, stroke_color=GREEN, stroke_width=4)
        top.move_to(UP * 2)
        bottom = Rectangle(width=4, height=2.6, stroke_color=RED, stroke_width=4)
        bottom.move_to(DOWN * 2)

        self.add(Blur(top, 26).set_z_index(2))
        self.add(IMGBlur(bottom, 26).set_z_index(2))

        self.add(
            Text("Blur", font_size=28, color=GREEN).move_to(LEFT * 6.2 + UP * 3.4),
            Text("IMGBlur", font_size=28, color=RED).move_to(LEFT * 5.9 + DOWN * 0.6),
        )

        live = Circle(radius=0.9, fill_opacity=1, stroke_width=0, color=BLUE)
        live.move_to(LEFT * 7 + UP * 2).set_z_index(1)
        static = live.copy().move_to(LEFT * 7 + DOWN * 2)
        self.add(live, static)

        self.play(
            live.animate.move_to(RIGHT * 7 + UP * 2),
            static.animate.move_to(RIGHT * 7 + DOWN * 2),
            run_time=3.5,
            rate_func=linear,
        )


class AnyShapeDemo(Scene):
    """Circles, stars and shapes with holes all mask correctly."""

    def construct(self):
        self.add(scenery())
        shapes = [
            Circle(radius=1.4, stroke_color=WHITE, stroke_width=4),
            Star(7, outer_radius=1.7, stroke_color=YELLOW, stroke_width=4),
            Annulus(inner_radius=0.7, outer_radius=1.6, stroke_width=0),
        ]
        for i, shape in enumerate(shapes):
            self.add(Blur(shape, 28).move_to([-4.4 + i * 4.4, 0, 0]).set_z_index(1))
        self.wait(2)


class CardDemo(Scene):
    """Glass cards in a range of outlines and colours."""

    def construct(self):
        self.add(scenery())
        cards = [
            BlurCard(
                Star(7, outer_radius=1.7),
                tint=GREEN,
                tint_opacity=0.35,
                border_color=GREEN,
                border_width=5,
            ).shift(LEFT * 4.6 + UP * 1.5),
            BlurCard(
                Circle(radius=1.6),
                tint=BLUE,
                tint_opacity=0.4,
                border_color=WHITE,
                border_width=5,
            ).shift(UP * 1.5),
            BlurCard(
                RegularPolygon(6).scale(1.7),
                tint=ORANGE,
                tint_opacity=0.4,
                border_color=ORANGE,
                border_width=5,
            ).shift(RIGHT * 4.6 + UP * 1.5),
            BlurCard(
                width=4.2,
                height=2.4,
                tint=PURPLE,
                tint_opacity=0.4,
                border_color=PURPLE,
                border_width=5,
            ).shift(LEFT * 3.4 + DOWN * 1.9),
            BlurCard(
                Triangle().scale(1.9),
                tint=RED,
                tint_opacity=0.35,
                border_color=RED,
                border_width=5,
            ).shift(RIGHT * 3.4 + DOWN * 1.9),
        ]
        for card in cards:
            card.set_z_index(1)
            self.add(card)
        self.wait(2)


class CameraBlurDemo(Scene):
    """The whole frame goes soft while the title stays sharp."""

    def construct(self):
        self.add(scenery())

        veil = CameraBlur(0)
        veil.set_z_index(1)
        self.add(veil)

        self.add(Text("Sharp Title", font_size=52).set_z_index(2))

        self.play(veil.to(32), run_time=2)
        self.wait(1)
      
