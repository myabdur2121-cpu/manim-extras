"""Visual demos for StreamAlongPath and ParticleStream.

manim -ql examples/animations/stream_along_path_demo.py StreamAlongPathDemo
manim -ql examples/animations/stream_along_path_demo.py DensityDemo
manim -ql examples/animations/stream_along_path_demo.py PathWindowDemo
manim -ql examples/animations/stream_along_path_demo.py SpawnStyleDemo
manim -ql examples/animations/stream_along_path_demo.py OrientationDemo
manim -ql examples/animations/stream_along_path_demo.py FacePointDemo
manim -ql examples/animations/stream_along_path_demo.py ParticleStreamDemo
manim -ql examples/animations/stream_along_path_demo.py EmissionRateDemo
"""

import numpy as np
from manim import *

from manim_extras import ParticleStream, StreamAlongPath


def arrow(color=WHITE, size=0.15):
    """A Triangle points UP, so pass reference_direction=UP when using it."""
    return Triangle().scale(size).set_fill(color, 1).set_stroke(width=0)


def lanes(labels, top=1.4, gap=0.95, left=3.2, right=4.8, font_size=18):
    """Horizontal tracks with a right-aligned label each."""
    tracks, tags = VGroup(), VGroup()
    for i, (text, color) in enumerate(labels):
        line = Line(LEFT * left, RIGHT * right).shift(UP * (top - i * gap))
        line.set_stroke(GREY, 1, opacity=0.2)
        tag = Text(text, font_size=font_size, color=color).next_to(line, LEFT, buff=0.15)
        tracks.add(line)
        tags.add(tag)
    return tracks, tags


class StreamAlongPathDemo(Scene):
    """The short version: a stream, a partial window, and arrows that turn."""

    def construct(self):
        title = Text("StreamAlongPath", font_size=40).to_edge(UP)
        self.play(Write(title))

        circle = Circle(radius=1.9).set_stroke(GREY, 2, opacity=0.3).shift(DOWN * 0.3)
        self.play(Create(circle))

        # a plain stream around the whole path
        self.play(
            StreamAlongPath(
                Dot(color=BLUE), circle, run_time=4, travel_time=2, spawn_lag_ratio=0.05
            )
        )

        # only part of the path, running backwards
        self.play(
            StreamAlongPath(
                Dot(color=YELLOW),
                circle,
                run_time=4,
                travel_time=2,
                start_proportion=0.9,
                end_proportion=0.1,
                spawn_lag_ratio=0.05,
            )
        )

        # arrows that follow the tangent
        self.play(
            StreamAlongPath(
                arrow(GREEN),
                circle,
                run_time=4,
                travel_time=2,
                particle_count=18,
                align_to_path=True,
                reference_direction=UP,
                spawn_style="fade",
            )
        )
        self.wait(0.5)


class DensityDemo(Scene):
    """spawn_lag_ratio derives the count; particle_count overrides it."""

    def construct(self):
        title = Text("density", font_size=32).to_edge(UP)
        self.add(title)

        ratios = [0.2, 0.1, 0.05]
        labels = []
        for lag in ratios:
            stream = StreamAlongPath(
                Dot(),
                Line(LEFT * 3, RIGHT * 4.8),
                run_time=6,
                travel_time=3,
                spawn_lag_ratio=lag,
            )
            labels.append((f"lag={lag}  ->  n={stream.particle_count}", BLUE_B))

        tracks, tags = lanes(labels)
        self.add(tracks, tags)
        self.play(
            *[
                StreamAlongPath(
                    Dot(radius=0.1, color=BLUE),
                    track,
                    run_time=6,
                    travel_time=3,
                    spawn_lag_ratio=lag,
                )
                for track, lag in zip(tracks, ratios)
            ]
        )
        self.remove(tracks, tags)

        counts = [5, 14, 30]
        tracks, tags = lanes([(f"count={n}  (lag ignored)", GREEN_B) for n in counts])
        self.add(tracks, tags)
        self.play(
            *[
                StreamAlongPath(
                    Dot(radius=0.1, color=GREEN),
                    track,
                    run_time=6,
                    travel_time=3,
                    spawn_lag_ratio=0.2,
                    particle_count=n,
                )
                for track, n in zip(tracks, counts)
            ]
        )
        self.wait(0.4)


class PathWindowDemo(Scene):
    """start_proportion / end_proportion pick any slice, either direction."""

    def construct(self):
        title = Text("path window", font_size=32).to_edge(UP)
        self.add(title)

        circle = Circle(radius=1.8).set_stroke(GREY, 2, opacity=0.3).shift(DOWN * 0.4)
        caption = Text("", font_size=20).next_to(circle, DOWN, buff=0.4)
        self.add(circle, caption)

        cases = [
            ("0.0 -> 1.0   (whole path)", 0.0, 1.0, BLUE),
            ("0.25 -> 0.75   (half)", 0.25, 0.75, YELLOW),
            ("0.9 -> 0.1   (backwards)", 0.9, 0.1, RED),
        ]
        for text, start, end, color in cases:
            caption.become(Text(text, font_size=20, color=color).next_to(circle, DOWN, buff=0.4))
            self.play(
                StreamAlongPath(
                    Dot(radius=0.1, color=color),
                    circle,
                    run_time=4,
                    travel_time=2,
                    start_proportion=start,
                    end_proportion=end,
                    spawn_lag_ratio=0.045,
                )
            )
        self.wait(0.4)


class SpawnStyleDemo(Scene):
    """How a particle is born and dies."""

    def construct(self):
        title = Text("spawn_style", font_size=32).to_edge(UP)
        self.add(title)

        styles = [("scale_fade", BLUE), ("scale", GREEN), ("fade", YELLOW), ("none", RED)]
        tracks, tags = lanes(styles, top=1.35, gap=0.95, font_size=19)
        self.add(tracks, tags)
        self.play(
            *[
                StreamAlongPath(
                    Dot(radius=0.13, color=color),
                    track,
                    run_time=6,
                    travel_time=3,
                    spawn_duration=0.8,
                    despawn_duration=0.8,
                    spawn_style=style,
                    spawn_lag_ratio=0.06,
                )
                for track, (style, color) in zip(tracks, styles)
            ]
        )
        self.wait(0.4)


class OrientationDemo(Scene):
    """align_to_path, face_direction and face_point, side by side."""

    def construct(self):
        title = Text("orientation", font_size=32).to_edge(UP)
        self.add(title)

        circles = VGroup(
            *[
                Circle(radius=1.2)
                .set_stroke(GREY, 1.5, opacity=0.28)
                .shift(LEFT * 4.3 + RIGHT * 4.3 * i + DOWN * 0.5)
                for i in range(3)
            ]
        )
        names = ["align_to_path=True", "face_direction=UP", "face_point=centre"]
        colors = [BLUE, YELLOW, GREEN]
        labels = VGroup(
            *[
                Text(name, font_size=17, color=color).next_to(circle, DOWN, buff=0.3)
                for name, color, circle in zip(names, colors, circles)
            ]
        )
        hub = Dot(circles[2].get_center(), color=GREEN, radius=0.07)
        self.add(circles, labels, hub)

        options = [
            {"align_to_path": True},
            {"face_direction": UP},
            {"face_point": circles[2].get_center()},
        ]
        self.play(
            *[
                StreamAlongPath(
                    arrow(color),
                    circle,
                    run_time=6,
                    travel_time=3,
                    particle_count=18,
                    reference_direction=UP,
                    spawn_style="fade",
                    **option,
                )
                for circle, color, option in zip(circles, colors, options)
            ]
        )
        self.wait(0.4)


class FacePointDemo(Scene):
    """face_point accepts a Mobject and tracks it while it moves."""

    def construct(self):
        title = Text("face_point tracks a moving target", font_size=30).to_edge(UP)
        self.add(title)

        circle = Circle(radius=2.0).set_stroke(GREY, 1.5, opacity=0.25).shift(DOWN * 0.3)
        target = Dot(color=RED, radius=0.12).move_to(circle.get_center() + LEFT * 3.6)
        halo = Circle(radius=0.25, color=RED).set_stroke(RED, 2.5, opacity=0.7)
        halo.add_updater(lambda m: m.move_to(target.get_center()))
        self.add(circle, target, halo)

        self.play(
            StreamAlongPath(
                arrow(BLUE),
                circle,
                run_time=8,
                travel_time=3.5,
                particle_count=24,
                face_point=target,
                reference_direction=UP,
                spawn_style="fade",
            ),
            Succession(
                target.animate(run_time=2.6).move_to(circle.get_center() + RIGHT * 3.8 + UP * 1.4),
                target.animate(run_time=2.6).move_to(circle.get_center() + DOWN * 2.6),
                target.animate(run_time=2.8).move_to(circle.get_center() + LEFT * 3.5 + UP * 1.1),
            ),
        )
        halo.clear_updaters()
        self.wait(0.4)


class ParticleStreamDemo(Scene):
    """Colour and randomness, one knob at a time."""

    def construct(self):
        title = Text("ParticleStream", font_size=32).to_edge(UP)
        self.add(title)

        options = [
            ("path_colors", {"path_colors": [BLUE, YELLOW, RED]}, GREY_A),
            ("particle_colors", {"particle_colors": [RED, GREEN, BLUE]}, GREY_A),
            ("position_jitter=0.3", {"position_jitter": 0.3}, BLUE),
            ("size_jitter=0.6", {"size_jitter": 0.6}, GREEN),
            ("wobble_amplitude=0.3", {"wobble_amplitude": 0.3, "wobble_frequency": 2.5}, RED),
        ]
        tracks, tags = lanes(
            [(name, color) for name, _, color in options],
            top=1.6,
            gap=0.8,
            left=2.9,
            font_size=17,
        )
        self.add(tracks, tags)
        self.play(
            *[
                ParticleStream(
                    Dot(radius=0.12, color=color),
                    track,
                    run_time=6,
                    travel_time=3,
                    spawn_lag_ratio=0.055,
                    seed=7,
                    **option,
                )
                for track, (_, option, color) in zip(tracks, options)
            ]
        )
        self.wait(0.4)


class EmissionRateDemo(Scene):
    """Same particle count everywhere -- only the birth timing changes."""

    def construct(self):
        title = Text("emission_rate_func", font_size=32).to_edge(UP)
        note = Text("same count = 26 in every lane", font_size=18, color=GREY_B)
        note.next_to(title, DOWN, buff=0.15)
        self.add(title, note)

        curves = [
            ("steady (None)", None, GREY_B),
            ("lambda t: t", lambda t: t, BLUE),
            ("lambda t: 1 - t", lambda t: 1 - t, YELLOW),
            ("lambda t: 0.1 + t**3", lambda t: 0.1 + t**3, RED),
        ]
        tracks, tags = lanes(
            [(name, color) for name, _, color in curves],
            top=1.1,
            gap=0.9,
            left=3.3,
            font_size=17,
        )
        self.add(tracks, tags)
        self.play(
            *[
                ParticleStream(
                    Dot(radius=0.11, color=color),
                    track,
                    run_time=8,
                    travel_time=2.6,
                    particle_count=26,
                    emission_rate_func=curve,
                )
                for track, (_, curve, color) in zip(tracks, curves)
            ]
        )
        self.wait(0.4)


class AnyPathDemo(Scene):
    """Any VMobject works as a path.

    manim -ql examples/animations/stream_along_path_demo.py AnyPathDemo
    """

    def construct(self):
        title = Text("any VMobject as a path", font_size=32).to_edge(UP)
        self.add(title)

        square = Square(side_length=1.9).shift(LEFT * 4.4 + DOWN * 0.2)
        star = Star(n=5, outer_radius=1.05).shift(LEFT * 1.5 + DOWN * 0.2)
        curve = FunctionGraph(lambda x: 0.55 * np.sin(2.2 * x), x_range=[-1.5, 1.5])
        curve.shift(RIGHT * 1.6 + DOWN * 0.2)
        # The third t_range entry is the sampling step. The default (0.01) gives
        # this spiral ~7500 points, and point_from_proportion walks them on every
        # frame for every particle -- which makes the scene crawl. A coarser step
        # looks the same here and renders roughly 20x faster.
        spiral = ParametricFunction(
            lambda t: np.array([0.16 * t * np.cos(t), 0.16 * t * np.sin(t), 0]),
            t_range=[0, 6 * PI, 0.2],
        ).shift(RIGHT * 4.5 + DOWN * 0.2)

        paths = [square, star, curve, spiral]
        colors = [BLUE, YELLOW, GREEN, RED]
        for path in paths:
            path.set_stroke(GREY, 1.5, opacity=0.3)
        labels = VGroup(
            *[
                Text(name, font_size=16, color=color).next_to(path, DOWN, buff=0.25)
                for name, color, path in zip(
                    ["Square", "Star", "FunctionGraph", "Spiral"], colors, paths
                )
            ]
        )
        self.add(*paths, labels)
        self.play(
            *[
                ParticleStream(
                    arrow(color, 0.11),
                    path,
                    run_time=6,
                    travel_time=3,
                    particle_count=16,
                    align_to_path=True,
                    reference_direction=UP,
                    spawn_style="fade",
                    path_colors=[color, WHITE],
                )
                for path, color in zip(paths, colors)
            ]
        )
        self.wait(0.4)
