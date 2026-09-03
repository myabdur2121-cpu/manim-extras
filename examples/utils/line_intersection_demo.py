"""Visual demos for the line intersection helpers.

manim -ql examples/utils/line_intersection_demo.py LineIntersectionDemo
manim -ql examples/utils/line_intersection_demo.py SegmentIntersectionDemo
"""

from manim import *

from manim_extras import GeometryOperations, ManimGeometryAdapter


class LineIntersectionDemo(Scene):
    """The canonical example: the crossing diagonals of a square meet at [1, 1]."""

    def construct(self):
        self.camera.background_color = BLACK
        title = Text("Line intersection", font_size=40).to_edge(UP)
        self.play(Write(title))

        l1 = Line([0, 0, 0], [4, 4, 0], color=BLUE)
        l2 = Line([0, 4, 0], [4, 0, 0], color=GREEN)
        self.play(Create(l1), Create(l2))

        pt = ManimGeometryAdapter.intersection_point(l1, l2)
        dot = Dot(pt, color=RED)
        label = Text("(2, 2)", font_size=24, color=RED).next_to(dot, UR)
        self.play(FadeIn(dot), Write(label))
        self.wait()


class SegmentIntersectionDemo(Scene):
    """Two segments that cross (left), and two that don't (right)."""

    def construct(self):
        self.camera.background_color = BLACK

        # left pair: they do intersect
        a1 = Line([-5, -2, 0], [-1, 2, 0], color=BLUE)
        a2 = Line([-5, 2, 0], [-1, -2, 0], color=GREEN)
        self.play(Create(a1), Create(a2))
        did = GeometryOperations.segments_intersect([[-5, -2], [-1, 2]], [[-5, 2], [-1, -2]])
        self.add(Text(f"intersect: {did}", font_size=28).move_to([-3, 3, 0]))

        # right pair: infinite lines cross, but the segments don't meet
        b1 = Line([1, -2, 0], [5, -2, 0], color=BLUE)
        b2 = Line([1, 2, 0], [1, 0, 0], color=GREEN)
        self.play(Create(b1), Create(b2))
        did = GeometryOperations.segments_intersect([[1, -2], [5, -2]], [[1, 2], [1, 0]])
        self.add(Text(f"intersect: {did}", font_size=28).move_to([3, 3, 0]))

        self.wait()
