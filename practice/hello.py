from manim import *

class Hello_World(Scene):
    def construct(self):
        text=TextMobject("Hello World")
        self.play(Write(text))
        self.wait()