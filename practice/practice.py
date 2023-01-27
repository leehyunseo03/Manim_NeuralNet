from manim import *

class Testing(Scene):
    
    def construct(self):
        ax = Axes(
            x_range=(0,5),
            y_range=(0,5),
            tips=False,
            x_length=7,
            y_length=7,
            axis_config={"include_numbers":True},
        )

        plane = NumberPlane(
            x_range = (0,5),
            y_range = (0,5),
            x_length = 7,
            axis_config={"include_numbers":True},
        )
        #plane.center()
        #self.add(plane)

        ax = Axes(
            x_range=[0, 20],
            y_range=[-5, 5], 
            x_length=10, 
            y_length=5,
            axis_config={"include_numbers":True},
        )

        labels = ax.get_axis_labels(x_label="x", y_label="y")
        graph = ax.plot(lambda x: x, x_range=[1,10],use_smoothing=False)
        self.play(FadeIn(ax, labels))
        self.play(SpinInFromNothing(graph))

        t = ValueTracker(5)
        
        def func(x):
            return (x-2)*(x-8)*(x-16) / 100

        def get_point():
            return ax.c2p(t.get_value(),func(t.get_value()))
        
        graph = ax.plot(func, color=MAROON)

        text, number = label = VGroup(
            Text("Value = "),
            DecimalNumber(
                5,
                num_decimal_places=2,
                include_sign=True,
            )
        )
        label.arrange(RIGHT)
        
        label.move_to(2*UP)
        self.play(Write(graph),runtime = 2)
        self.play(FadeIn(label))

        dot = always_redraw(
            lambda: Dot().move_to(ax.c2p(t.get_value(),func(t.get_value())))
        )
        
        number.add_updater(lambda x: x.set_value(t.get_value()))
        
        self.play(FadeIn(dot))
        self.wait(1)
        
        #self.play(t.animate.set_value(10), run_time = 5, rate_func=linear)
        self.play(t.animate.set_value(10), run_time = 5, rate_func=lambda t:smooth(t,16))
        t.set_value(10)
        self.wait(1)

        #manim practice.py Testing


