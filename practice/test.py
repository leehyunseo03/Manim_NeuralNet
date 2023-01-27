from manim import *

class Intro(Scene):
    def construct(self):
        with register_font("./nanum-myeongjo/NanumMyeongjo.ttf"):
            title = Text("NerualNet101").scale(1.4).move_to(UP)
            author = Text("VLAB", font="NanumMyeongjo")
        line = Line(UP, LEFT*4.5)
        self.wait(1)
        self.play(Write(title))
        self.wait(1)
        self.play(Write(author))
        self.wait(2)
        self.play(Unwrite(author))
        self.wait(1)
        self.play(ScaleInPlace(title, 1/1.4), MoveAlongPath(title, line))

        order_str = [
        "1. Gradient-Based Optimization",
        "2. Linear Regression",
        "3. Logistic Regression",
        "4. Softmax & Cross-Entropy Error",
        "5. Back Propagation"
        ]
        order_group = VGroup()
        for i in range(len(order_str)):
            order = Text(order_str[i]).move_to(2.2*RIGHT+(i-2)*DOWN).scale(0.8)
            order_group.add(order)
        b = Brace(order_group, LEFT)
        order_group.arrange(2*DOWN, center=False, aligned_edge=LEFT)
        self.play(FadeIn(b), Write(order_group))
        self.wait(3)

        order1 = order_group[0].copy()
        self.add(order1)
        self.play(Circumscribe(order1), FadeOut(order_group, b, title))
        order2 = order1.copy().scale(1/0.8).move_to(ORIGIN)
        self.play(Transform(order1, order2))
        self.wait(2)

class Optimization(Scene):
    def construct(self):
        title = Text("1. Gradient-Based Optimization", t2c={"Optimization":"#000000"})
        optim = Text("Optimization").move_to(title, aligned_edge=RIGHT)
        self.add(title, optim)
        self.play(FadeToColor(optim, "#00EEEE"))
        self.wait(3)
        self.play(LaggedStart(FadeOut(title), MoveAlongPath(optim, Line(optim.get_center(), ORIGIN)), lag_ratio=0.5))
        self.wait(2)
        self.play(FadeToColor(optim, "#FFFFFF"))
        self.play(MoveAlongPath(optim, Line(optim.get_center(), 3*UP)))
        self.wait(1)

        points = [
            "• Optimize the design of X to reach the maximum performance",
            "• Optimize the recommendation algorithm to show the best\n   individualized advertisement to each person",
            "• Optimize the factory, so we can spend lesser about 21%\n   than before"
        ]
        
        evs = VGroup()
        for i in range(len(points)):
            ev = Text(points[i], line_spacing=0.7).scale(0.7).move_to((1.7*i-1.7)*DOWN+6.5*LEFT, aligned_edge=LEFT)
            self.play(Write(ev))
            evs.add(ev)
            self.wait(5)
        
        self.play(Unwrite(evs))
        self.wait(1)

class MathOptimization(Scene):
    def construct(self):
        optim = Text("Optimization").move_to(3*UP)
        self.add(optim)
        self.wait(1)

        ax = Axes(
            x_range=[0, 20], y_range=[0, 10], axis_config={"include_tip": False},
            x_length=10, y_length=5
        )
        labels = ax.get_axis_labels(x_label="x", y_label="y")
        self.play(FadeIn(ax, labels))

        t = ValueTracker(0)

        def func(x):
            return (x+1)*(x-17)*(x-17)/144+2
        graph = ax.plot(func, color=MAROON)
        f_label = MathTex("y=f(x)").move_to(1.5*LEFT+2*UP)
        self.play(Write(graph), FadeIn(f_label))

        def get_point():
            return [ax.c2p(t.get_value(), func(t.get_value()))]
        dot = Dot(point=get_point())
        dot.add_updater(lambda x: x.move_to(get_point()))

        val_labels = MathTex("x &= 0.0000\\\\f(x) &= 0.0000000").move_to(3*RIGHT+1.5*UP)
        def get_labels() -> MathTex:
            return MathTex("x &= {0:0.4f}\\\\f(x) &= {1:0.7f}".format(t.get_value(), func(t.get_value()))).move_to(3*RIGHT+1.5*UP)
        val_labels.add_updater(lambda x: x.become(get_labels()))
        def get_dashed_lines() -> VGroup:
            tmp = VGroup()
            tmp.add(DashedLine(start=[ax.c2p(t.get_value(), 0)], end=get_point()))
            tmp.add(DashedLine(start=[ax.c2p(0, func(t.get_value()))], end=get_point()))
            return tmp
        vh_line = get_dashed_lines()
        vh_line.add_updater(lambda x: x.become(get_dashed_lines()))

        self.play(FadeIn(dot, vh_line), Write(val_labels))
        self.play(t.animate.set_value(5), run_time=5, rate_func=lambda t:smooth(t, 20))
        self.wait(3)
        self.play(t.animate.set_value(17), run_time=5, rate_func=lambda t:smooth(t, 20))
        self.wait(3)
        self.play(LaggedStart(Unwrite(val_labels), FadeOut(vh_line, ax, labels, graph, dot), MoveAlongPath(f_label, Line(f_label.get_center(), 2*UP)), lag_ratio=0.4))
        self.wait(2)

        f_label2 = MathTex(r"z=f(x,y)").move_to(2*UP)
        self.play(Transform(f_label, f_label2))
        self.wait(1)

class MathOptimization3D(ThreeDScene):
    def construct(self):
        optim = Text("Optimization").move_to(3*UP)
        f_label = MathTex(r"z=f(x,y)").move_to(2*UP)
        self.add_fixed_in_frame_mobjects(optim, f_label)
        self.wait(1)

        self.set_camera_orientation(phi=75 * DEGREES, theta=-60 * DEGREES)
        self.move_camera(frame_center=1*OUT)
        axes = ThreeDAxes(x_range=[-4, 4], y_range=[-4, 4],z_range=[-1,3], z_length=4)
        self.wait(1)

        def _gauss(u, v):
            return np.exp(-(u**2+v**2))
        def func(u, v):
            z = 2*_gauss(u, v) - 1*_gauss(u-1, v+1) + 1.5*_gauss(u-2, v)
            return z
        surf = Surface(
            func=lambda u, v: axes.c2p(u, v, func(u, v)),
            u_range=[-3, 3],
            v_range=[-3, 3],
            resolution=24
        )
        surf.set_fill_by_value(axes=axes, colorscale=[(BLUE, -0.5), (GREEN, 0), (RED, 1)], axis=2)
        self.play(FadeIn(surf))
        self.wait(1)
        self.move_camera(theta=245 * DEGREES, run_time=3)
        self.wait(1)
        self.play(FadeOut(surf))
        self.play(MoveAlongPath(f_label, Line(f_label.get_center(), ORIGIN)))

class MultivarFunction(Scene):
    def construct(self):
        optim = Text("Optimization").move_to(3*UP)
        f_label = MathTex(r"z=f(x,y)")
        self.add(optim, f_label)
        self.wait(1)

        multi_f_label = MathTex(r"y=f(x_{1},x_{2},\cdots,x_{n})")
        self.play(Transform(f_label, multi_f_label))
        self.wait(1)
        self.play(FadeOut(f_label), MoveAlongPath(optim, Line(optim.get_center(), ORIGIN)))

class Gradient(Scene):
    def construct(self):
        title = Text("Gradient-Based Optimization")
        others = Text("Gradient-Based"+("\t"*10)+"p", t2c={"p":"#000000"}).move_to(title, aligned_edge=LEFT)
        optim = Text("Optimization")
        optim_new = Text("Optimization").move_to(title, aligned_edge=RIGHT)

        self.add(optim)
        self.play(LaggedStart(Transform(optim, optim_new), FadeIn(others), lag_ratio=0.5))
        self.add(title)
        self.remove(others, optim)
        self.wait(1)
        self.play(MoveAlongPath(title, Line(title.get_center(), 3*UP)))

        ax = Axes(
            x_range=[-8, 8], y_range=[0, 10], axis_config={"include_tip": False},
            x_length=10, y_length=5
        ).move_to(0.7*UP)
        labels = ax.get_axis_labels(x_label="x", y_label="y")
        # self.play(FadeIn(ax, labels))

        def func(x):
            return (x**2)/10
        graph = ax.plot(func, color=MAROON)
        graph_pos = graph.get_center()
        f_label = MathTex(r"f(x)=x^2").move_to(2*UP)
        self.play(Write(graph), FadeIn(f_label))

        tangent = TangentLine(graph, alpha=0.5, length=4, color=WHITE)
        self.play(FadeIn(tangent))

        t = ValueTracker(0)
        # graph updater
        graph_updater = lambda x:x.move_to(graph_pos+ax.c2p(-t.get_value(),-func(t.get_value()))-ax.c2p(0,0)+np.array([0, 2*np.abs(np.tan(tangent.get_angle())), 0]))
        graph.add_updater(graph_updater)
        
        # tangent updatere
        def tangent_updater(x):
            angle = TangentLine(graph, alpha=(t.get_value()/16)+0.5, length=4, color=WHITE).get_angle()
            x.become(TangentLine(graph, alpha=(t.get_value()/16)+0.5, length=4/np.cos(angle), color=WHITE)).move_to(ax.c2p(0,0)+np.array([0, 2*np.abs(np.tan(tangent.get_angle())), 0]))
        tangent.add_updater(tangent_updater)

        # lines updater
        def line_updater(x):
            t.get_value()
            start_x, start_y = tangent.get_start()[:2]
            end_x, end_y = tangent.get_end()[:2]
            y = min(start_y, end_y)
            diff = np.array([0, 4*np.abs(np.tan(tangent.get_angle())), 0])
            x[0].become(Line(tangent.get_start(), [start_x, y, 0]))
            if t.get_value()==0:
                x[1].become(Line([end_x, y, 0], [start_x, y, 0]))
            x[2].become(Line(tangent.get_end(), [end_x, y, 0]))
            # auxilary parallel lines
            if t.get_value()>=0:
                x[3].become(Line(tangent.get_end(), tangent.get_start()).move_to(tangent.get_center()-diff))
            else:
                x[3].become(Line(tangent.get_end(), tangent.get_start()))
        lines = VGroup()
        for _ in range(4):
            lines.add(Line())
        line_updater(lines)
        self.add(lines)
        lines.add_updater(line_updater)

        angle = Angle(Line(10*UP, 11*UP), Line(10*UP, 11*UP).rotate(1))
        def angle_updater(x):
            if t.get_value()==0:
                x.become(Angle(Line(10*UP, 11*UP), Line(10*UP, 11*UP).rotate(1)))
            elif t.get_value()>0:
                x.become(Angle(lines[1], lines[3], radius=0.5))
            else:
                x.become(Angle(lines[3], lines[1], radius=0.5))
        angle.add_updater(angle_updater)
        self.add(angle)

        template = TexTemplate()
        template.add_to_preamble(r"\usepackage{gensymb}")
        label = MathTex("\\theta = 0.00 \\degree", tex_template=template)
        def label_updater(x):
            if t.get_value()==0:
                _str = "\\theta = 0.00 \\degree"
            else:
                _str = "\\theta = {0:0.2f} \\degree".format(np.rad2deg(-tangent.get_angle()))
            x.become(MathTex(_str, tex_template=template).move_to(lines[1].get_right()+2*RIGHT))
        label.add_updater(label_updater)
        self.play(Write(label))

        self.play(t.animate.set_value(3), run_time=2)
        self.play(t.animate.set_value(-5), run_time=16/3)
        graph_updater(graph)
        angle_updater(angle)
        tangent_updater(tangent)
        self.wait(3)

        self.play(
            FadeOut(*lines),
            FadeOut(angle),
            FadeOut(tangent),
            FadeOut(graph),
            Unwrite(label),
            Unwrite(f_label)
        )

        formula = MathTex("x'=x-f'(x)").scale(2)
        self.play(Write(formula), run_time=2)
        self.wait(3)
        self.play(FadeOut(formula))
        self.wait(1)

class GradientBallDrop(Scene):
    def construct(self):
        title = Text("Gradient-Based Optimization").move_to(3*UP)
        self.add(title)
        new_title = Text("Example 1").move_to(3*UP)
        self.play(LaggedStart(FadeOut(title, shift=LEFT), FadeIn(new_title, shift=LEFT), lag_ratio=0.2))
        self.wait(1)