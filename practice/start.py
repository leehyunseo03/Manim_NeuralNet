from manim import *

class test(Scene):
    def construct(self):
        square = Square()
        
        circle = Circle()
        circle.set_fill(BLUE, opacity=0.5)
        circle.set_stroke(BLUE_E, width=4)

        triangle = Triangle()

        dot1 = Dot(point = np.array([0.0,0.0,0.0]))
        dot2 = Dot(point = 1*RIGHT)
        dot3 = Dot(point = 2*RIGHT)
        self.add(dot1,dot2,dot3)

        text = Text("Show Creation") # 글씨쓰기
        text.shift(2*UP) #위로이동
        self.play(Write(text))

        text = Text("한국어 되나?") # 글씨쓰기
        text.shift(2*DOWN) #밑으로이동
        self.play(Write(text))
        self.play(FadeOut(text))

        self.play(Create(square))
        self.play(Create(triangle))
        self.play(ReplacementTransform(triangle,circle))  
        self.play(Uncreate(circle))
        circle = Circle()
        circle.set_fill(BLUE, opacity=0.5)
        circle.set_stroke(BLUE_E, width=4)
        
        
        self.wait()
        self.play(ReplacementTransform(square, circle)) #사각형 -> 원
        circle.generate_target()
        circle.target.shift(2* UP)
        self.play(MoveToTarget(circle))

        circle.target.shift(2* DOWN)
        self.play(MoveToTarget(circle))

        self.play(FadeToColor(circle,RED))
        

        self.play(ApplyPointwiseFunction(lambda x: 2*x, circle))
        self.play(ApplyPointwiseFunction(lambda x: 0.5*x, circle))

        circle.save_state()
        self.play(FadeToColor(circle,WHITE))
        self.play(ApplyPointwiseFunction(lambda x: 2*x, circle))
        self.play(Restore(circle))
        
        self.play(ScaleInPlace(circle, 2))
        self.play(ShrinkToCenter(circle))
        self.wait()

        square = Square()
        self.play(Create(square))
        square.save_state()

        arr = [[1,1],[0,1]]
        self.play(ApplyMatrix(arr,square))
        self.play(Restore(square))

        arr = [[0,0],[0,1]]
        self.play(ApplyMatrix(arr,square))
        self.play(Restore(square))

        circle = Circle()
        circle.shift(2*UP + 2*RIGHT)
        self.play(Create(circle))
        self.play(CyclicReplace(circle, square))
        self.play(FadeOut(square))
        self.play(FadeToColor(circle, WHITE))
        self.play(FadeTransformPieces(circle,square))
        self.play(FadeOut(square))
    
#   manim start.py test