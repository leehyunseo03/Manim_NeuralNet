from manim import *
import torch
import pandas as pd

class Intro(Scene):
    def construct(self):
        title = Text("Linear Regression")
        others = Text("Linear"+("\t"*10)+"p", t2c={"p":"#000000"}).move_to(title, aligned_edge=LEFT)
        optim = Text("Regression")
        optim_new = Text("Regression").move_to(title, aligned_edge=RIGHT)

        self.add(optim)
        self.play(LaggedStart(Transform(optim, optim_new), FadeIn(others), lag_ratio = 0.5))
        self.add(title)
        self.remove(others, optim)
        self.wait(1)
        self.play(MoveAlongPath(title, Line(title.get_center(), 3*UP)))

#manim NN101_2.py Intro

class define(Scene):
    def construct(self):
        linear_function = MathTex("f(x) = wx + b")
        self.play(Write(linear_function))
        self.wait(1)

#manim NN101_2.py define

class lossdefine(Scene):
    def construct(self):
        title = Text("Loss function")
        title.move_to(3*UP)
        self.play(Write(title))
        self.wait(1)

        arr = ["MAE","MBE","SVM","HUBER","LOG-COSH","QUANTILE","CrossEntropy","MSE"]
        nowloss = Text(arr[0])
        self.play(Write(nowloss))
        for i in range(len(arr)-1):
            nextloss = Text(arr[i+1])
            self.play(Transform(nowloss,nextloss,run_time = 0.3))
            self.wait(0.3)
        self.wait(2)
        self.play(MoveAlongPath(nowloss, Line(nowloss.get_center(), 2*UP)))
        self.wait(1)
        mse = Text("Mean Square Error")
        mse.move_to(2*UP)
        self.play(Transform(nowloss,mse),FadeOut(title))
        self.wait(1)

        loss_function = MathTex("L(w,b) = "+r"\frac{1}{2m} \sum\limits_{i=0}^m (",r"f(x_{i})", r"- y_{i})^{2}")
        
        self.play(Write(loss_function))
        self.wait(2)
        
        square = Square(side_length=0.3,color = RED)
        square.move_to(0.23*UP+3.2*RIGHT)
        self.play(ShowCreationThenFadeOut(square))
        self.wait(3)
        
        center = MathTex(r"f(x) = wx + b")
        losstolinearfunction = MathTex(r"f(x_{i})")
        self.play(TransformMatchingTex(loss_function,losstolinearfunction))
        linearfunction = MathTex(r"f(x)")
        self.play(Transform(losstolinearfunction, linearfunction))
        self.wait(2)

        others = MathTex(r"= wx + b").move_to(center, aligned_edge=RIGHT)
        newlinear = MathTex(r"f(x)").move_to(center, aligned_edge=LEFT)
        
        self.play(LaggedStart(Transform(losstolinearfunction, newlinear), FadeIn(others), lag_ratio = 0.5))
        self.wait(2)
        self.play(FadeOut(losstolinearfunction),FadeOut(others),FadeOut(mse))
        self.wait(1)

#manim NN101_2.py lossdefine   


class linear(Scene):
    
    def construct(self):
        ax = Axes(
            x_range=[0, 20],
            y_range=[0, 10], 
            x_length=10, 
            y_length=5,
            axis_config={"include_numbers":False},
        )

        labels = ax.get_axis_labels(x_label="x", y_label="y")
        self.play(FadeIn(ax, labels))

        xdot = [2,4,6,8,10,12,14,16]
        ydot = [1,3,2,4,6,5,7,8,7]
        for i in range(len(xdot)):
            dot = Dot(ax.c2p(xdot[i],ydot[i]))
            self.add(dot)
            self.wait(0.2)
        #self.play(SpinInFromNothing(graph))

        #manim Linear_regression.py linear

class predict(Scene):
    def construct(self):
        title = Text("Linear Regression")
        title.move_to(3*UP)
        self.add(title)

        ax = Axes(
            x_range=(0, 8),
            y_range=(0, 8), 
            x_length=10, 
            y_length=5,
            axis_config={"include_numbers":True},
        )

        labels = ax.get_axis_labels(x_label="x", y_label="y")
        self.play(FadeIn(ax, labels))

        xdot = [0,1,2,3,4,5]
        ydot = [0,1,2,3,4,5]
        dotarr = []
        for i in range(len(xdot)):
            dot = Dot(ax.c2p(xdot[i],ydot[i]))
            dotarr.append(dot)
            self.add(dot)
            self.wait(0.2)

        self.wait(1)

        graph = ax.plot(lambda x: x+1,x_range=[0,8],color=MAROON)
        
        line = ax.get_vertical_line(ax.i2gp(6,graph),color = YELLOW)
        self.play(Write(line))
        self.wait(1)

        t = ValueTracker(0)

        xvalue = Text("x =  6")
        xvalue.scale(0.7)
        xvalue.move_to(0.7 * UP + 3.7 * RIGHT)

        yvalue = Text("y = ")
        yvalue.scale(0.7)
        snum = DecimalNumber(t.get_value())
        yvalue.move_to(3.5* RIGHT)
        snum.next_to(yvalue,RIGHT)
        
        self.play(Write(xvalue))
        self.play(Write(yvalue),Write(snum))

        initialpoint =[ax.c2p(6,t.get_value())]

        newdot = Dot(point = initialpoint)
        self.play(FadeIn(newdot))
        self.wait(1)

        newdot.add_updater(lambda x : x.move_to(ax.c2p(6,t.get_value())))
        snum.add_updater(lambda x : x.set_value(t.get_value()))
        
        self.play(t.animate.set_value(7))
        self.wait(1)
        self.play(t.animate.set_value(3))
        self.wait(1)
        self.play(t.animate.set_value(6))
        self.wait(1)
        newdot.set_color(RED)
        self.wait(0.5)
        newdot.set_color(WHITE)
        self.wait(1)

        def func1(x):
            return x
        linegraph = ax.plot(func1,x_range=[0,7],color=MAROON)
        self.play(Write(linegraph))
        self.wait(1)

#manim NN101_2.py predict

class linearrealdata(Scene):        

    def construct(self):
        ax = Axes(
            x_range=(0, 60),
            y_range=(0, 4), 
            x_length=10, 
            y_length=5,
            axis_config={"include_numbers":False},
        )
        labels = ax.get_axis_labels(x_label="x", y_label="y")
        self.play(FadeIn(ax, labels))
        self.wait(1)

        data = pd.read_csv('C:\\Users\\이현서\\Documents\\dataset\\manim_exampledata.csv')
        x = torch.tensor(data['x'].values.tolist(), dtype=torch.float32)
        y = torch.tensor(data['y'].values.tolist(), dtype=torch.float32)

        w = torch.zeros(1, requires_grad=True, dtype = torch.float32)
        b = torch.zeros(1, requires_grad=True, dtype = torch.float32)

        optimizer = torch.optim.SGD([w,b], lr = 0.001)
        epochs = 30
        
        for i in range(len(x)):
            dot = Dot(ax.c2p(x[i],y[i]))
            self.add(dot)
            self.wait(0.05)
        
        self.wait(2)
        
        step = Text("Step ")
        step.scale(0.7)
        snum = MathTex("0")
        step.move_to(1.5 * DOWN + 2 * RIGHT)
        snum.next_to(step,RIGHT)

        text = Text("Cost = ")
        number = MathTex("?")
        text.move_to(2 * DOWN + 2 * RIGHT)
        number.next_to(text,RIGHT)
    
        self.play(Write(step),Write(snum))
        self.play(Write(text),Write(number))
        
        def func(x):
            return float(w[0])*x+float(b[0])

        oldgraph = ax.plot(func, x_range= [1,50],use_smoothing = False,color=MAROON)
        
        for i in range(epochs):
            H = x * w + b
            cost = torch.mean((H - y)**2)
            
            optimizer.zero_grad()
            cost.backward()
            optimizer.step()
            if (i < 8):
                graph = ax.plot(func, x_range= [1,50],use_smoothing = False,color=MAROON)
                self.play(Transform(oldgraph,graph,run_time = 1))
            elif(i < 15):
                graph = ax.plot(func, x_range= [1,50],use_smoothing = False,color=MAROON)
                self.play(Transform(oldgraph,graph,run_time = 0.6))
            else:
                graph = ax.plot(func, x_range= [1,50],use_smoothing = False,color=MAROON)
                self.play(Transform(oldgraph,graph,run_time = 0.3))


            costtext = MathTex(str(round(float(cost),4)))
            costtext.next_to(text,RIGHT)
            
            nnum = MathTex(str(i+1))
            nnum.next_to(step,RIGHT)
            self.play(ReplacementTransform(snum,nnum,run_time = 0.3),
                    ReplacementTransform(number,costtext,run_time = 0.3))
            
            number = costtext
            number.next_to(text,RIGHT)

            snum = nnum
            snum.next_to(step,RIGHT)

    #manim Linear_regression.py linearrealdata