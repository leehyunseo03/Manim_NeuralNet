from manim import *
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd

class logisticrealdata(Scene):        

    def construct(self):
        ax = Axes(
            x_range=(0,30),
            y_range=(0, 1), 
            x_length=10, 
            y_length=5,
            axis_config={"include_numbers":False},
        )
        labels = ax.get_axis_labels(x_label="x", y_label="y")
        self.play(FadeIn(ax, labels))
        self.wait(1)

        data = pd.read_csv('C:\\Users\\이현서\\Documents\\dataset\\Breast_cancer_train.csv')
        x = torch.tensor(data.loc[:, ['mean_radius']].values.tolist(), dtype=torch.float32)
        y = torch.tensor(data['diagnosis'].values.tolist(), dtype=torch.float32).reshape(-1,1)

        w = torch.zeros((1,1), requires_grad=True, dtype = torch.float32)
        b = torch.zeros(1, requires_grad=True, dtype = torch.float32)

        optimizer = torch.optim.SGD([w,b], lr = 0.08)
        epochs = 3500
        amount = 200
        
        for i in range(amount):
            dot = Dot(ax.c2p(float(x[i]),float(y[i])),color = RED)
            self.add(dot)
            self.wait(0.02)
        
        self.wait()
        
        step = Text("Step ")
        step.scale(0.7)
        snum = MathTex("0")
        step.move_to(2.3* UP + 2 * RIGHT)
        snum.next_to(step,RIGHT)

        text = Text("Cost = ")
        number = MathTex("?")
        text.move_to(1.8 * UP + 2 * RIGHT)
        number.next_to(text,RIGHT)
    
        self.play(Write(step),Write(snum))
        self.play(Write(text),Write(number))
        
        def func(x):
            return 1/(1+np.exp(-1*(float(w[0])*x+float(b[0]))))

        oldgraph = ax.plot(func,x_range= [0,30],use_smoothing = False,color=MAROON)
        
        a = 0

        for i in range(1,epochs):
            H = torch.sigmoid(x.matmul(w)+b)
            cost = F.binary_cross_entropy(H, y)

            optimizer.zero_grad()
            cost.backward()
            optimizer.step()
            if (a < 15):
                if(i % 100 == 0):
                    graph = ax.plot(func, x_range= [1,30],use_smoothing = False,color=MAROON)
                    self.play(Transform(oldgraph,graph,run_time = 0.3))

                    costtext = MathTex(str(round(float(cost),4)))
                    costtext.next_to(text,RIGHT)
                        
                    nnum = MathTex(str(i))
                    nnum.next_to(step,RIGHT)
                    self.play(ReplacementTransform(snum,nnum,run_time = 0.3),
                    ReplacementTransform(number,costtext,run_time = 0.3))
                        
                    number = costtext
                    number.next_to(text,RIGHT)

                    snum = nnum
                    snum.next_to(step,RIGHT)
                    a += 1
            elif (a < 20):
                if(i % 100 == 0):
                    graph = ax.plot(func, x_range= [1,30],use_smoothing = False,color=MAROON)
                    self.play(Transform(oldgraph,graph,run_time = 0.6))

                    costtext = MathTex(str(round(float(cost),4)))
                    costtext.next_to(text,RIGHT)
                        
                    nnum = MathTex(str(i))
                    nnum.next_to(step,RIGHT)
                    self.play(ReplacementTransform(snum,nnum,run_time = 0.3),
                    ReplacementTransform(number,costtext,run_time = 0.3))
                        
                    number = costtext
                    number.next_to(text,RIGHT)

                    snum = nnum
                    snum.next_to(step,RIGHT)
                    a += 1
            else:
                if(i % 100 == 0):
                    graph = ax.plot(func, x_range= [1,30],use_smoothing = False,color=MAROON)
                    self.play(Transform(oldgraph,graph,run_time = 0.3))

                    costtext = MathTex(str(round(float(cost),4)))
                    costtext.next_to(text,RIGHT)
                        
                    nnum = MathTex(str(i))
                    nnum.next_to(step,RIGHT)
                    self.play(ReplacementTransform(snum,nnum,run_time = 0.3),
                    ReplacementTransform(number,costtext,run_time = 0.3))
                        
                    number = costtext
                    number.next_to(text,RIGHT)

                    snum = nnum
                    snum.next_to(step,RIGHT)
                    a += 1

            
                
        print(cost)
        print(w)

        
            
    #manim NonLinear_regression.py logisticrealdata