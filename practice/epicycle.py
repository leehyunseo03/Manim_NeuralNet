import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

data = pd.read_csv('C:\\Users\\이현서\\Documents\\dataset\\Breast_cancer_train.csv')

xdata = torch.tensor(data.loc[:, ['mean_radius','mean_texture','mean_perimeter','mean_area','mean_smoothness']].values.tolist(), dtype=torch.float32)
diagnosis = torch.tensor(data['diagnosis'].values.tolist(), dtype=torch.float32).reshape(-1,1)

w = torch.zeros((5,1), requires_grad=True, dtype = torch.float32)
b = torch.zeros(1, requires_grad=True, dtype = torch.float32)

optimizer = torch.optim.SGD([w,b], lr = 0.00001)

nb_epochs = 10000

for i in range(nb_epochs):
    H = torch.sigmoid(xdata.matmul(w)+b)
    cost = F.binary_cross_entropy(H, diagnosis)
    
    optimizer.zero_grad()
    cost.backward()
    optimizer.step()

def get_prediction(x_predict):
    y_predict = torch.sigmoid(x_predict.matmul(w)+b)
    prediction = (y_predict >= torch.Tensor([0.5])).int()
    return prediction

if __name__=="__main__":
    x_test = torch.tensor([[17.99,10.38,122.8,1001,0.1184],[13.54,14.36,87.46,566.3,0.09779]], dtype=torch.float32)
    print(get_prediction(x_test))
    print(cost)
    print(w)