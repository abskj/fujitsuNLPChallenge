import numpy as np
import torch.nn as nn
import torch
from torch.autograd import Variable
from torch import optim

dtype = torch.float
device = torch.device("cuda:0")

class ConvNet(torch.nn.Module):
    def __init__(self,outDim):
        super(ConvNet, self).__init__()
        self.model=torch.nn.Sequential()
        self.model.add_module('conv_layer',torch.nn.Conv2d(3,outDim,5,stride=1))
    def forward(self,x):
        
        out=self.model(x)
        return out

X=torch.randn(100,3,200,200, device=device, dtype=dtype)
y=torch.randn(100,5,196,196, device=device, dtype=dtype)
cnn=ConvNet(5)
cnn=cnn.cuda()
learning_rate=1e-4
lossFunc=nn.MSELoss(size_average=False)
optimizer=torch.optim.Adam(cnn.parameters(),lr=learning_rate)
for t in range(10000):
    y_pred=cnn.model(X)
    loss=lossFunc(y_pred,y)
    print(t,loss.item())

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


