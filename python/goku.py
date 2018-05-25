import numpy as np
import torch.nn as nn
import torch
from torch.autograd import Variable
from torch import optim
dtype = torch.float
device = torch.device("cuda:0")

class selqa_net(nn.Module):
  def __init__(self,outDim):
		super(selqa_net, self)
		self.model=torch.nn.Sequential()

		self.conv1=Conv2d(1,10,5,stride=1,padding=1)
		self.pool1=torch.nn.MaxPool3d(2,stride=2,padding=0)
		self.dropout1=torch.nn.Dropout(p=0.1, inplace=False)
		self.linear=