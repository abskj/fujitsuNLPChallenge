import numpy as np
import torch.nn as nn
import torch
from torch.autograd import Variable
from torch import optim
import torch.nn.functional as F
dtype = torch.float
device = torch.device("cuda:0")

class selqa_net(nn.Module):
	def __init__(self):

		super(selqa_net, self).__init__()

		self.conv1=torch.nn.Conv2d(1,10,5,stride=1,padding=1)
		self.pool1=torch.nn.MaxPool3d(2,stride=2,padding=0)
		self.pool2=torch.nn.MaxPool3d(3,stride=2,padding=0)
		self.dropout1=torch.nn.Dropout(p=0.1, inplace=False)
		self.linear=torch.nn.Linear(2812,1)

	def forward(self,x):
		x.type(torch.FloatTensor)
		x.unsqueeze_(1)
		x=F.relu(self.conv1(x))
		# print(x.shape)
		x=self.pool1(x)
		# print(x.shape)
		x=self.dropout1(x)
		# print(x.shape)
		x=self.pool2(x)
		# print(x.shape)
		x=x.view(100,-1)
		# print(x.shape)
		x=self.linear(x)
		x=x*10
		return x

