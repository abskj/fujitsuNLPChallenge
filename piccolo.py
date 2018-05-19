import torch
from torch.utils import data
import numpy as np
# local data address: C:\Users\sauron\Downloads\word_embeddings_train01.csv

class Dataset(data.Dataset):
  def __init__(self,filename1):
    self.length=0
    self.filename=filename1
  def __len__(self):
    if self.length ==0:
      with open(self.filename) as infile:
        for line in infile:
          self.length+=1
      return self.length
    else:
      return self.length
  def __getitem__(self,idx):
    i=0
    with open(self.filename) as infile:
      for line in infile:
        i+=1
        if i==idx:
          data=np.fromstring(line,sep=',')
          data=data[:24000].reshape(80,300)
    x=torch.from_numpy(data)
    return x

  def __getitemy__(self,idx):
    i=0
    with open(self.filename) as infile:
      for line in infile:
        i+=1
        if i==idx:
          data=np.fromstring(line,sep=',')
          data=data[idx]
    x=data
    return x

ds=Dataset(r'C:\Users\sauron\Downloads\word_embeddings_train01.csv')
print(ds[5])


          
    