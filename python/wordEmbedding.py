# TODO:read the data from glove vectors
# TODO: create the tensor for use in pytorch
import io
import torch
import torch.nn as nn

with open('./../datasets/glove/glove.6B.50d.txt', encoding="utf8") as fp:
   for line in fp:
       list = {}
       line2arr = line.split(' ')
       for value in line2arr[1:]:
           list.append(float(value))
           pass
        my_tensor=torch.tensor(list)
        print(my_tensor)
                
