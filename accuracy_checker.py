from piccolo import myDataset
from python.goku import selqa_net
import torch.nn as nn
import torch.utils as utils
import torch.optim as optim
import torch
model=torch.load('.\models\krillin00.pt')
test_data=myDataset('.\datasets\selqa-evaluater\SelQA-ass-test.json')
test_loader=utils.data.DataLoader(test_data,batch_size=100,shuffle=False,num_workers=0)
total_correct=0
for i,data in enumerate(test_loader,0):
    inputs,labels=data
    inputs=inputs.cuda()
    labels=labels.cuda()
    labels=labels.type(torch.cuda.FloatTensor)
    labels.unsqueeze_(1)
    outputs = model(inputs).cuda()
    outputs=outputs.round()
    c=labels==outputs
    correct=torch.nonzero(c).size(0)
    print('for batch '+str(i)+' accuracy is '+str(correct/100))
print('Total accuracy is '+str(total_correct/len(test_data)))