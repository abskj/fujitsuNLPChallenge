from piccolo import myDataset
from python.goku import selqa_net
import torch.nn as nn
import torch.utils as utils
import torch.optim as optim

# if __name__ == 'main':
    
    
# test_data=myDataset()
learning_rate=1e-4
train_dataset=myDataset('.\datasets\selqa-evaluater\SelQA-ass-train.json')
trainloader=utils.data.DataLoader(train_dataset,batch_size=5,shuffle=True,num_workers=2)
criterion=nn.CrossEntropyLoss()
model=selqa_net()
model=model.cuda()
optimizer=optim.Adam(model.parameters(),lr=learning_rate)

for epoch in range(2):  # loop over the dataset multiple times

    running_loss = 0.0
    for i,j  in enumerate(trainloader, 0):
        print(i,j)
    print(trainloader)
    #print(enumerate(trainloader))
    # for i, data in enumerate(trainloader, 0):
    #     # get the inputs
    #     inputs, labels = data

    #     # zero the parameter gradients
    #     optimizer.zero_grad()

    #     # forward + backward + optimize
    #     outputs = net(inputs)
    #     loss = criterion(outputs, labels)
    #     loss.backward()
    #     optimizer.step()

    #     # print statistics
    #     running_loss += loss.item()
    #     print(str(i)+" "+str(loss.item()))
    #     if i % 20 == 19:    # print every 2000 mini-batches
    #         print('[%d, %5d] loss: %.3f' %
    #             (epoch + 1, i + 1, running_loss / 2000))
    #         running_loss = 0.0

print('Finished Training')