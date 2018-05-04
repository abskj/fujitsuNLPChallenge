# TODO:create a dictionary for our data
# TODO: create the tensor for use in pytorch
import io
import torch
import torch.nn as nn
import json

word_to_iy = {}
device=torch.device("cpu")
embeddings=torch.zeros(100,device=device,dtype=torch.float)
with open('./../datasets/glove/glove.6B.100d.txt', encoding="utf8") as fp:
    tensor_list=[]
    for line in fp:
        list = []
        
        line2arr = line.split(' ')
        word=line2arr[0]
        if word not in word_to_iy:
            word_to_iy[word]=len(word_to_iy)
        for value in line2arr[1:]:
           list.append(float(value))
           pass
        my_tensor=torch.tensor(list)
        tensor_list.append(my_tensor)
        
    embeddings=torch.stack((tensor_list),1)
    print(embeddings.shape)

print(embeddings)        
word_to_ix={}
with open('.\..\datasets\selqa-evaluater\SelQA-ass-train.json') as fp:
    s=fp.read()
arr=s.split('\n')
print(arr[0])
for i in range( len(arr)):
    try:
        obj=json.loads(arr[i])
        words=obj["question"].split(' ')
        for word in words:
            if word not in word_to_ix:
                word_to_ix[word]=len(word_to_ix)

        for sentence in obj["sentences"]:
            for word in sentence.split(' '):
                if word not in word_to_ix:
                    word_to_ix[word]=len(word_to_ix)
        pass
    except ValueError:
        pass

print('dict size of train data:'+str(len(word_to_ix)))
print('dict size of glove data:'+str(len(word_to_iy)))
print(str(word_to_ix.items()<=word_to_iy.items()))
print(word_to_ix)
print(word_to_iy.items())

for key,value in word_to_iy.items():
    if key not in word_to_ix:
        print(key)