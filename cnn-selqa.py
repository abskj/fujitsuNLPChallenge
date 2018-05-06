

import torch
import torch.nn as nn
import spacy
import numpy as np
import json

max_no_of_tokens=40
no_missed=0
total_tokens=0
total_sents=0
q_missed=0
word_to_ix={}

#read data into dataset

nlp=spacy.load("en_core_web_md")
list_of_sents=[]

with open('.\datasets\selqa-evaluater\SelQA-ass-train.json') as fp:
    s=fp.read()
arr=s.split('\n')
print(arr[0])
for i in range( 10):
    try:
        list_of_tokens=[]
        obj=json.loads(arr[i])

        total_sents+=len(obj["sentences"])+1
        print(i)
        q=nlp(obj["question"])
        for i in range(40):
            if(i<len(q)):
                temp=torch.Tensor(q[i].vector)
                # print(temp.shape)
                list_of_tokens.append(temp)
            else:
                list_of_tokens.append(torch.zeros(300))
        tokens_question=list_of_tokens.copy()
        print(len(obj["sentences"]))
        for sentence in obj["sentences"]:
            
            list_of_tokens=tokens_question.copy()
            y=nlp(sentence)
            for i in range(40):
                if(i<len(y)):
                    temp=torch.Tensor(y[i].vector)
                    list_of_tokens.append(temp)
                else:
                    list_of_tokens.append(torch.zeros(300))
                
            # print(len(list_of_tokens))
            sentence_to_embedding=torch.stack(list_of_tokens,dim=0)
            list_of_sents.append(sentence_to_embedding)
        x=torch.stack(list_of_sents,0)
        print(x.shape)


    except ValueError:
        pass