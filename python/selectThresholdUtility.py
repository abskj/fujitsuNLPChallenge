#read data from train
#use spacy word vector
#create input vector

#vary the number of max_number of_tokens to find number of missed sentences

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

nlp=spacy.load("en_core_web_lg")

with open('.\..\datasets\selqa-evaluater\SelQA-ass-train.json') as fp:
    s=fp.read()
arr=s.split('\n')
print(arr[0])
for i in range( 10):
    try:
        obj=json.loads(arr[i])

        total_sents+=len(obj["sentences"])+1
        #########with split
        # words=obj["question"].split(' ')
        # if(len(words)>max_no_of_tokens):
        #     no_missed+=1
        # for word in words:
        #     total_tokens+=1
        #     if word not in word_to_ix:
        #         word_to_ix[word]=len(word_to_ix)

        # for sentence in obj["sentences"]:
        #     if(len(sentence.split(' '))>max_no_of_tokens):
        #         no_missed+=1
        #     for word in sentence.split(' '):
        #         total_tokens+=1
        #         if word not in word_to_ix:
        #             word_to_ix[word]=len(word_to_ix)
        # pass
        ##with spacy
        # 
        print(i)
        q=nlp(obj["question"])
        if(len(q)>max_no_of_tokens):
            no_missed+=1
            q_missed+=1
        total_sents+=1
        for sentence in obj["sentences"]:
            y=nlp(sentence)
            for token in y:
                print(token.vector)
                print(token.vector.shape)
            if(len(y)>max_no_of_tokens):
                no_missed+=1
            total_sents+=1


    except ValueError:
        pass
print('no of missed sentences:'+str(no_missed)+" out of "+str(total_sents)+" "+str(q_missed))

# results
# for max=40
#     no of missed sentences:8226 out of 144474 1