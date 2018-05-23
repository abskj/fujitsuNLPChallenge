import numpy as np
import torch
import spacy
from torch.utils.data import Dataset, DataLoader
import json


class myDataset(Dataset):
	def __init__(self,filename,y=0):
		self.nlp=spacy.load("en_core_web_md")
		self.filename=filename
		self.length=0
		with open(filename) as fp:
			s=fp.read()
		self.arr=s.split('\n')
		self.y=y


	def __len__(self):
		if self.length == 0:
			for i in range(len(self.arr)-1):
				try:
					obj=json.loads(self.arr[i])
					self.length+=len(obj["sentences"])
				except ValueError:
					print('got a Value error(most probably improper JSON at '+str(i))
					pass
		return self.length
	
	def __getitem__(self,idx):
		index=0
		for i in range(len(self.arr)):
			obj=json.loads(self.arr[i])
			no_of_ans=len(obj["sentences"])
			
			if((index+no_of_ans)>idx) :
				break
			index+=no_of_ans
		print(i)
		if self.y :
			if (idx-index) in obj["candidates"]:
				return 1
			else:
				return 0
		question=self.nlp(obj["question"])
		answer=self.nlp(obj["sentences"][idx-index])
		list_of_tokens=[]
		for i in range(40):
			if(i<len(question)):
				list_of_tokens.append(torch.Tensor(question[i].vector).cuda(0))
			else:
				list_of_tokens.append(torch.zeros(300).cuda(0))
		for i in range(40):
			if(i<len(answer)):
    				list_of_tokens.append(torch.Tensor(answer[i].vector).cuda(0))
			else:
				list_of_tokens.append(torch.zeros(300).cuda(0))
		sentence_to_embedding=torch.stack(list_of_tokens,dim=0)
		return sentence_to_embedding
		
		

d=myDataset('.\datasets\selqa-evaluater\SelQA-ass-train.json',y=0)
print(len(d))
print(len(d))
print(len(d))
print(len(d))
print(len(d))
print(len(d))
print(d[50000])
print(d[2000])
print(d[20500])
print(d[52000])
for i in range(200):
	print(d[i])