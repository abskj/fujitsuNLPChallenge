import torch
from torch.utils.data import Dataset, DataLoader

class selqaDataset(Dataset):
    file_train='.\datasets\selqa-evaluater\SelQA-ass-train.json'
    file_test='.\datasets\selqa-evaluater\SelQA-ass-test.json'
    def int