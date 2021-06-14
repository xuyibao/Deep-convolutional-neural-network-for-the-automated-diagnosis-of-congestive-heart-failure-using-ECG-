# filenames是训练数据文件名称列表，labels是标签列表
import os

import torch
from PIL import Image
from torch.utils.data import Dataset
import numpy as np

class MyDataset(Dataset):
    def __init__(self, filepath, transform):
        self.filepath = filepath
        self.labels = {'normal':'0', 'chf':'1'}
        self.transform = transform

    def __len__(self):
        dirs = os.listdir(self.filepath)
        f1 = open(self.filepath + '/' + dirs[0] + '/' + dirs[0]+'.txt')
        f2 = open(self.filepath + '/' + dirs[1] + '/' + dirs[1]+'.txt')
        l1 = f1.readlines()
        l2 = f2.readlines()
        print(len(l1))
        print(len(l2))
        return len(l1)+len(l2)

    def __getitem__(self, idx):
        dirs = os.listdir(self.filepath)
        f1 = open(self.filepath + '/' + dirs[0] + '/' + dirs[0]+'.txt')
        f2 = open(self.filepath + '/' + dirs[1] + '/' + dirs[1]+'.txt')
        l1 = f1.readlines()
        l2 = f2.readlines()
        if(idx>=len(l1)):
            return self.transform(np.array([l2[idx-len(l1)].strip('\n').split(' ')], dtype=np.float32)), self.labels[dirs[1]]
        return self.transform(np.array([l1[idx].strip('\n').split(' ')], dtype=np.float32)), self.labels[dirs[0]]



class Dataset1(Dataset):
    def __init__(self, filepath, data_type, transform):
        self.filepath = filepath
        self.transform = transform
        self.labels = {'chf': 1, 'normal':0}
        self.data_type = data_type
        self.file = open(self.filepath + '/' +self.data_type + '/z_' +self.data_type+'.txt')
        self.records = self.file.readlines()

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        return self.transform(np.array([self.records[idx].strip('\n').split(' ')], dtype=np.float32)),(self.labels[self.data_type])



