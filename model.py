import torch
import torchvision
from torch import nn
from torch.nn import Conv2d, MaxPool2d, Flatten, Linear, Conv1d, MaxPool1d, ReLU, Softmax
import numpy as np
from sklearn import preprocessing
from data_loader import Dataset1


class Cnn_Model(nn.Module):
    def __init__(self):
        super(Cnn_Model,self).__init__()
        self.model1 = nn.Sequential(
            Conv1d(1,5,5),
            ReLU(),
            MaxPool1d(2,2),
            Conv1d(5,5,5),
            ReLU(),
            MaxPool1d(2, 2),
            Conv1d(5,10,3),
            ReLU(),
            MaxPool1d(2, 2),
            Conv1d(10, 10, 3),
            ReLU(),
            MaxPool1d(2, 2),
            Flatten(),
            ReLU(),
            Linear(290,40),
            ReLU(),
            Linear(40,20),
            Linear(20,2),
            Softmax(dim=1)

        )
    def forward(self,x):
        x = self.model1(x)
        return x


# test = Cnn_Model()
# print(test)
# input = torch.ones((5,1,500))
# tar = torch.ones(5,dtype=torch.long)
# print(tar.shape)
# out = test(input)
# # out = torch.max(test(input),1).indices
# loss = nn.CrossEntropyLoss()
# result_loss = loss(out, tar)
# # out = test(input)
# print(out.shape)
# data_path = './deep_cnn/pro_data/setD'
# data_set_chf = Dataset1(filepath=data_path, data_type='chf', transform=torchvision.transforms.ToTensor())
# data_set_normal = Dataset1(filepath=data_path, data_type='normal', transform=torchvision.transforms.ToTensor())
# for i in range(22000,23000):
#     re,le = data_set_normal[i]
#     print(i)
#     print(re)

# re ,l = data_set_normal[22000]
# print(re)
# re= re.numpy()

# re = re.reshape(500,1)

# # re = preprocessing.scale(re)

#
# re = re.reshape(1,1,500)

# input =[]
# tar = []
# r1,l1 = data_set_chf[0]
# r2,l2 = data_set_chf[1]
# r1 = np.array(r1)
# r2 = np.array(r2)
# print(type(l1))
# tar.append(np.array(l1))
# tar.append(np.array(l2))
# input.append(r1)
# input.append(r2)
# input = np.array(input)
# tar = np.array(tar)
# input = torch.tensor(np.squeeze(input,1))
# tar = torch.tensor(tar)
# print(input.shape)
# print(tar.shape)
