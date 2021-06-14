import torch
import torchvision

from data_loader import Dataset1
from logger import log_info

#定义记录对象，用于保存训练记录
log = log_info()

#加载模型
model = torch.load('./deep_cnn/model/model_satD.pth')

#加载数据库
data_path = './deep_cnn/pro_data/setD'
data_set_chf = Dataset1(filepath=data_path, data_type='chf', transform=torchvision.transforms.ToTensor())
data_set_normal = Dataset1(filepath=data_path, data_type='normal', transform=torchvision.transforms.ToTensor())

#进行交叉验证