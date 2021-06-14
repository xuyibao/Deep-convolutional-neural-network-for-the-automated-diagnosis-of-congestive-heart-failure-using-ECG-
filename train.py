import torch
from torch import nn
import tqdm
from data_loader import Dataset1
import torchvision
import numpy as np
from logger import log_info
from model import Cnn_Model


#日志对象。保存训练记录
log = log_info()


#获取k倍交叉验证的第i次数据 i=1,2,...,k
def get_k_fold_data(data1, data2, chf_data_len, normal_data_len, i):
    chf_one_len = chf_data_len // 10
    normal_one_len = normal_data_len // 10
    chf_cut_start = i * chf_one_len
    chf_cut_end = chf_one_len * (i + 1)
    normal_cut_start = i * normal_one_len
    normal_cut_end = normal_one_len * (i + 1)  # 不包含最后一位
    test_data = []
    test_lable = []
    train_data = []
    train_lable = []
    # 获取验证集
    for z in tqdm.tqdm(range(chf_cut_start, chf_cut_end)):
        re, le = data1[z]
        test_data.append(np.array(re))
        test_lable.append(le)
    for z in tqdm.tqdm(range(normal_cut_start, normal_cut_end)):
        re, le = data2[z]
        test_data.append(np.array(re))
        test_lable.append(le)
    # 获取训练集
    for z in tqdm.tqdm(range(chf_cut_start)):
        re, le = data1[z]
        train_data.append(np.array(re))
        train_lable.append(le)
    for z in tqdm.tqdm(range(chf_cut_end, chf_data_len)):
        re, le = data1[z]
        train_data.append(np.array(re))
        train_lable.append(le)
    for z in tqdm.tqdm(range(normal_cut_start)):
        re, le = data2[z]
        train_data.append(np.array(re))
        train_lable.append(le)
    for z in tqdm.tqdm(range(normal_cut_end, normal_data_len)):
        re, le = data2[z]
        train_data.append(np.array(re))
        train_lable.append(le)

    # 打乱数据
    np.random.seed(i)
    np.random.shuffle(train_data)
    np.random.seed(i)
    np.random.shuffle(train_lable)

    # 将数据转化为张量
    train_data = torch.tensor(np.squeeze(np.array(train_data), 1))
    train_lable = torch.tensor(np.array(train_lable)).long()
    test_data = torch.tensor(np.squeeze(np.array(test_data), 1))
    test_lable = torch.tensor(np.array(test_lable)).long()
    return train_data, train_lable, test_data, test_lable


#训练模型
def train_model(train_data, train_lable, test_data, test_lable, model):
    # 定义训练轮数
    epochs = 5
    # 定义批次大小
    batch_size = 10
    # 初始化数据
    inputs = train_data
    target = train_lable
    test_data = test_data
    test_lable = test_lable
    # 定义损失函数
    loss = nn.CrossEntropyLoss()
    # 定义优化器  学习率 lr = 0.0003 动量 m = 0.3
    optim = torch.optim.SGD(model.parameters(), lr=0.0003, momentum=0.3)
    running_loss = 0
    k = 0  # 记录读取了多少数据
    batch_records = []
    batch_tar = []
    #是被交叉验证的平均基准值
    ave_acc = 0
    ave_ppv = 0
    ave_sen = 0
    ave_spe = 0
    for t in range(epochs):
        for x in range(len(inputs)):
            k = k + 1
            re = inputs[x].numpy()
            tar = target[x].numpy()
            # print(type(re))
            batch_records.append(re)
            batch_tar.append(tar)

            if (k == batch_size):
                batch_records = torch.tensor(np.array(batch_records))
                batch_tar = torch.tensor(np.array(batch_tar))
                # 将数据输入到模型中
                outputs = model(batch_records) / 0.01
                # 计算损失，更新参数
                result_loss = loss(outputs, batch_tar)
                optim.zero_grad()
                result_loss.backward()
                optim.step()
                running_loss = result_loss
                # 清空记录
                k = 0
                batch_records = []
                batch_tar = []
        print("--第{}轮loss:{}-----------".format(t, running_loss))
        torch.save(model, './deep_cnn/model/model.pth')
        # '''    （Acc = (TP + TN) / (TP + FP + TN + FN)
        #     PPV = TP / (TP + FP)
        #     Sen = TP / (TP + FN)
        #     Spe = TN / (TN + FP) ）'''
        # 进行验证
        acc = 0
        tp = 0
        fp = 0
        fn = 0
        tn = 0

        test_output = np.array((torch.max(model(test_data), 1)).indices)
        for i in range(len(test_output)):
            if test_output[i] == 1 and test_lable[i] == 1:
                tp = tp + 1
                acc = acc + 1
            elif test_output[i] == 0 and test_lable[i] == 1:
                fn = fn + 1
            elif test_output[i] == 1 and test_lable[i] == 0:
                fp = fp + 1
            else:
                tn = tn + 1
                acc = acc + 1
        # for k in len(test_lable):
        ave_acc = ave_acc + acc / len(test_lable)
        ave_sen = ave_sen + tp / max(tp + fn, 1)
        ave_spe = ave_spe + tn / max(tn + fp, 1)
        ave_ppv = ave_ppv + tp / max(tp + fp, 1)
        print("第{}轮准确率acc={}".format(t, acc / len(test_lable)))
        print("第{}轮特异性spe={}".format(t, tn / max(tn + fp, 1)))
        print("第{}轮敏感性sen={}".format(t, tp / max(tp + fn, 1)))
        print("第{}轮ppv={}".format(t, tp / max(tp + fp, 1)))
    return ave_acc/epochs, ave_sen/epochs, ave_spe/epochs, ave_ppv/epochs


def k_fold_train(data1, data2, model, k):

    chf_data_len = len(data1)
    normal_data_len = len(data2)
    k_fold_acc = 0
    k_fold_ppv = 0
    k_fold_sen = 0
    k_fold_spe = 0

    for i in range(k):
        print("开始第{}组交叉数据获取".format(i))
        log.logger("开始第{}组交叉数据获取".format(i))
        train_data, train_lable, test_data, test_lable = get_k_fold_data(data1, data2, chf_data_len, normal_data_len, i)
        log.logger("第{}组交叉数据获取完成".format(i))
        print("开始第{}组交叉验证训练".format(i))
        log.logger("开始第{}组交叉验证训练".format(i))
        k_acc, k_sen, k_spe, k_ppv = train_model(train_data, train_lable, test_data, test_lable, model)
        k_fold_acc = k_fold_acc + k_acc
        k_fold_ppv = k_fold_ppv + k_ppv
        k_fold_sen = k_fold_sen + k_sen
        k_fold_spe = k_fold_spe + k_spe
        log.logger("第{}组准确率acc={}".format(i, k_acc))
        log.logger("第{}组敏感性sen={}".format(i, k_sen))
        log.logger("第{}组特异性spe={}".format(i, k_spe))
        log.logger("第{}组ppv={}".format(i, k_ppv))

    log.logger("{}倍交叉验证平均准确率acc={}".format(k, (k_fold_acc / k)))
    log.logger("{}倍交叉验证平均敏感性sen={}".format(k, (k_fold_sen / k)))
    log.logger("{}倍交叉验证平均特异性spe={}".format(k, (k_fold_spe / k)))
    log.logger("{}倍交叉验证平均ppv={}".format(k, (k_fold_ppv / k)))

if __name__ == '__main__':
    # 加载数据
    set_ = 'setC'
    data_path = './deep_cnn/pro_data/'+set_
    data_set_chf = Dataset1(filepath=data_path, data_type='chf', transform=torchvision.transforms.ToTensor())
    data_set_normal = Dataset1(filepath=data_path, data_type='normal', transform=torchvision.transforms.ToTensor())
    print("数据加载完成")
    print("开始训练")
    # print(data_set_chf[0])
    # 定义模型/加载模型
    model = Cnn_Model()
    # model = torch.load('./deep_cnn/model/model_setA.pth')
    # model = torch.load('./deep_cnn/model/model_setB.pth')
    # model = torch.load('./deep_cnn/model/model_setC.pth')
    # model = torch.load('./deep_cnn/model/model_setD.pth')

    # 进行十倍交叉验证
    k_fold_train(data_set_chf, data_set_normal, model, 10)
    torch.save(model, './deep_cnn/model/model_'+set_+'.pth')
