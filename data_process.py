import random

import wfdb
import numpy as np

# 读取编号为data的一条心电数据
from scipy.interpolate import interpolate


def read_ecg_data(path):
    '''
    读取心电信号文件
    sampfrom: 设置读取心电信号的起始位置，sampfrom=0表示从0开始读取，默认从0开始
    sampto：设置读取心电信号的结束位置，sampto = 1500表示从1500出结束，默认读到文件末尾
    channel_names：设置设置读取心电信号名字，必须是列表，channel_names=['MLII']表示读取MLII导联线
    channels：设置读取第几个心电信号，必须是列表，channels=[0, 3]表示读取第0和第3个信号，注意信号数不确定
    '''
    #path是读取的新店路径
    # 读取所有导联的信号
    record = wfdb.rdrecord(path, sampfrom=0, channel_names=['ECG2'])
    # 仅仅读取“MLII”导联的信号
    # record = wfdb.rdrecord('../ecg_data/' + data, sampfrom=0, sampto=1500, channel_names=['MLII'])
    # 仅仅读取第0个信号（MLII）
    # record = wfdb.rdrecord('../ecg_data/' + data, sampfrom=0, sampto=1500, channels=[0])

    # 查看record类型
    # print(type(record))
    # 查看类中的方法和属性
    # print(dir(record))

    # 获得心电导联线信号，本文获得是MLII和V1信号数据
    # print(record.p_signal)
    # print(np.shape(record.p_signal))
    # 查看导联线信号长度，本文信号长度1500
    # print(record.sig_len)
    # 查看文件名
    # print(record.record_name)
    # 查看导联线条数，本文为导联线条数2
    # print(record.n_sig)
    # 查看信号名称（列表），本文导联线名称['MLII', 'V1']
    # print(record.sig_name)
    # 查看采样率
    # print(record.fs)

    '''
    读取注解文件
    sampfrom: 设置读取心电信号的起始位置，sampfrom=0表示从0开始读取，默认从0开始
    sampto：设置读取心电信号的结束位置，sampto=1500表示从1500出结束，默认读到文件末尾
    '''
    # annotation = wfdb.rdann(path, 'atr')
    # # 查看annotation类型
    # print(type(annotation))
    # # 查看类中的方法和属性
    # print(dir(annotation))
    #
    # # 标注每一个心拍的R波的尖锋位置的信号点，与心电信号对应
    # print(annotation.sample)
    # # 标注每一个心拍的类型N，L，R等等
    # print(annotation.symbol)
    # # 被标注的数量
    # print(annotation.ann_len)
    # # 被标注的文件名
    # print(annotation.record_name)
    # # 查看心拍的类型
    # print(wfdb.show_ann_labels())

    # 画出数据
    # draw_ecg(record.p_signal)
	# 返回一个numpy二维数组类型的心电信号，shape=(65000,1)
    return record.p_signal


def build_data_set(path,database, records):
    for re in records:
        ecg_records = np.array(read_ecg_data(path+database+'/'+re))
        # cut = (len(ecg_records)//500)*500
        cut = (len(ecg_records) //128)
        x = np.array(np.linspace(1,cut*128,cut*128))
        # x = x.reshape(-1,1)
        print(x.shape)
        newx = np.array(np.linspace(1,cut*128,cut*250))
        # newx = newx.reshape(-1,1)
        # print(newx.shape)
        ecg_records = ecg_records[:cut*128]
        print(ecg_records.shape)
        ecg_records = data_up(x,newx,ecg_records)
        # print(ecg_records.shape)
        ecg_records = ecg_records.reshape(-1,500)
        # print(ecg_records)
        print(ecg_records.shape)
        np.savetxt('D:/deep_cnn/pro_data/'+database+'/'+re+'.txt',ecg_records,'%0.6f')


def data_up(x,newx,y):
    f = interpolate.interp1d(x, y, kind='quadratic',axis=-2)
    newy = f(newx)
    return newy


def build_set(path):
    file = open(path,'r')
    txt = file.readlines()
    print(len(txt))
    ran = random.sample(range(0,len(txt)), 3906)
    with open('D:/deep_cnn/pro_data/setA/normal/normal.txt',"a") as f:
        for i in ran:
            f.write(txt[i])
    f.close()
    file.close()



if __name__ == "__main__":
    path = "./deep_cnn/sourcedata"
    database = "/mit-bit"
    f = open(path+database+'/RECORDS')
    # read_ecg_data(path+database+'/'+'16265')
    records=[]
    for re in f.readlines():
        records.append(re.strip('\n'))
    print(records)
    for re in records:
        build_set('./deep_cnn/pro_data/mit-bit/'+re+'.txt')
    build_data_set(path, database, records)


