import tqdm
import numpy as np
from sklearn import preprocessing
data_path = './deep_cnn/pro_data/setC'
database = '/normal'
f = open(data_path+database+'/'+database+'.txt')
records =np.array(f.readlines())
# record = (records[0].strip('\n').split(' '))
# record = np.array([float(x) for x in record])
# print(record.shape)
# mean = np.mean(record)
# var = np.var(record)
# z_record = np.array([((x - mean) / var) for x in record])
# z_record = z_record.reshape(-1,500)
# # with open('D:/deep_cnn/pro_data/setD/normal/z_normal.txt',"a") as f:
# np.savetxt('D:/deep_cnn/pro_data/setD/normal/z_normal.txt',z_record,'%.6f')
#
# print(z_record.shape)
with open('D:/deep_cnn/pro_data/setC/normal/z_normal.txt',"a") as f:
    for re in tqdm.tqdm((records)):
        record = (re.strip('\n').split(' '))
        record = np.array([float(x) for x in record])
        z_record = record.reshape(500,1)
        z_record = preprocessing.scale(z_record)
        z_record = z_record.reshape(1,500)
        np.savetxt(f,z_record,'%.6f')
f.close()



