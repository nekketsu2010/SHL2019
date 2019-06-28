import numpy as np
import pandas as pd
from sklearn import svm
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
import pickle
import os

def zscore(x, axis = None, xmean = None, xstd = None):
    zscore = (x-xmean)/xstd
    return zscore


# 最初にFFTデータと平均分散データを結合する
folderMaster = 'D:\\Huawei_Challenge2019\\challenge-2019-validate_all\\'
positions = ['Bag', 'Hips', 'Torso', 'Hand']
FFT_Mag_xyz_FolderName = '\\FFT_sample_Mag_xyz'
FFT_Acc_z_FolderName = '\\FFT_sample_Acc_z'
i = 0
for position in positions:
    folder = folderMaster + position
    sampleNameList = os.listdir(folder + FFT_Acc_z_FolderName)
    mag_xyz = np.load(folder + FFT_Mag_xyz_FolderName + "\\" + sampleNameList[0])
    acc_z = np.load(folder + FFT_Acc_z_FolderName + "\\" + sampleNameList[0])
    print(mag_xyz.shape)
    print(acc_z.shape)
    mag_xyz = mag_xyz.flatten()
    acc_z = acc_z.flatten()
    np_array = np.hstack((mag_xyz, acc_z))
    if i == 0:
        X = np_array
    else:
        X = np.vstack((X, np_array))
    for sampleName in sampleNameList[1:]:
        print(sampleName)
        mag_xyz = np.load(folder + FFT_Mag_xyz_FolderName + "\\" + sampleName)
        acc_z = np.load(folder + FFT_Acc_z_FolderName + "\\" + sampleName)
        mag_xyz = mag_xyz.flatten()
        acc_z = acc_z.flatten()
        np_array = np.hstack((mag_xyz, acc_z))
        X = np.vstack((X, np_array))
    i += 1

Label = np.load("val_Label.npy")
Y = Label.copy()
Y = np.vstack(Y, Label)
Y = np.vstack(Y, Label)

#標準化をする
with open("stdFile3_1.binaryfile", 'rb') as f:
    stdData = pickle.load(f)
print(stdData[0].shape)
print(stdData[1].shape)
xmean = stdData[0]
xstd = stdData[1]
X_std = zscore(X, xmean=xmean, xstd=xstd)
print(X_std[0:5])

np.savez('val3_1', X=X_std, Y=Y)

#ここでnanの行は0にする
X_std[np.isnan(X_std)] = 0


# 車とバスを取り出す
learn_label = [6]
index = np.where(Y == 5)
X_2 = X_std[index]
Y_2 = Y[index]
for label in learn_label:
    index = np.where(Y == label)
    X_2 = np.vstack((X_2, X_std[index]))
    Y_2 = np.hstack((Y_2, Y[index]))
#carのラベルを１にbusのラベルを2にする
Y_2[Y_2==5] = 1
Y_2[Y_2==6] = 2

print(X_2.shape)
print(Y_2.shape)

with open('model3_1.binaryfile', 'rb') as f:
    model = pickle.load(f)
predict = model.predict(X_std)
with open('predict3_1.binaryfile', 'wb') as file:
    pickle.dump(predict, file)

print(accuracy_score(Y, predict))
print(confusion_matrix(Y, predict))