import numpy as np
import pandas as pd
from sklearn import svm
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
import pickle
import os

def zscore(x, axis = None, xmean = None, xstd = None):
    if xmean == None:
        xmean = x.mean(axis=axis, keepdims=True)
    if xstd == None:
        xstd  = np.std(x, axis=axis, keepdims=True)
    zscore = (x-xmean)/xstd

    # このときのxmeanとxstdを保存しておく
    with open("stdFile3_2.binaryfile", 'wb') as f:
        pickle.dump([xmean, xstd], f)
    return zscore


# 最初にFFTデータと平均分散データを結合する
folderMaster = 'D:\\Huawei_Challenge2019\\challenge-2019-train_'
positions = ['bag', 'hips', 'torso']
FFT_Mag_xyz_FolderName = '\\FFT_sample_Mag_xyz'
i = 0
for position in positions:
    folder = folderMaster + position
    sampleNameList = os.listdir(folder + FFT_Acc_z_FolderName)
    mag_xyz = np.load(folder + FFT_Mag_xyz_FolderName + "\\" + sampleNameList[0])
    print(mag_xyz.shape)
    mag_xyz = mag_xyz.flatten()
    np_array = mag_xyz
    if i == 0:
        X = np_array
    else:
        X = np.vstack((X, np_array))
    for sampleName in sampleNameList[1:]:
        print(sampleName)
        mag_xyz = np.load(folder + FFT_Mag_xyz_FolderName + "\\" + sampleName)
        mag_xyz = mag_xyz.flatten()
        np_array = mag_xyz
        X = np.vstack((X, np_array))
    i += 1

Label = np.load("train_Label.npy")
Y = Label.copy()
Y = np.vstack(Y, Label)
Y = np.vstack(Y, Label)

#標準化をする
X_std = zscore(X, axis=0)
print(X_std[0:5])

np.savez('train3_2', X=X_std, Y=Y)

#ここでnanの行は消す
deleteIndex = np.isnan(X_std)
X_std = np.delete(X_std, deleteIndex, 0)
Y = np.delete(Y, deleteIndex, 0)


# 電車と地下鉄を取り出す
learn_label = [8]
index = np.where(Y == 7)
X_2 = X_std[index]
Y_2 = Y[index]
for label in learn_label:
    index = np.where(Y == label)
    X_2 = np.vstack((X_2, X_std[index]))
    Y_2 = np.hstack((Y_2, Y[index]))
#trainのラベルを１にsubwayのラベルを2にする
Y_2[Y_2==7] = 1
Y_2[Y_2==8] = 2

print(X_2.shape)
print(Y_2.shape)

#学習開始！
clf3 = svm.SVC(verbose=True)
clf3.fit(X_2, Y_2)
with open('model3_2.binaryfile', 'wb') as file:
    pickle.dump(clf3, file)