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
    with open("stdFile.binaryfile", 'wb') as f:
        pickle.dump([xmean, xstd], f)
    return zscore

# 最初にFFTデータと平均分散データを結合する
folderMaster = 'C:\\Users\\ohyama\\Documents\SR\\train_'
positions = ['bag', 'hips', 'torso']
FFT_Acc_z_FolderName = '\\FFT_sample_Acc_z'
mean_variance_skew_Folder = '\\mean_variance_skew_Acc_Mag'
i = 0
for position in positions:
    folder = folderMaster + position
    sampleNameList = os.listdir(folder + FFT_Acc_z_FolderName)
    acc_z = np.load(folder + FFT_Acc_z_FolderName + "\\" + sampleNameList[0])
    mean_variance_skew = np.load(folder + mean_variance_skew_Folder + "\\" + sampleNameList[0])
    print(acc_z.shape)
    print(mean_variance_skew.shape)
    acc_z = acc_z.flatten()
    mean_variance_skew = mean_variance_skew[0:9].flatten()
    np_array = np.hstack((acc_z, mean_variance_skew))
    if i == 0:
        X = np_array
    else:
        X = np.vstack((X, np_array))
    for sampleName in sampleNameList[1:]:
        print(sampleName)
        acc_z = np.load(folder + FFT_Acc_z_FolderName + "\\" + sampleName)
        mean_variance_skew = np.load(folder + mean_variance_skew_Folder + "\\" + sampleName)
        acc_z = acc_z.flatten()
        mean_variance_skew = mean_variance_skew[0:9].flatten()
        np_array = np.hstack((acc_z, mean_variance_skew))
        X = np.vstack((X, np_array))
    i += 1

Label = np.load("train_Label.npy")
Y = Label.copy()
Y = np.vstack((Y, Label))
Y = np.vstack((Y, Label))

#標準化をする
X_std = zscore(X, axis=0)
print(X_std[0:5])

np.savez("train", X=X_std, Y=Y)


#ここでnanの行は消す
deleteIndex = np.isnan(X_std)
X_std = np.delete(X_std, deleteIndex, 0)
Y = np.delete(Y, deleteIndex, 0)

# 乗り物のラベルを全部Stopと同じにする
Y[Y==5] = 1
Y[Y==6] = 1
Y[Y==7] = 1
Y[Y==8] = 1

print(X_std.shape)
print(Y.shape)

# 学習開始！
clf = svm.SVC(verbose=True)
clf.fit(X_std, Y)
with open('model1.binaryfile', 'wb') as file:
    pickle.dump(clf, file)