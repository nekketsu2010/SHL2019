import numpy as np
import pandas as pd
from sklearn import svm
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
import pickle
import os

# 最初にFFTデータと平均分散データを結合する
FFT_Acc_z_FolderName = 'D:\\Huawei_Challenge2019\\challenge-2019-train_bag\\FFT_sample_Acc_z'
mean_variance_skew_Folder = 'D:\\Huawei_Challenge2019\\challenge-2019-train_bag\\mean_variance_skew_Acc_Mag'

sampleNameList = os.listdir(FFT_Acc_z_FolderName)
acc_z = np.load(FFT_Acc_z_FolderName + "\\" + sampleNameList[0])
mean_variance_skew = np.load(mean_variance_skew_Folder + "\\" + sampleNameList[0])
print(acc_z.shape)
print(mean_variance_skew.shape)
np_array = np.vstack((acc_z, mean_variance_skew))
X = np_array
acc_z = np.load(FFT_Acc_z_FolderName + "\\" + sampleNameList[1])
mean_variance_skew = np.load(mean_variance_skew_Folder + "\\" + sampleNameList[1])
np_array = np.vstack((acc_z, mean_variance_skew))
X = np.stack([X, np_array], axis=0)
for sampleName in sampleNameList[2:]:
    print(sampleName)
    acc_z = np.load(FFT_Acc_z_FolderName + "\\" + sampleName)
    mean_variance_skew = np.load(mean_variance_skew_Folder + "\\" + sampleName)
    np_array = np.vstack(acc_z, mean_variance_skew)
    np_array = np_array[np.newaxis, :, :]
    X = np.vstack((X, np_array))

Y = np.load("Label.npy")

#標準化をする
stdsc = StandardScaler()
X_std = stdsc.fit_transform(X)
print(X_std[0:5])
with open('stdFile.binaryfile', 'wb') as file:
    pickle.dump(stdsc, file)

#ここでnanの行は消す
deleteIndex = np.isnan(X_std)
X_std = np.delete(X_std, deleteIndex, 0)
Y = np.delete(Y, deleteIndex, 0)

# 乗り物のラベルを全部Stopと同じにする
Y[Y==5] = 1
Y[Y==6] = 1
Y[Y==7] = 1
Y[Y==8] = 1

print(X.shape)
print(Y.shape)
exit()

# 学習開始！
clf = svm.SVC(verbose=True)
clf.fit(X, Y)
with open('model1.binaryfile', 'wb') as file:
    pickle.dump(clf, file)