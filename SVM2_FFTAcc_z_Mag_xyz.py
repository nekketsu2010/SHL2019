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
FFT_Mag_xyz_FolderName = 'D:\\Huawei_Challenge2019\\challenge-2019-train_bag\\FFT_sample_Mag_xyz'
mean_variance_skew_Folder = 'D:\\Huawei_Challenge2019\\challenge-2019-train_bag\\mean_variance_skew_Acc_Mag'

sampleNameList = os.listdir(FFT_Acc_z_FolderName)
acc_z = np.load(FFT_Acc_z_FolderName + "\\" + sampleNameList[0])
mag_xyz = np.load(FFT_Mag_xyz_FolderName + '\\' + sampleNameList[0])
mean_variance_skew = np.load(mean_variance_skew_Folder + "\\" + sampleNameList[0])
print(acc_z.shape)
print(mean_variance_skew.shape)
np_array = np.vstack((mean_variance_skew, mag_xyz))
np_array = np.vstack((np_array, acc_z))
X = np_array
acc_z = np.load(FFT_Acc_z_FolderName + "\\" + sampleNameList[1])
mag_xyz = np.load(FFT_Mag_xyz_FolderName + '\\' + sampleNameList[1])
mean_variance_skew = np.load(mean_variance_skew_Folder + "\\" + sampleNameList[1])
np_array = np.vstack((mean_variance_skew, mag_xyz))
np_array = np.vstack((np_array, acc_z))
X = np.stack([X, np_array], axis=0)
for sampleName in sampleNameList[2:]:
    print(sampleName)
    acc_z = np.load(FFT_Acc_z_FolderName + "\\" + sampleName)
    mean_variance_skew = np.load(mean_variance_skew_Folder + "\\" + sampleName)
    np_array = np.vstack((mean_variance_skew, mag_xyz))
    np_array = np.vstack((np_array, acc_z))
    np_array = np_array[np.newaxis, :, :]
    X = np.vstack((X, np_array))

Y = np.load("Label.npy")

#標準化をする
stdsc = StandardScaler()
X_std = stdsc.fit_transform(X)
print(X_std[0:5])
with open('stdFile.binaryfile', 'wb') as file:
    pickle.dump(stdsc, file)

# まずStopと乗り物のみを取り出す
learn_label = [5, 6, 7, 8]
index = np.where(Y == 1)
X_2 = X_std[index]
Y_2 = Y[index]
for label in learn_label:
    index = np.where(Y == label)
    X_2 = np.vstack((X_2, X_std[index]))
    Y_2 = np.hstack((Y_2, Y[index]))
#carとbusのラベルを2にする
#同様にtrainとsubwayのラベルを3にする
Y_2[Y_2==5] = 2
Y_2[Y_2==6] = 2
Y_2[Y_2==7] = 3
Y_2[Y_2==8] = 3

print(X_2.shape)
print(Y_2.shape)
#学習開始！
clf2 = svm.SVC(verbose=True)
clf2.fit(X_2, Y_2)
with open('model2.binaryfile', 'wb') as file:
    pickle.dump(clf2, file)