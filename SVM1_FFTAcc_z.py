import numpy as np
import pandas as pd
from sklearn import svm
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
import pickle
import os

# 最初にFFTデータと平均分散データを結合する
folderMaster = 'C:\\Users\\ohyama\\Documents\SR\\train_'
positions = ['bag', 'hips', 'torso']
FFT_Acc_z_FolderName = '\\FFT_sample_Acc_z'
mean_variance_skew_Folder = '\\mean_variance_skew_Acc_Mag'
i = 0
x = []
for position in positions:
    folder = folderMaster + position
    sampleNameList = os.listdir(folder + FFT_Acc_z_FolderName)
    acc_z = np.load(folder + FFT_Acc_z_FolderName + "\\" + sampleNameList[0])
    print(acc_z.dtype)
    mean_variance_skew = np.load(folder + mean_variance_skew_Folder + "\\" + sampleNameList[0])
    print(acc_z.shape)
    print(mean_variance_skew.shape)
    acc_z = acc_z.flatten()
    mean_variance_skew = mean_variance_skew[0:9].flatten()
    np_array = np.hstack((acc_z, mean_variance_skew))
    print(type(np_array.tolist()[0]))
    exit()
    x.append(np_array.tolist())
    for sampleName in sampleNameList[1:]:
        print(sampleName)
        acc_z = np.load(folder + FFT_Acc_z_FolderName + "\\" + sampleName)
        mean_variance_skew = np.load(folder + mean_variance_skew_Folder + "\\" + sampleName)
        acc_z = acc_z.flatten()
        mean_variance_skew = mean_variance_skew[0:9].flatten()
        np_array = np.hstack((acc_z, mean_variance_skew))
        x.append(np_array.tolist())
    i += 1

X = np.asarray(x)
del x
Label = np.load("train_Label.npy")
Y = Label.copy()
Y = np.vstack((Y, Label))
Y = np.vstack((Y, Label))

#標準化をする
std = StandardScaler()
X_std = std.fit_transform(X)
with open("stdFile.binaryfile", 'wb') as f:
    pickle.dump(std, f)

#ここでnanとInfinityの行は消す
print(X.shape)
print(Y.shape)
deleteIndex = np.isnan(X_std)
X_std = np.delete(X_std, deleteIndex, 0)
Y = np.delete(Y, deleteIndex, 0)
deleteIndex = np.isinf(X_std)
X_std = np.delete(X_std, deleteIndex, 0)
Y = np.delete(Y, deleteIndex, 0)
print(X_std.shape)
print(Y.shape)
# 乗り物のラベルを全部Stopと同じにする
Y[Y==5] = 1
Y[Y==6] = 1
Y[Y==7] = 1
Y[Y==8] = 1

# print(X_std.shape)
# print(Y.shape)

# 学習開始！
clf = svm.SVC(verbose=True)
clf.fit(X_std, Y)
with open('model1.binaryfile', 'wb') as file:
    pickle.dump(clf, file)