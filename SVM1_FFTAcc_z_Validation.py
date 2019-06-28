import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.metrics import confusion_matrix
import pickle
import os

def zscore(x, axis = None, xmean = None, xstd = None):
    zscore = (x-xmean)/xstd
    return zscore

# 最初にFFTデータと平均分散データを結合する
folderMaster = 'D:\\Huawei_Challenge2019\\challenge-2019-validate_all\\'
positions = ['Bag', 'Hips', 'Torso', 'Hand']
FFT_Acc_z_FolderName = '\\FFT_sample_Acc_z'
mean_variance_skew_Folder = '\\mean_variance_skew_Acc_Mag'
i = 0

for position in positions:
    folder = folderMaster + position
    sampleNameList = os.listdir(FFT_Acc_z_FolderName)
    acc_z = np.load(FFT_Acc_z_FolderName + "\\" + sampleNameList[0])
    mean_variance_skew = np.load(mean_variance_skew_Folder + "\\" + sampleNameList[0])
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
        acc_z = np.load(FFT_Acc_z_FolderName + "\\" + sampleName)
        mean_variance_skew = np.load(mean_variance_skew_Folder + "\\" + sampleName)
        acc_z = acc_z.flatten()
        mean_variance_skew = mean_variance_skew[0:9].flatten()
        np_array = np.hstack((acc_z, mean_variance_skew))
        X = np.vstack((X, np_array))
    i += 1

Y = np.load("val_Label.npy")
Y = Label.copy()
Y = np.vstack(Y, Label)
Y = np.vstack(Y, Label)

#標準化をする
with open("stdFile.binaryfile", 'rb') as f:
    stdData = pickle.load(f)
print(stdData[0].shape)
print(stdData[1].shape)
xmean = stdData[0]
xstd = stdData[1]
X_std = zscore(X, xmean=xmean, xstd=xstd)
print(X_std[0:5])

np.savez("val1", X=X_std, Y=Y)

#ここでnanの行は0にする
X_std[np.isnan(X_std)] = 0

# 乗り物のラベルを全部Stopと同じにする
Y[Y==5] = 1
Y[Y==6] = 1
Y[Y==7] = 1
Y[Y==8] = 1

print(X_std.shape)
print(Y.shape)

with open('model1.binaryfile', 'rb') as f:
    model = pickle.load(f)
predict = model.predict(X_std)
with open('predict1.binaryfile', 'wb') as file:
    pickle.dump(predict, file)

print(accuracy_score(Y, predict))
print(confusion_matrix(Y, predict))