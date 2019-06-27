import numpy as np
import pandas as pd
from sklearn import svm
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
import pickle
import decimal


fileNames = ['train_hips.npy', 'train_torso.npy']

data = np.load('train_bag.npy')

X = data
Y = #ここにLabelのcsvとかね

for fileName in fileNames:
    data = np.load(fileName)
    label = data[:, 0]
    feautures = data[:, 1:]
    X = np.vstack((X, feautures))
    Y = np.hstack((Y, label))

#Yは横行列になおす
Y = Y.reshape((Y.shape[0], ))

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

# 学習開始！
clf = svm.SVC(verbose=True)
clf.fit(X, Y)
with open('model1.binaryfile', 'wb') as file:
    pickle.dump(clf, file)