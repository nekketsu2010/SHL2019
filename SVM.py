import numpy as np
import pandas as pd
from sklearn import svm
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
import pickle
import decimal

fileNames = ['train_hips.npy', 'train_torso.npy']

data = np.loadtxt('data_0615/train_bag.npy', delimiter=',', skiprows=1)
label = data[:, 0]
feautures = data[:, 1:]
X = feautures
Y = label

for fileName in fileNames:
    data = np.loadtxt("data_0615/" + fileName, delimiter=',', skiprows=1)
    label = data[:, 0]
    feautures = data[:, 1:]
    X = np.vstack((X, feautures))
    Y = np.hstack((Y, label))


#Yは横行列になおす
Y = Y.reshape((Y.shape[0], ))

#標準化のみ（画像のRGBに正規化はいいらしいので）
# stdsc = StandardScaler()
# X_std = stdsc.fit_transform(X)
# print(X_std[0:5])
with open('data_0615/stdFile.binaryfile', 'wb') as file:
    pickle.dump(stdsc, file)

#ここでnanをなおす
# X_std[np.isnan(X_std)] = 0
X_std[np.isnan(X_std)] = -1

with open('model_next.binaryfile', 'rb') as file:
    clf1 = pickle.load(file)
scoring = "f1_macro"
scores = cross_val_score(clf1, X_std, Y, cv=10, scoring=scoring)
print("{}:{:.3f}+/-{:.3f}".format(scoring, scores.mean(), scores.std()))
exit()

#学習開始！
clf1 = svm.SVC(verbose=True)
clf1.fit(X_std, Y)
with open('model_next.binaryfile', 'wb') as file:
    pickle.dump(clf1, file)