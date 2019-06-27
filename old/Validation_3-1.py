import numpy as np
import pandas as pd
from sklearn import svm
from sklearn.preprocessing import StandardScaler
import pickle
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.metrics import confusion_matrix
from matplotlib import pyplot as plt
import seaborn as sns


fileNames = ['val_hips.npy', 'val_torso.npy']#, 'val_Hand.npy']

data = np.load('val_bag.npy')
label = data[:, 0]
feautures = data[:, 1:]
X = feautures
Y = label

for fileName in fileNames:
    data = np.load(fileName)
    label = data[:, 0]
    feautures = data[:, 1:]
    X = np.vstack((X, feautures))
    Y = np.hstack((Y, label))

#Yは横行列になおす
Y = Y.reshape((Y.shape[0], ))

#標準化のみ（画像のRGBに正規化はいいらしいので）
with open('stdFile.binaryFile', 'rb') as file:
    stdsc = pickle.load(file)
X_std = stdsc.transform(X)
print(X_std[0:5])

#ここでnanの行は消す
deleteIndex = np.isnan(X_std)
X_std = np.delete(X_std, deleteIndex, 0)
Y = np.delete(Y, deleteIndex, 0)

print(X_std.shape)
print(Y.shape)

learn_label = [6]  # car, bus
index = np.where(Y == 5)
X_2 = X_std[index]
Y_2 = Y[index]
for label in learn_label:
    index = np.where(Y == label)
    X_2 = np.vstack((X_2, X_std[index]))
    Y_2 = np.hstack((Y_2, Y[index]))

# carを1，busを2にする
Y_2[Y_2==5] = 1
Y_2[Y_2==6] = 2

# センサは地磁気とジャイロを使う
X_2 = X_2[:, [6,7,8,9]]

with open('model3_1.binaryFile', 'rb') as file:
    clf = pickle.load(file)
print("predictするよ")
predict = clf.predict(X_2)
with open('predict3_1.binaryFile', 'wb') as file:
    pickle.dump(predict, file)
print(accuracy_score(Y_2, predict))
print(confusion_matrix(Y_2, predict))