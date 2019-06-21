import numpy as np
import pandas as pd
from sklearn import svm
from sklearn.preprocessing import StandardScaler
import pickle
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.metrics import confusion_matrix
from matplotlib import pyplot as plt
import seaborn as sns


fileNames = ['data_0615/val_hips.csv', 'data_0615/val_torso.csv']#, 'val_Hand.npy']

data = np.loadtxt('data_0615/val_bag.csv', delimiter=',', skiprows=1)
label = data[:, 0]
feautures = data[:, 1:]
X = feautures
Y = label

for fileName in fileNames:
    data = np.loadtxt(fileName, delimiter=',', skiprows=1)
    label = data[:, 0]
    feautures = data[:, 1:]
    X = np.vstack((X, feautures))
    Y = np.hstack((Y, label))

#Yは横行列になおす
Y = Y.reshape((Y.shape[0], ))
Y = np.where(Y > 4, 1, Y)
print(X.shape)
print(Y.shape)
with open('data_0615/predict1.binaryFile', 'rb') as file:
    predict = pickle.load(file)
    print(accuracy_score(Y, predict))
    cofusion = confusion_matrix(Y, predict)
    print(cofusion)
    plt.figure(figsize=(12, 8))
    sns.heatmap(cofusion)
    plt.show()
exit()

#標準化のみ（画像のRGBに正規化はいいらしいので）
with open('stdFile.binaryFile', 'rb') as file:
    stdsc = pickle.load(file)
X_std = stdsc.transform(X)
print(X_std[0:5])

#ここでnanをなおす
# X_std[np.isnan(X_std)] = 0
X_std[np.isnan(X_std)] = -1

with open('model_next.binaryFile', 'rb') as file:
    clf = pickle.load(file)
print("predictするよ")
predict = clf.predict(X_std)
with open('predict.binaryFile', 'wb') as file:
    pickle.dump(predict, file)
print(accuracy_score(Y, predict))
print(confusion_matrix(Y, predict))