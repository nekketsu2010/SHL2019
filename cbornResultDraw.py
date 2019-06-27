from matplotlib import pyplot as plt
import seaborn as sns
import pickle
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.metrics import confusion_matrix

folder = "data_0615"

fileNames = ['train_hips.csv', 'train_torso.csv']

data = np.loadtxt(folder + "/train_bag.csv", delimiter=',', skiprows=1)
label = data[:, 0]
feautures = data[:, 1:]
X = feautures
Y = label
for fileName in fileNames:
    data = np.loadtxt(folder + "/" + fileName, delimiter=',', skiprows=1)
    label = data[:, 0]
    feautures = data[:, 1:]
    X = np.vstack((X, feautures))
    Y = np.hstack((Y, label))

#Yは横行列になおす
Y = Y.reshape((Y.shape[0], ))

with open(folder + "/predict2.binaryfile", 'rb') as f:
    predict = pickle.load(f)
print(Y.shape)
print(predict.shape)

cmx_data =[[  425,  4273,   489],
 [  118, 10287,  1420],
 [  303,  4872,  5661]]

plt.figure(figsize=(12,8))
sns.heatmap(cmx_data)
plt.show()