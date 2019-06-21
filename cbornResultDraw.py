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

cmx_data =[[3546,3,0,4,416,957,244, 38],
 [13, 3686, 65,957, 29, 32, 25, 23],
 [ 0,8,783,9,1,0,0,0],
 [ 118,426,0, 1715,332,201,112, 75],
 [ 912, 16,0, 95, 1954, 1962,931, 85],
 [1244, 38,5, 50, 2097, 1972,369,126],
 [1432,3,0,8,487,700, 2614,141],
 [1417, 18,0, 28,779,747, 2042,438]]

plt.figure(figsize=(12,8))
sns.heatmap(cmx_data)
plt.show()