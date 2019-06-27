import pickle

import keras.backend as K
from keras.engine.saving import load_model
import numpy as np
from sklearn.metrics import confusion_matrix
from matplotlib import pyplot as plt
import seaborn as sns

def f1(y_true, y_pred):
    def recall(y_true, y_pred):
        """Recall metric.
        Only computes a batch-wise average of recall.
        Computes the recall, a metric for multi-label classification of
        how many relevant items are selected.
        """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + K.epsilon())
        return recall

    def precision(y_true, y_pred):
        """Precision metric.
        Only computes a batch-wise average of precision.
        Computes the precision, a metric for multi-label classification of
        how many selected items are relevant.
        """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision
    precision = precision(y_true, y_pred)
    recall = recall(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))


model = load_model("19サンプル目\\model.ep29_val_loss0.58.hdf5", custom_objects={'f1':f1})
load_array = np.load("D:\\Huawei_Challenge2019\\challenge-2019-validate_all\\Bag\\val_bag0.npz")
x = load_array['x']
with open('stdFile.binaryfile', 'rb') as file:
    stdsc = pickle.load(file)
    for i in range(len(x)):
        x[i] = stdsc.transform(x[i])
y = load_array['y']

predict_classes = model.predict_classes(x, verbose=1)
cofusion = confusion_matrix(y, predict_classes)
plt.figure(figsize=(12,8))
sns.heatmap(cofusion)
plt.show()