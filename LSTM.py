import pickle

import keras
import numpy as np
from keras import Sequential
from keras.initializers import he_normal
from keras.layers import Embedding, LSTM, Dense, Masking
from keras.optimizers import Adamax
from keras.utils import np_utils
import keras.backend as K
from sklearn.preprocessing import StandardScaler
from tensorflow.python.keras import callbacks
import os


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

def BuildLSTM(ipshape=(32, 2400000, 19)):
    model = Sequential()
    model.add(Masking(input_shape=(None, 12), mask_value=0))
    model.add(LSTM(units=128, return_sequences=False))
    model.add(Dense(8, activation='softmax', kernel_initializer=he_normal()))
    adam = Adamax(lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
    model.compile(loss='categorical_crossentropy',
                  optimizer=adam,
                  metrics=['accuracy', f1])
    model.summary()
    return model

model = keras.models.load_model('1サンプル目/model.ep30_val_loss0.75.hdf5', {"f1": f1})
for num in range(2, 20):
    load_array = np.load("D:\\Huawei_Challenge2019\\challenge-2019-train_bag\\train_bag" + str(num) + ".npz")
    x_train = load_array['x']
    with open('stdFile.binaryfile', 'rb') as file:
        stdsc = pickle.load(file)
    for i in range(len(x_train)):
        x_train[i] = stdsc.transform(x_train[i])
    # with open('stdFile.binaryfile', 'wb') as file:
    #     pickle.dump(stdsc, file)
    y_train = load_array['y']

    if not os.path.exists(str(num) + "サンプル目"):
        os.mkdir(str(num) + "サンプル目")

    # one-hot-encoding
    y_train = np_utils.to_categorical(y=y_train, num_classes=8)
    cp_cb = callbacks.ModelCheckpoint(
        filepath=str(num) + "サンプル目/model.ep{epoch:02d}_val_loss{val_loss:.2f}.hdf5", monitor='loss',
        save_best_only=True)
    print(">>>学習開始")
    history = model.fit(x_train, y_train,
                        batch_size=32,
                        verbose=1,
                        epochs=30,
                        validation_split=0.2,
                        callbacks=[cp_cb])