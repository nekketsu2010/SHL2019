import numpy as np
import pandas as pd
from sklearn import svm
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
import pickle
import os


FFT_Acc_z_FolderName = 'D:\\Huawei_Challenge2019\\challenge-2019-train_bag\\FFT_sample_Acc_z'
sampleNameList = os.listdir(FFT_Acc_z_FolderName)
