import numpy as np
import pandas as pd
import csv

def suffix(i):
    # 6桁の０埋め
    return "{:06d}".format(i)

# 時系列データに変換

labels = np.loadtxt("Label.txt", delimiter=" ", dtype=int)
labels = labels.flatten()

#各種センサを読み込むよ
acc_x = np.loadtxt("Acc_x.txt", delimiter=" ")
acc_x = acc_x.flatten()
print("ok")
acc_y = np.loadtxt("Acc_y.txt", delimiter=" ")
acc_y = acc_y.flatten()
print("ok")
acc_z = np.loadtxt("Acc_z.txt", delimiter=" ")
acc_z = acc_z.flatten()
print("ok")
gra_x = np.loadtxt("Gra_x.txt", delimiter=" ")
gra_x = gra_x.flatten()
print("ok")
gra_y = np.loadtxt("Gra_y.txt", delimiter=" ")
gra_y = gra_y.flatten()
print("ok")
gra_z = np.loadtxt("Gra_z.txt", delimiter=" ")
gra_z = gra_z.flatten()
print("ok")
mag_x = np.loadtxt("Mag_x.txt", delimiter=" ")
mag_x = mag_x.flatten()
print("ok")
mag_y = np.loadtxt("Mag_y.txt", delimiter=" ")
mag_y = mag_y.flatten()
print("ok")
mag_z = np.loadtxt("Mag_z.txt", delimiter=" ")
mag_z = mag_z.flatten()
print("ok")
gyr_x = np.loadtxt("Gyr_x.txt", delimiter=" ")
gyr_x = gyr_x.flatten()
print("ok")
gyr_y = np.loadtxt("Gyr_y.txt", delimiter=" ")
gyr_y = gyr_y.flatten()
print("ok")
gyr_z = np.loadtxt("Gyr_z.txt", delimiter=" ")
gyr_z = gyr_z.flatten()
print("ok")
# lacc_x = np.loadtxt("LAcc_x.txt", delimiter=" ")
# lacc_x = lacc_x.flatten()
# print("ok")
# lacc_y = np.loadtxt("LAcc_y.txt", delimiter=" ")
# lacc_y = lacc_y.flatten()
# print("ok")
# lacc_z = np.loadtxt("LAcc_z.txt", delimiter=" ")
# lacc_z = lacc_z.flatten()
# print("ok")
# ori_x = np.loadtxt("Ori_x.txt", delimiter=" ")
# ori_x = ori_x.flatten()
# print("ok")
# ori_y = np.loadtxt("Ori_y.txt", delimiter=" ")
# ori_y = ori_y.flatten()
# print("ok")
# ori_z = np.loadtxt("Ori_z.txt", delimiter=" ")
# ori_z = ori_z.flatten()
# print("ok")
# ori_w = np.loadtxt("Ori_w.txt", delimiter=" ")
# ori_w = ori_w.flatten()
# print("ok")

# Labelの切れ目を見つける
temp_label = labels[0]
change_index = []
for i in range(0, len(labels)):
    if labels[i] != temp_label:
        change_index.append(i)
        print("ChangeIndex is " + str(i))
    temp_label = labels[i]

#numpy ndarrayでサンプル保存すればよさそう（csvは最悪の末路迎えそう）
before_list = [] #numpyにする前のリスト
num = 1
SampleName = "Sample" + suffix(num)
with open("Label.tsv", mode='w', newline='') as tsv:
    f = csv.writer(tsv, delimiter='\t')
    f.writerow(["fileName", "label"])
for i in range(0, len(acc_x)):
    list = [acc_x[i], acc_y[i], acc_z[i], gra_x[i], gra_y[i], gra_z[i], mag_x[i], mag_y[i], mag_z[i],
            gyr_x[i], gyr_y[i], gyr_z[i]]#, lacc_x[i], lacc_y[i], lacc_z[i], ori_x[i], ori_y[i], ori_z[i], ori_w[i]]
    if i in change_index or len(before_list) >= 500:
        if len(before_list) < 500:
            before_list = []
            continue
        #changeIndexだったら違うサンプルへと変更
        #保存
        with open("Label.tsv", mode='a', newline='') as tsv:
            f = csv.writer(tsv, delimiter='\t')
            f.writerow([SampleName, labels[i-1]])
        after_list = np.array(before_list)
        np.save("LSTM_Sample/" + SampleName, after_list)
        print(SampleName + "保存しました")
        num += 1
        SampleName = "Sample" + suffix(num)
        before_list = []
    print("あと" + str(len(acc_x) - i) + "で完了")
    before_list.append(list)

    if i == len(acc_x)-1:
        with open("Label.tsv", mode='a', newline='') as tsv:
            f = csv.writer(tsv, delimiter='\t')
            f.writerow([SampleName, labels[i-1]])
        after_list = np.array(before_list)
        np.save("LSTM_Sample/" + SampleName, after_list)
        print(SampleName + "保存しました")
print("満了")

tsvName = 'Label.tsv'
meta_data = pd.read_table(tsvName)
labels, uniques = pd.factorize(meta_data['label'])
meta_data['label'] = labels
x = list(meta_data.loc[:, "fileName"])
y = list(meta_data.loc[:, 'label'])

max_len = 1000
np_targets = np.zeros(len(y))
arrlist = np.load("LSTM_Sample/" + x[0] + ".npy")
pad_width = [(max_len - len(arrlist), 0), (0, 0)]
arrlist = np.pad(arrlist, pad_width, 'constant', constant_values=0)
np_targets[0] = y[0]
array = np.load("LSTM_Sample/" + x[1] + ".npy")
pad_width = [(max_len - len(array), 0), (0, 0)]
array = np.pad(array, pad_width, 'constant', constant_values=0)
arrlist = np.stack([arrlist, array], axis=0)
print(arrlist.shape)
# arrlist.append(array)
np_targets[1] = y[1]
for num in range(2, len(y)):
    array = np.load("LSTM_Sample/" + x[i] + ".npy")
    pad_width = [(max_len-len(array), 0), (0, 0)]
    array = np.pad(array, pad_width, 'constant', constant_values=0)
    array = array[np.newaxis, :, :]
    arrlist = np.vstack((arrlist, array))
    print(arrlist.shape)
    # arrlist.append(array)
    np_targets[i] = y[i]
    print(str(i) + "個のデータを処理しました")
    # nplist = np.asarray(arrlist)
np.savez("val_bag", x=arrlist, y=np_targets)
