import numpy as np
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
lacc_x = np.loadtxt("LAcc_x.txt", delimiter=" ")
lacc_x = lacc_x.flatten()
print("ok")
lacc_y = np.loadtxt("LAcc_y.txt", delimiter=" ")
lacc_y = lacc_y.flatten()
print("ok")
lacc_z = np.loadtxt("LAcc_z.txt", delimiter=" ")
lacc_z = lacc_z.flatten()
print("ok")
ori_x = np.loadtxt("Ori_x.txt", delimiter=" ")
ori_x = ori_x.flatten()
print("ok")
ori_y = np.loadtxt("Ori_y.txt", delimiter=" ")
ori_y = ori_y.flatten()
print("ok")
ori_z = np.loadtxt("Ori_z.txt", delimiter=" ")
ori_z = ori_z.flatten()
print("ok")
ori_w = np.loadtxt("Ori_w.txt", delimiter=" ")
ori_w = ori_w.flatten()
print("ok")



# Labelの切れ目を見つける
temp_label = labels[0]
change_index = []
for i in range(0, len(labels)):
    if labels[i] != temp_label:
        change_index.append(i)

num = 1
sample_name = "Sample" + str(suffix(num) + ".csv")
#tsvでサンプルネームとラベルを一致させる
with open("Label.tsv", 'a' , newline='') as label_tsv:
    writer_tsv = csv.writer(label_tsv, delimiter='\t')
    writer_tsv.writerow(["file_name", 'label'])
    writer_tsv.writerow([sample_name, labels[0]])
    for i in range(0, len(acc_x)):
        if i in change_index:
            num += 1
            sample_name = "Sample" + str(suffix(num) + ".csv")
            print("Change!")
            writer_tsv.writerow([sample_name, labels[i]])
        with open("LSTM_Sample/" + sample_name, mode='a', newline='') as sample_csv:
            writer_csv = csv.writer(sample_csv, delimiter=',')
            writer_csv.writerow([acc_x[i], acc_y[i], acc_z[i], gra_x[i], gra_y[i], gra_z[i],
                             mag_x[i], mag_y[i], mag_z[i], gyr_x[i], gyr_y[i], gyr_z[i], lacc_x[i], lacc_y[i], lacc_z[i],
                             ori_x[i], ori_y[i], ori_z[i], ori_w[i]])
        print(str(i) + "書き込んだ")