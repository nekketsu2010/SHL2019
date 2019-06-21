import sys
import numpy as np

def suffix(i):
    # 6桁の０埋め
    return "{:06d}".format(i)

folders = ['Stil', 'Walking', 'Run', 'Bike', 'Car', 'Bus', 'Train', 'Subway']
directoryName = 'D:\\Huawei_Challenge2019\\challenge-2019-\\'
print(len(sys.argv))
if len(sys.argv) > 2:
    folderName = sys.argv[2]

# with open("Glo_Acc_x.txt") as f:
# 	acc_x = f.readlines()
# print("ok")
# with open("Glo_Acc_y.txt") as f:
# 	acc_y = f.readlines()
# print("ok")
# with open("Glo_Acc_z.txt") as f:
# 	acc_z = f.readlines()
# print("ok")
# with open("Gra_x.txt") as f:
# 	gra_x = f.readlines()
# print("ok")
# with open("Gra_y.txt") as f:
# 	gra_y = f.readlines()
# print("ok")
# with open("Gra_z.txt") as f:
# 	gra_z = f.readlines()
# print("ok")
# with open("Gyr_x.txt") as f:
# 	gyr_x = f.readlines()
# print("ok")
# with open("Gyr_y.txt") as f:
# 	gyr_y = f.readlines()
# print("ok")
# with open("Gyr_z.txt") as f:
# 	gyr_z = f.readlines()
# print("ok")
# with open("LAcc_x.txt") as f:
# 	lacc_x = f.readlines()
# print("ok")
# with open("LAcc_y.txt") as f:
# 	lacc_y = f.readlines()
# print("ok")
# with open("LAcc_z.txt") as f:
# 	lacc_z = f.readlines()
# print("ok")
# with open("Mag_x.txt") as f:
# 	mag_x = f.readlines()
# print("ok")
# with open("Mag_y.txt") as f:
# 	mag_y = f.readlines()
# print("ok")
# with open("Mag_z.txt") as f:
# 	mag_z = f.readlines()
# print("ok")
# with open("Ori_w.txt") as f:
# 	ori_w = f.readlines()
# print("ok")
# with open("Ori_x.txt") as f:
# 	ori_x = f.readlines()
# print("ok")
# with open("Ori_y.txt") as f:
# 	ori_y = f.readlines()
# print("ok")
# with open("Ori_z.txt") as f:
# 	ori_z = f.readlines()
# print("ok")
# with open("Pressure.txt") as f:
# 	press = f.readlines()
# print("ok")
with open("Label.txt") as f:
	labels = f.readlines()
print("ok")

#ここでLabelを一行ずつ確認し，すべて同じラベルかどうかを見る
#同じラベルでない行は使わないことにする
#ラベルごとにサンプルを管理する
NG_num = [] #使わない行を格納する配列
i = 0
for label in labels:
    print(label)
    for j in range(len(label)):
        if label[j] != label[0]:
            # print("NG行　" + str(j))
            NG_num.append(i)
            break
    i += 1

i = 0
for acc_x_s, acc_y_s, acc_z_s, gra_x_s, gra_y_s, gra_z_s, gyr_x_s, gyr_y_s, gyr_z_s, lacc_x_s, lacc_y_s, lacc_z_s, mag_x_s, mag_y_s, mag_z_s, ori_x_s, ori_y_s, ori_z_s, ori_w_s, press_s, label in zip(
        acc_x, acc_y, acc_z, gra_x, gra_y, gra_z, gyr_x, gyr_y, gyr_z, lacc_x, lacc_y, lacc_z, mag_x, mag_y, mag_z, ori_x, ori_y, ori_z, ori_w, press, labels):
    if i in NG_num:
        i += 1
        continue
    label_s = label[0]
    outTexts = []
    acc_x_s = acc_x_s.split(" ")
    acc_y_s = acc_y_s.split(" ")
    acc_z_s = acc_z_s.split(" ")
    gra_x_s = gra_x_s.split(" ")
    gra_y_s = gra_y_s.split(" ")
    gra_z_s = gra_z_s.split(" ")
    gyr_x_s = gyr_x_s.split(" ")
    gyr_y_s = gyr_y_s.split(" ")
    gyr_z_s = gyr_z_s.split(" ")
    lacc_x_s = lacc_x_s.split(" ")
    lacc_y_s = lacc_y_s.split(" ")
    lacc_z_s = lacc_z_s.split(" ")
    mag_x_s = mag_x_s.split(" ")
    mag_y_s = mag_y_s.split(" ")
    mag_z_s = mag_z_s.split(" ")
    ori_x_s = ori_x_s.split(" ")
    ori_y_s = ori_y_s.split(" ")
    ori_z_s = ori_z_s.split(" ")
    ori_w_s = ori_w_s.split(" ")
    press_s = press_s.split(" ")

    for acc_x_s_s, acc_y_s_s, acc_z_s_s, gra_x_s_s, gra_y_s_s, gra_z_s_s, gyr_x_s_s, gyr_y_s_s, gyr_z_s_s, lacc_x_s_s, lacc_y_s_s, lacc_z_s_s, mag_x_s_s, mag_y_s_s, mag_z_s_s, ori_x_s_s, ori_y_s_s, ori_z_s_s, ori_w_s_s, press_s_s in zip(
        acc_x_s, acc_y_s, acc_z_s, gra_x_s, gra_y_s, gra_z_s, gyr_x_s, gyr_y_s, gyr_z_s, lacc_x_s, lacc_y_s, lacc_z_s, mag_x_s, mag_y_s, mag_z_s,
        ori_x_s, ori_y_s, ori_z_s, ori_w_s, press_s):
        s = labels.strip() + "," + acc_x_s_s.strip() + "," + acc_y_s_s.strip() + "," + acc_z_s_s.strip() + "," + gra_x_s_s.strip() + "," + \
            gra_y_s_s.strip() + "," + gra_z_s_s.strip() + "," + gyr_x_s_s.strip() + "," + gyr_y_s_s.strip() + "," + \
            gyr_z_s_s.strip() + "," + lacc_x_s_s.strip() + "," + lacc_y_s_s.strip() + "," + lacc_z_s_s.strip() + "," + \
            mag_x_s_s.strip() + "," + mag_y_s_s.strip() + "," + mag_z_s_s.strip() + "," + ori_w_s_s.strip() + "," + \
            ori_x_s_s.strip() + "," + ori_y_s_s.strip() + "," + ori_z_s_s.strip() + "," + press_s_s.strip()
        outTexts.append(s)

    # 出力する
    with open("SVM_Data/Sample" + suffix(j), 'w') as file:
        file.write(
            "Label,Acc_x,Acc_y,Acc_z,Gra_x,Gra_y,Gra_z,Gyr_x,Gyr_y,Gyr_z,LAcc_x,LAcc_y,LAcc_z,Mag_x,Mag_y,Mag_z,Ori_x,Ori_y,Ori_z,Ori_w,Pressure\n")
        j = 0
        for text in outTexts:
            file.write(text + "\n")
            j += 1
    print(str(i) + "できた")
    i += 1
print("できた")