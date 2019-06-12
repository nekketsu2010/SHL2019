import pandas as pd
import numpy as np
from statistics import mean, median, variance, stdev
import math
import os
import sys

import ConvertWorld

folderName = sys.argv[1]

folders = ['Stil', 'Walking', 'Run', 'Bike', 'Car', 'Bus', 'Train', 'Subway']

directoryName = 'D:\\Huawei_Challenge2019\\challenge-2019-' + folderName + "\\Raw\\"

if len(sys.argv) > 1:
    folderName = sys.argv[2]

with open(directoryName + folderName + ".csv", mode="w") as newcsv:
    newcsv.write('Label,XY_accel_mean,XY_accel_variance,Z_accel_mean,Z_accel_variance,XYZ_mag_mean,XYZ_mag_variance,pressure_difference,XYZ_gyro_mean,XYZ_gyro_variance\n')

for num in range(len(folders)):
    dirName = directoryName + folders[num]
    files = os.listdir(dirName)

    for file in files:
        df = pd.read_csv(directoryName + folders[num] + '\\' + file)

        X_Acc = df.Acc_x.values
        Y_Acc = df.Acc_y.values
        Z_Acc = df.Acc_z.values
        X_Gra = df.Gra_x.values
        Y_Gra = df.Gra_y.values
        Z_Gra = df.Gra_z.values
        X_Gyr = df.Gyr_x.values
        Y_Gyr = df.Gyr_y.values
        Z_Gyr = df.Gyr_z.values
        X_Mag = df.Mag_x.values
        Y_Mag = df.Mag_y.values
        Z_Mag = df.Mag_z.values
        Pressure = df.Pressure.values

        出XY_accel = []
        出XY_accel_variance = 0
        出Z_accel = []
        出Z_accel_variance = 0
        出XYZ_mag = []
        出XYZ_mag_variance = 0
        出pressure = []
        出XYZ_gyro = []
        出XYZ_gyro_variance = 0

        for x_acc, y_acc, z_acc, x_gra, y_gra, z_gra, x_gyr, y_gyr, z_gyr, x_mag, y_mag, z_mag, pressure in zip(X_Acc, Y_Acc, Z_Acc, X_Gra, Y_Gra, Z_Gra, X_Gyr, Y_Gyr, Z_Gyr, X_Mag, Y_Mag, Z_Mag, Pressure):
            inR = [0] * 16
            gravityes = [x_gra, y_gra, z_gra]
            geomagnetics = [x_mag, y_mag, z_mag]
            inR = ConvertWorld.getRotationMatrix(R=inR, I=None, gravity=gravityes, geomagnetic=geomagnetics)
            outR = [0] * 16
            outR = ConvertWorld.remapCoodinateSystem(inR=inR, X=1, Y=2, outR=outR)
            temp = [0] * 4
            temp[0] = x_acc
            temp[1] = y_acc
            temp[2] = z_acc
            temp[3] = 0
            temp = np.reshape(temp, (4, 1))
            outR = np.reshape(outR, (4, 4))
            try:
                inv = np.linalg.inv(outR)
            except np.linalg.linalg.LinAlgError:
                inv = np.identity(4, dtype=float)
            globalValues = np.dot(inv, temp)
            出XY_accel.append(math.sqrt(globalValues[0] * globalValues[0] + globalValues[1] * globalValues[1]))
            出Z_accel.append(globalValues[2,0])
            出XYZ_mag.append(math.sqrt(x_mag * x_mag + y_mag * y_mag + z_mag * z_mag))
            出pressure.append(pressure)
            出XYZ_gyro.append(math.sqrt(x_gyr * x_gyr + y_gyr * y_gyr + z_gyr * z_gyr))
        #平均、分散を求める
        出XY_accel_variance = variance(出XY_accel)
        出Z_accel_variance = variance(出Z_accel)
        出XYZ_mag_variance = variance(出XYZ_mag)
        出XYZ_gyro_variance = variance(出XYZ_gyro)
        出XY_accel = mean(出XY_accel)
        出Z_accel = mean(出Z_accel)
        出XYZ_mag = mean(出XYZ_mag)
        出pressure = max(出pressure) - min(出pressure)
        出XYZ_gyro = mean(出XYZ_gyro)

        #出力する
        with open(directoryName + folderName + ".csv", mode="a") as newcsv:
            newcsv.write(str((num+1)) + ',' + str(出XY_accel) + ',' + str(出XY_accel_variance) + ',' + str(出Z_accel) + ',' + str(出Z_accel_variance) + ',' + str(出XYZ_mag) + ',' + str(出XYZ_mag_variance) + ',' + str(出pressure) + ',' + str(出XYZ_gyro) + ',' + str(出XYZ_gyro_variance) + '\n')
        print(folders[num] + ': ' + file + 'いってるよ')