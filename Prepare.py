import pandas as pd
import numpy as np
import math
import os

import ConvertWorld


def main(folderName):
    folder = ['Stil', 'Walking', 'Run', 'Bike', 'Car', 'Bus', 'Train', 'Subway']

    directoryName = 'D:\\Huawei_Challenge2019\\' + folderName
    files = os.listdir(directoryName)

    出 = ['Label,XY_accel,XY_accel_dispersion,Z_accel,Z_accel_dispersion']

    for file in files:
        df = pd.read_csv(directoryName + "\\" + file)

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

        出xy_acc = []
        出z_acc = []
        出xyz_mag = []
        出pressure = []
        出xyz_gyro = []

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
            inv =  np.linalg.inv(outR)
            globalValues = np.dot(inv, temp)
            出xy_acc.append(math.sqrt(globalValues[0] * globalValues[0] + globalValues[1] * globalValues[1]))
            出z_acc.append(globalValues[2])
            出xyz_mag.append(math.sqrt(x_mag * x_mag + y_mag * y_mag + z_mag * z_mag))
            出pressure.append(pressure)
            出xyz_gyro.append(math.sqrt(x_gyr * x_gyr + y_gyr * y_gyr + z_gyr * z_gyr))