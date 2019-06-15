import numpy as np
import ConvertWorld

def calGlobalAcc(accels, gravities, geomagnetics):
    inR = [0] * 16
    inR = ConvertWorld.getRotationMatrix(R=inR, I=None, gravity=gravities, geomagnetic=geomagnetics)
    outR = [0] * 16
    outR = ConvertWorld.remapCoodinateSystem(inR=inR, X=1, Y=2, outR=outR)
    temp = [0] * 4
    temp[0] = accels[0]
    temp[1] = accels[1]
    temp[2] = accels[2]
    temp[3] = 0
    temp = np.reshape(temp, (4, 1))
    outR = np.reshape(outR, (4, 4))
    try:
        inv = np.linalg.inv(outR)
    except np.linalg.linalg.LinAlgError:
        inv = np.identity(4, dtype=float)
    globalValues = np.dot(inv, temp)
    return globalValues

with open("Acc_x.txt") as f:
	acc_x = f.readlines()
print("ok")
with open("Acc_y.txt") as f:
	acc_y = f.readlines()
print("ok")
with open("Acc_z.txt") as f:
	acc_z = f.readlines()
print("ok")
with open("Gra_x.txt") as f:
	gra_x = f.readlines()
print("ok")
with open("Gra_y.txt") as f:
	gra_y = f.readlines()
print("ok")
with open("Gra_z.txt") as f:
	gra_z = f.readlines()
print("ok")
with open("Mag_x.txt") as f:
	mag_x = f.readlines()
print("ok")
with open("Mag_y.txt") as f:
	mag_y = f.readlines()
print("ok")
with open("Mag_z.txt") as f:
	mag_z = f.readlines()
print("ok")


GloAcc_x = []
GloAcc_y = []
GloAcc_z = []
for i in range(0, len(acc_x)):
    acc_x_s = acc_x[i].split(' ')
    acc_y_s = acc_y[i].split(' ')
    acc_z_s = acc_z[i].split(' ')
    gra_x_s = gra_x[i].split(" ")
    gra_y_s = gra_y[i].split(" ")
    gra_z_s = gra_z[i].split(" ")
    mag_x_s = mag_x[i].split(" ")
    mag_y_s = mag_y[i].split(" ")
    mag_z_s = mag_z[i].split(" ")

    for j in range(0, len(acc_x_s)):
        glovalAcc = calGlobalAcc([float(acc_x_s[j]), float(acc_y_s[j]), float(acc_z_s[j])], [float(gra_x_s[j]), float(gra_y_s[j]), float(gra_z_s[j])], [float(mag_x_s[j]), float(mag_y_s[j]), float(mag_z_s[j])])
        with open('GloAcc_x.txt', mode='a') as f:
            if j != 0:
                f.write(' ')
            f.write(str(glovalAcc[0][0]))
        with open('GloAcc_y.txt', mode='a') as f:
            if j != 0:
                f.write(' ')
            f.write(str(glovalAcc[1][0]))
        with open('GloAcc_z.txt', mode='a') as f:
            if j != 0:
                f.write(' ')
            f.write(str(glovalAcc[2][0]))
    with open('GloAcc_x.txt', mode='a') as f:
        f.write('\n')
    with open('GloAcc_y.txt', mode='a') as f:
        f.write('\n')
    with open('GloAcc_z.txt', mode='a') as f:
        f.write('\n')
    print(str(i) + "終わった")