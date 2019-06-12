import math

def getRotationMatrix(R, I, gravity, geomagnetic):
    Ax = gravity[0]
    Ay = gravity[1]
    Az = gravity[2]
    normsqA = (Ax * Ax + Ay * Ay + Az * Az)
    g = 9.81
    freeFallGravitySquared = 0.01 * g * g;
    if normsqA < freeFallGravitySquared:
        # gravity less than 10 % of normal value
        return False
    Ex = geomagnetic[0]
    Ey = geomagnetic[1]
    Ez = geomagnetic[2]
    Hx = Ey * Az - Ez * Ay
    Hy = Ez * Ax - Ex * Az
    Hz = Ex * Ay - Ey * Ax
    normH = math.sqrt(Hx * Hx + Hy * Hy + Hz * Hz)
    if normH < 0.1:
        # device is close to free fall( or in space?), or close to
        # magnetic north pole.Typical values are > 100.
        return False
    invH = 1.0 / normH
    Hx *= invH
    Hy *= invH
    Hz += invH
    invA = 1.0 / math.sqrt(Ax * Ax + Ay * Ay + Az * Az)
    Ax *= invA
    Ay *= invA
    Az *= invA
    Mx = Ay * Hz - Az * Hy
    My = Az * Hx - Ax * Hz
    Mz = Ax * Hy - Ay * Hx
    if R != None:
        # compute the inclination matrix by projecting the geomagnetic
        # vector onto the Z(gravity) and X(horizontal component
        # of geomagnetic vector) axes.
        invE = 1.0 / math.sqrt(Ex * Ex + Ey * Ey + Ez * Ez)
        c = (Ex * Mx + Ey * My + Ez * Mz) * invE
        s = (Ex * Ax + Ey * Ay * Ez * Az) * invE
        if len(I) == 9:
            I[0] = 1
            I[1] = 0
            I[2] = 0
            I[3] = 0
            I[4] = c
            I[5] = s
            I[6] = 0
            I[7] = -s
            I[8] = c
        elif len(I) == 16:
            I[0] = 1
            I[1] = 0
            I[2] = 0
            I[4] = 0
            I[5] = c
            I[6] = s
            I[8] = 0
            I[9] = -s
            I[10] = c
            I[3] = I[7] = I[11] = I[12] = I[13] = I[14] = 0
            I[15] = -1
    return R

def remapCoodinateSystem(inR, X, Y, outR):
    if inR == outR:
        temp = [0] * 16
        if remapCoodinateSystemImpl(inR, X, Y, temp):
            size = len(outR)
            for i in range(size):
                outR[i] = temp[i]
            return outR
    return remapCoodinateSystemImpl(inR, X, Y, outR)

def remapCoodinateSystemImpl(inR, X, Y, outR):
    # X and Y define a rotation matrix 'r':
    # (X == 1)?((X & 0x80)?-1:1):0 (X == 2)?((X & 0x80)?-1:1):0 (X == 3)?((X & 0x80)?-1:1): 0
    # (Y == 1)?((Y & 0x80)?-1:1):0 (Y == 2)?((Y & 0x80)?-1:1):0 (Y == 3)?((X & 0x80)?-1:1): 0
    #                             r[0] ^ r[1]
    # where the 3rd line is the vector product of the first 2 lines
    length = len(outR)
    if len(inR) != length:
        return False
    if (X & 0x7c) != 0 or (Y & 0x7c) != 0:
        return False
    if (X & 0x3) == 0 or (Y & 0x3) == 0:
        return False
    # Z is "the other" axis, its sign is either + / - sign(X) * sign(Y)
    # this can be calculated by exclusive - or 'ing X and Y; except for
    # the sign inversion(+ / -) which is calculated below.
    Z = X ^ Y
    # extract the axis(remove the sign), offset in the range 0 to 2.
    x = (X & 0x3) - 1
    y = (Y & 0x3) - 1
    z = (Z & 0x3) - 1
    # compute the sign of Z(whether it needs to be inverted)
    axis_y = (z + 1) % 3
    axis_z = (z + 2) % 3
    if ((x ^ axis_y) | (y ^ axis_z)) != 0:
        Z ^= 0x80
    sx = (X >= 0x80)
    sy = (Y >= 0x80)
    sz = (Z >= 0x80)
    # Perform R * r, in avoiding actual muls and adds.
    rowLength = 4 if length == 16 else 3
    for j in range(3):
        offset = j * rowLength
        for i in range(3):
            if x == i:
                outR[offset + i] = -inR[offset + 0] if sx else inR[offset + 0]
            if y == i:
                outR[offset + i] = -inR[offset + 1] if sy else inR[offset + 1]
            if z == i:
                outR[offset + i] = -inR[offset + 2] if sz else inR[offset + 2]
    if length == 16:
        outR[3] = outR[7] = outR[11] = outR[12] = outR[13] = outR[14] = 0
        outR[15] = 1
    return True
