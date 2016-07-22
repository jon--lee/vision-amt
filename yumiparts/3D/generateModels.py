from stl import mesh
from meshPCA import getPC, getRod, getCenteredPoints
import numpy as np

for i in range(1, 9):
    m = mesh.Mesh.from_file("/Users/chrispowers/Desktop/research/vision-amt/yumiparts/STL/Part" + str(i) + ".STL")
    points, centeredMesh = getCenteredPoints(m)
    volume, cog, inertia = centeredMesh.get_mass_properties()
    pAxis, sAxis = getPC(points)
    pAxis, sAxis = pAxis/np.linalg.norm(pAxis), sAxis/np.linalg.norm(sAxis)
    sides = centeredMesh.points
    """output file format:
    cog
    pAxis
    sAxis
    points
    """
    outFile = open("obj" + str(i) + ".txt", "w")
    def asString(ta):
        return "[" + str(ta[0]) + " " + str(ta[1]) + " " + str(ta[2]) + "]"
    outFile.write(asString(cog) + "\n")
    outFile.write(asString(pAxis) + "\n")
    outFile.write(asString(sAxis) + "\n")
    for p in sides:
        outFile.write("[" + str(p[0]) + " " + str(p[1]) + " " + str(p[2]) + "\n"
                        + str(p[3]) + " " + str(p[4]) + " " + str(p[5]) + "\n"
                        + str(p[6]) + " " + str(p[7]) + " " + str(p[8]) + "]" + "\n")
