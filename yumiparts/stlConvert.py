import numpy as np
from stl import mesh
import math
from PIL import Image
from quadtree import quadtree
from geometry import isInside, avgD, getZ
from meshPCA import getPC, getRod, getCenteredPoints

"""Generates png of mesh in binary, and optionally fully rendered also"""
def makeRaster(partNum, centeredMesh, pAxis, sAxis, cog, fullRender, num, outputName, wid):
    #vision axis is perpendicular to both basis vectors
    vAxis = np.cross(pAxis, sAxis)
    #origin at some distance along vAxis from cog of object, f = further distance to camera
    #w, h must be integers (dimensions of output image)
    #w, h, unitsperpixel should be 1 / a power of 2 (for quadtree); w * unitsperpixel = 256
    originD, f, w, h, unitsPerPixel = -100, -40, wid, wid, 256.0/wid
    origin = cog + originD * vAxis
    camera = origin + f * vAxis
    im = Image.new("RGB", (w, h), "white")
    mInv = np.linalg.inv(np.array([pAxis, sAxis, vAxis]).T)
    #Each side consists of three triplets specifying the vertexes of a triangle in R3
    sides = centeredMesh.points
    #Convert sides to faces in camera reference frame
    faces = []
    minZ, maxZ = 100000, -1
    for s in sides:
        face = []
        for i in range(3):
            points = np.array(s[i * 3: i * 3 + 3])
            transformed = np.dot(mInv, points - origin)
            vertex = [transformed[i] for i in range(3)]
            z = transformed[2]
            if (z < minZ):
                minZ = z
            if (z > maxZ):
                maxZ = z
            face.append(vertex)
        faces.append(face)
    #puts z from part into range 0 to 255
    def fitZ(z):
        return (z - minZ)/(maxZ - minZ) * 255
    #create quadtree with dim = width of screen in units, min width = width of pixel in units
    qtree = quadtree(w * unitsPerPixel, unitsPerPixel)
    #add faces to quadtree
    for f in faces:
        qtree.addTriangle(f)
    for xp in range(int(-w/2), int(w/2)):
        for yp in range(int(-h/2), int(h/2)):
            #each pixel corresponds to some number of units
            x, y = xp * unitsPerPixel, yp * unitsPerPixel
            #find all triangles in quadtree node that x is in
            closeTriangles = qtree.getTrianglesOfPoint([x, y])
            validTriangles = []
            for tri in closeTriangles:
                if isInside(tri, (x, y)):
                    validTriangles.append(tri)
            #default white if no triangles contained point
            col = (255, 255, 255)
            if (len(validTriangles) != 0):
                if (not fullRender):
                    col = (255, 0, 0)
                else:
                    #if there are multiple valid triangles, choose smallest average z
                    minTri = None
                    for tri in validTriangles:
                        if (minTri == None or avgD(minTri, 2) > avgD(tri, 2)):
                            minTri = tri
                    #maps to range 55 to 200
                    zVal = fitZ(getZ(minTri, (x, y)))
                    #improve by making dist = dist to camera, not just z value
                    c = int(max(min(255, 255 - zVal), 0))
                    col = (c, 0, 0)
            im.putpixel((xp + int(w/2), yp + int(h/2)), col)
    def makeBinary(img, n):
        img = img.resize((128, 128))
        back = Image.new("RGB", (420, 420), "white")
        st = int((420 - 128)/2)
        back.paste(img, (st, st))
        back.save(str(outputName) + "-" + str(n + 1) + ".png")

    if (fullRender):
        im.save(str(outputName) + "-" + str(num) + ".png")
        #create binary version if not already created
        bim = Image.new("RGB", (w, h), "white")
        for xp in range(w):
            for yp in range(h):
                oldC = im.getpixel((xp, yp))
                if (oldC == (255, 255, 255)):
                    bim.putpixel((xp, yp), oldC)
                else:
                    bim.putpixel((xp, yp), (255, 0, 0))
        makeBinary(bim, num)
    else:
        makeBinary(im, num)
"""Outputs two images of STL specified by partNum at specified angle:
outputName-1.png - fully rendered from view angle (w x h)
outputName-2.png - binary image from view angle (shrunk and placed onto backdrop)
"""
def generatePose(partNum, tilt1, tilt2, outputName, wid):
    m = mesh.Mesh.from_file("/Users/chrispowers/Desktop/research/vision-amt/yumiparts/STL/Part" + str(partNum) + ".STL")
    points, centeredMesh = getCenteredPoints(m)
    volume, cog, inertia = centeredMesh.get_mass_properties()
    # first 2 principal components are basis vectors of image plane
    pAxis, sAxis = getPC(points)
    pAxis, sAxis = pAxis/np.linalg.norm(pAxis), sAxis/np.linalg.norm(sAxis)
    #rod1, rod2 = getRod(pAxis, cog), getRod(sAxis, cog)
    #final = mesh.Mesh(np.concatenate([centeredMesh.data.copy(), rod1.data.copy(),rod2.data.copy()]))
    #final.save("final" + str(partNum) + ".stl")
    centeredMesh.rotate(pAxis, math.radians(tilt1))
    centeredMesh.rotate(sAxis, math.radians(tilt2))
    makeRaster(partNum, centeredMesh, pAxis, sAxis, cog, True, 1, outputName, wid)
    #Gets view from different aaxis
    # centeredMesh.rotate(sAxis, math.radians(90))
    # makeRaster(partNum, centeredMesh, pAxis, sAxis, cog, False, 3, outputName, wid)
