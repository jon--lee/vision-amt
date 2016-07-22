import numpy as np
from stl import mesh
import math
from PIL import Image

"""Returns two largest principal component vectors of given matrix"""
def getPC(data):
    U, s, Vt = np.linalg.svd(data, full_matrices=False)
    V = Vt.T
    #principal components are columns of Vt, or rows of V
    return V[0], V[1]
"""Returns a rod added along the given axis and through the mesh's center of gravity"""
def getRod(axis, cog):
    startPoint = cog + axis * 150
    endPoint = cog - axis * 150
    #rectangular axis with small square faces on each end
    vertices = np.array([startPoint, startPoint + [0, 0, 2], startPoint + [0, 2, 0], startPoint + [0, 2, 2],
                        endPoint, endPoint + [0, 0, 2], endPoint + [0, 2, 0], endPoint + [0, 2, 2]])
    vertInds = [i for i in range(8)]
    #triangular faces (two per rectangle)
    #specified by triplets of indices of vertices
    #order of indices matters for normal vectors (RH rule)
    faces = np.array([[0, 1, 3], [0, 3, 2], [4, 7, 5], [4, 6, 7], [0, 5, 1], [0, 4, 5], [2, 3, 7],
                     [2, 7, 6], [0, 6, 4], [0, 2, 6], [1, 5, 7], [1, 7, 3]])
    rod = mesh.Mesh(np.zeros(faces.shape[0], dtype=mesh.Mesh.dtype))
    for i, f in enumerate(faces):
        for j in range(3):
            rod.vectors[i][j] = vertices[f[j],:]
    return rod
"""Centers mesh at origin then returns all points in mesh"""
def getCenteredPoints(oldMesh):
    dim = 3
    points = []
    #unique points contains triplets of points (lists not hashable)
    uniquePoints = set(())
    totals = [0 for i in range(dim)]
    numPoints = 0
    #each point of mesh is a triplet containing each vertex of triangle
    vert = 3
    f = 0
    for triplet in oldMesh.points:
        #each point consists of x, y, z coordinates
        for i in range(0, vert):
            ind = i * dim
            p = [0 for i in range(dim)]
            for j in range(len(p)):
                p[j] += triplet[ind+j]
            if tuple(p) not in uniquePoints:
                numPoints += 1
                totals = [totals[i] + p[i] for i in range(dim)]
                points.append(p)
                uniquePoints.add(tuple(p))
    means = [totals[i]/numPoints for i in range(dim)]
    #center mesh at origin so that principal axis is a vector starting at origin
    data = oldMesh.data.copy()
    #i/j refer to different vectors, k refers to coordinate
    for i in range(len(data['vectors'])):
        for j in range(3):
            for k in range(3):
                data['vectors'][i][j][k] -= means[k]
    centeredMesh = mesh.Mesh(data.copy())
    #center mean of data at 0 for pca
    for p in points:
        for i in range(len(p)):
            p[i] -= means[i]
    return np.array(points), centeredMesh
"""mergesorts array of arrays
   comparisons by elements at inner index = axis
   compares after applying conversion to all elements"""
def mergeConvert(points, transform, shift, axis):
    if (len(points) == 1):
        return [np.dot(transform, points[0] - shift)]
    else:
        splitInd = int(len(points)/2)
        m1 = mergeConvert(points[0:splitInd], transform, shift, axis)
        m2 = mergeConvert(points[splitInd:len(points)], transform, shift, axis)
        combined = []
        while (len(m1) > 0 or len(m2) > 0):
            if (len(m1) == 0):
                combined.append(m2[0])
                m2 = m2[1:]
            elif (len(m2) == 0):
                combined.append(m1[0])
                m1 = m1[1:]
            else:
                if m1[0][axis] < m2[0][axis]:
                    combined.append(m1[0])
                    m1 = m1[1:]
                else:
                    combined.append(m2[0])
                    m2 = m2[1:]
        return combined
"""binary searches an array of arrays for a single value
   sorted/searches by elements at inner index = axis
   returns outer index of closest match"""
def findClosestInd(points, val, axis):
    low = 0
    high = len(points) - 1
    while (high - low > 1):
        mid = int((low + high)/2)
        if points[mid][axis] < val:
            low = mid
        else:
            high = mid
    if (val - points[low][axis] <= points[high][axis] - val):
        return low
    return high
"""input array of arrays sorted by inner index = axis
   checks that val is within lowest/highest values at inner index = axis"""
def inRange(points, val, axis):
    return not (val < points[0][axis] or val > points[-1][axis])
"""returns list of points with value at inner index = axis within tolerance of val
   begins search at startInd, closest point in sorted points array to val"""
def nearPoints(points, startInd, axis, val, tolerance, axis2, val2, tolerance2):
    validPoints = []
    currInd = startInd
    while (currInd > -1 and points[currInd][axis] >= val - tolerance and points[currInd][axis] <= val + tolerance):
        currP = points[currInd]
        oVal = currP[axis2]
        if (oVal <= val2 + tolerance2 and oVal >= val2 - tolerance2):
            validPoints.append(currP)
        currInd -= 1
    currInd = startInd + 1
    while (currInd < len(points) and points[currInd][axis] <= val + tolerance and points[currInd][axis] >= val - tolerance):
        currP = points[currInd]
        oVal = currP[axis2]
        if (oVal <= val2 + tolerance2 and oVal >= val2 - tolerance2):
            validPoints.append(currP)
        currInd += 1
    return validPoints
"""finds squared xy distance between two points in R3"""
def distSquared(p0, p1):
    return (p0[0] - p1[0])**2 + (p0[1] - p1[1])**2
for partNum in range(1, 2):
    m = mesh.Mesh.from_file("/Users/chrispowers/Desktop/research/reyumiparts/Part" + str(partNum) + ".STL")
    points, centeredMesh = getCenteredPoints(m)
    volume, cog, inertia = centeredMesh.get_mass_properties()
    # get primary and secondary principal component axes
    pAxis, sAxis = getPC(points)
    # rotate to align principal axes with y and z axes
    z, y = np.array([0, 0, 1]), np.array([0, 1, 0])
    pAxis, sAxis = pAxis/np.linalg.norm(pAxis), sAxis/np.linalg.norm(sAxis)
    rod1, rod2 = getRod(pAxis, cog), getRod(sAxis, cog)
    final = mesh.Mesh(np.concatenate([centeredMesh.data.copy(), rod1.data.copy(),rod2.data.copy()]))
    final.save("final" + str(partNum) + ".stl")
    #vision axis is perpendicular to both principal components (basis vectors of image plane)
    vAxis = np.cross(pAxis, sAxis)
    #origin at some distance along vAxis from cog of object, f = further distance to camera
    #w and h should be integers (dimensions of output image)
    originD, f, w, h = -100, -40, 300, 300
    origin = cog + originD * vAxis
    camera = origin + f * vAxis
    im = Image.new("RGB", (w, h), "white")
    ma = np.array([pAxis, sAxis, vAxis]).T
    mInv = np.linalg.inv(ma)
    def rToCam(p):
        return np.dot(mInv, p - origin)
    #convert coordinates of and sort points by x coordinates
    pointsX = mergeConvert(points, mInv, origin, 0)
    #Resort points by y coordinates
    pointsY = mergeConvert(pointsX, np.identity(3), 0, 1)
    #Can increase image size by mapping 1 pixel to 1/2 unit
    for x in range(int(-w/2), int(w/2)):
        for y in range(int(-h/2), int(h/2)):
            validPoints = []
            tolerance, cutoff = 3, 3
            #do not include points if x coord is right or left of all points, or if y coord is above or below all points
            if (inRange(pointsX, x, 0) and inRange(pointsY, y, 1)):
                #find all points within tolerance in x and cutoff of y, or within tolerance of y and cutoff of x
                closestXInd = findClosestInd(pointsX, x, 0)
                closestYInd = findClosestInd(pointsY, y, 1)
                validPoints += nearPoints(pointsX, closestXInd, 0, x, tolerance, 1, y, cutoff)
                validPoints += nearPoints(pointsY, closestYInd, 1, y, tolerance, 0, x, cutoff)
            #of remaining points, take one with closest xy distance, breaking ties by minimum z values
            minInd = 0
            if (len(validPoints) > 0):
                pStart = validPoints[minInd]
                minDist = distSquared((x, y), pStart)
                for i in range(1, len(validPoints)):
                    p = validPoints[i]
                    d = distSquared((x, y), p)
                    if (d == minDist and validPoints[minInd][2] > p[2]) or (d < minDist):
                        minInd = i
                        minDist = d
            #default blue if no points were within range
            col = (0, 0, 255)
            if (len(validPoints) != 0):
                minZ = validPoints[minInd][2]
                #expands range of z values to make depth difference clearer
                minZ = (minZ - 80) * 5
                c = int(max(min(255, 255 - minZ), 0))
                col = (c, 0, 0)
            im.putpixel((x + int(w/2), y + int(h/2)), col)
    im.save('test' + str(partNum) + '.png')
