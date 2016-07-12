from stl import mesh
import numpy as np
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
