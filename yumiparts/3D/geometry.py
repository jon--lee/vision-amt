"""Finds the barycentric coordinates of a point with respect to a triangle"""
def baryCoord(tri, p):
    p1, p2, p3 = tri[0], tri[1], tri[2]
    #can cache 4 of these values with triangle
    v13y, v23y, vp3y = p1[1] - p3[1], p2[1] - p3[1], p[1] - p3[1]
    v13x, v32x, vp3x = p1[0] - p3[0], p3[0] - p2[0], p[0] - p3[0]
    #denominator should not be zero if it is a valid triangle
    denom = v23y * v13x + v32x * v13y
    invDenom = 1/denom
    alphaNum = v23y * vp3x + v32x * vp3y
    betaNum = -v13y * vp3x + v13x * vp3y
    alpha = alphaNum * invDenom
    beta = betaNum * invDenom
    gamma = 1 - alpha - beta
    return alpha, beta, gamma
"""triangle: array of 3 points, each with 2-3 coordinates
   point: array of 2-3 coordinates
   checks if point is inside xy projection of triangle; on line coutns as inside
   Uses http://stackoverflow.com/questions/13300904/determine-whether-point-lies-inside-triangle"""
def isInside(triangle, point):
    alpha, beta, gamma = baryCoord(triangle, point)
    return (alpha >= 0) and (beta >= 0) and (gamma >= 0)
"""returns true if rectangle is completely inside triangle"""
def inTri(tri, rect):
    #checks that every vertex of rectangle is inside triangle
    result = isInside(tri, rect[0:2]) and isInside(tri, rect[2:4])
    return result and isInside(tri, [rect[0], rect[3]]) and isInside(tri, [rect[2], rect[1]])
"""input is triangle of list of three vertices in r3
   rectangle as [x1, y1, x2, y2](non-rotated)"""
def overlap(tri, rect):
    result = False
    #return true if any vertex of triangle is in rect
    result = result or contains(tri[0], rect) or contains(tri[1], rect) or contains(tri[2], rect)
    #check if any of rectangle's lines intersect any of triangle's lines
    result = result or lineRect(tri[0][0], tri[1][0], tri[0][1], tri[1][1], rect)
    result = result or lineRect(tri[2][0], tri[1][0], tri[2][1], tri[1][1], rect)
    result = result or lineRect(tri[0][0], tri[2][0], tri[0][1], tri[2][1], rect)
    #check if rectangle is completely inside triangle (if any of rectangle's vertices are inside triangle)
    result = result or isInside(tri, rect[0:2]) or isInside(tri, rect[2:4])
    result = result or isInside(tri, [rect[0], rect[3]]) or isInside(tri, [rect[2], rect[1]])
    return result
"""Checks if the line specified by the given endpoints intersects rectangle as [x1, y1, x2, y2] (non-rotated)
   Uses http://seb.ly/2009/05/super-fast-trianglerectangle-intersection-test/"""
def lineRect(x0, x1, y0, y1, rect):
    denom = x1 - x0
    #vertical line special case
    if (denom == 0 and not(rect[0] < x0 and rect[2] > x0)):
        return False
    ilow = rect[1]
    ihigh = rect[3]
    if (denom != 0):
        m = (y1 - y0)/denom
        c = y0 - (m * x0)
        #y coordinates where projected sides of rectangle intersect line
        ilow, ihigh = c + m * rect[0], c + m * rect[2]
        if ilow > ihigh:
            ilow, ihigh = ihigh, ilow
    ylow, yhigh = y0, y1
    if ylow > yhigh:
        ylow, yhigh = yhigh, ylow
    low = max(ylow, ilow)
    high = min(yhigh, ihigh)
    return (low < high) and (not (low > rect[3] or high < rect[1]))
"""input is point as [x, y] and rectangle as [x1, y1, x2, y2] (non-rotated)"""
def contains(p, rect):
    inX = p[0] >= rect[0] and p[0] <= rect[2]
    inY = p[1] >= rect[1] and p[1] <= rect[3]
    return inX and inY
"""returns the average of one coordinates of each vertex of triangular face"""
def avgD(face, axis):
    return (face[0][axis] + face[1][axis] + face[2][axis])/3.0
"""finds squared xy distance between two points with 2-3 dimensions"""
def distSquared(p0, p1):
    return (p0[0] - p1[0])**2 + (p0[1] - p1[1])**2
"""gets the z coordinate of the triangle at the given (x, y) position"""
def getZ(tri, p):
    alpha, beta, gamma = baryCoord(tri, p)
    #calculate z as weighted average where weights are barycentric coordinates
    return alpha * tri[0][2] + beta * tri[1][2] + gamma * tri[2][2]
