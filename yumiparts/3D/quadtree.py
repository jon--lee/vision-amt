from geometry import overlap, contains, isInside, inTri

"""node for quadtree"""
class node():
    def __init__(self, lx, hx, ly, hy, minWidth):
        self.lx, self.hx, self.ly, self.hy = lx, hx, ly, hy
        self.children = []
        self.triangles = []
        self.asRect = [self.lx, self.ly, self.hx, self.hy]
        if ((self.hx - self.lx) > minWidth):
            self.createChildren(minWidth)
    """Generates 4 new squares on each layer"""
    def createChildren(self, minWidth):
        cx, cy = (self.lx + self.hx)/2, (self.ly + self.hy)/2
        for i in range(4):
            nlx, nly, nhx, nhy = self.lx, self.ly, cx, cy
            if (i % 2 == 0):
                nlx, nhx = cx, self.hx
            if (i < 2):
                nly, nhy = cy, self.hy
            self.children.append(node(nlx, nhx, nly, nhy, minWidth))
    """input is triangle as a list of three vertices in r3"
       encompasses = flag indicating that triangle completely encompasses current layer (optimization)"""
    def addTriangle(self, tri, encompasses):
        #base case is lowest level node
        if len(self.children) == 0:
            self.triangles.append(tri)
        else:
            for c in self.children:
                #triangle in rectangle if any vertex is in it or if rectangle is in vertex
                if encompasses or overlap(tri, c.asRect):
                    encompasses = inTri(tri, c.asRect)
                    c.addTriangle(tri, encompasses)
    """input is point as x, y pair; if point is on an edge, choose arbitrarily"""
    def getNodeOfPoint(self, p):
        #base case is lowest level node
        if len(self.children) == 0:
            return self
        else:
            for c in self.children:
                if contains(p, c.asRect):
                    return c.getNodeOfPoint(p)
"""recursively divides region into quadrants to improve searchability"""
class quadtree():
    """dim should be a power of 2 so it subdivides evenly"""
    def __init__(self, dim, minWidth):
        self.dim = dim
        self.frame = node(-dim/2, dim/2, -dim/2, dim/2, minWidth)
    """input is triangle as a list of three vertices in r3
       stores triangle in any low level nodes that interesect its projection"""
    def addTriangle(self, tri):
        self.frame.addTriangle(tri, False)
    """Gets lowest level node containing point
       checks if there are any triangles in that node that contain the point"""
    def getTrianglesOfPoint(self, p):
        node = self.frame.getNodeOfPoint([p[0], p[1]])
        return node.triangles
