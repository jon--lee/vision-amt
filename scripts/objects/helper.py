from PIL import Image
import numpy as np
import math
import random

"""Helper class to ensure valid object placement"""
class Rectangle(object):
    def __init__ (self, lowX, width, lowY, height):
        self.left = lowX
        self.right = lowX + width
        self.bot = lowY
        self.top = lowY + height
        self.cenX = (self.left + self.right)/2
        self.cenY = (self.bot + self.top)/2
    def doesCollideWith(self, rect):
        """returns true if two rectangles have overlap"""
        return (self.left < rect.right and self.right > rect.left and self.bot < rect.top and self.top > rect.bot)
    def hasCollision(self, rects):
        """returns true if rectangle has overlap with any other rectangle in array"""
        for r in rects:
            if self.doesCollideWith(r):
                return True
        return False
    def within(self, maxD, p1, p2):
        """returns true if points are within maximum distance of each other"""
        dist = math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)
        return dist < maxD
    def insideCircle(self, radius, centerX, centerY):
        """returns true if no area of rectangle is outside of bounding circle"""
        center = (centerX, centerY)
        vertices = ((self.left, self.bot), (self.left, self.top), (self.right, self.bot), (self.right, self.top))
        isInside = True
        for v in vertices:
            isInside = isInside and self.within(radius, center, v)
        return isInside

    def insideRectangle(self, topL, bottomR):
        """returns true if no area of rectangle is outside of bounding circle"""
        vertices = ((self.left, self.bot), (self.left, self.top), (self.right, self.bot), (self.right, self.top))
        isInside = True
        for v in vertices:
            isInside = isInside and v[0] < bottomR[0] and v[0] > topL[0] and v[1] > bottomR[1] and v[1] < topL[1] 
        return isInside
# Helper functions
sign = lambda x: math.copysign(1, x)
cosine = lambda theta: math.cos(math.radians(theta))
sine = lambda theta: math.sin(math.radians(theta))
w = lambda img: img.size[0]
h = lambda img: img.size[1]
getRot = lambda rotLimit: int(np.random.normal(0, rotLimit/3))
def pasteOn(back, add, x, y):
    back.paste(add, (int(x), int(y)), add.convert('RGBA'))
