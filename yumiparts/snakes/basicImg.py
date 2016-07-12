import numpy as np
import math
def cos(t):
    return math.cos(math.radians(t))
def sin(t):
    return math.sin(math.radians(t))
def makeTransform(a1, a2, a3, a4, b1, b2, c1, c2):
    return np.array([np.array([a1, a2, b1]), np.array([a3, a4, b2]), np.array([c1, c2, 1])])
def translate(x, y):
    return makeTransform(1, 0, 0, 1, x, y, 0, 0)
def rotate(t):
    return makeTransform(cos(t), -sin(t), sin(t), cos(t), 0, 0, 0, 0)
def affine(a1, a2, a3, a4, b1, b2):
    return makeTransform(a1, a2, a3, a4, b1, b2, 0, 0)
def project(a1, a2, a3, a4, b1, b2, c1, c2):
    return makeTransform(a1, a2, a3, a4, b1, b2, c1, c2)
def compose(ts):
    result = np.identity(3)
    for t in ts:
        result = np.dot(t, result)
    return result
def transform(m, v):
    result = np.dot(m, np.array(v + [1]))
    w = result[2]
    return (result/w)[0:2]
m = project(1, 0, 0, 1, 0, 0, 0.5, 0.5)
print(transform(m, [0, 5]))
