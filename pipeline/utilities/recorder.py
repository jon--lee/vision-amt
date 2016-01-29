import const
import cv2
import numpy as np

def generate(shape):
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter('original.mov', fourcc, 10.0, shape)
    return writer
