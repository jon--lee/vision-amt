import numpy as np
import cv2
from options import AMTOptions

def im2tensor(im,channels=3):
    """
        convert 3d image (height, width, 3-channel) where values range [0,255]
        to appropriate pipeline shape and values of either 0 or 1
        cv2 --> tf
    """
    shape = np.shape(im)
    h, w = shape[0], shape[1]
    zeros = np.zeros((h, w, channels))
    for i in range(channels):
        zeros[:,:,i] = np.round(im[:,:,i] / 255.0 - .25, 0)
    return zeros



if __name__ == '__main__':
    im = cv2.imread(AMTOptions.amt_dir + 'comparisons/pictures_CASE2017/characteristic_errors.png')
    # im = cv2.imread(AMTOptions.amt_dir + 'comparisons/pictures_CASE2017/correcting.png')

    im2 = im2tensor(im, channels=3)
    cv2.imshow('preview', im2)
    cv2.waitKey(0)

    im3 = np.zeros(im2.shape)
    for i in range(im2.shape[0]):
        for j in range(im2.shape[1]):
            if im2[i, j, 0] > .3 or im2[i, j, 1] > .3 or im2[i, j, 2] > .3:
                im3[i, j, :] = im[i, j, :]
            else:
                im3[i, j, :] = [255.0, 255.0, 255.0]
                
    cv2.imshow('preview', im3 / 255.0)
    cv2.waitKey(0)
    cv2.imwrite('preview.jpg', im3)
