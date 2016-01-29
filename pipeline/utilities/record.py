import const
import cv2
import numpy

vc = cv2.VideoCapture(0)
fourcc = cv2.VideoWriter_fourcc(*'mp4v')



if vc.isOpened():
    rval, frame = vc.read()
else:
    rval = False

rval, frame = vc.read()

original_writer = cv2.VideoWriter('original.mov', fourcc, 10.0, np.shape(frame))
cropped_writer = cv2.VideoWriter('cropped.mov', fourcc,  10.0, (const.WIDTH, const.HEIGHT))



while 1:
    original_writer.write(frame)
    frame = frame[0+const.OFFSET_Y:const.HEIGHT+const.OFFSET_Y, 0+const.OFFSET_X:const.WIDTH+const.OFFSET_X]
    cropped_writer.write(frame)
    rval, frame = vc.read()

    key = cv2.waitKey(20)
    if key == 27:
        break

original_writer.release()
cropped_writer.release()

cv2.destroyAllWindows()
