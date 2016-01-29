# Vision pipeline interface

##### calibrate.py

Calibration is required as a first step in order for the images
to be properly segmented. Clear the table aside from a single red
dot in the middle of the turn table. Run:

`$ python calibrate.py`

Adjust width, height, x, y of the frame in `constants/__init__.py` appropriately
so that the square frame roughly circumscribes the circular turn table.

Ensure that the red dot and a point on the turn table's border are located as a bright red group and a
bright green group of pixels respectively.

Calibration information is saved in `meta.txt`

##### BinaryCamera

Once calibrated, you can use BinaryCamera.

- `bc = bincam.BinaryCamera()`  Create a new instance
- `bc.pipe(frame)`              Given a frame (np.array) resize to 125x125 and apply binary segmentation
- `bc.open()`                   If necessary, open the camera (must be called before using any read methods)
- `bc.read_frame(show=False)`             Returns the original color frame captured by camera
- `bc.read_binary_frame(show=False)`      Applies and returns result of `pipe` a frame captured on camera.

By setting `show=True`, the read frame will be displayed in cv2.
