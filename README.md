# GoDec
Python implementation of the GoDec algorithm from [Zhou and Tao (ICML 2011)](http://www.icml-2011.org/papers/41_icmlpaper.pdf) for low-rank and sparse representation.

<p align="center"><img src="https://github.com/andrewssobral/godec/raw/master/doc/images/results_highway.png" width="50%" /></p>

# Requirements
* numpy
* scipy
* sklearn
* arparse, time (for demos)
* matplotlib (optional, to visualize the results)
* opencv (optional, to read video files)

# Demo using OpenCV
Simple demo using data from video file with OpenCV:
```
# Import libraries
import cv2 as cv
import time
from numpy import array, column_stack
from godec import godec
from utils import play_2d_results

# Open video file
cap = cv.VideoCapture('dataset/demo.avi')

# Read until video is completed
M = None
height = None
width = None
i = 0
print("Press 'q' to stop...")
while cap.isOpened():
    # Capture frame-by-frame
    ret, frame = cap.read()
    if ret:
        frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        height, width = frame.shape

        # Stack frame as a column vector
        F = frame.T.reshape(-1)

        if i == 0:
            M = array([F]).T
        else:
            M = column_stack((M, F))

        # Display the resulting frame
        cv.imshow('Frame', frame)

        # Press Q on keyboard to exit
        if cv.waitKey(25) & 0xFF == ord('q'):
            break

        i = i + 1

    # Break the loop
    else:
        break

# Release the video capture object
cap.release()

# Closes all the windows
cv.destroyAllWindows()

# Decompose
print("Number of frames: ", i, " with size ", (width, height))
print("Processing M with shape ", M.shape)
t = time.time()
L, S, LS, _ = godec(M)
elapsed = time.time() - t
print(elapsed, "sec elapsed")

# Play results
play_2d_results(M, LS, L, S, width, height)
```
For more info, see `godec_opencv.py`

# Demo using MATLAB data
Simple demo using data from MATLAB:
```
# Import libraries
import time
import scipy.io as sio

from godec import godec
from utils import play_2d_video, play_2d_results

# Load data
mat = sio.loadmat('dataset/demo.mat')
M, height, width = mat['M'], int(mat['m']), int(mat['n'])

# Play input data
play_2d_video(M, width, height)

# Decompose
t = time.time()
L, S, LS, _ = godec(M)
elapsed = time.time() - t
print(elapsed, "sec elapsed")

# Play results
play_2d_results(M, LS, L, S, width, height)
```
For more info, see `godec_matlab.py`

# More examples
```
python godec_demo.py dataset/highway.mpg True
python godec_demo.py dataset/demo.avi True
python godec_demo.py dataset/demo.mat True
```

# License
MIT
