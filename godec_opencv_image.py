#%% Import libraries
import time
import numpy as np

try:
    import cv2
except ImportError:
    raise ImportError('OpenCV is requires to read video files')

from godec import godec
from utils import *

# Read until video is completed
M = None
height = None
width = None
i = 0

img = cv2.imread("/Users/onyekachukwuokonji/Desktop/godec/dataset/3.jpg")
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
height, width = img.shape

# flatten the 2-D array, produces a row vector
frame = img.T.reshape(-1)

# convert row vector to column vector
M = np.array([frame]).T

cv2.imshow('frame', frame)

assert M is not None
assert width is not None
assert height is not None

print("Number of frames: ", i, " with size ", (width, height))
print("Processing M with shape ", M.shape)

#%% Decompose
t = time.time()
L, S, LS, _ = godec(M)
elapsed = time.time() - t
print(elapsed, "sec elapsed")

#%% Display results
plot_2d_results(M, LS, L, S, height, width)
