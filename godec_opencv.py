#%% Import libraries
import time

from numpy import array, column_stack

try:
    import cv2 as cv
except ImportError:
    raise ImportError('OpenCV is requires to read video files')

from godec import godec
from utils import play_2d_results

#%% Open video file
cap = cv.VideoCapture('dataset/demo.avi')

# Check if video was opened successfully
if not cap.isOpened():
    raise("Error opening video stream or file")

# Read until video is completed
M = None
height = None
width = None
i = 0
debug = True
print("Press 'q' to stop...")
while cap.isOpened():
    # Capture frame-by-frame
    ret, frame = cap.read()
    if ret:
        frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        height, width = frame.shape

        # Stack frames as column vectors
        F = frame.T.reshape(-1)

        if i == 0:
            M = array([F]).T
        else:
            M = column_stack((M, F))

        if debug:
            # Display the resulting frame
            cv.imshow('Frame', frame)

            # Press Q on keyboard to exit
            if cv.waitKey(25) & 0xFF == ord('q'):
                break

        i = i + 1
        # print(i)

    # Break the loop
    else:
        break

# Release the video capture object
cap.release()

# Closes all the frames
if debug:
    cv.destroyAllWindows()

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

#%% Play results
play_2d_results(M, LS, L, S, width, height)
