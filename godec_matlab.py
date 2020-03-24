#%% Import libraries
import time
import scipy.io as sio

from godec import godec
from utils import play_2d_video, play_2d_results

#%% Load data
mat = sio.loadmat('dataset/demo.mat')
M, height, width = mat['M'], int(mat['m']), int(mat['n'])

#%% Play input data
play_2d_video(M, width, height)

#%% Decompose
t = time.time()
L, S, LS, _ = godec(M)
elapsed = time.time() - t
print(elapsed, "sec elapsed")

#%% Play results
play_2d_results(M, LS, L, S, width, height)
