#%% Import libraries
import scipy.io as sio

from godec import godec
from utils import play_2d_video, plot_2d_results, plot_2d_results_mean, play_2d_results

#%% Load data
mat = sio.loadmat('dataset/demo.mat')
M, height, width = mat['M'], int(mat['m']), int(mat['n'])

#%% Play input data
play_2d_video(M, width, height)

#%% Decompose
L, S, LS,  _ = godec(M)

#%% Plot results
plot_2d_results(M, LS, L, S, width, height)

#%% Plot mean results
plot_2d_results_mean(M, LS, L, S, width, height)

#%% Play results
play_2d_results(M, LS, L, S, width, height)
