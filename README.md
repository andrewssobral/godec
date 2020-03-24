# GoDec
Python implementation of the GoDec algorithm from [Zhou and Tao (ICML 2011)](http://www.icml-2011.org/papers/41_icmlpaper.pdf) for low-rank and sparse representation.

<p align="center"><img src="https://github.com/andrewssobral/godec/raw/master/doc/images/results_highway.png" width="50%" /></p>

# Requirements
* numpy
* scipy
* sklearn
* arparse, time (for godec_demo.py)
* matplotlib (optional, to visualize the results)
* opencv (optional, to read video files)

# Usage
Simple demo using data from MATLAB (see dataset folder):
```
#%% Import libraries
import scipy.io as sio

from godec import godec
from utils import play_2d_video, plot_2d_results

#%% Load data
mat = sio.loadmat('dataset/demo.mat')
M, height, width = mat['M'], int(mat['m']), int(mat['n'])

#%% Play input data
play_2d_video(M, width, height)

#%% Decompose
L, S, LS,  _ = godec(M)

#%% Plot results
plot_2d_results(M, LS, L, S, width, height)
```
For more info, see `godec_test.py`

# Examples
```
python godec_demo.py dataset/highway.mpg True
python godec_demo.py dataset/demo.avi True
python godec_demo.py dataset/demo.mat True
```

# License
MIT
