import numpy as np

def find_closest_timestamp(arr, t):
    ind = np.nanargmin(np.abs(arr - t))
    approx_t = arr[ind]
    return ind, approx_t

def boxcar_smooth(data, window_size):
  if window_size <= 0:
    raise ValueError("Window size must be positive")
  kernel = np.ones(window_size) / window_size
  smoothed_data = np.convolve(data, kernel, mode='same')
  return smoothed_data