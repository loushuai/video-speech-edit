import numpy as np


def interp_array1d(arr, target_len):
    """Interpolate a 1D array to the target length using linear interpolation."""
    original_len = len(arr)
    if original_len == target_len:
        return arr
    x_original = np.linspace(0, 1, original_len)
    x_target = np.linspace(0, 1, target_len)
    interpolated_idx = np.interp(x_target, x_original, np.arange(original_len))
    interpolated_arr = arr[np.floor(interpolated_idx).astype(int)]
    return interpolated_arr

