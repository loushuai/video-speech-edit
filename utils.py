import numpy as np
import subprocess
import sys


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


def run_subprocess(command, env=None):
    if env is not None:
        command = ["conda",  "run", "-n", env, "--live-stream"] + command

    try:
        process = subprocess.Popen(command, stdout=subprocess.PIPE, text=True, bufsize=1)
    except subprocess.CalledProcessError as e:
        print(f"Error executing command: {e}")
        raise e

    for line in process.stdout:
        sys.stdout.write(line) # Prints each line as it's received
        sys.stdout.flush()
    process.wait()
