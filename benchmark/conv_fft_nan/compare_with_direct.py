import numpy as np
import time
from ait.conv import convolve
from astropy.convolution import Gaussian2DKernel

arr = np.random.randn(1000, 1000)
kernel = Gaussian2DKernel(15)
arr[40:60, 40:60] = np.nan

start_time = time.time()
arr_direct = convolve(arr,
                      kernel,
                      method="direct",
                      boundary="fill",
                      fill_value=np.nan,
                      nan_treatment="fill")
end_time = time.time()
execution_time = end_time - start_time
print(f"direct took {execution_time} seconds to complete.")

try:
    arr_fft = convolve(arr,
                       kernel,
                       method="fft",
                       boundary="fill",
                       fill_value=np.nan,
                       nan_treatment="fill")
except Exception as e:
    print("fft failed.")
    print(e)

start_time = time.time()
arr_fft_nan = convolve(arr, kernel, method="fft_nan")
end_time = time.time()
execution_time = end_time - start_time
print(f"fft_nan took {execution_time} seconds to complete.")

print("direct and fft_nan are equal?",
      np.allclose(arr_direct, arr_fft_nan, equal_nan=True))
