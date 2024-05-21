import numpy as np
import time
from ait.conv import conv_cube_2d_p
from astropy.convolution import Gaussian2DKernel

n = 1000

cube = np.random.randn(100, n, n)
kernel = Gaussian2DKernel(11)

start_time = time.time()
conv_p_fft = conv_cube_2d_p(cube,
                            kernel,
                            n_jobs=5,
                            method='fft',
                            boundary='fill')
end_time = time.time()
execution_time = end_time - start_time
print(f"fft took {execution_time} seconds to complete.")

start_time = time.time()
conv_p_direct = conv_cube_2d_p(cube,
                               kernel,
                               n_jobs=5,
                               method='direct',
                               boundary='fill')
end_time = time.time()
execution_time = end_time - start_time
print(f"direct took {execution_time} seconds to complete.")
