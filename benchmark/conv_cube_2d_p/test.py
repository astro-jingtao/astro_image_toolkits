import numpy as np
import time
from ait.conv import conv_cube_2d_p
from astropy.convolution import Gaussian2DKernel

n = 1000

cube = np.random.randn(100, n, n)
kernel = Gaussian2DKernel(11)

start_time = time.time()
conv_p = conv_cube_2d_p(cube, kernel, n_jobs=5)
end_time = time.time()
execution_time = end_time - start_time
print(f"batch_size = 'auto' took {execution_time} seconds to complete.")

start_time = time.time()
conv_p = conv_cube_2d_p(cube, kernel, n_jobs=5, batch_size=20)
end_time = time.time()
execution_time = end_time - start_time
print(f"batch_size = 20 took {execution_time} seconds to complete.")
