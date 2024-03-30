import numpy as np
from astropy.convolution import convolve, convolve_fft
from astropy.convolution import Gaussian2DKernel, Gaussian1DKernel
from joblib import Parallel, delayed




def convole_cube_1d(cube, fwhm, **kwargs):
    '''
    cube: 3D numpy array, (n_pix1, n_pix2, n_vel)
    fwhm: float, in unit of pixel
    '''

    if fwhm < 1:
        return cube

    n_pix1, n_pix2, _ = cube.shape

    kernel = Gaussian1DKernel(fwhm / 2.355)

    # if large kernel use fft
    if fwhm > 10:
        conv_method = convolve_fft
    else:
        conv_method = convolve

    cube_conv = np.zeros_like(cube)
    for i in range(n_pix1):
        for j in range(n_pix2):
            cube_conv[i, j, :] = conv_method(cube[i, j, :], kernel, **kwargs)

    return cube_conv
