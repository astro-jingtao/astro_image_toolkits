import numpy as np
from astropy.convolution import Gaussian2DKernel, Kernel
from astropy.convolution import convolve as _convolve, convolve_fft as _convolve_fft
from skimage.measure import label
from asa.loess2d import LOESS2D
from .utils import get_isolate

# TODO: test all functions


def convolve(arr, kernel, method='direct', **kwargs):
    if method == 'direct':
        return _convolve(arr, kernel, **kwargs)
    elif method == 'fft':
        return _convolve_fft(arr, kernel, **kwargs)
    else:
        raise ValueError('method must be direct or fft')


def convolve_err(err, kernel, method='direct', **kwargs):
    if isinstance(kernel, Kernel):
        kernel = kernel.array
    kernel = np.square(kernel)
    var = np.square(err)
    kwargs['normalize_kernel'] = False
    return np.sqrt(convolve(var, kernel, method=method, **kwargs))


def match_psf_gaussian(width_from,
                       width_to,
                       data_from,
                       is_err=False,
                       patch_nan=True,
                       threshold=2,
                       width_type='fwhm',
                       fft_method='direct',
                       **kwargs):

    if width_to < width_from:
        raise ValueError('width_to must be greater than width_from')
    elif width_to == width_from:
        print('width_to is equal to width_from, no need to convolve')
        return data_from

    if width_type == 'fwhm':
        sigma_from = width_from / 2.355
        sigma_to = width_to / 2.355
    elif width_type == 'sigma':
        sigma_from = width_from
        sigma_to = width_to

    kernel_size_in_pix = np.sqrt(sigma_to**2 - sigma_from**2)
    kernel = Gaussian2DKernel(x_stddev=kernel_size_in_pix,
                              y_stddev=kernel_size_in_pix)

    if patch_nan:
        data_from = patch_image(data_from, np.isnan(data_from), threshold)

    if is_err:
        return convolve_err(data_from, kernel, method=fft_method, **kwargs)
    else:
        return convolve(data_from, kernel, method=fft_method, **kwargs)


def patch_image(data, mask, threshold):
    '''
    masked = 1, unmasked = 0
    '''

    if mask.dtype == bool:
        mask = mask.astype(np.uint8)

    # is small cluster and not background
    can_patch = get_isolate(
        mask, larger_cut=threshold, connectivity=1, background=0) & (mask != 0)

    # interpolate the can patched region
    x, y = np.meshgrid(np.arange(data.shape[1]), np.arange(data.shape[0]))
    data_new = data.copy()
    loess = LOESS2D(x[mask == 0], y[mask == 0], data[mask == 0], 100)
    data_new[can_patch == 1] = loess(x[can_patch == 1], y[can_patch == 1])

    return data_new
