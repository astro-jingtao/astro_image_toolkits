from functools import partial
import warnings
import numpy as np
from astropy.convolution import Gaussian2DKernel, Kernel
from astropy.convolution import convolve as _convolve, convolve_fft as _convolve_fft
from joblib import Parallel, delayed

from asa.loess2d import LOESS2D
from .utils import get_isolate

# TODO: test all functions


def convolve(arr, kernel, is_err=False, method='direct', **kwargs):
    if is_err:
        return convolve_err(arr, kernel, method='direct', **kwargs)
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
    return np.sqrt(convolve(var, kernel, is_err=False, method=method,
                            **kwargs))


def conv_image():
    ...


def conv_cube_2d(cube, kernel, method='direct', vel_axis=0, **kwargs):

    if vel_axis not in [0, 2]:
        raise ValueError('vel_axis must be 0 or 2')

    conv_method = partial(convolve, kernel=kernel, method=method, **kwargs)
    cube_conv = np.zeros_like(cube)

    if vel_axis == 0:
        for i in range(cube.shape[0]):
            cube_conv[i, :, :] = conv_method(cube[i, :, :])
    elif vel_axis == 2:
        for i in range(cube.shape[2]):
            cube_conv[:, :, i] = conv_method(cube[:, :, i])

    return cube_conv


def conv_cube_2d_p(cube,
                   kernel,
                   method='direct',
                   vel_axis=0,
                   n_jobs=1,
                   batch_size='auto',
                   **kwargs):

    if vel_axis not in [0, 2]:
        raise ValueError('vel_axis must be 0 or 2')

    conv_method = partial(convolve, kernel=kernel, method=method, **kwargs)

    if vel_axis == 0:
        cube_conv_stack = Parallel(n_jobs=n_jobs, batch_size=batch_size)(
            delayed(conv_method)(cube[i, :, :]) for i in range(cube.shape[0]))
        cube_conv = np.array(cube_conv_stack)

    elif vel_axis == 2:
        cube_conv_stack = Parallel(n_jobs=n_jobs, batch_size=batch_size)(
            delayed(conv_method)(cube[:, :, i]) for i in range(cube.shape[2]))
        cube_conv = np.array(cube_conv_stack).transpose(1, 2, 0)
    else:
        raise ValueError('vel_axis must be 0 or 2')

    return cube_conv


def conv_cube_1d(cube, kernel, method='direct', vel_axis=0, **kwargs):

    if vel_axis not in [0, 2]:
        raise ValueError('vel_axis must be 0 or 2')

    conv_method = partial(convolve, kernel=kernel, method=method, **kwargs)
    cube_conv = np.zeros_like(cube)

    if vel_axis == 0:
        _, n_pix1, n_pix2 = cube.shape
        for i in range(n_pix1):
            for j in range(n_pix2):
                cube_conv[:, i, j] = conv_method(cube[:, i, j])
    elif vel_axis == 2:
        n_pix1, n_pix2, _ = cube.shape
        for i in range(n_pix1):
            for j in range(n_pix2):
                cube_conv[i, j, :] = conv_method(cube[i, j, :])

    return cube_conv


def conv_cube_1d_p(cube,
                   kernel,
                   method='direct',
                   vel_axis=0,
                   n_jobs=1,
                   **kwargs):

    if vel_axis not in [0, 2]:
        raise ValueError('vel_axis must be 0 or 2')

    conv_method = partial(convolve, kernel=kernel, method=method, **kwargs)

    if vel_axis == 0:
        _, n_pix1, n_pix2 = cube.shape
        cube_conv_stack = Parallel(n_jobs=n_jobs)(delayed(conv_method)(cube[:,
                                                                            i,
                                                                            j])
                                                  for i in range(n_pix1)
                                                  for j in range(n_pix2))
        cube_conv = np.array(cube_conv_stack).reshape(
            (n_pix1, n_pix2, -1)).transpose(2, 0, 1)

    elif vel_axis == 2:
        n_pix1, n_pix2, _ = cube.shape
        cube_conv_stack = Parallel(n_jobs=n_jobs)(
            delayed(conv_method)(cube[i, j, :]) for i in range(n_pix1)
            for j in range(n_pix2))
        cube_conv = np.array(cube_conv_stack).reshape((n_pix1, n_pix2, -1))
    else:
        raise ValueError('vel_axis must be 0 or 2')

    return cube_conv


def match_psf_gaussian_cube_2d():
    ...


def _parse_input_width(width_from, width_to, width_type):

    if width_to < width_from:
        raise ValueError('width_to must be greater than width_from')
    else:
        if width_to == width_from:
            is_equal = True
        else:
            is_equal = False

    if width_type == 'fwhm':
        sigma_from = fwhm2sigma(width_from)
        sigma_to = fwhm2sigma(width_to)
    elif width_type == 'sigma':
        sigma_from = width_from
        sigma_to = width_to
    else:
        raise ValueError('width_type must be fwhm or sigma')

    return sigma_from, sigma_to, is_equal


# TODO: support width_in_pixel_unit=False
# TODO: support error correction
def match_psf_gaussian(data_from,
                       width_from,
                       width_to,
                       is_err=False,
                       patch_nan=True,
                       threshold=2,
                       width_in_pixel_unit=True,
                       width_type='fwhm',
                       conv_method='direct',
                       **kwargs):

    if not width_in_pixel_unit:
        raise NotImplementedError(
            'width_in_pixel_unit=False is not supported yet')

    sigma_from, sigma_to, is_equal = _parse_input_width(
        width_from, width_to, width_type)

    if is_equal:
        print('width_to is equal to width_from, no need to convolve')
        return data_from

    kernel_size_in_pix = np.sqrt(sigma_to**2 - sigma_from**2)
    kernel = Gaussian2DKernel(x_stddev=kernel_size_in_pix,
                              y_stddev=kernel_size_in_pix)

    if patch_nan:
        data_from = patch_image(data_from, np.isnan(data_from), threshold)

    if is_err:
        return convolve_err(data_from, kernel, method=conv_method, **kwargs)
    else:
        return convolve(data_from, kernel, method=conv_method, **kwargs)


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


def sigma2fwhm(sigma):
    return 2.355 * sigma


def fwhm2sigma(fwhm):
    return fwhm / 2.355


def get_error_correction(width_from,
                         width_to,
                         width_type='fwhm',
                         d=2,
                         small_value_warning=True):
    # https://iopscience.iop.org/article/10.3847/2515-5172/abe8df

    sigma_from, sigma_to, is_equal = _parse_input_width(
        width_from, width_to, width_type)

    if is_equal:
        print('width_to is equal to width_from, no need to correct')
        return 1

    b = sigma_from
    theta = np.sqrt(sigma_to**2 - sigma_from**2)

    res = np.sqrt(
        np.power((2 * np.sqrt(np.pi) * theta * b) / (np.sqrt(theta**2 + b**2)),
                 d))

    if res < 1:
        if small_value_warning:
            warnings.warn('''
                The calculated error correction is less than 1,
                which indicates that the the original PSF or (and) the kernel is not well-sampled. 
                The behavior of error propagation in this case can not be handled by this formula. 
                Returning 1 (usually a good approximation for).
                ''')
        return 1

    return res
