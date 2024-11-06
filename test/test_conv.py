import numpy as np
from astropy.convolution import Gaussian1DKernel, Gaussian2DKernel
import pytest
from ait.conv import (_convolve, conv_cube_1d, conv_cube_1d_p, conv_cube_2d,
                      conv_cube_2d_p, convolve, match_psf_gaussian,
                      get_error_correction)

avoid_cpu_heavy = True

class TestConvolve:

    def test_basic(self):
        arr = np.random.randn(100, 100)
        kernel = Gaussian2DKernel(5)
        conv = convolve(arr, kernel)
        assert conv.shape == arr.shape
        assert np.allclose(conv, _convolve(arr, kernel))


class TestConvolveFFTNaN:

    def test_basic(self):
        arr = np.random.randn(100, 100)
        kernel = Gaussian2DKernel(1)
        arr[40:60, 40:60] = np.nan

        arr_conv_direct = convolve(arr,
                                   kernel,
                                   boundary="fill",
                                   fill_value=np.nan,
                                   nan_treatment="fill")

        arr_conv_nanfft = convolve(arr, kernel, method="fft_nan")

        assert np.allclose(arr_conv_direct, arr_conv_nanfft, equal_nan=True)

        arr_conv_err_direct = convolve(np.abs(arr),
                                       kernel,
                                       boundary="fill",
                                       fill_value=np.nan,
                                       nan_treatment="fill",
                                       is_err=True)

        arr_conv_err_nanfft = convolve(np.abs(arr),
                                       kernel,
                                       method="fft_nan",
                                       is_err=True)

        assert np.allclose(arr_conv_err_direct,
                           arr_conv_err_nanfft,
                           equal_nan=True)


@pytest.mark.skipif(avoid_cpu_heavy, reason="avoid cpu-heavy tests")
class TestConvCube1D:

    def test_basic(self):
        cube = np.random.randn(100, 30, 30)
        kernel = Gaussian1DKernel(5)
        conv = conv_cube_1d(cube, kernel)
        assert conv.shape == cube.shape

        for i in range(cube.shape[1]):
            for j in range(cube.shape[2]):
                assert np.allclose(conv[:, i, j],
                                   _convolve(cube[:, i, j], kernel))

    def test_parallel(self):

        cube = np.random.randn(100, 30, 30)
        kernel = Gaussian1DKernel(5)
        conv = conv_cube_1d(cube, kernel)
        conv_p = conv_cube_1d_p(cube, kernel, n_jobs=1)
        assert conv.shape == conv_p.shape
        assert np.allclose(conv, conv_p)

        cube = np.random.randn(30, 30, 100)
        kernel = Gaussian1DKernel(5)
        conv = conv_cube_1d(cube, kernel, vel_axis=2)
        conv_p = conv_cube_1d_p(cube, kernel, n_jobs=1, vel_axis=2)
        assert conv.shape == conv_p.shape
        assert np.allclose(conv, conv_p)

    def test_data_type(self):

        cube = np.random.randn(100, 30, 30).astype('float32')
        kernel = Gaussian1DKernel(5)
        conv = conv_cube_1d(cube, kernel)
        conv_p = conv_cube_1d_p(cube, kernel, n_jobs=1)
        assert conv.dtype == 'float32'
        assert conv_p.dtype == 'float32'

        cube = np.random.randn(100, 30, 30).astype('float64')
        kernel = Gaussian1DKernel(5)
        conv = conv_cube_1d(cube, kernel)
        conv_p = conv_cube_1d_p(cube, kernel, n_jobs=1)
        assert conv.dtype == 'float64'
        assert conv_p.dtype == 'float64'


@pytest.mark.skipif(avoid_cpu_heavy, reason="avoid cpu-heavy tests")
class TestConvCube2D:

    def test_basic(self):
        cube = np.random.randn(100, 30, 30)
        kernel = Gaussian2DKernel(5)
        conv = conv_cube_2d(cube, kernel)
        assert conv.shape == cube.shape

        for i in range(cube.shape[0]):
            assert np.allclose(conv[i, :, :], _convolve(cube[i, :, :], kernel))

    def test_parallel(self):

        cube = np.random.randn(100, 30, 30)
        kernel = Gaussian2DKernel(5)
        conv = conv_cube_2d(cube, kernel)
        conv_p = conv_cube_2d_p(cube, kernel, n_jobs=1)
        assert conv.shape == conv_p.shape
        assert np.allclose(conv, conv_p)

        cube = np.random.randn(30, 30, 100)
        kernel = Gaussian2DKernel(5)
        conv = conv_cube_2d(cube, kernel, vel_axis=2)
        conv_p = conv_cube_2d_p(cube, kernel, n_jobs=1, vel_axis=2)
        assert conv.shape == conv_p.shape
        assert np.allclose(conv, conv_p)

    def test_data_type(self):

        cube = np.random.randn(20, 30, 30).astype('float32')
        kernel = Gaussian2DKernel(5)
        conv = conv_cube_2d(cube, kernel)
        conv_p = conv_cube_2d_p(cube, kernel, n_jobs=1)
        assert conv.dtype == 'float32'
        assert conv_p.dtype == 'float32'

        cube = np.random.randn(20, 30, 30).astype('float64')
        kernel = Gaussian2DKernel(5)
        conv = conv_cube_2d(cube, kernel)
        conv_p = conv_cube_2d_p(cube, kernel, n_jobs=1)
        assert conv.dtype == 'float64'
        assert conv_p.dtype == 'float64'


class TestMatchPSF:

    def test_consistency(self):
        arr = np.random.randn(100, 100)

        kernel = Gaussian2DKernel(3)
        arr_conv = convolve(arr, kernel)
        arr_conv_match = match_psf_gaussian(arr, 0, 3, width_type='sigma')
        assert np.allclose(arr_conv, arr_conv_match)

        kernel2 = Gaussian2DKernel(4)
        arr_conv_conv = convolve(arr_conv, kernel2)
        arr_conv_conv_match = match_psf_gaussian(arr_conv,
                                                 3,
                                                 5,
                                                 width_type='sigma')
        assert np.allclose(arr_conv_conv, arr_conv_conv_match)

    def test_error_correction(self):
        arr = np.random.randn(100, 100)

        arr_not_corrected = match_psf_gaussian(arr,
                                               2,
                                               4,
                                               width_type='fwhm',
                                               is_err=True,
                                               to_correct_error=False)
        arr_corrected = match_psf_gaussian(arr,
                                           2,
                                           4,
                                           width_type='fwhm',
                                           is_err=True,
                                           to_correct_error=True)

        assert np.allclose(arr_not_corrected * get_error_correction(2, 4),
                           arr_corrected)
