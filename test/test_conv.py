import numpy as np
from ait.conv import convolve, _convolve, conv_cube_1d, conv_cube_1d_p, conv_cube_2d, conv_cube_2d_p
from astropy.convolution import Gaussian2DKernel, Gaussian1DKernel


class TestConvolve:
    def test_basic(self):
        arr = np.random.randn(100, 100)
        kernel = Gaussian2DKernel(5)
        conv = convolve(arr, kernel)
        assert conv.shape == arr.shape
        assert np.allclose(conv, _convolve(arr, kernel))


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
