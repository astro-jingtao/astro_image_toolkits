import numpy as np
from utils_test.data_generator import generate_wcs

from ait.resample import resample_array, zoom_wcs, resample_wcs


def gaussian2d(x, y, x0, y0, sx, sy, A):
    """
    2D Gaussian function.

    Parameters
    ----------
    x : float or array_like
        x-coordinate.
    y : float or array_like
        y-coordinate.
    x0 : float
        x-coordinate of the center.
    y0 : float
        y-coordinate of the center.
    sx : float
        Standard deviation in x-direction.
    sy : float
        Standard deviation in y-direction.
    A : float
        Amplitude.

    Returns
    -------
    float or array_like
        Value of the 2D Gaussian function.
    """
    return A * np.exp(-((x - x0)**2 / (2 * sx**2) + (y - y0)**2 / (2 * sy**2)))


def get_2d_gaussian():
    x = np.linspace(-3, 3, 100)
    y = np.linspace(-3, 3, 100)
    X, Y = np.meshgrid(x, y)
    return gaussian2d(X, Y, 0, 0, 1, 1, 1)


class TestResampleArray:

    def test_different_inupt_consistency(self):
        Z = get_2d_gaussian()
        Z_up = resample_array(Z, zoom_factor=0.5)
        Z_up_pixel_scale = resample_array(Z,
                                          from_pixel_scale=1,
                                          to_pixel_scale=2)
        assert np.allclose(Z_up, Z_up_pixel_scale)

    def test_up_down_consistency(self):
        Z = get_2d_gaussian()
        Z_up = resample_array(Z, zoom_factor=0.5)
        Z_up_down = resample_array(Z_up, zoom_factor=2)
        # The tolerance is set to 0.1 because the resampling is not exact.
        assert np.allclose(Z_up_down, Z, rtol=0.1)

    def test_scaler(self):
        Z = get_2d_gaussian()

        Z_manual = resample_array(Z, zoom_factor=0.5) * 4
        Z_scaler_4 = resample_array(Z, zoom_factor=0.5, scaler=4)
        Z_scaler_area = resample_array(Z, zoom_factor=0.5, scaler='area')
        assert np.allclose(Z_manual, Z_scaler_4)
        assert np.allclose(Z_manual, Z_scaler_area)


class TestZoomWCS:

    def test_cdelt_case(self):

        wcs = generate_wcs()

        wcs_up = zoom_wcs(wcs, zoom_factor=2)
        assert np.allclose(wcs_up.wcs.cdelt, wcs.wcs.cdelt / 2)
        assert np.allclose(wcs_up.proj_plane_pixel_area(),
                           wcs.proj_plane_pixel_area() / 4)

        wcs_down = zoom_wcs(wcs, zoom_factor=0.5)
        assert np.allclose(wcs_down.wcs.cdelt, wcs.wcs.cdelt * 2)
        assert np.allclose(wcs_down.proj_plane_pixel_area(),
                           wcs.proj_plane_pixel_area() * 4)

    def test_cd_case(self):

        wcs = generate_wcs()
        wcs.wcs.cd = wcs.wcs.cdelt * np.array([[1, 0], [0, 1]])

        wcs_up = zoom_wcs(wcs, zoom_factor=2)
        assert np.allclose(wcs_up.wcs.cd, wcs.wcs.cd / 2)
        assert np.allclose(wcs_up.proj_plane_pixel_area(),
                           wcs.proj_plane_pixel_area() / 4)

        wcs_down = zoom_wcs(wcs, zoom_factor=0.5)
        assert np.allclose(wcs_down.wcs.cd, wcs.wcs.cd * 2)
        assert np.allclose(wcs_down.proj_plane_pixel_area(),
                           wcs.proj_plane_pixel_area() * 4)


class TestResampleWCS:

    def test_consistency_with_resample_array(self):

        wcs = generate_wcs()
        Z = get_2d_gaussian()

        self._test_with_given_zoom_factor(2, Z, wcs)
        self._test_with_given_zoom_factor(0.5, Z, wcs)

    def _test_with_given_zoom_factor(self, zoom_factor, Z, wcs):
        Z_up = resample_array(Z, zoom_factor=zoom_factor)
        Z_up_wcs = resample_wcs(Z, wcs=wcs, zoom_factor=zoom_factor)
        assert np.allclose(Z_up_wcs, Z_up, atol=1e-3)

    def test_scaler(self):
        wcs = generate_wcs()
        Z = get_2d_gaussian()

        Z_up = resample_wcs(Z, wcs=wcs, zoom_factor=2)
        Z_up_scaler = resample_wcs(Z, wcs=wcs, zoom_factor=2, scaler=1 / 4)
        Z_up_area = resample_wcs(Z, wcs=wcs, zoom_factor=2, scaler='area')
        assert np.allclose(Z_up / 4, Z_up_scaler)
        assert np.allclose(Z_up / 4, Z_up_area)
