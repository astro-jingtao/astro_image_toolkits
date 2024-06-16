import os

import numpy as np
from astropy.wcs import WCS
import astropy.units as u
from astropy.cosmology import Planck15
import pytest

from ait.map import Map
from utils_test.gen_map import generate_map_instance

TMP_PATH = "./tmp_map.h5"


class TestSaveLoad:

    def test_save_load(self):
        _map = generate_map_instance()

        # sourcery skip: no-conditionals-in-tests
        if not os.path.exists(f'./test_data/map_{_map.DATA_VERSION}.h5'):
            _map.save(f'./test_data/map_{_map.DATA_VERSION}.h5')

        _map.save(TMP_PATH)
        _map_loaded = Map.load(TMP_PATH)

        assert _map.metadata == _map_loaded.metadata
        assert _map.pixel_scale == _map_loaded.pixel_scale
        assert _map.redshift == _map_loaded.redshift
        assert np.allclose(_map.layers['image'], _map_loaded.layers['image'])
        assert np.allclose(_map.layers['image_err'],
                           _map_loaded.layers['image_err'])
        assert _map.wcs.to_header_string() == _map_loaded.wcs.to_header_string(
        )

        os.remove(TMP_PATH)

        # with None

        _map.redshift = None

        _map.save(TMP_PATH)
        _map_loaded = Map.load(TMP_PATH)

        assert _map_loaded.redshift is None

        os.remove(TMP_PATH)


class TestUnitConversion:

    def test_pixel_to_area(self):
        _map = generate_map_instance()

        image_area = _map.per_pixel_to_per_area('image',
                                                in_unit=u.erg / u.s / u.cm**2 *
                                                1e-17)

        image_err_area = _map.per_pixel_to_per_area('image',
                                                    in_unit=u.erg / u.s /
                                                    u.cm**2 * 1e-17,
                                                    kind='err')

        # from emilkit
        image_true = _map.layers['image'] * 4 * np.pi / (np.deg2rad(
            _map.pixel_scale / 3600)**2) * (3.0856776e21)**2 * 1e-17

        image_err_true = _map.get_err('image') * 4 * np.pi / (np.deg2rad(
            _map.pixel_scale / 3600)**2) * (3.0856776e21)**2 * 1e-17

        assert np.allclose(image_area, image_true)
        assert np.allclose(image_err_area, image_err_true)

        # cosmological

        image_area_cosmo = _map.per_pixel_to_per_area('image',
                                                      in_unit=u.erg / u.s /
                                                      u.cm**2 * 1e-17,
                                                      cosmological=True)

        scaler = ((Planck15.luminosity_distance(_map.redshift) /
                   Planck15.angular_diameter_distance(
                       _map.redshift))**2).decompose().value

        assert np.allclose(image_area_cosmo, image_true * scaler)

        # 3d layer

        _map.layers['cube'] = np.random.rand(20, 10, 10)

        cube_area = _map.per_pixel_to_per_area('cube',
                                               in_unit=u.erg / u.s / u.cm**2 *
                                               1e-17)

        cube_true = _map.layers['cube'] * 4 * np.pi / (np.deg2rad(
            _map.pixel_scale / 3600)**2) * (3.0856776e21)**2 * 1e-17

        assert np.allclose(cube_area, cube_true)


class TestSelfCheck:

    def test_check_shape(self):
        _map = generate_map_instance()
        _map.check_shape()

        _map.layers['image_err'] = np.random.rand(5, 5)
        # should raise error
        with pytest.raises(ValueError):
            _map.check_shape()

        _map.layers['image_err'] = np.random.rand(5, 10, 10)
        _map.check_shape()
