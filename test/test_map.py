import os

import numpy as np
from astropy.wcs import WCS

from ait.map import Map

TMP_PATH = "./tmp_map.h5"


def generate_map_instance(with_err=True, with_snr=False):

    layers = {'image': np.random.rand(10, 10)}

    snr = np.random.uniform(1, 5, size=(10, 10))
    err = layers['image'] / snr

    if with_err:
        layers['image_err'] = err

    if with_snr:
        layers['image_snr'] = snr

    pixel_size = 0.5
    metadata = {'aaa': 123, 'bbb': 456}


    # https://docs.astropy.org/en/stable/wcs/example_create_imaging.html
    wcs = WCS(naxis=2)
    wcs.wcs.crpix = [-234.75, 8.3393]
    wcs.wcs.cdelt = np.array([-0.066667, 0.066667])
    wcs.wcs.crval = [0, -90]
    wcs.wcs.ctype = ["RA---AIR", "DEC--AIR"]
    wcs.wcs.set_pv([(2, 1, 45.0)])


    redshift = 0.1

    return Map(layers=layers,
               pixel_size=pixel_size,
               metadata=metadata,
               wcs=wcs,
               redshift=redshift)


class TestSaveLoad:

    def test_save_load(self):
        _map = generate_map_instance()
        _map.save(TMP_PATH)
        _map_loaded = Map.load(TMP_PATH)

        assert _map.metadata == _map_loaded.metadata
        assert _map.pixel_size == _map_loaded.pixel_size
        assert _map.redshift == _map_loaded.redshift
        assert np.allclose(_map.layers['image'], _map_loaded.layers['image'])
        assert np.allclose(_map.layers['image_err'],
                           _map_loaded.layers['image_err'])
        assert _map.wcs.to_header_string() == _map_loaded.wcs.to_header_string(
        )

        os.remove(TMP_PATH)
