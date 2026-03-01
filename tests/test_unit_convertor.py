import astropy.units as u
import numpy as np
from astropy.cosmology import FlatLambdaCDM

from ait.unit_convertor import ang2phy_size, phy2ang_size


class TestPhyAngConversion:

    def test_phy2ang(self):

        D = FlatLambdaCDM(H0=70, Om0=0.3).angular_diameter_distance(0.1)
        kpc_per_arcmin = FlatLambdaCDM(H0=70,
                                       Om0=0.3).kpc_proper_per_arcmin(0.1)

        # distance
        theta = phy2ang_size(2 * u.kpc,
                             angular_distance=D,
                             return_with_unit=False)
        assert np.allclose(theta, (2 * u.kpc / kpc_per_arcmin).value * 60)

        # redshift
        theta = phy2ang_size(2 * u.kpc, redshift=0.1, return_with_unit=False)
        assert np.allclose(theta, (2 * u.kpc / kpc_per_arcmin).value * 60)

        # vector
        theta = phy2ang_size(np.array([2, 3]) * u.kpc,
                             angular_distance=D,
                             return_with_unit=False)
        assert np.allclose(
            theta, (np.array([2, 3]) * u.kpc / kpc_per_arcmin).value * 60)

        theta = phy2ang_size(np.array([2, 3]),
                             angular_distance=D,
                             return_with_unit=False)
        assert np.allclose(
            theta, (np.array([2, 3]) * u.kpc / kpc_per_arcmin).value * 60)

    def test_ang2phy(self):

        D = FlatLambdaCDM(H0=70, Om0=0.3).angular_diameter_distance(0.1)
        kpc_per_arcmin = FlatLambdaCDM(H0=70,
                                       Om0=0.3).kpc_proper_per_arcmin(0.1)

        # distance
        size = 2
        distance = ang2phy_size(size,
                                angular_distance=D,
                                return_with_unit=True)
        assert np.allclose(distance, 2 * u.arcsec * kpc_per_arcmin)

        # redshift
        distance = ang2phy_size(size, redshift=0.1, return_with_unit=True)
        assert np.allclose(distance, 2 * u.arcsec * kpc_per_arcmin)

        # vector
        size = np.array([2, 3])
        distance = ang2phy_size(size * u.arcsec,
                                angular_distance=D,
                                return_with_unit=True)
        assert np.allclose(distance,
                           np.array([2, 3]) * u.arcsec * kpc_per_arcmin)

        distance = ang2phy_size(size,
                                angular_distance=D,
                                return_with_unit=True)
        assert np.allclose(distance,
                           np.array([2, 3]) * u.arcsec * kpc_per_arcmin)
