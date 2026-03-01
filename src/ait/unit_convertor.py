import warnings

import astropy.units as u
import numpy as np
from astropy.cosmology import FlatLambdaCDM
from astropy.units import Unit


def luminosity_to_flux(luminosity,
                       luminosity_distance=None,
                       redshift=None,
                       luminosity_distance_unit: Unit = u.Mpc,
                       in_unit: Unit = u.erg / u.s,
                       out_unit: Unit = u.erg / u.s / u.cm**2,
                       return_with_unit=False,
                       cosmo=FlatLambdaCDM(H0=70, Om0=0.3)):
    """
    Convert luminosity to observed flux.
    """

    if isinstance(luminosity, u.Quantity):
        L = luminosity
    else:
        L = luminosity * in_unit

    luminosity_distance = _get_luminosity_distance(luminosity_distance,
                                                   redshift, cosmo)

    if isinstance(luminosity_distance, u.Quantity):
        D = luminosity_distance
    else:
        D = luminosity_distance * luminosity_distance_unit

    flux = (L / (4 * np.pi * D**2)).to(out_unit)

    if return_with_unit:
        return flux
    else:
        return flux.value


def flux_to_luminosity(flux,
                       luminosity_distance=None,
                       redshift=None,
                       luminosity_distance_unit: Unit = u.Mpc,
                       in_unit: Unit = u.erg / u.s / u.cm**2,
                       out_unit: Unit = u.erg / u.s,
                       return_with_unit=False,
                       cosmo=FlatLambdaCDM(H0=70, Om0=0.3)):
    """
    Convert observed flux to luminosity.
    """

    if isinstance(flux, u.Quantity):
        F = flux
    else:
        F = flux * in_unit

    luminosity_distance = _get_luminosity_distance(luminosity_distance,
                                                   redshift, cosmo)

    if isinstance(luminosity_distance, u.Quantity):
        D = luminosity_distance
    else:
        D = luminosity_distance * luminosity_distance_unit

    luminosity = (4 * np.pi * D**2 * F).to(out_unit)

    if return_with_unit:
        return luminosity
    else:
        return luminosity.value


def flux_to_surface_brightness(flux,
                               angular_size=None,
                               angular_size_unit: Unit = u.arcsec,
                               in_unit: Unit = u.erg / u.s / u.cm**2,
                               out_unit: Unit = u.erg / u.s / u.kpc**2,
                               cosmological=False,
                               redshift=None,
                               return_with_unit=False):
    """
    Convert flux per angular element to surface brightness.

    from F = [U] / [S_rec]
    to SB = [U] / [S_phy]

    SB = F * (4 pi) * Dl^2 / (Da^2 * (Omega / rad^2))
       = F * (1 + z)^4 * (4 pi / (Omega / rad^2))
    """

    if angular_size is None:
        raise ValueError("angular_size must be provided")

    # convert angular_size to radian
    if isinstance(angular_size, u.Quantity):
        theta = angular_size.to(u.radian).value
    else:
        theta = angular_size * (angular_size_unit.to(u.radian))

    if isinstance(flux, u.Quantity):
        F = flux
    else:
        F = flux * in_unit

    omega = theta**2
    scaler = (4 * np.pi) / omega

    if cosmological:
        if redshift is None:
            raise ValueError(
                "redshift must be provided when cosmological=True")
        scaler = (1 + redshift)**4

    SB = (F * scaler).to(out_unit)

    if return_with_unit:
        return SB
    else:
        return SB.value


def luminosity_to_surface_brightness(luminosity,
                                     physical_size=None,
                                     angular_size=None,
                                     angular_distance=None,
                                     redshift=None,
                                     physical_size_unit: Unit = u.kpc,
                                     angular_size_unit: Unit = u.arcsec,
                                     angular_distance_unit: Unit = u.Mpc,
                                     in_unit: Unit = u.erg / u.s,
                                     out_unit: Unit = u.erg / u.s / u.kpc**2,
                                     return_with_unit=False,
                                     cosmo=FlatLambdaCDM(H0=70, Om0=0.3)):
    """
    Convert luminosity to surface brightness using angular size.

    from L = [U]
    to SB = [U] / [S_phy]

    SB = L / (Da^2 * (Omega / rad^2))

    """

    if isinstance(luminosity, u.Quantity):
        L = luminosity
    else:
        L = luminosity * in_unit

    if physical_size is None:
        if angular_size is None:
            raise ValueError("Either physical_size or angular_size must be provided")
        d = ang2phy_size(angular_size,
                         angular_distance,
                         angular_size_unit=angular_size_unit,
                         angular_distance_unit=angular_distance_unit,
                         to_unit=u.kpc,
                         return_with_unit=True,
                         redshift=redshift,
                         cosmo=cosmo)
    else:
        if angular_size is not None:
            warnings.warn("Both physical_size and angular_size are provided, angular_size will be ignored")
        if isinstance(physical_size, u.Quantity):
            d = physical_size
        else:
            d = physical_size * physical_size_unit

    area = d**2
    SB = (L / area).to(out_unit)

    if return_with_unit:
        return SB
    else:
        return SB.value


def luminosity_to_intensity(luminosity,
                            physical_size,
                            redshift=None,
                            physical_size_unit: Unit = u.kpc,
                            in_unit: Unit = u.erg / u.s,
                            out_unit: Unit = u.erg / u.s / u.cm**2 /
                            u.arcsec**2,
                            cosmological=False,
                            return_with_unit=False):
    """
    Convert luminosity to intensity.

    from L = [U]
    to I = [U] / [S_rec] / [Omega]

    I = L / (4 pi * Dl^2) / ((d_phy / Da) * rad)^2
      = L * (Da / Dl)^2 / (4 pi * rad^2 * d_phy^2)
      = L * (1 + z)^-4 / (4 pi * rad^2 * d_phy^2)

    """
    if isinstance(luminosity, u.Quantity):
        L = luminosity
    else:
        L = luminosity * in_unit

    if isinstance(physical_size, u.Quantity):
        d_phy = physical_size
    else:
        d_phy = physical_size * physical_size_unit

    scaler = 1 / (4 * np.pi)

    if cosmological:
        if redshift is None:
            raise ValueError(
                "redshift must be provided when cosmological=True")
        scaler /= (1 + redshift)**4

    intensity = (L * scaler / (d_phy**2 * u.radian**2)).to(out_unit)

    if return_with_unit:
        return intensity
    else:
        return intensity.value

def intensity_to_surface_brightness():
    ...

def surface_brightness_to_intensity():
    ...


def ang2phy_size(angular_size,
                 angular_distance=None,
                 angular_size_unit: Unit = u.arcsec,
                 angular_distance_unit: Unit = u.Mpc,
                 to_unit: Unit = u.kpc,
                 return_with_unit=False,
                 redshift=None,
                 cosmo=FlatLambdaCDM(H0=70, Om0=0.3)):
    """
    Convert angular size to physical size

    Parameters
    ----------
    angular_size : float or astropy.units.Quantity
        Angular size in arcsec or astropy.units.Quantity
    angular_distance : float or astropy.units.Quantity, optional
        Angular distance in Mpc or astropy.units.Quantity, by default None.
        If None, redshift must be provided
    angular_size_unit: astropy.units.Unit, optional
        Angular size unit, by default u.arcsec
    angular_distance_unit: astropy.units.Unit, optional
        Angular distance unit, by default u.Mpc
    to_unit : astropy.units.Unit, optional
        Physical unit to convert to, by default u.kpc
    with_unit : bool, optional
        Whether to return the result with astropy.units.Quantity, by default False
    redshift : float, optional
        Redshift of the source, by default None
        If None, angular_distance must be provided
    cosmo : astropy.cosmology.FLRW, optional
        Cosmology to use, by default FlatLambdaCDM(H0=70, Om0=0.3)

    Returns
    -------
    float or astropy.units.Quantity
        Physical size in the given unit
    """

    if isinstance(angular_size, u.Quantity):
        theta = angular_size.to(u.radian).value
    else:
        # (angular_size_unit.to(u.radian)) give a float value
        theta = angular_size * (angular_size_unit.to(u.radian))

    angular_distance = _get_angular_distance(angular_distance, redshift, cosmo)

    if isinstance(angular_distance, u.Quantity):
        D = angular_distance
    else:
        D = angular_distance * angular_distance_unit

    d = (theta * D).to(to_unit)

    if return_with_unit:
        return d
    else:
        return d.value


def phy2ang_size(physical_size,
                 angular_distance=None,
                 physical_size_unit: Unit = u.kpc,
                 angular_distance_unit: Unit = u.Mpc,
                 to_unit: Unit = u.arcsec,
                 return_with_unit=False,
                 redshift=None,
                 cosmo=FlatLambdaCDM(H0=70, Om0=0.3)):

    if isinstance(physical_size, u.Quantity):
        d = physical_size
    else:
        d = physical_size * physical_size_unit

    angular_distance = _get_angular_distance(angular_distance, redshift, cosmo)

    if isinstance(angular_distance, u.Quantity):
        D = angular_distance
    else:
        D = angular_distance * angular_distance_unit

    theta = ((d / D).decompose() * u.radian).to(to_unit)

    if return_with_unit:
        return theta
    else:
        return theta.value


def _get_angular_distance(angular_distance, redshift, cosmo):
    if angular_distance is None:
        if redshift is None:
            raise ValueError("Both angular_distance and redshift are None")
        angular_distance = cosmo.angular_diameter_distance(redshift)
    else:
        if redshift is not None:
            warnings.warn(
                "Both angular_distance and redshift are provided, redshift will be ignored"
            )

    return angular_distance


def _get_luminosity_distance(luminosity_distance, redshift, cosmo):
    if luminosity_distance is None:
        if redshift is None:
            raise ValueError("Both distance and redshift are None")
        luminosity_distance = cosmo.luminosity_distance(redshift)
    else:
        if redshift is not None:
            warnings.warn(
                "Both distance and redshift are provided, redshift will be ignored"
            )

    return luminosity_distance
