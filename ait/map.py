import pickle
from typing import Dict, Any

import h5py
import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
import astropy.units as u
from astropy.cosmology import FlatLambdaCDM


class Map:

    # only updated when the data structure do not compatible with the previous version
    DATA_VERSION = 2

    ATTRS_TO_SAVE = ['pixel_scale', 'wcs', 'redshift', 'PSF_FWHM', 'shape']

    def __init__(self,
                 *,
                 layers,
                 pixel_scale,
                 wcs=None,
                 redshift=None,
                 PSF_FWHM=None,
                 shape=None,
                 metadata=None) -> None:

        self.layers: Dict[
            str, np.
            ndarray] = layers  # if 3D, the first dimension is the additional dimension
        self.pixel_scale: float = pixel_scale  # in arcsec
        self.wcs: WCS = wcs
        self.redshift: float = redshift
        self.PSF_FWHM: float = PSF_FWHM  # in arcsec
        self.shape: tuple = tuple(shape) if shape is not None else None

        if metadata is None:
            metadata = {}
        self.metadata: Dict[str, Any] = metadata

        self.check_shape()

    @classmethod
    def load(cls, filename):
        with h5py.File(filename, 'r') as file:

            if file.attrs['data_version'] != cls.DATA_VERSION:
                print(
                    f"Data version mismatch: {file.attrs['data_version']} != {cls.DATA_VERSION}"
                )

            # sourcery skip: de-morgan
            layers = {
                key: value[()]
                for key, value in file.items() if not key == 'metadata'
            }

            attributes = {
                key: value
                for key, value in file.attrs.items()
                if not key == 'data_version'
            }
            metadata_bytes = bytes(file['metadata'][()])
            attributes['metadata'] = pickle.loads(metadata_bytes)

        # Deserialize WCS
        if 'wcs' in attributes and attributes['wcs']:
            wcs_header = fits.Header.fromstring(attributes['wcs'])
            attributes['wcs'] = WCS(wcs_header)

        # shape as tuple
        if 'shape' in attributes:
            attributes['shape'] = tuple(attributes['shape'])

        return cls(layers=layers, **attributes)

    def check_shape(self, set_shape=True):

        def get_shape(layer):
            return layer.shape if len(layer.shape) == 2 else layer.shape[1:]

        shape = self.shape
        for key, value in self.layers.items():
            if shape is None:
                shape = get_shape(value)
            else:
                if shape != get_shape(value):
                    raise ValueError(
                        f"Shape mismatch: {shape} != {value.shape} for {key}")

        if (set_shape) and (self.shape is None):
            self.shape = shape

    def save(self, filename):
        with h5py.File(filename, 'w') as file:
            for key, value in self.layers.items():
                file.create_dataset(key, data=value)

            # set attribute values

            for arrts in self.ATTRS_TO_SAVE:
                if this_arrts := getattr(self, arrts):
                    if arrts == 'wcs':
                        this_arrts = self.wcs.to_header_string()
                    file.attrs[arrts] = this_arrts

            file.attrs['data_version'] = self.DATA_VERSION

            # Serialize other attributes using pickle, including WCS as string

            metadata_bytes = pickle.dumps(self.metadata)
            file.create_dataset('metadata', data=np.void(metadata_bytes))

    def get_snr(self, name):
        name_snr = f"{name}_snr"
        if name_snr in self.layers:
            return self.layers[name_snr]
        name_err = f"{name}_err"
        if name_err in self.layers:
            return np.abs(self.layers[name] / self.layers[name_err])
        return None

    def get_err(self, name):
        name_err = f"{name}_err"
        if name_err in self.layers:
            return self.layers[name_err]
        name_snr = f"{name}_snr"
        if name_snr in self.layers:
            return np.abs(self.layers[name] / self.layers[name_snr])
        return None

    def get_pixel_area(self,
                       D=None,
                       unit=u.kpc**2,
                       cosmo=FlatLambdaCDM(H0=70, Om0=0.3)):
        '''
        Return the pixel area in given unit
        '''

        if D is None:
            D = cosmo.angular_diameter_distance(self.redshift)

        return ((D * np.deg2rad(self.pixel_scale / 3600))**2).to(unit).value

    def flux_to_surface_brightness(self,
                                   name,
                                   in_unit=u.erg / u.s / u.cm**2,
                                   out_unit=u.erg / u.s / u.kpc**2,
                                   kind='value',
                                   cosmological=False):
        '''
        Assume the input map is flux like, and the unit of the input map is per pixel. 
        e.g. erg/s/cm^2/pixel
        
        The output map is surface brightness like. e.g. erg/s/kpc^2
        '''

        scaler = (4 * np.pi) / (np.deg2rad(self.pixel_scale / 3600)**2)

        if cosmological:
            scaler *= (1 + self.redshift)**4

        if name is None:
            data = 1
        elif kind == 'snr':
            return self.get_snr(name)
        elif kind == 'value':
            data = self.layers[name]
        elif kind == 'err':
            data = self.get_err(name)

        return (data * in_unit * scaler).to(out_unit).value
