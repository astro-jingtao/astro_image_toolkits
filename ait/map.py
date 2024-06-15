import pickle
from typing import Dict, Any, Callable

import h5py
import numpy as np
import pandas as pd
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


def layers_to_df(layers,
                 postfix_3d=None,
                 add_pos: bool = True,
                 global_feature: Dict = None,
                 drop_nan: bool = True,
                 nan_threshold: float = 0.8):

    if postfix_3d is None:

        def postfix_3d(layer_name, i):
            return f"{layer_name}_{i}"

    df = pd.DataFrame()

    for layer_name, layer in layers.items():
        if len(layer.shape) == 2:
            df[layer_name] = layer.flatten()
        else:
            for i in range(layer.shape[0]):
                new_layer_name = postfix_3d(layer_name, i)

                if new_layer_name in df.columns:
                    raise ValueError(
                        f"Duplicate key: {new_layer_name}, consider using a new postfix_3d to rename the keys"
                    )
                else:
                    df[new_layer_name] = layer[i].flatten()

    if drop_nan:
        # get nan fraction for each row
        nan_fraction = df.isna().mean(axis=1)
        to_drop = nan_fraction > nan_threshold
        df = df[~to_drop].reindex()

    if add_pos:
        ii, jj = np.indices(layers[list(layers.keys())[0]].shape)

        ii_name = 'pos_ii'
        jj_name = 'pos_jj'

        _k = 0
        while (ii_name in df.columns) or (jj_name in df.columns):
            ii_name = f'pos_ii_{_k}'
            jj_name = f'pos_jj_{_k}'
            _k += 1

        if drop_nan:
            df[ii_name] = ii.flatten()[~to_drop]
            df[jj_name] = jj.flatten()[~to_drop]
        else:
            df[ii_name] = ii.flatten()
            df[jj_name] = jj.flatten()

    if global_feature is not None:
        for key, value in global_feature.items():
            if key in df.columns:
                raise ValueError(
                    f"Duplicate key: {key}, consider using a new key to add the globals"
                )
            else:
                df[key] = value

    return df
