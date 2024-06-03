import pickle

import h5py
import numpy as np
from astropy.io import fits
from astropy.wcs import WCS


class Map:

    def __init__(self,
                 *,
                 layers,
                 pixel_size,
                 info=None,
                 wcs=None,
                 redshift=None) -> None:

        self.layers = layers
        self.pixel_size = pixel_size  # in arcsec
        self.info = info
        self.wcs = wcs
        self.redshift = redshift

    @classmethod
    def load(cls, filename):
        with h5py.File(filename, 'r') as file:
            layers = {
                key: file[key][()]
                for key in file.keys() if not key == 'metadata'
            }
            metadata_bytes = bytes(file['metadata'][()])
            attributes = pickle.loads(metadata_bytes)

        # Deserialize WCS
        if 'wcs' in attributes and attributes['wcs']:
            wcs_header = fits.Header.fromstring(attributes['wcs'])
            attributes['wcs'] = WCS(wcs_header)

        return cls(layers=layers, **attributes)

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

    def save(self, filename):
        with h5py.File(filename, 'w') as file:
            for key, value in self.layers.items():
                file.create_dataset(key, data=value)

            # Handle WCS specifically if it's not None
            wcs_string = self.wcs.to_header_string() if self.wcs else None

            # Serialize other attributes using pickle, including WCS as string
            metadata_bytes = pickle.dumps({
                'pixel_size': self.pixel_size,
                'info': self.info,
                'wcs': wcs_string,
                'redshift': self.redshift
            })
            file.create_dataset('metadata', data=np.void(metadata_bytes))
