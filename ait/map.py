import pickle

import h5py
import numpy as np
from astropy.io import fits
from astropy.wcs import WCS


class Map:

    DATA_VERSION = 1

    def __init__(self,
                 *,
                 layers,
                 pixel_size,
                 wcs=None,
                 redshift=None,
                 metadata=None) -> None:

        self.layers = layers
        self.pixel_size = pixel_size  # in arcsec
        self.wcs = wcs
        self.redshift = redshift
        self.metadata = metadata

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
                for key, value in file.attrs.items() if not key == 'data_version'
            }
            metadata_bytes = bytes(file['metadata'][()])
            attributes['metadata'] = pickle.loads(metadata_bytes)

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

            # set attribute values
            file.attrs['pixel_size'] = self.pixel_size
            file.attrs['wcs'] = wcs_string
            file.attrs['redshift'] = self.redshift
            file.attrs['data_version'] = self.DATA_VERSION

            # Serialize other attributes using pickle, including WCS as string

            metadata_bytes = pickle.dumps(self.metadata)
            file.create_dataset('metadata', data=np.void(metadata_bytes))
