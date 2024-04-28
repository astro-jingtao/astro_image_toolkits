import numpy as np


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
