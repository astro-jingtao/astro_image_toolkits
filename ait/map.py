import os
import pickle
import warnings
from typing import Any, Dict, Union

import astropy.units as u
import h5py
import numpy as np
import pandas as pd
from astropy.cosmology import FlatLambdaCDM
from astropy.io import fits
from astropy.wcs import WCS

from .utils import get_unique_name, is_string_series


# TODO: distance
class Map:

    # only updated when the data structure do not compatible with the previous version
    DATA_VERSION = 3

    ALLOWED_OLD_DATA_VERSION = [2]

    # layers and units are saved as datasets in hdf5 file
    ATTRS_TO_SAVE = ['pixel_scale', 'wcs', 'redshift', 'PSF_FWHM', 'shape']

    def __init__(self,
                 *,
                 layers,
                 pixel_scale,
                 units=None,
                 wcs=None,
                 redshift=None,
                 PSF_FWHM=None,
                 shape=None,
                 metadata=None) -> None:

        self.layers: Dict[
            str, np.
            ndarray] = layers  # if 3D, the first dimension is the additional dimension
        self.pixel_scale: float = pixel_scale  # in arcsec
        self.units: Dict[str, str] = units if units is not None else {}
        self.wcs: WCS = wcs
        self.redshift: float = redshift
        self.PSF_FWHM: float = PSF_FWHM  # in arcsec
        self.shape: Union[tuple,
                          None] = tuple(shape) if shape is not None else None

        if metadata is None:
            metadata = {}
        self.metadata: Dict[str, Any] = metadata

        self.check_shape()

    @classmethod
    def load(cls, filename):
        return load_map_v3(filename)

    @classmethod
    def load_old(cls, filename, version):
        if version not in cls.ALLOWED_OLD_DATA_VERSION:
            raise ValueError(
                f"Unsupported data version: {version}, only support {cls.ALLOWED_OLD_DATA_VERSION}"
            )
        elif version == 2:
            return load_map_v2(filename)

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

    def save(self, filename, replace=False):

        if os.path.exists(filename):
            if replace:
                os.remove(filename)
            else:
                raise ValueError(f"File already exists: {filename}")

        with h5py.File(filename, 'w') as file:

            layers_group = file.create_group('layers')
            for key, value in self.layers.items():
                layers_group.create_dataset(key, data=value)

            units_group = file.create_group('units')
            for key, value in self.units.items():
                units_group.create_dataset(key, data=value)

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

        pixel_scale_phy = ang2phy_size(self.pixel_scale,
                                       angular_distance=D,
                                       return_with_unit=True,
                                       redshift=self.redshift,
                                       cosmo=cosmo)

        return (pixel_scale_phy**2).to(unit).value

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
        else:
            raise ValueError(f"Invalid kind: {kind}")

        return (data * in_unit * scaler).to(out_unit).value


def layers_to_df(layers,
                 postfix_3d=None,
                 add_pos: bool = True,
                 global_feature: Union[Dict, None] = None,
                 drop_nan: bool = True,
                 nan_threshold: float = 0.8):

    if postfix_3d is None:

        def postfix_3d(layer_name, i):
            return f"{layer_name}_{i}"

    # df = pd.DataFrame()

    # for layer_name, layer in layers.items():
    #     if len(layer.shape) == 2:
    #         df[layer_name] = layer.flatten()
    #     else:
    #         for i in range(layer.shape[0]):
    #             new_layer_name = postfix_3d(layer_name, i)

    #             if new_layer_name in df.columns:
    #                 raise ValueError(
    #                     f"Duplicate key: {new_layer_name}, consider using a new postfix_3d to rename the keys"
    #                 )
    #             else:
    #                 df[new_layer_name] = layer[i].flatten()

    # PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
    # Construct a list of Pandas Series
    series_list = []
    name_list = []

    for layer_name, layer in layers.items():
        if len(layer.shape) == 2:
            series = pd.Series(layer.flatten(), name=layer_name)
            series_list.append(series)
            name_list.append(layer_name)
        else:
            for i in range(layer.shape[0]):
                new_layer_name = postfix_3d(layer_name, i)

                if new_layer_name in name_list:
                    raise ValueError(
                        f"Duplicate key: {new_layer_name}, consider using a new postfix_3d to rename the keys"
                    )

                series = pd.Series(layer[i].flatten(), name=new_layer_name)
                series_list.append(series)
                name_list.append(layer_name)

    # Concatenate the list of series into a single DataFrame
    df = pd.concat(series_list, axis=1)

    if drop_nan:
        # get nan fraction for each row
        nan_fraction = df.isna().mean(axis=1)
        to_drop = nan_fraction > nan_threshold
        df = df[~to_drop].reset_index(drop=True)

    if add_pos:
        ii, jj = np.indices(layers[list(layers.keys())[0]].shape)

        ii_name, jj_name = get_unique_name(['pos_ii', 'pos_jj'], df.columns)

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


def df_to_layers(df: pd.DataFrame,
                 i_name='pos_ii',
                 j_name='pos_jj',
                 shape=None,
                 name_3d=None,
                 postfix_3d=None,
                 drop_unique=False,
                 set_masked=True,
                 masked_mapper=None):

    if postfix_3d is None:

        def postfix_3d(layer_name, i):
            return f"{layer_name}_{i}"

    if i_name not in df.columns:
        raise ValueError(f"Cannot find i_name: {i_name} in the DataFrame")

    if j_name not in df.columns:
        raise ValueError(f"Cannot find j_name: {j_name} in the DataFrame")

    if shape is None:
        shape = (df[i_name].max() + 1, df[j_name].max() + 1)
    else:
        if shape[0] < df[i_name].max() + 1 or shape[1] < df[j_name].max() + 1:
            raise ValueError(
                f"The given shape {shape} is smaller than the shape according to index in the DataFrame: ({df[i_name].max() + 1}, {df[j_name].max() + 1})"
            )

    if masked_mapper is None:

        masked_mapper = masked_mapper_default

    layers = {}

    mask_name = get_unique_name('mask', df.columns)

    layers[mask_name] = np.ones(shape, dtype=bool)
    layers[mask_name][df[i_name], df[j_name]] = False

    for name in df.columns:

        if name in [i_name, j_name]:
            continue

        if drop_unique and len(df[name].unique()) == 1:
            continue

        # if dtype is str

        # print(name, df[name].dtype)
        if is_string_series(df[name]):

            max_len = 64
            for _str in df[name].unique():
                max_len = max(max_len, len(_str))
            this_arr = np.full(shape, '', dtype=f'U{max_len}')
        else:
            this_arr = np.zeros(shape, dtype=df[name].dtype)
        # fill data into this_arr according to i_name and j_name
        this_arr[df[i_name], df[j_name]] = df[name]
        if set_masked:
            this_arr = masked_mapper(this_arr, layers[mask_name])
        layers[name] = this_arr

    if name_3d is not None:
        for name in name_3d:

            if name in layers:
                raise ValueError(f"The 3d cube name '{name}' already exists")

            this_name_lst = []

            _k = 0
            df_name = postfix_3d(name, _k)
            while df_name in layers:
                this_name_lst.append(df_name)
                _k += 1
                df_name = postfix_3d(name, _k)

            layers[name] = np.stack(
                [layers[this_name] for this_name in this_name_lst])

            for this_name in this_name_lst:
                del layers[this_name]

    return layers


def masked_mapper_default(arr, mask):

    # if str, set to 'masked'
    if np.issubdtype(arr.dtype, np.str_):
        arr = np.where(mask, 'masked', arr)
    # if float, set to nan
    elif np.issubdtype(arr.dtype, np.floating):
        arr = np.where(mask, np.nan, arr)
    # if int, set to -999
    elif np.issubdtype(arr.dtype, np.integer):
        arr = np.where(mask, -999, arr)
    # if bool, convert to int, and set to -1
    elif np.issubdtype(arr.dtype, np.bool_):
        arr = np.where(mask, -1, arr.astype(int))
    else:
        print(f"Unknown dtype: {arr.dtype}")

    return arr


def ang2phy_size(angular_size,
                 angular_distance=None,
                 to_unit=u.kpc,
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
        # assume in arcsec
        theta = np.deg2rad(angular_size / 3600)

    angular_distance = get_angular_distance(angular_distance, redshift, cosmo)

    if isinstance(angular_distance, u.Quantity):
        D = angular_distance
    else:
        D = angular_distance * u.Mpc  # assume in Mpc

    d = (theta * D).to(to_unit)

    if return_with_unit:
        return d
    else:
        return d.value


def phy2ang_size(physical_size,
                 angular_distance=None,
                 to_unit=u.arcsec,
                 return_with_unit=False,
                 redshift=None,
                 cosmo=FlatLambdaCDM(H0=70, Om0=0.3)):

    if isinstance(physical_size, u.Quantity):
        d = physical_size
    else:
        d = physical_size * u.kpc  # assume in kpc

    angular_distance = get_angular_distance(angular_distance, redshift, cosmo)

    if isinstance(angular_distance, u.Quantity):
        D = angular_distance
    else:
        D = angular_distance * u.Mpc  # assume in Mpc

    theta = ((d / D).decompose() * u.radian).to(to_unit)

    if return_with_unit:
        return theta
    else:
        return theta.value


def get_angular_distance(angular_distance, redshift, cosmo):
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


# --- load map ---


def load_map_v3(filename):
    with h5py.File(filename, 'r') as file:

        if file.attrs['data_version'] != 3:
            raise ValueError(
                f"Data version mismatch: {file.attrs['data_version']} != 3")

        # sourcery skip: de-morgan
        layers = {key: value[()] for key, value in file['layers'].items()}

        units = {
            key: value[()].decode('utf-8')
            for key, value in file['units'].items()
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

    # shape as tuple
    if 'shape' in attributes:
        attributes['shape'] = tuple(attributes['shape'])

    return Map(layers=layers, units=units, **attributes)


def load_map_v2(filename):
    with h5py.File(filename, 'r') as file:

        if file.attrs['data_version'] != 2:
            raise ValueError(
                f"Data version mismatch: {file.attrs['data_version']} != 2")

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

    # shape as tuple
    if 'shape' in attributes:
        attributes['shape'] = tuple(attributes['shape'])

    return Map(layers=layers, **attributes)
