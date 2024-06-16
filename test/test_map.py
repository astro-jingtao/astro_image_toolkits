import os

import numpy as np
import pandas as pd
from astropy.wcs import WCS
import astropy.units as u
from astropy.cosmology import Planck15, FlatLambdaCDM
import pytest

from ait.map import Map, layers_to_df, df_to_layers
from utils_test.gen_map import generate_map_instance

TMP_PATH = "./tmp_map.h5"
OLD_FILE_TEMP = "./map_data/map_{ver}.h5"

def is_map_all_same(map1, map2):
    assert map1.metadata == map2.metadata
    assert map1.pixel_scale == map2.pixel_scale
    assert map1.redshift == map2.redshift

    assert map1.layers.keys() == map2.layers.keys()
    
    for k in map1.layers.keys():
        assert np.allclose(map1.layers[k], map2.layers[k])    

    assert map1.wcs.to_header_string() == map2.wcs.to_header_string()


class TestSaveLoad:

    def test_save_load(self):
        _map = generate_map_instance()

        # sourcery skip: no-conditionals-in-tests
        if not os.path.exists(OLD_FILE_TEMP.format(ver=_map.DATA_VERSION)):
            _map.save(OLD_FILE_TEMP.format(ver=_map.DATA_VERSION))

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

        _map.units['image'] = 'erg/s'
        _map.save(TMP_PATH)
        _map_loaded = Map.load(TMP_PATH)

        assert _map_loaded.units == _map.units

        os.remove(TMP_PATH)

    def test_load_old(self):

        this_map = generate_map_instance()

        for ver in Map.ALLOWED_OLD_DATA_VERSION:
            old_file_path = OLD_FILE_TEMP.format(ver=ver)
            old_map = Map.load_old(old_file_path, version=ver)

            is_map_all_same(this_map, old_map)
            


class TestUnitConversion:

    def test_get_pixel_area(self):

        _map = generate_map_instance()

        pixel_area = _map.get_pixel_area(unit=u.kpc**2)

        D = FlatLambdaCDM(H0=70, Om0=0.3).angular_diameter_distance(0.1)

        assert np.allclose(pixel_area,
                           (D.to(u.kpc).value *
                            np.deg2rad(_map.pixel_scale / 3600))**2)

    def test_pixel_to_area(self):
        _map = generate_map_instance()

        image_sb = _map.flux_to_surface_brightness('image',
                                                   in_unit=u.erg / u.s /
                                                   u.cm**2 * 1e-17)

        image_err_sb = _map.flux_to_surface_brightness('image',
                                                       in_unit=u.erg / u.s /
                                                       u.cm**2 * 1e-17,
                                                       kind='err')

        # from emilkit
        image_true = _map.layers['image'] * 4 * np.pi / (np.deg2rad(
            _map.pixel_scale / 3600)**2) * (3.0856776e21)**2 * 1e-17

        image_err_true = _map.get_err('image') * 4 * np.pi / (np.deg2rad(
            _map.pixel_scale / 3600)**2) * (3.0856776e21)**2 * 1e-17

        assert np.allclose(image_sb, image_true)
        assert np.allclose(image_err_sb, image_err_true)

        # cosmological

        image_area_cosmo = _map.flux_to_surface_brightness(
            'image', in_unit=u.erg / u.s / u.cm**2 * 1e-17, cosmological=True)

        scaler = ((Planck15.luminosity_distance(_map.redshift) /
                   Planck15.angular_diameter_distance(
                       _map.redshift))**2).decompose().value

        assert np.allclose(image_area_cosmo, image_true * scaler)

        # 3d layer

        _map.layers['cube'] = np.random.rand(20, 10, 10)

        cube_area = _map.flux_to_surface_brightness('cube',
                                                    in_unit=u.erg / u.s /
                                                    u.cm**2 * 1e-17)

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


class TestLayers2DF:

    def test_add_global_feature(self):

        layers = {
            'layer1': np.random.rand(10, 10),
            'layer2': np.random.rand(10, 10)
        }

        global_feature = {'feature1': 1}

        result_df = layers_to_df(layers=layers, global_feature=global_feature)

        assert 'feature1' in result_df.columns
        assert result_df['feature1'].unique() == [1]

        global_feature = {'layer1': 1}

        with pytest.raises(ValueError) as e:
            layers_to_df(layers=layers, global_feature=global_feature)
        assert str(
            e.value
        ) == "Duplicate key: layer1, consider using a new key to add the globals"

    def test_add_pos(self):

        def check(result_df):
            assert 'pos_ii' in result_df.columns
            assert 'pos_jj' in result_df.columns

            selected = (result_df['pos_ii'] == 8) & (result_df['pos_jj'] == 4)

            assert result_df['layer1'][selected].values[0] == layers['layer1'][
                8, 4]
            assert result_df['layer2'][selected].values[0] == layers['layer2'][
                8, 4]

        layers = {
            'layer1': np.random.rand(10, 10),
            'layer2': np.random.rand(10, 10)
        }

        result_df = layers_to_df(layers=layers, add_pos=True)

        check(result_df)

        result_df_1 = layers_to_df(layers=layers, add_pos=True, drop_nan=False)

        check(result_df_1)

        layers = {
            'pos_ii': np.random.rand(10, 10),
            'pos_jj': np.random.rand(10, 10)
        }
        result_df_2 = layers_to_df(layers=layers, add_pos=True)

        assert 'pos_ii_0' in result_df_2.columns
        assert 'pos_jj_0' in result_df_2.columns
        assert np.allclose(result_df_2['pos_ii_0'], result_df_1['pos_ii'])
        assert np.allclose(result_df_2['pos_jj_0'], result_df_1['pos_jj'])

    def test_nan_treatment(self):
        # Create dummy input data with NaN values for testing
        layers = {
            'layer1': np.array([[1, np.nan], [3, 4]]),
            'layer2': np.array([[[5, 6], [7, np.nan]], [[9, np.nan], [11,
                                                                      12]]])
        }
        global_feature = {'feature1': 1}

        # Call the function with the test input and drop_nan set to False to observe NaN treatment
        result_df = layers_to_df(layers=layers,
                                 add_pos=False,
                                 global_feature=global_feature,
                                 drop_nan=True,
                                 nan_threshold=0.5)
        print(result_df)
        # Perform assertions to verify NaN treatment
        assert np.allclose(result_df['layer1'], [1, 3, 4])
        assert np.allclose(result_df['layer2_0'], [5, 7, np.nan],
                           equal_nan=True)

    def test_3d_layer(self):
        layers = {
            'layer1': np.random.rand(10, 10),
            'layer2': np.random.rand(10, 10),
            'layer3': np.random.rand(10, 10)
        }

        layers['layer3d'] = np.random.rand(3, 10, 10)

        result_df = layers_to_df(layers=layers)

        assert 'layer3d_0' in result_df.columns
        assert 'layer3d_1' in result_df.columns
        assert 'layer3d_2' in result_df.columns
        assert 'layer3d_3' not in result_df.columns

        print(result_df)

        assert result_df.shape[1] == 3 + 3 + 2

        # test postfix_3d
        def postfix_3d(layer_name, i):
            return f"{layer_name}_{i}_3d"

        result_df = layers_to_df(layers=layers, postfix_3d=postfix_3d)

        assert 'layer3d_0_3d' in result_df.columns
        assert 'layer3d_1_3d' in result_df.columns
        assert 'layer3d_2_3d' in result_df.columns

        assert result_df.shape[1] == 3 + 3 + 2

        def postfix_3d_bad(layer_name, i):
            return f"layer{i}"

        with pytest.raises(ValueError) as e:
            layers_to_df(layers=layers, postfix_3d=postfix_3d_bad)
        assert str(
            e.value
        ) == "Duplicate key: layer1, consider using a new postfix_3d to rename the keys"


class TestDfToLayers:

    def test_basic_functionality(self):
        # Create a simple DataFrame for testing
        df = pd.DataFrame({
            'pos_ii': [0, 1, 2],
            'pos_jj': [0, 1, 2],
            'value': [10, 20, 30]
        })

        # Call the function with default parameters
        layers = df_to_layers(df, set_masked=False)

        # Check if the layers dictionary is created
        assert isinstance(layers, dict)

        # Check if the mask layer is created correctly
        assert 'mask' in layers
        assert layers['mask'].shape == (3, 3)
        assert np.array_equal(
            layers['mask'],
            np.array([[False, True, True], [True, False, True],
                      [True, True, False]]))

        # Check if the value layer is created correctly
        assert 'value' in layers
        assert layers['value'].shape == (3, 3)
        assert np.array_equal(layers['value'],
                              np.array([[10, 0, 0], [0, 20, 0], [0, 0, 30]]))

        # Add more tests for other parameters and edge cases

    def test_drop_unique(self):
        # Create a DataFrame with a column that has only one unique value
        df = pd.DataFrame({
            'pos_ii': [0, 1, 2],
            'pos_jj': [0, 1, 2],
            'value1': [10, 10, 10],
            'value2': [1, 2, 3]
        })

        # Call the function with drop_unique=True
        layers = df_to_layers(df, drop_unique=True)

        # Check if the layer with the unique value is not included
        assert 'value' not in layers

    def test_shape_parameter(self):
        # Create a DataFrame
        df = pd.DataFrame({
            'pos_ii': [0, 1, 2],
            'pos_jj': [0, 1, 2],
            'value': [10, 20, 30]
        })

        # Call the function with a custom shape
        layers = df_to_layers(df, shape=(4, 4))

        # Check if the shape is respected
        assert 'value' in layers
        assert layers['value'].shape == (4, 4)

        assert np.array_equal(
            layers['mask'],
            np.array([[False, True, True, True], [True, False, True, True],
                      [True, True, False, True], [True, True, True, True]]))

    def test_set_masked(self):
        # Create a simple DataFrame for testing
        df = pd.DataFrame({
            'pos_ii': [0, 1, 2],
            'pos_jj': [0, 1, 2],
            'value_float': [10., 20., 30.],
            'value_int': [10, 20, 30],
            'value_str': ['10', '20', '30'],
            'value_bool': [True, True, True]
        })

        # Call the function with default parameters
        layers = df_to_layers(df, set_masked=True)

        assert np.array_equal(layers['value_float'],
                              np.array([[10, np.nan, np.nan],
                                        [np.nan, 20, np.nan],
                                        [np.nan, np.nan, 30]]),
                              equal_nan=True)
        assert layers['value_float'].dtype == np.float64

        assert np.array_equal(
            layers['value_int'],
            np.array([[10, -999, -999], [-999, 20, -999], [-999, -999, 30]]))
        assert layers['value_int'].dtype == int

        # print(layers['value_str'])
        assert np.array_equal(
            layers['value_str'],
            np.array([['10', 'masked', 'masked'], ['masked', '20', 'masked'],
                      ['masked', 'masked', '30']]))

        assert np.issubdtype(layers['value_str'].dtype, np.dtype('U'))

        assert np.array_equal(
            layers['value_bool'],
            np.array([[1, -1, -1], [-1, 1, -1], [-1, -1, 1]]))
        assert layers['value_bool'].dtype == int
