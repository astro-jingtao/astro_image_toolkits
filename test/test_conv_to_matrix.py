from itertools import product
import pytest
import numpy as np
from ait.conv_to_matrix import kernel_to_matrix
from ait.conv import conv_cube_2d

# I * K = int I(tau) K(t - tau) d tau
# so kernel should be reversed
KERNEL_ARR = {
    'identity': np.array([[1]]),
    'xsobel': np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]]),
    'ysobel': np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]]),
    'xgradf': np.array([[1, -1, 0]]),
    'xgradb': np.array([[0, 1, -1]]),
    'xgradc': np.array([[1, 0, -1]]),
    'ygradf': np.array([[1], [-1], [0]]),
    'ygradb': np.array([[0], [1], [-1]]),
    'ygradc': np.array([[1], [0], [-1]]),
}

all_kernel_types = list(KERNEL_ARR.keys())


class TestConsistencyWithConv:

    @pytest.mark.parametrize("kernel_type, normalize",
                             product(all_kernel_types, [False, True]))
    def test_different_kernel_type(self, kernel_type, normalize):
        m, n, p = 20, 20, 3
        X = np.random.normal(size=(m, n, p))
        A = kernel_to_matrix(m,
                             n,
                             p,
                             kernel_type=kernel_type,
                             normalize=normalize)
        X_conv = (A.toarray() @ X.ravel()).reshape(m, n, p)
        assert X_conv.shape == X.shape
        kernel = KERNEL_ARR[kernel_type]
        if normalize:
            if kernel_type in ['xsobel', 'ysobel']:
                scaler = 8
            elif kernel_type in ['xgradc', 'ygradc']:
                scaler = 2
            else:
                scaler = 1
            kernel = kernel / scaler
        X_conv_ref = conv_cube_2d(X,
                                  kernel,
                                  vel_axis=2,
                                  normalize_kernel=False)
        assert np.allclose(X_conv, X_conv_ref)

    @pytest.mark.parametrize(
        "kernel_type, boundary_type",
        product(all_kernel_types, ['zero', 'reflect', 'edge']))
    def test_boundary(self, kernel_type, boundary_type):
        m, n, p = 20, 20, 3
        X = np.random.normal(size=(m, n, p))
        A = kernel_to_matrix(m,
                             n,
                             p,
                             kernel_type=kernel_type,
                             normalize=False,
                             boundary_type=boundary_type)
        X_conv = (A.toarray() @ X.ravel()).reshape(m, n, p)
        assert X_conv.shape == X.shape
        kernel = KERNEL_ARR[kernel_type]
        if boundary_type == 'zero':
            pad_boundary = 'constant'
        else:
            pad_boundary = boundary_type

        X_conv_ref = conv_cube_2d(X,
                                  kernel,
                                  vel_axis=2,
                                  normalize_kernel=False,
                                  pad=True,
                                  pad_boundary=pad_boundary)
        assert np.allclose(X_conv, X_conv_ref)
