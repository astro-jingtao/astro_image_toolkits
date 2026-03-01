from functools import partial

import numpy as np
from scipy.sparse import coo_matrix

# TODO: mask

# I * K = int I(tau) K(t - tau) d tau
# it should be considered as K(-x)
KERNEL_COORD_DATA = {
    'identity': ([(0, 0, 0, 1)], 1),
    'xsobel': ([(0, -1, 0, -2), (-1, -1, 0, -1), (1, -1, 0, -1), (0, 1, 0, 2),
                (-1, 1, 0, 1), (1, 1, 0, 1)], 8),
    'ysobel': ([(-1, 0, 0, -2), (-1, -1, 0, -1), (-1, 1, 0, -1), (1, 0, 0, 2),
                (1, -1, 0, 1), (1, 1, 0, 1)], 8),
    'xgradf': ([(0, 0, 0, -1), (0, 1, 0, 1)], 1),  # x gradient forward
    'xgradb': ([(0, 0, 0, 1), (0, -1, 0, -1)], 1),  # x gradient backward
    'xgradc': ([(0, 1, 0, 1), (0, -1, 0, -1)], 2),  # x gradient central
    'ygradf': ([(0, 0, 0, -1), (1, 0, 0, 1)], 1),  # y gradient forward
    'ygradb': ([(0, 0, 0, 1), (-1, 0, 0, -1)], 1),  # y gradient backward
    'ygradc': ([(1, 0, 0, 1), (-1, 0, 0, -1)], 2),  # y gradient central
}


def boundary_handler(coord, shape, boundary_type='zero'):
    '''
    boundary_type: 'zero', 'reflect', 'edge'
    '''
    i, j, k, v = coord
    m, n, p = shape
    if boundary_type == 'zero':
        if i >= 0 and i < m and j >= 0 and j < n:
            return i, j, k, v
        else:
            return None
    elif boundary_type == 'reflect':
        if i < 0:
            i = -i
        elif i >= m:
            i = 2 * (m - 1) - i
        if j < 0:
            j = -j
        elif j >= n:
            j = 2 * (n - 1) - j
        return i, j, k, v
    elif boundary_type == 'edge':
        if i < 0:
            i = 0
        elif i >= m:
            i = m - 1
        if j < 0:
            j = 0
        elif j >= n:
            j = n - 1
        return i, j, k, v


def get_kernel_coord(coord,
                     shape,
                     kernel_type='identity',
                     normalize=True,
                     boundary_type='zero'):
    i, j, k = coord
    m, n, p = shape
    coord_basic_lst, norm = KERNEL_COORD_DATA[kernel_type]
    norm = norm if normalize else 1
    coord_lst = []
    ijk_idx_dict = {}
    ijk_n = 0
    for this_coord in coord_basic_lst:
        ijkv = boundary_handler((i + this_coord[0], j + this_coord[1],
                                 k + this_coord[2], this_coord[3] / norm),
                                shape,
                                boundary_type=boundary_type)
        if ijkv is None:
            continue
        ijk = (ijkv[0], ijkv[1], ijkv[2])
        if ijk in ijk_idx_dict:
            ijk_idx = ijk_idx_dict[ijk]
            coord_lst[ijk_idx][3] += ijkv[3]  # accumulate value
        else:
            ijk_idx_dict[ijk] = ijk_n
            ijk_n += 1
            coord_lst.append([ijkv[0], ijkv[1], ijkv[2], ijkv[3]])
    # remove zero-valued coordinates
    coord_lst = [coord for coord in coord_lst if coord[3] != 0]
    return coord_lst


def kernel_to_matrix(m,
                     n,
                     p,
                     kernel_coord_getter=None,
                     kernel_type='identity',
                     normalize=True,
                     boundary_type='zero'):

    if kernel_coord_getter is None:
        kernel_coord_getter = partial(get_kernel_coord,
                                      kernel_type=kernel_type,
                                      normalize=normalize,
                                      boundary_type=boundary_type)

    size = m * n * p
    shape = (m, n, p)
    i, j, k = np.unravel_index(np.arange(size), (m, n, p))
    ijk = zip(list(i), list(j), list(k))
    ijk_nbrs = map(lambda coord: kernel_coord_getter(coord, shape),
                   ijk)  # neighbors of each pixel
    ijk_nbrs_to_index = map(lambda l: _change_to_ravel_index(l, shape),
                            ijk_nbrs)
    # we get a list of idx, values for a particular idx
    # we have got the complete list now, map it to actual index
    i_lst = []
    j_lst = []
    v_lst = []

    # i: index of center pixel
    # j: index of neighbor pixel
    # v: value of neighbor pixel
    for i, list_of_coords in enumerate(ijk_nbrs_to_index):
        for (j, v) in list_of_coords:
            i_lst.append(i)
            j_lst.append(j)
            v_lst.append(v)

    return coo_matrix((v_lst, (i_lst, j_lst)), shape=(size, size))


def _change_to_ravel_index(li, shape):
    if len(li) == 0:
        return zip([], [])
    i, j, k, v = zip(*li)
    return zip(np.ravel_multi_index((i, j, k), shape), v)
