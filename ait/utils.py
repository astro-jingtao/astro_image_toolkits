import numpy as np
from PIL import Image

def get_r(input_array):
    '''
    Get the r values for input array with given dimensions.
    Return a cube with r in each cell
    '''
    dim = len(input_array.shape)
    if dim == 1:
        return get_r_1d(input_array)
    elif dim == 2:
        return get_r_2d(input_array)
    elif dim == 3:
        return get_r_3d(input_array)


def get_r_1d(input_array):
    x = np.arange(len(input_array))
    center = x.max() / 2.
    rx = (x - center)
    return [rx], rx


def get_r_2d(input_array):
    x, y = np.indices(input_array.shape)
    center = np.array([(x.max() - x.min()) / 2, (y.max() - y.min()) / 2])
    rx = (x - center[0])
    ry = (y - center[1])
    r = np.sqrt(rx**2 + ry**2)
    return [rx, ry], r


def get_r_3d(input_array):
    nx, ny, nz = input_array.shape
    x, y, z = np.indices(input_array.shape)
    center = np.array([nx/2 if nx%2==0 else (nx-1)/2, ny/2 if ny%2==0 else (ny-1)/2, \
                        nz/2 if nz%2==0 else (nz-1)/2])
    rx = (x - center[0])
    ry = (y - center[1])
    rz = (z - center[2])

    r = np.sqrt(rx**2 + ry**2 + rz**2)
    return [rx, ry, rz], r



def read_png(path):
    img = Image.open(path)
    img = np.array(img)
    return img

def save_png(img, path):
    # img should be [0, 255]
    img = Image.fromarray(np.clip(img, 0, 255).astype(np.uint8))
    img.save(path)