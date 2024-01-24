import numpy as np


def fft2d(arr):
    fft_arr = np.fft.fft2(arr)
    fft_arr = np.fft.fftshift(fft_arr)
    return fft_arr


def ifft2d(fft_arr):
    ifft_arr = np.fft.ifftshift(fft_arr)
    ifft_arr = np.fft.ifft2(ifft_arr)
    return ifft_arr


def polar2complex(r, theta):
    return r * np.exp(1j * theta)


def complex2polar(z):
    return np.abs(z), np.angle(z)


# for RGB img

def fft2d_img(img):
    '''
    do fft in three channels one by one
    '''
    fft_img = np.zeros(img.shape, dtype=np.complex128)
    for i in range(3):
        fft_img[:, :, i] = fft2d(img[:, :, i])
    return fft_img

def ifft2d_img(fft_img):
    '''
    do ifft in three channels one by one
    '''
    ifft_img = np.zeros(fft_img.shape, dtype=np.complex128)
    for i in range(3):
        ifft_img[:, :, i] = ifft2d(fft_img[:, :, i])

    # convert to int8
    ifft_img = np.real(ifft_img)
    ifft_img = np.clip(ifft_img, 0, 255)
    ifft_img = np.array(ifft_img, dtype=np.uint8)

    return ifft_img
