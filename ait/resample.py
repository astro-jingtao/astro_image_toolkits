from scipy.ndimage import zoom


def resample_image(image,
                   scale=None,
                   from_pixel_scale=None,
                   to_pixel_scale=None,
                   **kwargs):
    if scale is None:
        if from_pixel_scale is None or to_pixel_scale is None:
            raise ValueError(
                'Either scale or from_pixsize and to_pixsize must be provided')
        scale = from_pixel_scale / to_pixel_scale
    else:
        if from_pixel_scale is not None or to_pixel_scale is not None:
            Warning('Both scale and from_pixsize and to_pixsize are provided. '
                    'from_pixsize and to_pixsize will be ignored.')

    return zoom(image, scale, **kwargs)
