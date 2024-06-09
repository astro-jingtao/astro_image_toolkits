from reproject import reproject_interp, reproject_adaptive, reproject_exact


def reproject(arr,
              from_wcs,
              to_wcs,
              to_shape,
              method='interp',
              scaler=1,
              return_footprint=False,
              **kwargs):
    if method == 'interp':
        reproject_func = reproject_interp
    elif method == 'adaptive':
        reproject_func = reproject_adaptive
    elif method == 'exact':
        reproject_func = reproject_exact
    else:
        raise ValueError(f'Invalid resampling method: {method}')

    array_new, footprint = reproject_func((arr, from_wcs),
                                          to_wcs,
                                          shape_out=to_shape,
                                          return_footprint=True,
                                          **kwargs)
    if return_footprint:
        return array_new * scaler, footprint

    return array_new * scaler
