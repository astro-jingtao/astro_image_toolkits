from reproject import reproject_interp, reproject_adaptive, reproject_exact


def reproject(arr,
              from_wcs,
              to_wcs,
              to_shape,
              method='interp',
              scaler=1,
              return_footprint=False,
              **kwargs):
    """
    Reprojects an array from one WCS to another.

    Parameters
    ----------
    arr : array-like
        The array to be reprojected. 
        If the data array contains more dimensions than are described by the
        input header or WCS, the extra dimensions (assumed to be the first
        dimensions) are taken to represent multiple images with the same
        coordinate information. (This is handled by the reproject function.)
    from_wcs : `~astropy.wcs.WCS`
        The WCS object from which the array is currently projected.
    to_wcs : `~astropy.wcs.WCS`
        The WCS object to which the array should be reprojected.
    to_shape : tuple of int
        The shape of the output array.
    method : {'interp', 'adaptive', 'exact'}, optional
        The method to use for reprojection. Default is 'interp'.
    scaler : float, optional
        A scaling factor to apply to the reprojected array. Default is 1.
    return_footprint : bool, optional
        If True, return the footprint of the reprojected array. Default is False.
    **kwargs : dict, optional
        Additional keyword arguments to pass to the reproject function.

    Returns
    -------
    array : array-like
        The reprojected array, optionally scaled by `scaler`.
    footprint : `~astropy.nddata.NDUncertainty`, optional
        The footprint of the reprojected array, returned only if `return_footprint` is True.

    Raises
    ------
    ValueError
        If an invalid `method` is provided.
    """

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
