import astropy.units as u
from reproject import reproject_adaptive, reproject_exact, reproject_interp
from scipy.ndimage import zoom


def reproject(array,
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
    array : array-like
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
    scaler : float or string, optional
        A scaling factor to apply to the reprojected array. Default is 1.
        If 'area', the reprojected array will be scaled by the ratio of the
        areas of the input and output array.
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

    if scaler == 'area':
        area_ratio = to_wcs.proj_plane_pixel_area(
        ) / from_wcs.proj_plane_pixel_area()
        scaler = area_ratio.to(u.dimensionless_unscaled).value

    array_new, footprint = reproject_func((array, from_wcs),
                                          to_wcs,
                                          shape_out=to_shape,
                                          return_footprint=True,
                                          **kwargs)
    if return_footprint:
        return array_new * scaler, footprint

    return array_new * scaler


def zoom_wcs(wcs, zoom_factor):
    """
    Zooms a WCS by a given factor.

    Parameters
    ----------
    wcs : `~astropy.wcs.WCS`
        The WCS object to be zoomed.
    zoom_factor : float
        The zoom factor along the axes.
        zoom_factor > 1 means upsampling, zoom_factor < 1 means downsampling.

    Returns
    -------
    wcs_new : `~astropy.wcs.WCS`
        The zoomed WCS object.

    """

    wcs_new = wcs.deepcopy()
    if wcs.wcs.has_cd():
        wcs_new.wcs.cd /= zoom_factor
    else:
        wcs_new.wcs.cdelt /= zoom_factor
    # -0.5 and +0.5 are used to center the image at the new pixel centers
    wcs_new.wcs.crpix = (wcs.wcs.crpix - 0.5) * zoom_factor + 0.5
    return wcs_new


def resample_wcs(array,
                 wcs,
                 zoom_factor=None,
                 from_pixel_scale=None,
                 to_pixel_scale=None,
                 method='interp',
                 scaler=1,
                 **kwargs):
    """
    Resamples an array by a given zoom factor or pixel scale, by using WCS.

    Parameters
    ----------
    array : array-like
        The array to be resampled.
    wcs : `~astropy.wcs.WCS`
        The WCS object of the input array.
    zoom_factor : float
        The zoom factor along the axes.
        zoom_factor > 1 means upsampling, zoom_factor < 1 means downsampling.
        `to_pixel_scale` = `from_pixel_scale` / `zoom_factor`.
        If not provided, `from_pixel_scale` and `to_pixel_scale` must be provided.
    from_pixel_scale : float, optional
        The pixel scale of the input array.
    to_pixel_scale : float, optional
        The pixel scale of the output array.
    method : {'interp', 'adaptive', 'exact'}, optional
        The method to use for reprojection. Default is 'interp'.
    scaler : float or string, optional
        A scaling factor to apply to the reprojected array. Default is 1.
        If 'area', the reprojected array will be scaled by the ratio of the
        areas of the input and output array.
    **kwargs : dict, optional
        Additional keyword arguments to pass to the reproject function.

    Returns
    -------
    array : array-like
        The resampled array.
    
    Notes
    -----
    The results equivalent to `resample_array` with `grid_mode='True'`.
    """

    zoom_factor = get_zoom_factor(zoom_factor, from_pixel_scale,
                                  to_pixel_scale)
    wcs_new = zoom_wcs(wcs, zoom_factor)
    shape_new = [int(s * zoom_factor) for s in array.shape]

    return reproject(array,
                     wcs,
                     wcs_new,
                     shape_new,
                     method=method,
                     scaler=scaler,
                     return_footprint=False,
                     **kwargs)


def resample_array(array,
                   zoom_factor=None,
                   from_pixel_scale=None,
                   to_pixel_scale=None,
                   scaler=1,
                   mode='nearest',
                   grid_mode=True,
                   **kwargs):
    '''
    Resamples an array by a given zoom factor or pixel scale.

    Parameters
    ----------
    array : array-like
            The array to be resampled.
    zoom_factor : float or sequence
        The zoom factor along the axes.
        If a float, `zoom_factor` is the same for each axis. 
        If a sequence, `zoom_factor` should contain one value for each axis.
        zoom_factor > 1 means upsampling, zoom_factor < 1 means downsampling.
        `to_pixel_scale` = `from_pixel_scale` / `zoom_factor`.
        If not provided, `from_pixel_scale` and `to_pixel_scale` must be provided.
    from_pixel_scale : float, optional
        The pixel scale of the input array.
    to_pixel_scale : float, optional
        The pixel scale of the output array.
    scaler : float or string, optional
        A scaling factor to apply to the reprojected array. Default is 1.
        If 'area', the reprojected array will be scaled by the ratio of the
        areas of the input and output array.
    mode : optional
        The interpolation mode for `scipy.ndimage.zoom`. Default is 'nearest'.
    grid_mode : bool, optional
        If False, the distance from the pixel centers is zoomed. Otherwise, the
        distance including the full pixel extent is used. 
        Assuming the input array is an 1d signalwith 3 pixels
        If `zoom=2`, and `grid_mode=True`
        .. code-block:: text
            |ooo|ooo|ooo|
            |<-   6   ->|

        If `zoom=2`, and `grid_mode=False`
        .. code-block:: text
            |ooo|ooo|ooo|
              |<- 6 ->|

        The `resample_wcs` works as `grid_mode=True`. Thus we set default `grid_mode=True` here.
        Note that `grid_mode=False` is default for `scipy.ndimage.zoom`.
    **kwargs : dict, optional
        Additional keyword arguments to pass to the `scipy.ndimage.zoom` function.

    Returns
    -------
    output_array : array-like
        The resampled array.

    '''

    zoom_factor = get_zoom_factor(zoom_factor, from_pixel_scale,
                                  to_pixel_scale)

    if scaler == 'area':
        scaler = 1 / (zoom_factor**2)

    return zoom(array, zoom_factor, mode=mode, grid_mode=grid_mode, **
                kwargs) * scaler


def get_zoom_factor(zoom_factor, from_pixel_scale, to_pixel_scale):
    if zoom_factor is None:
        if from_pixel_scale is None or to_pixel_scale is None:
            raise ValueError(
                'Either scale or from_pixsize and to_pixsize must be provided if scale is not provided.'
            )
        else:
            zoom_factor = from_pixel_scale / to_pixel_scale
    elif from_pixel_scale is not None or to_pixel_scale is not None:
        raise Warning(
            'Both scale and from_pixsize and to_pixsize are provided. '
            'from_pixsize and to_pixsize will be ignored.')

    return zoom_factor
