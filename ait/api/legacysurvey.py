# https://www.legacysurvey.org/dr10/description/

import contextlib
import os
import requests

from astropy.coordinates import SkyCoord
import astropy.coordinates.name_resolve as name_resolve


def resolve_name(name):
    with contextlib.suppress(Exception):
        return name_resolve.get_icrs_coordinates(name)
    with contextlib.suppress(Exception):
        return name_resolve.get_icrs_coordinates(name, 'SIMBAD')
    with contextlib.suppress(Exception):
        # Try to resolve using Sesame
        coord = SkyCoord.from_name(name)
        return coord.icrs
    return None


def download_cutout_image(name=None,
                          ra=None,
                          dec=None,
                          layer=None,
                          pixscale=None,
                          size=None,
                          save_to_file=True,
                          print_info=False,
                          root=".",
                          fits=False,
                          timeout=300,
                          filename=None):
    # sourcery skip: merge-else-if-into-elif
    '''
    This Python function allows users to download cutout images from the Legacy Survey website. The cutouts are of a specific sky location and are saved as either a JPG or FITS file. The following parameters can be set:

    - `name`: If `None`, the right ascension (`ra`) and declination (`dec`) of the sky location must be given.
    - `ra`: The right ascension of the sky location.
    - `dec`: The declination of the sky location.
    - `layer`: The Legacy Survey layer to use.
    - `pixscale`: The angular size of the pixels in arc seconds.
    - `size`: The length of each side of the cutout image in pixels.
    - `save_to_file`: Whether or not to save the downloaded image to a file.
    - `print_info`: Whether or not to print information about the download process.
    - `root`: The directory in which to save the downloaded image file.
    - `fits`: Whether to save the cutout image as a FITS file. If not, it is saved as a JPG file.

    If `name` is not `None`, but `ra` and `dec` are `None`, `resolve_name` is called to determine the `ra` and `dec` of the given name. The URL for the image is then generated based on the specified parameters and a GET request is made to download the image data. If `save_to_file` is `True`, the image is saved to a file in the specified directory, and if `print_info` is `True`, information about the saved file is printed. The function returns the image data if the download was successful, and `None` otherwise.
    '''

    if (name is not None) and (ra is None) and (dec is None):
        ra_dec = resolve_name(name)
        ra = ra_dec.ra.value
        dec = ra_dec.dec.value
    if fits:
        url = "https://www.legacysurvey.org/viewer/cutout.fits"
    else:
        url = "https://www.legacysurvey.org/viewer/jpeg-cutout"
    response = requests.get(url,
                            timeout=timeout,
                            params={
                                'ra': ra,
                                'dec': dec,
                                'layer': layer,
                                'pixscale': pixscale,
                                'size': size
                            })
    if response.status_code == 200:
        if save_to_file:
            if filename is None:
                if fits:
                    filename = f"{layer}_ra{ra:.4f}_dec{dec:.4f}_scale{pixscale}_size{size}.fits"
                else:
                    filename = f"{layer}_ra{ra:.4f}_dec{dec:.4f}_scale{pixscale}_size{size}.jpg"
            file_path = os.path.join(root, filename)
            with open(file_path, "wb") as f:
                f.write(response.content)
            if print_info:
                print(
                    f"Downloaded {layer} cutout image for RA {ra}, Dec {dec} with a pixscale of {pixscale} and size of {size} to {file_path}"
                )
        else:
            if print_info:
                print(
                    f"Image not saved to file. Content: {response.content[:50]}..."
                )
        return response.content
    else:
        if print_info:
            print("Failed to download image")
        return None
