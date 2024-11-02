import hashlib
import os

import requests


class Printer:

    def __init__(self, verbose=False, quiet=False):
        self.verbose = verbose
        self.quiet = quiet

    def print_verbose(self, *args, **kwargs):
        if self.verbose:
            print(*args, **kwargs)

    def print_quiet(self, *args, **kwargs):
        if not self.quiet:
            print(*args, **kwargs)

    def print(self, *args, **kwargs):
        print(*args, **kwargs)


def download_url(url,
                 file_path=None,
                 file_name=None,
                 file_root='./',
                 force=False,
                 quiet=False,
                 verbose=False,
                 max_attempts=3,
                 timeout=None):
    """
    Download a file from a given URL and save it to the specified path.

    Parameters
    ----------
    url : str
        The URL of the file to download.
    file_path : str, optional
        The full path to save the file. If not provided, it will be constructed with file_root and file_name.
    file_name : str, optional
        The name of the file to save. If not provided, it is derived from the URL. If file_path is provided, this parameter is ignored.
    file_root : str, optional
        The root directory where the file will be saved. Defaults to the current directory. If file_path is provided, this parameter is overwritten by the root of file_path.
    force : bool, optional
        If True, overwrite the file if it already exists. Defaults to False.
    quiet : bool, optional
        If True, suppress all non-error messages. Defaults to False.
    verbose : bool, optional
        If True, print verbose messages about the download process. Defaults to False.
    max_attempts : int, optional
        The maximum number of attempts to download the file. Defaults to 3.
    timeout : float or tuple, optional
        The timeout value for the request. If a tuple is provided, it must have two elements (connect timeout, read timeout).

    Returns
    -------
    tuple
        A tuple containing two elements: the first is an integer indicating the success status (0 for success, 1 for failure), and the second is the HTTP status code of the last attempt.
    """

    printer = Printer(verbose=verbose, quiet=quiet)

    if file_path is None:
        if file_name is None:
            file_name = url.split('/')[-1]
        file_path = os.path.join(file_root, file_name)
        printer.print_verbose(f"File path not provided. Using {file_path}.")
    else:
        file_root = os.path.dirname(file_path)

    if not os.path.exists(file_root):
        os.makedirs(file_root)

    if os.path.exists(file_path):
        if force:
            printer.print_verbose(
                f"File {file_path} already exists. Forcing download (replace existed file)."
            )
        else:
            printer.print_verbose(
                f"File {file_path} already exists. Skipping download.")
            return -1, -1

    for attempt in range(max_attempts):
        response = requests.get(url, stream=True, timeout=timeout)

        if response.status_code == 200:
            with open(file_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            printer.print_verbose(
                f"Download completed successfully for {file_path}!")
            return 0, response.status_code
        else:
            printer.print_verbose(
                f"Failed to download {url} to {file_path} on attempt {attempt + 1}."
            )
            printer.print_verbose(f"The state code was {response.status_code}")

    printer.print_quiet(
        f"Failed to download {url} to {file_path} after {max_attempts} attempts"
    )
    printer.print_quiet(
        f"The state code of the last attempt was {response.status_code}")

    return 1, response.status_code


def verify_sha1sum(file_path, sha1sum_path, verbose=False, quiet=False):
    """
    Verifies the SHA1 checksum of a file against a provided checksum file.

    Parameters
    ----------
    file_path : str
        The path to the file to be verified.
    sha1sum_path : str
        The path to the checksum file containing the expected SHA1 checksums.
    verbose : bool, optional
        If True, print messages indicating the verification status.

    Returns
    -------
    int
        0 if the checksum matches, 1 if it does not match, and 2 if no checksum is found for the file.
    """

    printer = Printer(verbose=verbose, quiet=quiet)

    file_name = os.path.basename(file_path)

    expected_checksum = None
    with open(sha1sum_path, 'r', encoding='utf-8') as file:
        for line in file:
            if file_name in line:
                expected_checksum = line.split()[0]
                break

    if expected_checksum:
        hasher = hashlib.sha1()
        with open(file_path, 'rb') as a_file:
            buf = a_file.read()
            hasher.update(buf)
        file_checksum = hasher.hexdigest()

        if file_checksum == expected_checksum:
            printer.print_verbose("Checksum is correct for", file_path)
            return 0
        else:
            printer.print_verbose("Checksum mismatch for", file_path)
            return 1

    else:
        printer.print_verbose("No checksum found for", file_path)
        return 2
