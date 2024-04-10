from pathlib import Path
import urllib.request
import zipfile
import numpy as np
from typing import Dict, Union


# Quantities in daemonflux non-prefixed are conventional
quantities = [
    "muflux",
    "muratio",
    "numuflux",
    "numuratio",
    "nueflux",
    "nueratio",
    "flavorratio",
    "mu+",
    "mu-",
    "numu",
    "antinumu",
    "nue",
    "antinue",
]
# Total are conventional + prompt if available
quantities += ["total_" + q for q in quantities]


def format_angle(ang: Union[float, str]) -> str:
    """
    Format the given angle to a string with 4 decimal places.

    Parameters
    ----------
    ang : float
        The angle to be formatted.

    Returns
    -------
    str
        The formatted angle as a string.
    """
    return "{:4.4f}".format(float(ang))


def grid_cov(jac: np.ndarray, invcov: np.ndarray) -> np.ndarray:
    """
    Chi2 matrix expression.

    Parameters
    ----------
    jac : np.ndarray
        The Jacobian matrix.
    invcov : np.ndarray
        The inverse of the covariance matrix.

    Returns
    -------
    np.ndarray

    """
    return np.dot(jac, np.dot(invcov, jac.T))


def is_iterable(arg) -> bool:
    """
    Check if an argument is iterable.

    Parameters
    ----------
    arg : Any
        The argument to check.

    Returns
    -------
    bool
        Whether the argument is iterable or not.
    """
    from collections.abc import Iterable

    return isinstance(arg, Iterable) and not isinstance(arg, str)


def _download_file(outfile, url):
    """Download a file from 'url' to 'outfile'"""
    from rich.progress import (
        Progress,
        TextColumn,
        BarColumn,
        SpinnerColumn,
        DownloadColumn,
        TimeRemainingColumn,
    )

    fname = Path(url).name
    try:
        response = urllib.request.urlopen(url)  # type: ignore
    except BaseException:
        raise ConnectionError(
            f"_download_file: probably something wrong with url = '{url}'"
        )
    total_size = response.getheader("content-length")

    min_blocksize = 4096
    if total_size:
        total_size = int(total_size)
        blocksize = max(min_blocksize, total_size // 100)
    else:
        blocksize = min_blocksize

    wrote = 0
    with Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn() if total_size else SpinnerColumn(),
        DownloadColumn(),
        TimeRemainingColumn(),
        transient=True,
    ) as bar:
        task_id = bar.add_task(f"Downloading {fname}", total=total_size)

        with open(outfile, "wb") as f:
            chunk = True
            while chunk:
                chunk = response.read(blocksize)
                f.write(chunk)
                nchunk = len(chunk)
                wrote += nchunk
                bar.advance(task_id, nchunk)

    if total_size and wrote != total_size:
        raise ConnectionError(f"{fname} has not been downloaded")


# Function to check and download dababase files on github
def _cached_data_dir(url):
    """Checks for existence of version file
    "model_name_vxxx.zip". Downloads and unpacks
    zip file from url in case the file is not found

    Args:
        url (str): url for zip file
    """

    base_dir = Path(__file__).parent.absolute() / "data"
    base_dir.mkdir(parents=True, exist_ok=True)

    fname = Path(url).stem + ".pkl"
    full_path = base_dir / fname
    if not full_path.exists():
        zip_fname = base_dir / Path(url + ".zip").name
        _download_file(zip_fname, url + ".zip")
        if zipfile.is_zipfile(zip_fname):
            with zipfile.ZipFile(zip_fname, "r") as zf:
                zf.extractall(base_dir)
            zip_fname.unlink()

    return str(full_path)


def rearrange_covariance(
    original_order: Dict[str, int], new_order: list, cov: np.ndarray
) -> np.ndarray:
    """Rearrange the covariance matrix to match the new ordering of parameters.

    Parameters
    ----------
    original_order: Dict[str, int]
        Map of current parameter order of the covariance matrix `cov`.
    new_order: list
        The new ordering of the parameters.
    cov: np.ndarray
        The covariance matrix with the original ordering of the parameters.

    Returns
    -------
    cov_new: np.ndarray
        The rearranged covariance matrix with the new ordering of the parameters.
    """
    cov_new = np.zeros((len(new_order), len(new_order)))
    remap = original_order
    for i in range(cov_new.shape[0]):
        for j in range(cov_new.shape[1]):
            cov_new[i, j] = cov[remap[new_order[i]], remap[new_order[j]]]
    return cov_new
