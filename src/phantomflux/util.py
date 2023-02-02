from pathlib import Path
import urllib
import shutil
import zipfile


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
        response = urllib.request.urlopen(url)
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

    vname = Path(url).stem
    model_dir = base_dir / vname.split("_v")[0]
    version_file = model_dir / vname
    if not version_file.exists():
        zip_file = base_dir / Path(url).name
        temp_dir = Path(model_dir.parent / f".{model_dir.name}")
        shutil.rmtree(temp_dir, ignore_errors=True)
        if model_dir.exists():
            shutil.move(str(model_dir), str(temp_dir))
        _download_file(zip_file, url)
        if zipfile.is_zipfile(zip_file):
            with zipfile.ZipFile(zip_file, "r") as zf:
                zf.extractall(base_dir.as_posix())
            zip_file.unlink()
            shutil.rmtree(temp_dir, ignore_errors=True)

        version_glob = vname.split("_v")[0]
        for vfile in model_dir.glob(f"{version_glob}_v*"):
            vfile.unlink
        with open(version_file, "w") as vf:
            vf.write(url)
    return str(model_dir) + "/"
