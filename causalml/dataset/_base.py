"""Cache and download helpers shared by the ``fetch_*`` benchmark loaders.

The benchmark datasets are fetched from their original sources at first use and
cached on disk; none of them are redistributed with CausalML. See
``docs/datasets.rst`` for where each one comes from and on what terms.
"""

import hashlib
import logging
import os
import shutil
from pathlib import Path
from urllib.request import urlretrieve

logger = logging.getLogger("causalml")

DATA_HOME_ENV = "CAUSALML_DATA"
DEFAULT_DATA_HOME = "~/causalml-data"
_CHUNK = 1 << 20


def get_data_home(data_home=None) -> Path:
    """Return the directory the benchmark loaders cache datasets in.

    Resolution order: the ``data_home`` argument, then the ``CAUSALML_DATA``
    environment variable, then ``~/causalml-data``. The directory is created if
    it does not exist.

    Args:
        data_home (str or Path, optional): an explicit cache directory

    Returns:
        pathlib.Path, the cache directory
    """
    if data_home is None:
        data_home = os.environ.get(DATA_HOME_ENV, DEFAULT_DATA_HOME)
    data_home = Path(data_home).expanduser()
    data_home.mkdir(parents=True, exist_ok=True)
    return data_home


def clear_data_dir(data_home=None) -> None:
    """Delete the cache directory, so a bad download can be recovered from.

    Args:
        data_home (str or Path, optional): an explicit cache directory
    """
    shutil.rmtree(get_data_home(data_home))


def _sha256(path: Path) -> str:
    """Return the SHA256 hex digest of a file, read in chunks."""
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(_CHUNK), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fetch_remote(
    url: str,
    filename: str,
    sha256: str,
    data_home=None,
    download_if_missing: bool = True,
    dataset_name: str = "dataset",
) -> Path:
    """Return the local path to a cached dataset file, downloading it if needed.

    The digest is verified on download and on every cache hit. A benchmark file
    that has been truncated, replaced or served from a redirect page would
    otherwise be read as data and reported as a result; the digest is what makes
    that an error instead of a wrong number.

    Args:
        url (str): where to download the file from
        filename (str): the name to cache it under
        sha256 (str): the expected hex digest
        data_home (str or Path, optional): an explicit cache directory
        download_if_missing (bool): if False, raise instead of downloading
        dataset_name (str): used in error messages

    Returns:
        pathlib.Path, the path to the verified local file

    Raises:
        OSError: if the file is absent and ``download_if_missing`` is False
        OSError: if the digest does not match after downloading
    """
    path = get_data_home(data_home) / filename

    if path.exists():
        if _sha256(path) == sha256:
            return path
        logger.warning(
            f"cached {dataset_name} at {path} does not match its expected digest; "
            "re-downloading"
        )
        path.unlink()

    if not download_if_missing:
        raise OSError(
            f"{dataset_name} is not cached at {path} and download_if_missing=False. "
            f"Call with download_if_missing=True, or place the file there yourself "
            f"(source: {url})."
        )

    logger.info(f"downloading {dataset_name} from {url}")
    try:
        urlretrieve(url, path)
    except Exception as exc:
        if path.exists():
            path.unlink()
        raise OSError(
            f"failed to download {dataset_name} from {url}. The benchmark datasets "
            f"are fetched from their original hosts, which do move; see "
            f"docs/datasets.rst for the current source."
        ) from exc

    digest = _sha256(path)
    if digest != sha256:
        path.unlink()
        raise OSError(
            f"{dataset_name} downloaded from {url} has SHA256 {digest}, expected "
            f"{sha256}. The file at that URL has changed, so it is not the version "
            f"these results were produced with."
        )
    return path
