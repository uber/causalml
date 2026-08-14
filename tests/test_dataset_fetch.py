"""Tests for the benchmark dataset cache and loaders.

The tests here never reach the network: the cache, the digest check and the
offline path are exercised against files written into ``tmp_path``. The loaders
themselves are covered by ``test_benchmark_loaders.py``, which is marked
``network`` and skipped unless the datasets are already cached.
"""

import hashlib

import pytest

from causalml.dataset import clear_data_dir, get_data_home
from causalml.dataset._base import DATA_HOME_ENV, DEFAULT_DATA_HOME, fetch_remote

CONTENT = b"benchmark,data\n1,2\n"
DIGEST = hashlib.sha256(CONTENT).hexdigest()


@pytest.fixture
def data_home(tmp_path, monkeypatch):
    """Point the cache at a temporary directory for the duration of a test."""
    monkeypatch.setenv(DATA_HOME_ENV, str(tmp_path / "cache"))
    return tmp_path / "cache"


def _seed(path, content=CONTENT):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    return path


def test_get_data_home_prefers_the_argument_then_the_environment(tmp_path, monkeypatch):
    """Resolution order is argument, then CAUSALML_DATA, then the default.

    The default itself is asserted on the constant rather than by calling with no
    environment set, which would create a directory in the developer's home.
    """
    monkeypatch.setenv(DATA_HOME_ENV, str(tmp_path / "from_env"))

    assert get_data_home() == tmp_path / "from_env"
    assert get_data_home(tmp_path / "explicit") == tmp_path / "explicit"
    assert (tmp_path / "explicit").is_dir()
    assert DEFAULT_DATA_HOME == "~/causalml-data"


def test_fetch_remote_returns_the_cached_file_without_downloading(data_home):
    """A cached file whose digest matches is used as is."""
    _seed(data_home / "cached.csv")

    path = fetch_remote(
        url="https://example.invalid/cached.csv",
        filename="cached.csv",
        sha256=DIGEST,
        download_if_missing=False,
    )

    assert path.read_bytes() == CONTENT


def test_fetch_remote_raises_when_missing_and_offline(data_home):
    """`download_if_missing=False` names the file it wanted and the source."""
    with pytest.raises(OSError, match="download_if_missing=False"):
        fetch_remote(
            url="https://example.invalid/absent.csv",
            filename="absent.csv",
            sha256=DIGEST,
            download_if_missing=False,
            dataset_name="Absent",
        )


def test_fetch_remote_rejects_a_cached_file_with_the_wrong_digest(data_home):
    """A corrupted cache entry is not silently used.

    The digest is the whole point of the check: a truncated download, or an HTML
    error page saved under the dataset's name, is still a readable file. Without
    verification it would be parsed and reported as a benchmark result.
    """
    _seed(data_home / "corrupt.csv", b"not the data you wanted")

    with pytest.raises(OSError):
        fetch_remote(
            url="https://example.invalid/corrupt.csv",
            filename="corrupt.csv",
            sha256=DIGEST,
            download_if_missing=False,
            dataset_name="Corrupt",
        )


def test_fetch_remote_verifies_the_digest_after_downloading(data_home, monkeypatch):
    """A download whose digest does not match is deleted rather than returned."""

    def fake_urlretrieve(url, path):
        with open(path, "wb") as f:
            f.write(b"something else entirely")

    monkeypatch.setattr("causalml.dataset._base.urlretrieve", fake_urlretrieve)

    with pytest.raises(OSError, match="expected"):
        fetch_remote(
            url="https://example.invalid/wrong.csv",
            filename="wrong.csv",
            sha256=DIGEST,
            dataset_name="Wrong",
        )

    assert not (data_home / "wrong.csv").exists()


def test_fetch_remote_keeps_a_download_whose_digest_matches(data_home, monkeypatch):
    """The happy path writes the file to the cache and returns it."""

    def fake_urlretrieve(url, path):
        with open(path, "wb") as f:
            f.write(CONTENT)

    monkeypatch.setattr("causalml.dataset._base.urlretrieve", fake_urlretrieve)

    path = fetch_remote(
        url="https://example.invalid/good.csv", filename="good.csv", sha256=DIGEST
    )

    assert path == data_home / "good.csv"
    assert path.read_bytes() == CONTENT


def test_clear_data_dir_removes_the_cache(data_home):
    """A bad cache is one call away from being recoverable."""
    _seed(data_home / "cached.csv")

    clear_data_dir()

    assert not (data_home / "cached.csv").exists()
