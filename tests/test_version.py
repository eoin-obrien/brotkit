from importlib.metadata import version

from packaging.version import Version

import brotkit


def test_version_matches_metadata() -> None:
    assert brotkit.__version__ == version("brotkit")


def test_version_is_pep440() -> None:
    Version(brotkit.__version__)
