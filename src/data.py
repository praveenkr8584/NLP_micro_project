"""Load the raw Brown Corpus.

The corpus zip is stored in the repository (data/raw/brown.zip), so nothing has
to be downloaded. `download_brown` is only a fallback if the file is missing.
"""
import re
import urllib.request
import zipfile
from pathlib import Path
from typing import Iterator, Tuple

from . import config

# Brown file names look like "ca01", "cr09" ... (c + genre letter + 2 digits)
_FILE_RE = re.compile(r"c[a-z]\d\d")


def download_brown(dest: Path = config.RAW_ZIP, url: str = config.BROWN_URL) -> Path:
    """Download the Brown Corpus zip if it is not already on disk."""
    if dest.exists():
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(url, dest)
    return dest


def iter_raw_sentences(zip_path: Path = config.RAW_ZIP) -> Iterator[Tuple[str, list]]:
    """Yield (file_name, words) for every line (= one sentence) of the corpus.

    Brown is POS-tagged: each token looks like ``The/at``. We keep only the
    word and throw the tag away.
    """
    download_brown(zip_path)
    with zipfile.ZipFile(zip_path) as zf:
        for name in sorted(zf.namelist()):
            base = name.split("/")[-1]
            if not _FILE_RE.fullmatch(base):
                continue
            text = zf.read(name).decode("latin-1")
            for line in text.splitlines():
                line = line.strip()
                if not line:
                    continue
                words = [tok.rsplit("/", 1)[0] for tok in line.split()]
                yield base, words
