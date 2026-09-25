"""MNIST training images of selected digits."""

from __future__ import annotations

import gzip
import struct
import urllib.request
from pathlib import Path

import numpy as np

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "mnist"
MIRROR = "https://ossci-datasets.s3.amazonaws.com/mnist/"
_RAW: dict = {}


def _fetch(fname: str) -> Path:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    path = DATA_DIR / fname
    if not path.exists():
        print(f"[mnist] downloading {fname} ...", flush=True)
        urllib.request.urlretrieve(MIRROR + fname, path)
    return path


def read_idx(path: Path) -> np.ndarray:
    with gzip.open(path, "rb") as fh:
        magic, = struct.unpack(">I", fh.read(4))
        ndim = magic & 0xFF
        dims = struct.unpack(">" + "I" * ndim, fh.read(4 * ndim))
        return np.frombuffer(fh.read(), dtype=np.uint8).reshape(dims)


def train_images():
    """The 60,000 training images (uint8, 60000 x 28 x 28) and labels (int64)."""
    if not _RAW:
        _RAW["images"] = read_idx(_fetch("train-images-idx3-ubyte.gz"))
        _RAW["labels"] = read_idx(_fetch("train-labels-idx1-ubyte.gz")).astype(np.int64)
    return _RAW["images"], _RAW["labels"]


def load_digits(digits, per_class: int | None = None):
    """The first ``per_class`` training images of each digit, in dataset order (``None``: the size of the smallest
    of the chosen classes).  Returns X (n, 784) float64 with pixels in [0, 1] and labels (n,) in 0..K-1, where label k
    is ``digits[k]``; the rows are grouped by class."""
    images, labels = train_images()
    idx = [np.nonzero(labels == int(dg))[0] for dg in digits]
    take = min(i.size for i in idx) if per_class is None else int(per_class)
    if any(i.size < take for i in idx):
        raise ValueError(f"digits {tuple(digits)}: only {tuple(i.size for i in idx)} images, need {take} per class")
    idx = [i[:take] for i in idx]
    X = np.concatenate([images[i].reshape(i.size, -1) for i in idx])
    X = np.ascontiguousarray(X.astype(np.float64) / 255.0)
    y = np.concatenate([np.full(i.size, k, dtype=np.int64) for k, i in enumerate(idx)])
    return X, y
