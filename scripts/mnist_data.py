"""Download and cache MNIST as a compressed .npz under the data directory.

Source: the public OSSCI mirror of the original IDX files (the same mirror
torchvision uses). The four gzipped IDX files are parsed with NumPy and cached
as data/mnist.npz; the data directory is gitignored.
"""

from __future__ import annotations

import gzip
import os
import urllib.request
from pathlib import Path

import numpy as np

MIRROR = "https://ossci-datasets.s3.amazonaws.com/mnist/"
FILES = {
    "x_train": "train-images-idx3-ubyte.gz",
    "y_train": "train-labels-idx1-ubyte.gz",
    "x_test": "t10k-images-idx3-ubyte.gz",
    "y_test": "t10k-labels-idx1-ubyte.gz",
}


def data_dir() -> Path:
    root = Path(__file__).resolve().parent.parent
    return Path(os.environ.get("MLFS_DATA_DIR", root / "data"))


def parse_idx(raw: bytes) -> np.ndarray:
    """Parse an IDX file: 2 zero bytes, a dtype code, ndim, then big-endian dims."""
    if raw[0:2] != b"\x00\x00" or raw[2] != 0x08:  # 0x08 = unsigned byte
        raise ValueError("not an unsigned-byte IDX file")
    ndim = raw[3]
    dims = np.frombuffer(raw, dtype=">u4", count=ndim, offset=4)
    data = np.frombuffer(raw, dtype=np.uint8, offset=4 + 4 * ndim)
    return data.reshape(tuple(int(d) for d in dims))


def load_mnist() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return (x_train, y_train, x_test, y_test); images are uint8 of shape (n, 28, 28)."""
    cache = data_dir() / "mnist.npz"
    if not cache.exists():
        cache.parent.mkdir(parents=True, exist_ok=True)
        arrays = {}
        for key, name in FILES.items():
            print(f"downloading {MIRROR + name}")
            with urllib.request.urlopen(MIRROR + name, timeout=60) as resp:  # noqa: S310
                arrays[key] = parse_idx(gzip.decompress(resp.read()))
        np.savez_compressed(cache, **arrays)
    with np.load(cache) as f:
        return f["x_train"], f["y_train"], f["x_test"], f["y_test"]
