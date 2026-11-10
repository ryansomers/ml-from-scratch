"""Offline test of the IDX parser used by scripts/train_mnist.py (no download)."""

import struct
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
from mnist_data import parse_idx  # noqa: E402


def test_parse_idx_roundtrip():
    images = np.arange(2 * 3 * 4, dtype=np.uint8).reshape(2, 3, 4)
    raw = b"\x00\x00\x08\x03" + struct.pack(">III", 2, 3, 4) + images.tobytes()
    np.testing.assert_array_equal(parse_idx(raw), images)


def test_parse_idx_rejects_other_dtypes():
    with pytest.raises(ValueError):
        parse_idx(b"\x00\x00\x0d\x01" + struct.pack(">I", 1) + b"\x00" * 4)
