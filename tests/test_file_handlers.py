"""Unit tests for file_handlers."""

from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np
import pytest

from reactifptm.file_handlers import (
    FileBase,
    FileTypes,
    JsonFile,
    NpyFile,
    NpzFile,
    PklFile,
)


@pytest.fixture
def pae_array() -> np.ndarray:
    rng = np.random.default_rng(42)
    return rng.uniform(0, 30, size=(20, 20)).astype(float)


def test_filetypes_values_lists_all_supported() -> None:
    assert set(FileTypes.values()) == {"npz", "npy", "pkl", "json"}


def test_filebase_str_and_repr(tmp_path: Path, pae_array: np.ndarray) -> None:
    p = tmp_path / "x.json"
    p.write_text("{}")
    obj = JsonFile(p)
    # JsonFile inherits FileBase.__str__ / __repr__
    assert str(obj) == str(p)
    assert repr(obj) == f"JsonFile({p})"
    assert isinstance(obj, FileBase)


def test_npz_loader_returns_dict(tmp_path: Path, pae_array: np.ndarray) -> None:
    p = tmp_path / "scores.npz"
    np.savez(p, predicted_aligned_error=pae_array)
    f = NpzFile(p)
    assert isinstance(f.data, dict)
    assert "predicted_aligned_error" in f.data
    np.testing.assert_array_equal(f.data["predicted_aligned_error"], pae_array)


def test_npy_loader_returns_array(tmp_path: Path, pae_array: np.ndarray) -> None:
    p = tmp_path / "scores.npy"
    np.save(p, pae_array)
    f = NpyFile(p)
    assert isinstance(f.data, np.ndarray)
    np.testing.assert_array_equal(f.data, pae_array)


def test_json_loader_returns_dict(tmp_path: Path, pae_array: np.ndarray) -> None:
    p = tmp_path / "scores.json"
    p.write_text(json.dumps({"pae": pae_array.tolist()}))
    f = JsonFile(p)
    assert isinstance(f.data, dict)
    assert "pae" in f.data


def test_pkl_loader_returns_object(tmp_path: Path, pae_array: np.ndarray) -> None:
    p = tmp_path / "scores.pkl"
    with p.open("wb") as fp:
        pickle.dump({"predicted_aligned_error": pae_array}, fp)
    f = PklFile(p)
    assert "predicted_aligned_error" in f.data
    np.testing.assert_array_equal(f.data["predicted_aligned_error"], pae_array)


def test_loader_accepts_str_path(tmp_path: Path, pae_array: np.ndarray) -> None:
    p = tmp_path / "scores.npy"
    np.save(p, pae_array)
    f = NpyFile(str(p))  # not a Path
    np.testing.assert_array_equal(f.data, pae_array)
