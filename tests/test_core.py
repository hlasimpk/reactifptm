"""Unit tests for the Reactifptm class."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from reactifptm import Reactifptm

# --- construction ---------------------------------------------------------


def test_init_populates_attributes(sample_pair: tuple[Path, Path]) -> None:
    scores, pdb = sample_pair
    r = Reactifptm(str(scores), str(pdb))

    assert isinstance(r.pae_matrix, np.ndarray)
    assert r.pae_matrix.ndim == 2
    assert r.pae_matrix.shape[0] == r.pae_matrix.shape[1]

    assert isinstance(r.contact_map, np.ndarray)
    assert r.contact_map.shape == r.pae_matrix.shape

    assert isinstance(r.asym_id, np.ndarray)
    assert r.asym_id.shape[0] == r.pae_matrix.shape[0]

    assert isinstance(r.chain_lengths, list)
    assert sum(r.chain_lengths) == r.pae_matrix.shape[0]
    assert all(n > 0 for n in r.chain_lengths)

    # results not yet computed
    assert r.reactifptm is None
    assert r.reactifptm_pairwise is None
    assert r.reactifptm_pairwise_max is None


def test_threshold_changes_contact_map(sample_pair: tuple[Path, Path]) -> None:
    scores, pdb = sample_pair
    tight = Reactifptm(str(scores), str(pdb), threshold=4.0)
    loose = Reactifptm(str(scores), str(pdb), threshold=12.0)
    assert loose.contact_map.sum() > tight.contact_map.sum()


# --- parse_pae_file -------------------------------------------------------


def test_parse_pae_unsupported_format_raises(tmp_path: Path, sample_pair) -> None:
    scores, pdb = sample_pair
    r = Reactifptm(str(scores), str(pdb))
    bogus = tmp_path / "scores.txt"
    bogus.write_text("nope")
    with pytest.raises(ValueError, match="Unsupported PAE file format"):
        r.parse_pae_file(bogus)


def test_parse_pae_dict_without_pae_key_raises(tmp_path: Path, sample_pair) -> None:
    scores, pdb = sample_pair
    r = Reactifptm(str(scores), str(pdb))
    p = tmp_path / "x.json"
    p.write_text(json.dumps({"unrelated": [1, 2, 3]}))
    with pytest.raises(ValueError, match="PAE matrix not found"):
        r.parse_pae_file(p)


def test_parse_pae_handles_npy(tmp_path: Path, sample_pair) -> None:
    scores, pdb = sample_pair
    r = Reactifptm(str(scores), str(pdb))
    arr = np.arange(25, dtype=float).reshape(5, 5)
    p = tmp_path / "scores.npy"
    np.save(p, arr)
    out = r.parse_pae_file(p)
    np.testing.assert_array_equal(out, arr)


def test_parse_pae_handles_pae_key(tmp_path: Path, sample_pair) -> None:
    scores, pdb = sample_pair
    r = Reactifptm(str(scores), str(pdb))
    arr = [[1.0, 2.0], [3.0, 4.0]]
    p = tmp_path / "x.json"
    p.write_text(json.dumps({"pae": arr}))
    out = r.parse_pae_file(p)
    np.testing.assert_array_equal(out, np.asarray(arr))


# --- _score ---------------------------------------------------------------


def test_score_returns_nan_when_no_contacts() -> None:
    r = object.__new__(Reactifptm)  # don't run __init__
    n = 5
    tm = np.ones((n, n))
    pair_mask = np.ones((n, n))
    asym = np.array([0, 0, 0, 1, 1])
    contacts = np.zeros((n, n))  # nothing in contact
    s = r._score(tm, pair_mask, asym, interface=True, contacts=contacts)
    assert np.isnan(s)


def test_score_interface_only_excludes_intra_chain() -> None:
    r = object.__new__(Reactifptm)
    n = 4
    asym = np.array([0, 0, 1, 1])
    # tm = identity-ish: only intra-chain (i,i) and (j,j) are 1
    tm = np.eye(n)
    pair_mask = np.ones((n, n))
    contacts = np.ones((n, n))
    # With interface=True, intra-chain entries are masked off so tm·normed
    # ends up zero everywhere, but mask.sum() > 0 so we still get a number.
    s = r._score(tm, pair_mask, asym, interface=True, contacts=contacts)
    assert s == pytest.approx(0.0)


def test_score_row_chain_picks_correct_max() -> None:
    r = object.__new__(Reactifptm)
    n = 4
    asym = np.array([0, 0, 1, 1])
    # Construct a tm matrix where chain 0 rows score higher than chain 1 rows
    tm = np.array(
        [
            [0.0, 0.0, 0.9, 0.9],
            [0.0, 0.0, 0.8, 0.8],
            [0.5, 0.5, 0.0, 0.0],
            [0.4, 0.4, 0.0, 0.0],
        ]
    )
    pair_mask = np.ones((n, n))
    contacts = np.ones((n, n))
    s_a = r._score(tm, pair_mask, asym, interface=True, contacts=contacts, row_chain=0)
    s_b = r._score(tm, pair_mask, asym, interface=True, contacts=contacts, row_chain=1)
    assert s_a > s_b


# --- compute_reactifptm ---------------------------------------------------


def test_compute_returns_overall_and_pairwise(sample_pair: tuple[Path, Path]) -> None:
    scores, pdb = sample_pair
    r = Reactifptm(str(scores), str(pdb))
    overall, pairwise = r.compute_reactifptm()

    assert isinstance(overall, float)
    assert 0.0 <= overall <= 1.0
    assert isinstance(pairwise, dict)
    # 2-chain → both directions present
    assert set(pairwise.keys()) == {"A-B", "B-A"}
    assert all(0.0 <= v <= 1.0 for v in pairwise.values())


def test_compute_sets_pairwise_max(sample_pair: tuple[Path, Path]) -> None:
    scores, pdb = sample_pair
    r = Reactifptm(str(scores), str(pdb))
    r.compute_reactifptm()
    assert set(r.reactifptm_pairwise_max.keys()) == {"A-B"}
    assert r.reactifptm_pairwise_max["A-B"] == max(
        r.reactifptm_pairwise["A-B"], r.reactifptm_pairwise["B-A"]
    )


# --- save_results ---------------------------------------------------------


def test_save_results_writes_valid_json(tmp_path: Path, sample_pair) -> None:
    scores, pdb = sample_pair
    r = Reactifptm(str(scores), str(pdb))
    r.compute_reactifptm()

    out = tmp_path / "out.json"
    r.save_results(out)
    assert out.exists()

    written = json.loads(out.read_text())
    assert set(written.keys()) == {"actifptm", "pairwise_actifptm", "pairwise_actifptm_max"}
    assert written["actifptm"] == r.reactifptm
    assert written["pairwise_actifptm"] == r.reactifptm_pairwise
    assert written["pairwise_actifptm_max"] == r.reactifptm_pairwise_max
