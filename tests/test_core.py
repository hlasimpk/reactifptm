"""Unit tests for the Reactifptm class."""

from __future__ import annotations

import json
from pathlib import Path

import gemmi
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


def test_compute_omits_noncontacting_pairs() -> None:
    """Chain pairs with no interface contacts are left out entirely (no NaN)."""
    r = object.__new__(Reactifptm)
    r.pae_matrix = np.zeros((3, 3))
    r.asym_id = np.array([0, 1, 2])
    # A-B touch; C is isolated from both.
    r.contact_map = np.array(
        [
            [1.0, 1.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    _, pairwise = r.compute_reactifptm()
    assert set(pairwise.keys()) == {"A-B", "B-A"}
    assert set(r.reactifptm_pairwise_max.keys()) == {"A-B"}
    assert not any(np.isnan(v) for v in pairwise.values())


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


# --- chain / residue classification (ligand & nucleic-acid prefiltering) --


def _residue(name: str, atom_names: list[str]) -> gemmi.Residue:
    """Build a minimal gemmi residue with the named atoms at the origin."""
    res = gemmi.Residue()
    res.name = name
    for atom_name in atom_names:
        atom = gemmi.Atom()
        atom.name = atom_name
        atom.pos = gemmi.Position(0.0, 0.0, 0.0)
        res.add_atom(atom)
    return res


def _chain(name: str, residues: list[gemmi.Residue]) -> gemmi.Chain:
    chain = gemmi.Chain(name)
    for res in residues:
        chain.add_residue(res)
    return chain


def test_classify_chain_protein() -> None:
    chain = _chain("A", [_residue("ALA", ["CA", "CB"]), _residue("GLY", ["CA"])])
    assert Reactifptm._classify_chain(chain) == "protein"


def test_classify_chain_nucleic_keeps_modified() -> None:
    # A standard DC alongside a modified 6MA -> still a nucleic-acid chain.
    chain = _chain("C", [_residue("DC", ["C1'"]), _residue("6MA", ["C1'", "N9"])])
    assert Reactifptm._classify_chain(chain) == "nucleic"


def test_classify_chain_atp_is_ligand() -> None:
    # ATP carries a ribose C1' but no standard polymer monomer -> ligand chain.
    chain = _chain("D", [_residue("ATP", ["PA", "C1'", "N9"])])
    assert Reactifptm._classify_chain(chain) == "ligand"


def test_residue_atoms_protein_contact_cb_frame_ca() -> None:
    res = _residue("ALA", ["N", "CA", "C", "O", "CB"])
    contact, frame = Reactifptm._residue_atoms(res, "protein")
    assert contact.name == "CB"
    assert frame.name == "CA"


def test_residue_atoms_protein_falls_back_to_ca() -> None:
    res = _residue("GLY", ["N", "CA", "C", "O"])
    contact, frame = Reactifptm._residue_atoms(res, "protein")
    assert contact.name == "CA" and frame.name == "CA"


def test_residue_atoms_nucleic_uses_c1prime() -> None:
    res = _residue("6MA", ["P", "OP1", "C1'", "N9"])
    contact, frame = Reactifptm._residue_atoms(res, "nucleic")
    assert contact.name == "C1'" and frame.name == "C1'"


def test_residue_atoms_ligand_chain_returns_none() -> None:
    res = _residue("ATP", ["PA", "C1'", "N9"])
    assert Reactifptm._residue_atoms(res, "ligand") == (None, None)


# --- _align_pae (PAE reconciliation after dropping tokens) ----------------
# token_plan entries are (n_tokens, keep, frame_offset).


def test_align_pae_per_residue_passthrough() -> None:
    r = object.__new__(Reactifptm)
    pae = np.arange(9.0).reshape(3, 3)
    plan = [(1, True, 0)] * 3  # all standard, all kept -> identity
    np.testing.assert_array_equal(r._align_pae(pae, plan), pae)


def test_align_pae_drops_single_token_ligand() -> None:
    r = object.__new__(Reactifptm)
    pae = np.arange(9.0).reshape(3, 3)
    plan = [(1, True, 0), (1, False, 0), (1, True, 0)]  # drop middle residue
    out = r._align_pae(pae, plan)
    idx = [0, 2]
    assert out.shape == (2, 2)
    np.testing.assert_array_equal(out, pae[np.ix_(idx, idx)])


def test_align_pae_reconstructs_per_atom_tokens() -> None:
    """Protein (1 tok) + 3-atom ligand (drop) + 3-atom modified residue (keep)."""
    r = object.__new__(Reactifptm)
    pae = np.arange(49.0).reshape(7, 7)
    # res0: 1 token kept; res1: 3 atom-tokens dropped; res2: 3 atom-tokens,
    # representing atom at offset 1 -> token index 4 + 1 = 5.
    plan = [(1, True, 0), (3, False, 0), (3, True, 1)]
    out = r._align_pae(pae, plan)
    idx = [0, 5]
    assert out.shape == (2, 2)
    np.testing.assert_array_equal(out, pae[np.ix_(idx, idx)])


def test_align_pae_token_metadata_fallback() -> None:
    """When reconstruction can't size the PAE, fall back to token ids."""
    r = object.__new__(Reactifptm)
    r._token_chain_ids = ["A", "A", "B", "B", "B"]
    r._token_res_ids = [1, 2, 1, 1, 1]
    pae = np.arange(25.0).reshape(5, 5)
    # Deliberately wrong token counts so reconstruction (sum=3) != 5.
    plan = [(1, True, 0), (1, True, 0), (1, False, 0)]
    out = r._align_pae(pae, plan)
    idx = [0, 1]
    assert out.shape == (2, 2)
    np.testing.assert_array_equal(out, pae[np.ix_(idx, idx)])


def test_align_pae_unreconcilable_raises() -> None:
    r = object.__new__(Reactifptm)
    pae = np.zeros((5, 5))  # 3 residues, 3 tokens, no metadata -> 5 unexplained
    plan = [(1, True, 0), (1, True, 0), (1, True, 0)]
    with pytest.raises(ValueError, match="no corresponding coordinates"):
        r._align_pae(pae, plan)
