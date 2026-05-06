"""Core reactifPTM scoring."""

from __future__ import annotations

import json
import string
from collections import OrderedDict
from pathlib import Path
from typing import Union

import gemmi
import numpy as np

from .file_handlers import FileTypes, JsonFile, NpyFile, NpzFile, PklFile

PathLike = Union[str, Path]


class Reactifptm:
    """Compute reactifPTM scores from a PAE matrix and a predicted structure.

    Args:
        pae_file: Path to the PAE file (npz, npy, json, or pickle) containing
            the predicted aligned error matrix.
        model_path: Path to the predicted structure file (.pdb or .cif).
        threshold: Distance threshold in Angstroms for defining contacts
            (default 8.0).

    Attributes:
        contact_map: Binary contact map based on Cb distances.
        asym_id: Integer array of chain assignments.
        chain_lengths: List of residue counts per chain.
        pae_matrix: Predicted aligned error matrix.
        reactifptm: Overall reactifPTM score for the complex.
        reactifptm_pairwise: Directional pairwise reactifPTM scores
            (e.g. ``{"A-B": 0.95, "B-A": 0.94}``).
        reactifptm_pairwise_max: Per-unordered-pair max of the two directions
            (e.g. ``{"A-B": 0.95}``).

    Example:
        >>> r = Reactifptm("scores.json", "model.pdb")
        >>> overall, pairwise = r.compute_reactifptm()
        >>> r.save_results("results.json")
    """

    def __init__(self, pae_file: PathLike, model_path: PathLike, threshold: float = 8.0):
        self.pae_matrix = self.parse_pae_file(pae_file)
        self.struct = gemmi.read_structure(str(model_path))
        contact_map, asym_id, chain_lengths = self.parse_input_model(threshold)

        self.contact_map = contact_map
        self.asym_id = asym_id
        self.chain_lengths = chain_lengths

        self.reactifptm: float | None = None
        self.reactifptm_pairwise: dict[str, float] | None = None
        self.reactifptm_pairwise_max: dict[str, float] | None = None

    def parse_pae_file(self, pae_path: PathLike) -> np.ndarray:
        """Parse a PAE matrix from one of several supported formats."""
        suffix = Path(pae_path).suffix[1:]
        if suffix == FileTypes.NPZ.value:
            file_ = NpzFile(pae_path)
        elif suffix == FileTypes.NPY.value:
            file_ = NpyFile(pae_path)
        elif suffix == FileTypes.JSON.value:
            file_ = JsonFile(pae_path)
        elif suffix == FileTypes.PKL.value:
            file_ = PklFile(pae_path)
        else:
            raise ValueError(
                f"Unsupported PAE file format '.{suffix}'. Supported: {FileTypes.values()}"
            )

        if isinstance(file_.data, list):
            file_.data = file_.data[0]

        if isinstance(file_.data, np.ndarray):
            return file_.data

        if isinstance(file_.data, dict):
            if "predicted_aligned_error" in file_.data:
                return np.asarray(file_.data["predicted_aligned_error"])
            elif "pae" in file_.data:
                return np.asarray(file_.data["pae"])

        raise ValueError(f"PAE matrix not found in file: {pae_path}")

    def parse_input_model(self, threshold: float = 8.0):
        """Compute Cb-Cb contact map and chain assignments from the structure.

        Args:
            threshold: Distance threshold in Angstroms (default 8.0).

        Returns:
            contact_map: ``[N, N]`` binary contact map based on Cb distances.
            asym_id:     ``[N]`` integer array of chain assignments.
            chain_lengths: list of residue counts per chain.
        """
        coords = []
        chain_residues: OrderedDict[str, int] = OrderedDict()
        for model in self.struct:
            for chain in model:
                chain_id = chain.name
                residue_count = 0
                for residue in chain:
                    if residue.is_water():
                        continue
                    cb = residue.find_atom("CB", "*") or residue.find_atom("CA", "*")
                    if cb:
                        residue_count += 1
                        coords.append([cb.pos.x, cb.pos.y, cb.pos.z])
                if residue_count > 0:
                    chain_residues[chain_id] = residue_count
            break

        coords = np.array(coords)
        diff = coords[:, None, :] - coords[None, :, :]
        distances = np.sqrt(np.sum(diff**2, axis=-1))
        contact_map = (distances < threshold).astype(float)

        chain_lengths = list(chain_residues.values())
        asym_id = np.concatenate([np.full(length, i) for i, length in enumerate(chain_lengths)])
        return contact_map, asym_id, chain_lengths

    def compute_reactifptm(self):
        """Compute the global and pairwise reactifPTM scores.

        Returns:
            (reactifptm, reactifptm_pairwise) where the second value is the
            directional pairwise dict. ``reactifptm_pairwise_max`` (the
            per-unordered-pair max of both directions) is set as an
            attribute on the instance.
        """
        num_tokens = self.pae_matrix.shape[0]
        pair_mask = np.ones((num_tokens, num_tokens), dtype=float)
        d0 = 1.24 * (max(num_tokens, 19) - 15) ** (1.0 / 3) - 1.8
        tm_matrix = 1.0 / (1 + np.square(self.pae_matrix) / d0**2)

        # Global actifPTM: single TM-score over the whole complex's interface
        # (per-row normalization spans contacts to ALL other chains).
        reactifptm = self._score(
            tm_matrix,
            pair_mask,
            self.asym_id,
            interface=True,
            contacts=self.contact_map,
        )

        chain_labels = list(string.ascii_uppercase)
        unique_chains = list(np.unique(self.asym_id))
        pairwise: dict[str, float] = {}
        for i, chain_i in enumerate(unique_chains):
            for j, chain_j in enumerate(unique_chains):
                if chain_i == chain_j:
                    continue
                mask = (self.asym_id == chain_i) | (self.asym_id == chain_j)
                (indices,) = np.where(mask)
                idx = np.ix_(indices, indices)
                # row_chain restricts the row max to chain_i so A-B and B-A
                # are asymmetric (best aligned residue from each side).
                score = self._score(
                    tm_matrix[idx],
                    pair_mask[idx],
                    self.asym_id[mask],
                    interface=True,
                    contacts=self.contact_map[idx],
                    row_chain=chain_i,
                )
                key = f"{chain_labels[i % 26]}-{chain_labels[j % 26]}"
                pairwise[key] = round(float(score), 3)

        # Per-unordered-pair max of both directions.
        pairwise_max: dict[str, float] = {}
        seen: set[tuple[str, str]] = set()
        for key, value in pairwise.items():
            a, b = key.split("-")
            unordered = tuple(sorted((a, b)))
            if unordered in seen:
                continue
            seen.add(unordered)
            other = pairwise.get(f"{b}-{a}", value)
            candidates = [v for v in (value, other) if not np.isnan(v)]
            pairwise_max[f"{unordered[0]}-{unordered[1]}"] = (
                round(max(candidates), 3) if candidates else float("nan")
            )

        self.reactifptm = round(float(reactifptm), 3)
        self.reactifptm_pairwise = pairwise
        self.reactifptm_pairwise_max = pairwise_max
        return self.reactifptm, self.reactifptm_pairwise

    def _score(self, tm_matrix, pair_mask, asym_id, interface=False, contacts=None, row_chain=None):
        """Compute the TM-score from a pre-computed tm_matrix and mask.

        If ``row_chain`` is provided, the per-residue alignment score is
        maxed only over rows belonging to that chain (i.e. directional
        score from chain ``row_chain`` to its partner).
        """
        mask = pair_mask.copy()
        if interface:
            mask *= (asym_id[:, None] != asym_id[None, :]).astype(float)
        if contacts is not None:
            mask *= contacts
        if mask.sum() == 0:
            return np.nan
        normed = mask / (1e-8 + mask.sum(axis=-1, keepdims=True))
        per_row = np.sum(tm_matrix * normed, axis=-1)
        if row_chain is not None:
            row_mask = asym_id == row_chain
            if not row_mask.any() or mask[row_mask].sum() == 0:
                return np.nan
            per_row = per_row[row_mask]
        return float(per_row.max())

    def save_results(self, output_path: PathLike) -> None:
        """Save actifPTM results to a JSON file."""
        results = {
            "actifptm": self.reactifptm,
            "pairwise_actifptm": self.reactifptm_pairwise,
            "pairwise_actifptm_max": self.reactifptm_pairwise_max,
        }
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to: {output_path}")
