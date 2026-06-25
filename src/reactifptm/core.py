"""Core reactifPTM scoring."""

from __future__ import annotations

import json
import string
from pathlib import Path
from typing import Union

import gemmi
import numpy as np

from .file_handlers import FileTypes, JsonFile, NpyFile, NpzFile, PklFile

PathLike = Union[str, Path]

# Canonical (AlphaFold3-style) residues that occupy a single PAE token. Any
# residue not in this set is tokenised per-atom in the PAE (modified residues
# and ligands), which matters when collapsing the PAE back to one row per
# residue. UNK / N / DN are the unknown-monomer codes for each polymer type.
STANDARD_AMINO_ACIDS = frozenset(
    {
        "ALA",
        "ARG",
        "ASN",
        "ASP",
        "CYS",
        "GLN",
        "GLU",
        "GLY",
        "HIS",
        "ILE",
        "LEU",
        "LYS",
        "MET",
        "PHE",
        "PRO",
        "SER",
        "THR",
        "TRP",
        "TYR",
        "VAL",
        "UNK",
    }
)
STANDARD_RNA = frozenset({"A", "C", "G", "U", "N"})
STANDARD_DNA = frozenset({"DA", "DC", "DG", "DT", "DN"})
STANDARD_NUCLEOTIDES = STANDARD_RNA | STANDARD_DNA
STANDARD_RESIDUES = STANDARD_AMINO_ACIDS | STANDARD_NUCLEOTIDES


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
        contact_map, asym_id, chain_lengths, token_plan = self.parse_input_model(threshold)

        # Drop ligand/ion/water tokens from the PAE so it lines up with the
        # filtered (protein + nucleic-acid) contact map and chain assignments.
        self.pae_matrix = self._align_pae(self.pae_matrix, token_plan)

        self.contact_map = contact_map
        self.asym_id = asym_id
        self.chain_lengths = chain_lengths

        self.reactifptm: float | None = None
        self.reactifptm_pairwise: dict[str, float] | None = None
        self.reactifptm_pairwise_max: dict[str, float] | None = None

    def parse_pae_file(self, pae_path: PathLike) -> np.ndarray:
        """Parse a PAE matrix from one of several supported formats.

        If the source carries per-token chain/residue ids (e.g. an AlphaFold3
        ``*_full_data_*.json``), they are stashed on the instance so that
        per-atom ligand tokens can be collapsed and dropped during alignment.
        """
        self._token_chain_ids = None
        self._token_res_ids = None

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
            self._token_chain_ids = file_.data.get("token_chain_ids")
            self._token_res_ids = file_.data.get("token_res_ids")
            if "predicted_aligned_error" in file_.data:
                return np.asarray(file_.data["predicted_aligned_error"])
            elif "pae" in file_.data:
                return np.asarray(file_.data["pae"])

        raise ValueError(f"PAE matrix not found in file: {pae_path}")

    @staticmethod
    def _classify_chain(chain: gemmi.Chain) -> str:
        """Classify a chain as ``"protein"``, ``"nucleic"`` or ``"ligand"``.

        A chain is treated as a polymer if it contains at least one *standard*
        amino acid or nucleotide; everything else (a lone ATP, a metal ion, a
        small molecule) is a ligand. Deciding at the chain level is what lets us
        keep a modified nucleotide such as 6MA - which sits inside a nucleic-acid
        chain - while dropping an ATP ligand, even though both carry a C1' atom.
        """
        names = {residue.name for residue in chain}
        if names & STANDARD_AMINO_ACIDS:
            return "protein"
        if names & STANDARD_NUCLEOTIDES:
            return "nucleic"
        return "ligand"

    @staticmethod
    def _residue_atoms(residue: gemmi.Residue, chain_kind: str):
        """Return ``(contact_atom, frame_atom)`` for a residue in a polymer chain.

        ``contact_atom`` positions the residue in the contact map (Cb with a Ca
        fallback for proteins, C1' for nucleic acids); ``frame_atom`` is the atom
        whose PAE token represents the residue when it is tokenised per-atom (Ca
        for proteins, C1' for nucleic acids). Returns ``(None, None)`` for ligand
        chains or residues lacking the relevant atom, which drops them.
        """
        if chain_kind == "protein":
            contact = residue.find_atom("CB", "*") or residue.find_atom("CA", "*")
            frame = residue.find_atom("CA", "*") or residue.find_atom("CB", "*")
            return contact, frame
        if chain_kind == "nucleic":
            # Glycosidic anchor; some files spell the prime as '*'.
            c1 = residue.find_atom("C1'", "*") or residue.find_atom("C1*", "*")
            return c1, c1
        return None, None

    def parse_input_model(self, threshold: float = 8.0):
        """Compute the contact map, chain assignments and PAE token plan.

        Proteins are represented by their Cb (Ca fallback) and nucleic acids by
        their C1' atom; water, ions and ligands are dropped. Each non-water
        residue also contributes an entry to ``token_plan`` describing how it
        maps onto the PAE so the matrix can be realigned (see :meth:`_align_pae`).

        Args:
            threshold: Distance threshold in Angstroms (default 8.0).

        Returns:
            contact_map: ``[N, N]`` binary contact map over kept residues.
            asym_id:     ``[N]`` integer array of chain assignments.
            chain_lengths: list of residue counts per chain.
            token_plan: list of ``(n_tokens, keep, frame_offset)`` per non-water
                residue (structure order). ``n_tokens`` is 1 for standard
                residues and the atom count otherwise; ``frame_offset`` is the
                index of the representing atom within an atom-tokenised residue.
        """
        model = self.struct[0]
        chain_kinds = {chain.name: self._classify_chain(chain) for chain in model}

        coords = []
        token_plan: list[tuple[int, bool, int]] = []
        chain_lengths: list[int] = []
        for chain in model:
            kind = chain_kinds[chain.name]
            residue_count = 0
            for residue in chain:
                if residue.is_water():
                    continue
                atoms = list(residue)
                n_tokens = 1 if residue.name in STANDARD_RESIDUES else len(atoms)
                contact_atom, frame_atom = self._residue_atoms(residue, kind)
                keep = contact_atom is not None
                frame_offset = 0
                if keep:
                    residue_count += 1
                    coords.append([contact_atom.pos.x, contact_atom.pos.y, contact_atom.pos.z])
                    # Only atom-tokenised (non-standard) residues need an offset
                    # into their token block; standard residues are one token.
                    if n_tokens > 1 and frame_atom is not None:
                        frame_offset = next(
                            (i for i, a in enumerate(atoms) if a.name == frame_atom.name), 0
                        )
                token_plan.append((n_tokens, keep, frame_offset))
            if residue_count > 0:
                chain_lengths.append(residue_count)

        coords = np.array(coords)
        diff = coords[:, None, :] - coords[None, :, :]
        distances = np.sqrt(np.sum(diff**2, axis=-1))
        contact_map = (distances < threshold).astype(float)

        asym_id = np.concatenate([np.full(length, i) for i, length in enumerate(chain_lengths)])
        return contact_map, asym_id, chain_lengths, token_plan

    def _align_pae(self, pae: np.ndarray, token_plan: list[tuple[int, bool, int]]) -> np.ndarray:
        """Filter the PAE matrix down to one row/column per kept residue.

        Tries, in order:

        * **Reconstruction** - replay the per-residue token plan (standard
          residues = 1 token, others = 1 token per atom). If the running total
          equals the PAE size, the layout is understood exactly, so the
          representing token of each kept residue is selected and ligand /
          per-atom tokens are dropped. This covers AlphaFold3-style mixed
          granularity as well as plain per-residue PAEs (AF2/ColabFold).
        * **Per-residue** - ``n_pae == n_residues`` -> drop the ligand rows.
        * **Already filtered** - ``n_pae == n_kept`` -> returned unchanged.
        * **Token metadata** - collapse ``token_chain_ids``/``token_res_ids``
          runs when their length matches the PAE.

        If none apply, a :class:`ValueError` is raised rather than risk a
        silently misaligned score.
        """
        n_pae = int(pae.shape[0])
        n_residues = len(token_plan)
        n_kept = sum(1 for _, keep, _ in token_plan if keep)

        # 1. Reconstruct the token layout from the structure.
        cursor = 0
        selected: list[int] = []
        for n_tokens, keep, frame_offset in token_plan:
            if keep:
                selected.append(cursor + frame_offset)
            cursor += n_tokens
        if cursor == n_pae:
            return pae[np.ix_(selected, selected)]

        # 2. One token per non-water residue.
        if n_pae == n_residues:
            keep_idx = [i for i, (_, keep, _) in enumerate(token_plan) if keep]
            return pae[np.ix_(keep_idx, keep_idx)]

        # 3. PAE already restricted to the kept residues.
        if n_pae == n_kept:
            return pae

        # 4. Per-token chain/residue metadata, if it matches the PAE.
        meta = self._select_polymer_tokens(token_plan, n_pae)
        if meta is not None:
            return pae[np.ix_(meta, meta)]

        extra = n_pae - n_kept
        if extra > 0:
            raise ValueError(
                f"PAE has {n_pae} tokens but the structure has {n_kept} residues. "
                f"The PAE contains {extra} token(s) with no corresponding coordinates — "
                f"this usually means the input sequence contained non-standard residues "
                f"(e.g. 'X') that AlphaFold could not place. Remove or replace those "
                f"residues in your input sequence and re-run the prediction."
            )

        raise ValueError(
            "PAE matrix size cannot be reconciled with the structure: "
            f"PAE has {n_pae} tokens, but the structure has {n_residues} "
            f"non-water residues ({n_kept} kept after dropping ligands/ions, "
            f"reconstructed token total {cursor}). The PAE token layout could "
            "not be determined; check that the structure and PAE come from the "
            "same prediction."
        )

    def _select_polymer_tokens(
        self, token_plan: list[tuple[int, bool, int]], n_pae: int
    ) -> list[int] | None:
        """Map per-token PAE metadata to one token per kept residue.

        Uses ``token_chain_ids``/``token_res_ids`` (when present and the same
        length as the PAE) to collapse runs of tokens sharing a (chain, residue)
        into a residue group, then selects the representing token of each kept
        group. Returns ``None`` if the metadata is missing or does not line up
        with the structure's residues.
        """
        chain_ids = getattr(self, "_token_chain_ids", None)
        res_ids = getattr(self, "_token_res_ids", None)
        if chain_ids is None or res_ids is None:
            return None
        if len(chain_ids) != n_pae or len(res_ids) != n_pae:
            return None

        groups: list[tuple[tuple[object, object], list[int]]] = []
        for idx, key in enumerate(zip(chain_ids, res_ids)):
            if groups and groups[-1][0] == key:
                groups[-1][1].append(idx)
            else:
                groups.append((key, [idx]))

        if len(groups) != len(token_plan):
            return None

        selected: list[int] = []
        for (_, tokens), (_, keep, frame_offset) in zip(groups, token_plan):
            if keep:
                offset = frame_offset if frame_offset < len(tokens) else 0
                selected.append(tokens[offset])
        return selected

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
                # Pairs with no interface contacts score as NaN; omit them so
                # the output only lists chain pairs that actually interface.
                if np.isnan(score):
                    continue
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

        self.reactifptm = 0.0 if np.isnan(reactifptm) else round(float(reactifptm), 3)
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
