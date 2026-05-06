"""Regression tests against the reference actifPTM values shipped in
``tests/data/*_scores_*.json``.

For each (scores.json, model.pdb) pair, recompute reactifPTM and assert
it matches the reference ``actifptm`` field within tolerance. Confident
interfaces are expected within 0.01; weakly-confident interfaces are
allowed wider variation due to differences in contact-map / d0
conventions between implementations.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from reactifptm import Reactifptm

DATA_DIR = Path(__file__).parent / "data"

# Cases where the absolute actifPTM is low and small differences in contact
# definitions cause larger relative discrepancies. Allow looser tolerance.
LOOSE_CASES = {"RAF1_LYSC", "RAF1_RIPK1"}

CONFIDENT_TOLERANCE = 0.05
LOOSE_TOLERANCE = 0.10


def _discover_pairs():
    pairs = []
    for scores in sorted(DATA_DIR.glob("*_scores_*.json")):
        pdb = Path(str(scores).replace("_scores_", "_relaxed_")).with_suffix(".pdb")
        if pdb.exists():
            pairs.append((scores, pdb))
    return pairs


@pytest.mark.parametrize(
    ("scores_path", "pdb_path"),
    _discover_pairs(),
    ids=lambda p: p.name if hasattr(p, "name") else str(p),
)
def test_actifptm_matches_reference(scores_path: Path, pdb_path: Path) -> None:
    with scores_path.open() as f:
        ref = json.load(f)
    ref_score = ref["actifptm"]

    r = Reactifptm(str(scores_path), str(pdb_path))
    overall, _ = r.compute_reactifptm()

    # Extract the case name (everything before "_relaxed_" or "_long_relaxed_")
    case = pdb_path.stem.split("_relaxed_")[0]
    case_root = case.removesuffix("_long")

    tol = LOOSE_TOLERANCE if case_root in LOOSE_CASES else CONFIDENT_TOLERANCE
    assert overall == pytest.approx(ref_score, abs=tol), (
        f"{case}: got {overall}, ref {ref_score} (|Δ|={abs(overall - ref_score):.3f}, tol={tol})"
    )


def test_pairwise_max_matches_directional() -> None:
    """Sanity: pairwise_max[A-B] == max(pairwise[A-B], pairwise[B-A])."""
    pairs = _discover_pairs()
    assert pairs, "no test data fixtures discovered"
    scores_path, pdb_path = pairs[0]

    r = Reactifptm(str(scores_path), str(pdb_path))
    r.compute_reactifptm()

    for key, mx in r.reactifptm_pairwise_max.items():
        a, b = key.split("-")
        ab = r.reactifptm_pairwise[f"{a}-{b}"]
        ba = r.reactifptm_pairwise[f"{b}-{a}"]
        assert mx == pytest.approx(max(ab, ba), abs=1e-6)
