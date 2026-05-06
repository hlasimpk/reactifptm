"""Shared fixtures for the test suite."""

from __future__ import annotations

from pathlib import Path

import pytest

DATA_DIR = Path(__file__).parent / "data"


@pytest.fixture
def sample_pair() -> tuple[Path, Path]:
    """A small, fast (1YCR) (scores.json, model.pdb) pair."""
    scores = DATA_DIR / "1YCR_scores_rank_001_alphafold2_multimer_v3_model_1_seed_000.json"
    pdb = DATA_DIR / "1YCR_relaxed_rank_001_alphafold2_multimer_v3_model_1_seed_000.pdb"
    if not scores.exists() or not pdb.exists():
        pytest.skip("1YCR test fixtures not available")
    return scores, pdb
