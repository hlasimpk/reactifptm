# reactifptm

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/hlasimpk/reactifptm/blob/main/notebooks/reactifptm.ipynb)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.22960229.svg)](https://doi.org/10.5281/zenodo.22960229)

A reimplementation of the **actifpTM** score for assessing AlphaFold-Multimer
interface confidence.

reactifpTM is a variant of ipTM that restricts the per-residue alignment score
to **interface contacts** only — pairs of residues whose representative atoms
(Cβ–Cβ for protein residues, C1′ for nucleic acids) fall under a contact
threshold. This focuses the score on the residues that actually mediate the
predicted interaction.

> **Note:** protein–protein interfaces have been validated against actifpTM
> in the accompanying benchmark; nucleic acid support (protein–nucleic acid
> and nucleic acid–nucleic acid interfaces) is implemented but not yet
> independently benchmarked.

See the [preprint](https://www.biorxiv.org/content/10.64898/2026.08.24.746624v1)
for more details.

## Try it in Colab

No installation required — upload a PAE file and a structure (mmCIF/PDB) and
run reactifpTM directly in your browser:

[Open the reactifptm Colab notebook](https://colab.research.google.com/github/hlasimpk/reactifptm/blob/main/notebooks/reactifptm.ipynb)

## Installation

```bash
pip install reactifptm
```

Or, from a clone:

```bash
git clone https://github.com/adamsimpkin/reactifptm
cd reactifptm
pip install -e .
```

## Quickstart

### Command line

```bash
reactifptm scores.json model.pdb -o results.json
```

`scores.json` is a PAE file (any of `.json`, `.npz`,
`.npy`, or `.pkl` containing the predicted aligned error matrix). `model.pdb`
is the corresponding predicted structure (`.pdb` or `.cif`).

Output:

```
reactifpTM: 0.951

pairwise reactifpTM (directional):
  A-B: 0.951
  B-A: 0.940

pairwise reactifpTM (max of both directions):
  A-B: 0.951
```

### Python

```python
from reactifptm import Reactifptm

r = Reactifptm("scores.json", "model.pdb")
overall, pairwise = r.compute_reactifptm()

print(overall)  # global reactifpTM
print(pairwise)  # {"A-B": 0.951, "B-A": 0.940}
print(r.reactifptm_pairwise_max)  # {"A-B": 0.951}

r.save_results("results.json")
```

## Output

- **`reactifptm`** — the global reactifpTM score, computed over interface
  contacts across the entire complex.
- **`reactifptm_pairwise`** — directional pairwise scores. `A-B` is the
  best-aligned residue from chain A's perspective onto its partner; `B-A`
  is the same from chain B's side. Useful for diagnosing which side of an
  interface dominates the score.
- **`reactifptm_pairwise_max`** — per unordered pair, the max of the two
  directional scores. This is what you'd typically compare against
  reference single-direction implementations.

## Options

| Flag | Default | Description |
|---|---|---|
| `-t`, `--threshold` | `8.0` | Contact threshold in Ångström (Cβ–Cβ for protein residues, C1′ for nucleic acids) |
| `-o`, `--output` | — | Path to write JSON results |

## Development

```bash
pip install -e ".[dev]"
pytest
```

The test suite verifies reactifPTM values against reference `actifptm`
fields shipped in `tests/data/`.

## License

MIT — see [LICENSE](LICENSE).
