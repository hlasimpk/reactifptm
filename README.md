# reactifptm

A reimplementation of the **actifpTM** score for assessing AlphaFold-Multimer
interface confidence.

actifpTM is a variant of ipTM that restricts the per-residue alignment score
to **interface contacts** only — pairs of residues whose Cβ–Cβ distance falls
under a contact threshold. This focuses the score on the residues that
actually mediate the predicted interaction.

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

print(overall)                       # global reactifpTM
print(pairwise)                      # {"A-B": 0.951, "B-A": 0.940}
print(r.reactifptm_pairwise_max)     # {"A-B": 0.951}

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
| `-t`, `--threshold` | `8.0` | Cβ–Cβ contact threshold in Ångström |
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
