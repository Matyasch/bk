# Integrating Background Knowledge for Scalable Causal Discovery

Paper: https://arxiv.org/abs/2607.10456 \
Website: https://matyasch.github.io/bk

This is the official code repository for **Integrating Background Knowledge for Scalable Causal Discovery** (NeurIPS 2026) by Mátyás Schubert, Theofanis Aslanidis, Tom Claassen and Sara Magliacane.

> [!NOTE]
> This branch contains a minimal and portable implementation of the algorithms. Check out the [neurips2026 branch](https://github.com/Matyasch/bk/tree/neurips2026) to reproduce all results presented in the paper.

## Repository Structure

All algorithms are implemented in the [`algorithms/`](algorithms/) directory:

- [`algorithms/pc.py`](algorithms/pc.py): `pc`, the PC-BK algorithm
- [`algorithms/snap.py`](algorithms/snap.py): `snap`, the SNAP($\infty$)-BK algorithm
- [`algorithms/load.py`](algorithms/load.py): `load`, the LOAD-BK algorithm
- [`algorithms/mb_by_mb.py`](algorithms/mb_by_mb.py): `mb_by_mb_plus`, the MB-by-MB-BK algorithm, extended with the checks needed to return adjustment sets for a pair of targets
- [`algorithms/ldecc.py`](algorithms/ldecc.py): `ldecc_plus`, the LDECC-BK algorithm, extended with the checks needed to return adjustment sets for a pair of targets
- [`algorithms/orient.py`](algorithms/orient.py): Shared orientation steps (v-structures, Meek's rules, and applying BK to a CPDAG), not an algorithm itself

Each algorithm takes the data matrix, a CI test `ci_test(x, y, S)` that returns a p-value, a significance level `alpha`, and the background knowledge as a matrix `bk` where:

- `bk[i, j] = -1` and `bk[j, i] = -1` indicates adjacency: `i - j`.
- `bk[i, j] = -1` and `bk[j, i] =  1` indicates orientation: `i -> j`.
- `bk[i, j] =  1` and `bk[j, i] =  1` indicates gap: `i -/- j`.
- `bk[i, j] =  0` and `bk[j, i] =  0` indicates no knowledge about `i` and `j`.

The local algorithms `snap`, `load`, `mb_by_mb_plus` and `ldecc_plus` also take the `targets` (a pair of nodes for `load`, `mb_by_mb_plus` and `ldecc_plus`).

All dependencies are listed in [`requirements.txt`](requirements.txt):

```bash
pip install -r requirements.txt
```

## Citation

```bibtex
@inproceedings{schubert2026integrating,
  title={Integrating Background Knowledge for Scalable Causal Discovery},
  author={M{\'a}ty{\'a}s Schubert and Theofanis Aslanidis and Tom Claassen and Sara Magliacane},
  booktitle={The Fortieth Annual Conference on Neural Information Processing Systems},
  year={2026}
}
```
