# Diversity-Aware Recursive Feature Multiple Kernel Learning

<p align="center">
  <a href="https://openreview.net/"><img src="https://img.shields.io/badge/OpenReview-DARFMMKL-blue"></a>
  <a href="https://proceedings.mlr.press/"><img src="https://img.shields.io/badge/PMLR-2026-darkred"></a>
  <img src="https://img.shields.io/badge/ICML-2026-success">
</p>

Official implementation of **DARFMMKL** — "Diversity-Aware Recursive Feature Multiple Kernel Learning" (ICML 2026).

Nan Cao\*, Xu Zhao\*, Teng Zhang &nbsp;(\*equal contribution) — Huazhong University of Science and Technology

> **TL;DR.** We give multiple kernel learning two missing ingredients: a *data-driven* RFM kernel family whose feature importance is learned via AGOP, and a selector that *jointly* optimizes kernel quality (CKA) and kernel diversity — solved as an LP via Glover linearization and scaled by Nyström sketching.

## Abstract

Multiple kernel learning (MKL) combines several base kernels in the spirit of ensemble learning, yet existing methods rarely model kernel diversity—a known cornerstone of ensembles—and most traditional kernels weight all features uniformly, ignoring feature-level discriminability. We address both gaps with DARFMMKL: a data-driven kernel family (Recursive Feature Machine kernels) that learns feature importance directly from data, paired with a kernel selection method that jointly optimizes diversity and quality. The resulting NP-hard binary quadratic program is reformulated via Glover linearization and continuous relaxation into a linear program, and accelerated by Nyström sketching, yielding a selector whose cost is decoupled from the sample size. We provide a covering-number generalization bound that explicitly relates kernel diversity to estimation error. Experiments on 12 benchmark datasets show that DARFMMKL consistently outperforms 9 state-of-the-art MKL methods.

## Method

DARFMMKL follows four steps:

1. Generate candidate kernels from both traditional kernels (Gaussian / polynomial) and AGOP/RFM-induced kernels whose feature-importance matrix is learned from data.
2. Estimate kernel quality (centered kernel alignment) and pairwise diversity (prediction disagreement) on a sampled validation subset, using Nyström sketching to keep the cost independent of the sample size.
3. Select a fixed-size kernel subset by solving the diversity-quality objective, an NP-hard binary quadratic program reformulated via Glover linearization and continuous relaxation into a linear program.
4. Train the downstream MKL classifier with the uniformly averaged selected kernels.

The implementation entry point is `DASMKL.py`; `main.py` is a thin convenience wrapper.

## Setup

```bash
conda create -n darfmmkl python=3.10
conda activate darfmmkl
pip install -r requirements.txt
```

Datasets are fetched in LIBSVM format through `libsvmdata`. The selection solver uses `gurobipy`, so a working Gurobi installation/license is required for the full optimization routine.

## Datasets

Twelve LIBSVM benchmark datasets are used, spanning sizes from 128 to 49,749 samples and dimensions from 6 to 300:

`mux6-ver1`, `sonar`, `heart`, `semeion`, `ionosphere`, `breast-cancer`, `australian`, `diabetes`, `german.numer`, `splice`, `a8a`, `w7a`.

All features are normalized to `[0, 1]`. They are downloaded on first use via `libsvmdata`; no manual placement is needed.

## Usage

Run the default demo on `breast-cancer`:

```bash
python main.py            # or: python DASMKL.py
python main.py small      # small-dataset demo mode
```

The script writes a timestamped console log and a result file (e.g. `log-YYYYMMDD_HHMMSS.txt`, `results_rfm_YYYYMMDD_HHMMSS.txt`).

### Hyperparameters

The experiments select hyperparameters by 3-fold cross-validation (repeated 10 times) over the following grid; the demo sweeps it internally rather than via command-line flags:

| Hyperparameter | Symbol | Search grid | Recommended default |
|---|---|---|---|
| Candidate pool size | `M` | {100, 200, 300, 400} | 400 |
| Selected kernels | `m` | {10, 20, ..., 100} | 10 |
| Diversity-quality trade-off | `λ` | {M/16, M/8, M/4, M/2} | M/4 |
| SVM regularization | `C` | {1, 10, 100, 1000} | 1000 |
| Nyström sketch size | `s` | {10, 20, ..., 100} | 90 |
| RFM bandwidth scale | `β` | {0.25, 0.5, ..., 10} | tuned per dataset |

The candidate pool mixes traditional kernels (Gaussian with `τ ∈ [2⁻⁸, 2⁸]`, polynomial with degree `∈ [1, 20]`) and RFM-Gaussian / RFM-Laplacian kernels whose feature-importance matrix `W` is learned via AGOP.

## Project Structure

```text
.
├── DASMKL.py             # Main DARFMMKL pipeline (kernel generation, selection, evaluation)
├── ablation_study.py     # Kernel-family ablation script
├── main.py               # Thin wrapper around DASMKL.py
├── rfm/                  # Recursive Feature Machine utilities (AGOP, kernels, solver)
├── requirements.txt
└── LICENSE
```

## Citation

```bibtex
@inproceedings{cao2026darfmmkl,
  title     = {Diversity-Aware Recursive Feature Multiple Kernel Learning},
  author    = {Cao, Nan and Zhao, Xu and Zhang, Teng},
  booktitle = {Proceedings of the 43rd International Conference on Machine Learning},
  series    = {Proceedings of Machine Learning Research},
  volume    = {306},
  year      = {2026},
  publisher = {PMLR}
}
```

## Acknowledgement

The RFM kernel family draws on the AGOP / Recursive Feature Machine framework of Radhakrishnan et al. (*Science*, 2024).

## TODO

- [ ] Replace the OpenReview and PMLR badge links with the final paper URLs once available
- [ ] Add an arXiv preprint link
- [ ] Add ICML 2026 slides / poster
- [ ] Update the BibTeX with the final volume and page numbers after the PMLR proceedings are published
- [ ] Expand dataset and reproduction documentation
