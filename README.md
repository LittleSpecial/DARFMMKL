# Diversity-Aware Recursive Feature Multiple Kernel Learning

<p align="center">
  <a href="https://openreview.net/"><img src="https://img.shields.io/badge/OpenReview-DARFMMKL-blue"></a>
  <a href="https://proceedings.mlr.press/"><img src="https://img.shields.io/badge/PMLR-2026-darkred"></a>
  <img src="https://img.shields.io/badge/ICML-2026-success">
</p>

Official implementation of **"Diversity-Aware Recursive Feature Multiple Kernel Learning"** (ICML 2026).

**Nan Cao\*, Xu Zhao\*, Teng Zhang** &nbsp;(\*equal contribution)  
Huazhong University of Science and Technology

> **TL;DR.** DARFMMKL builds a data-driven RFM kernel family using AGOP-based feature importance, then selects a compact and diverse subset of kernels by jointly optimizing kernel quality and diversity.

<p align="center">
  <img src="assets/darfmmkl_overview_clean.png" width="90%">
</p>

## Abstract

Multiple kernel learning (MKL) combines several base kernels in the spirit of ensemble learning, yet existing methods rarely model kernel diversity, a known cornerstone of ensembles, and most traditional kernels weight all features uniformly. We propose **DARFMMKL**, which augments MKL with Recursive Feature Machine (RFM) kernels whose feature importance is learned from data through average gradient outer products (AGOP), and a diversity-aware selector that balances kernel quality with pairwise kernel disagreement. The resulting binary quadratic selection problem is reformulated with Glover linearization and continuous relaxation, and is accelerated with Nyström sketching. Experiments on 12 benchmark datasets show that DARFMMKL consistently improves over representative MKL baselines.

## Method

DARFMMKL follows four steps:

1. Generate candidate kernels from both traditional kernels and AGOP/RFM-induced kernels.
2. Estimate kernel quality and diversity on a sampled validation subset.
3. Select a fixed-size kernel subset by solving the relaxed diversity-quality objective.
4. Train the downstream MKL classifier with the selected kernels.

The implementation entry point is `DASMKL.py`; `main.py` is provided as a convenience wrapper.

## Setup

```bash
conda create -n darfmmkl python=3.10
conda activate darfmmkl
pip install -r requirements.txt
```

The demo uses LIBSVM-format datasets fetched through `libsvmdata`. The selection solver depends on `gurobipy`, so a working Gurobi installation/license is required for the full optimization routine.

## Run

Run the default demo on `breast-cancer_scale`:

```bash
python main.py
```

Equivalent direct entry point:

```bash
python DASMKL.py
```

The script writes a timestamped console log and result file, for example `log-YYYYMMDD_HHMMSS.txt` and `results_rfm_YYYYMMDD_HHMMSS.txt`.

To run the small demo mode explicitly:

```bash
python main.py small
```

## Project Structure

```text
.
├── DASMKL.py             # Main DARFMMKL demo and optimization pipeline
├── ablation_study.py     # Kernel-family ablation script
├── main.py               # Thin wrapper around DASMKL.py
├── rfm/                  # Recursive Feature Machine utilities
├── assets/               # Method overview figure
└── requirements.txt
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
