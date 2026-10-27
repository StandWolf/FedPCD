# FedPCD: Prototype-Cross Distillation for Multimodal Federated Learning

Code release accompanying the FedPCD paper.

FedPCD targets multimodal federated learning with **client-level modality
missingness**: every client permanently observes a subset of its modalities,
and different clients may observe different subsets. The method combines

- **GKD (Global Knowledge Distillation)** — logit and feature distillation
  from a frozen global teacher;
- **CPR (Cross-Modal Projection Recovery)** — directional projection heads
  trained with instance-level cross-modal recovery;
- **PCA (Prototype-Cross Alignment)** — server-side class prototypes per
  modality with prototype contrastive supervision.

No public/proxy dataset and no extra inference module are required.

> **Attribution.** This repository is a derivative work of the public
> [FedMultimodal](https://github.com/usc-sail/fed-multimodal) benchmark and
> retains its Apache-2.0 license. Files that contain modified upstream code
> carry a notice at the top. See Section 8 and [`NOTICE`](NOTICE) for the
> full list of derived files, modifications, and new components.

---

## 1. Repository layout

```
constants/                     dataset constants and class counts
dataloader/dataload_manager.py data loading, partitions, missing-modality masks
model/                         HAR encoders, fusion modules, MFCPL model
trainers/                      server/client trainers (FedAvg, FedProx, FedOpt,
                               SCAFFOLD, FedMEKT, MFCPL)
utils/                         distillation scheduler, prototype manager,
                               double-distillation and cross-modal losses
partitions/                    fixed Dirichlet client partitions
features/
  data_partitioning/           UCI-HAR partition script and shared partition manager
  feature_processing/          feature extraction for the three HAR datasets
  simulation_features/         client-level missing-modality masks
experiment/
  uci-har/  train.py           FedPCD and FL baselines (this dataset)
            train_mfcpl.py     MFCPL baseline
            train_fedmekt.py   FedMEKT baseline
  mhealth/  ...                same three entry points
  ku-har/   ...                same three entry points
scripts/                       download, preparation, and run scripts
```

The code keeps the data/model interfaces of the public
[FedMultimodal](https://github.com/usc-sail/fed-multimodal) benchmark and adds
the FedPCD components.

---

## 2. Requirements

- Python 3.9+ (tested with Python 3.10)
- PyTorch 2.x with CUDA
- A single GPU is sufficient
- `curl` or `wget` plus `unzip` for the dataset download script

```bash
pip install -r requirements.txt
```

All commands below are run from the repository root.

---

## 3. Data preparation

The three datasets are public. The raw data are **not** redistributed; the
fixed client partitions used in the paper are included under `partitions/`.

```bash
bash scripts/download_data.sh all          # or: uci-har / mhealth / ku-har
PYTHON=python bash scripts/prepare_uci_har.sh
PYTHON=python bash scripts/prepare_mhealth.sh
PYTHON=python bash scripts/prepare_ku_har.sh
```

Each preparation script writes three things under `./output`:

```
output/partition/<dataset>/...        client partitions / folds
output/feature/{acc,gyro}/<dataset>/  extracted modality features
output/simulation_feature/<dataset>/  missing-modality masks
```

All three datasets use client-level masks loaded with
`--missing_protocol clientlevel`. The files are `mmXX_alphaYY_cl.json`
(UCI-HAR) and `mmXX_cl.json` (mHealth, KU-HAR).

The paper's client partitions are fixed **Dirichlet** splits and are included
under `partitions/`, because the original one-off partition generator is not
part of the release. The preparation scripts copy them into
`output/partition/`, extract features, and build the masks.

| Dataset | Clients | Classes | Partition |
|---|---|---|---|
| UCI-HAR | 105 | 6 | Dirichlet β = 0.05 |
| mHealth | 15 per fold | 12 | Dirichlet β = 0.1 |
| KU-HAR | 63 per fold (64 in fold 5) | 8 | Dirichlet β = 0.3 |

The client-to-modality assignment used in the paper is stored in small fixed
tables (`features/simulation_features/*/fixed_masks*.json`) and applied after
the mask generators, so it does not depend on the NumPy version. For UCI-HAR
the Gyro-only / Acc-only / both-present counts are 40/35/30, 55/43/7,
48/54/3, and 50/55/0 at m = 0.5/0.7/0.8/1.0. Dev and test clients are always
evaluated with complete inputs; masks affect training clients.

To use a different data location, edit `system.cfg` or pass `--data_dir`
(training expects a directory containing both `partition/` and
`simulation_feature/`; it defaults to `./output`).

---

## 4. Reproducing the main results

`scripts/run_main.sh` selects a dataset and a method (or `all`), and loops
over the four missing ratios.

```bash
# FedPCD
bash scripts/run_main.sh uci-har fedpcd
bash scripts/run_main.sh mhealth fedpcd
bash scripts/run_main.sh ku-har  fedpcd

# All six methods (FedAvg, FedProx, FedOpt, MFCPL, FedMEKT, FedPCD)
bash scripts/run_main.sh uci-har all
```

Environment variables: `PYTHON` (interpreter), `FOLDS` (default 3), and
`LOGDIR` (default `logs/main`). Per-fold checkpoints are written under
`result/<fed_alg>/<dataset>/...`.

Each training run reports the best dev Macro-F1 checkpoint and its test
Macro-F1. Average the three runs to obtain the table below.

### Main results (Macro-F1, %, three runs)

| Method | UCI-HAR 0.5 | 0.7 | 0.8 | 1.0 | mHealth 0.5 | 0.7 | 0.8 | 1.0 | KU-HAR 0.5 | 0.7 | 0.8 | 1.0 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| FedAvg  | 77.31 | 75.67 | 74.61 | 75.39 | 54.79 | 55.59 | 44.45 | 43.74 | 57.59 | 49.72 | 56.04 | 55.49 |
| FedProx | 78.02 | 75.43 | 75.57 | 75.35 | 52.30 | 47.61 | 45.50 | 43.28 | 59.51 | 54.54 | 55.93 | 55.39 |
| FedOpt  | 74.17 | 74.88 | 69.51 | 71.13 | 44.43 | 30.47 | 32.65 | 35.78 | 15.14 | 10.39 | 15.70 | 9.86 |
| MFCPL   | 67.21 | 64.43 | 64.09 | 65.39 | 49.29 | 43.47 | 37.87 | 38.29 | 51.06 | 45.43 | 48.54 | 44.15 |
| FedMEKT | 77.33 | 73.44 | 73.49 | 74.33 | 51.36 | 43.07 | 49.78 | 43.51 | 59.68 | 52.87 | 53.44 | 49.59 |
| FedPCD  | 77.56 | 77.38 | 75.80 | 76.98 | 54.49 | 56.01 | 56.68 | 54.79 | 63.49 | 62.13 | 61.22 | 55.98 |

Small deviations are expected from GPU nondeterminism and library versions.

---

## 5. Ablations

```bash
# High-level: FedAvg vs. FedPCD without PCA vs. Full FedPCD (four ratios)
bash scripts/run_ablation.sh uci-har high
bash scripts/run_ablation.sh mhealth high
bash scripts/run_ablation.sh ku-har  high

# Fine-grained loss ablation on mHealth at m = 0.7
bash scripts/run_ablation.sh mhealth loss

# Federated sharing of the projection heads
bash scripts/run_ablation.sh uci-har proj    # m = 0.7
bash scripts/run_ablation.sh mhealth proj    # m = 1.0
```

`FedPCD without PCA` keeps GKD and CPR but disables prototype alignment
(`--leakfree_components rec`, no `--en_proto_cross`). The fine-grained
ablation disables the leak-free path and selects DD terms with
`--dd_components logit`, `feat`, `cross`, or a `+` combination.

### Evaluation with incomplete test inputs

`scripts/evaluate_incomplete_inputs.py` evaluates a saved checkpoint under
four test-input conditions: both modalities, accelerometer only, gyroscope
only, and random modality missingness. Point it at a result directory
containing `model_final_foldN.pt`:

```bash
python scripts/evaluate_incomplete_inputs.py \
    --dataset uci-har --mm 1.0 --method fedpcd --protocol clientlevel --alpha 0.05 \
    --result_dir result/fed_avg/uci-har/acc_gyro/fuse_base/<run>
```

Without `--result_dir`, the script searches `result/` using the method and
missing rate.

---

## 6. Hyperparameters

| Setting | Value |
|---|---|
| Communication rounds `R` | 200 |
| Local epochs `E` | 1 |
| Batch size | 16 |
| Optimizer | SGD, momentum 0.9, weight decay 1e-5 |
| Client participation `q` | 10% (UCI-HAR, KU-HAR), 25% (mHealth) |
| Local learning rate | 0.05 (UCI-HAR, KU-HAR), 0.01 (mHealth) |
| FedOpt global learning rate | 0.025 (UCI-HAR, KU-HAR), 0.005 (mHealth) |
| Distillation weight | `α_{r,k} = α_max (r/R) max(0, 1 − ρ m̂_k)`, `α_max = 0.7` |
| Missing-rate coefficient `ρ` | 0.2 (`--distill_gamma`) |
| Temperature | `τ_r = τ_init − (τ_init − 1) r/R`, `τ_init = 5.0`, `τ_min = 1.0` |
| Feature distillation weight | 0.2 (`--alpha_feat`) |
| Cross-modal recovery weight | 1.0 (`--lambda_rec`) |
| Prototype weight | 0.05 (`--lambda_proto`) |
| Prototype EMA momentum | 0.97 (`--proto_ema`) |
| Contrastive temperature | 0.07 (`--proto_temperature`) |
| Projection heads | linear, dimension `d = 128` |
| Encoder | 1D convolutions + unidirectional GRU, `d = 128` |

FedProx uses μ = 0.001 on UCI-HAR and μ = 0.01 on mHealth and KU-HAR
(argument defaults in the training scripts). FedMEKT uses 10 server KD steps,
server KD learning rate 0.01, temperature 3.0, KD weight 0.5, and a 10%
proxy set drawn from training clients.

### Key training flags

| Flag | Meaning |
|---|---|
| `--en_att --att_name fuse_base` | cross-modal temporal attention (FedPCD) |
| `--en_missing_modality --missing_modailty_rate m` | client-level missingness at ratio `m` |
| `--missing_protocol clientlevel` | load client-level masks (all datasets) |
| `--en_distill --en_double_distill` | logit/feature (GKD) and cross-modal (CPR) losses |
| `--en_proto_cross` | PCA: prototype cross-modal contrastive |
| `--no_federated_proj` | keep projection heads local (ablation) |
| `--leakfree_components rec` | CPR only (FedPCD without PCA) |

---

## 7. Notes

- Masks are assigned per client, not per sample, and all datasets use
  `--missing_protocol clientlevel`. At `m = 1.0` every client keeps exactly
  one modality.
- The distillation schedule in Eq. (19) uses the per-client missing rate
  m̂_k, computed from the client's modality mask: m̂_k = 0 for a
  complete-modality client and m̂_k = 1 for a single-modality client.
- For UCI-HAR, the three runs use the same partition and differ only in the
  training seed (seeds 8, 16, 24 for folds 1–3). For mHealth and KU-HAR the
  three runs use the first three cross-validation folds.
- The code supports an optional fast mode (`FEDPCD_FAST=1`), which enables
  cuDNN benchmarking. It is **not** used for the reported numbers because it
  makes results less reproducible.
- `train.py` in each dataset directory handles FedAvg, FedProx, FedOpt,
  SCAFFOLD and FedRS through `--fed_alg`; MFCPL and FedMEKT have their own
  entry points.

---

## 8. Attribution and license

This repository is a derivative work of **FedMultimodal**:

> Feng, Bose, Zhang, Hebbar, Ramakrishna, Gupta, Zhang, Avestimehr, and
> Narayanan. *FedMultimodal: A Benchmark for Multimodal Federated Learning.*
> KDD 2023, pp. 4035–4045.
> <https://github.com/usc-sail/fed-multimodal>

**Derived from FedMultimodal and modified for FedPCD.** The data-loading and
partition/feature-extraction scaffolding (`constants/`, `dataloader/`,
`features/data_partitioning/partition_manager.py`,
`features/feature_processing/feature_manager.py`,
`features/simulation_features/simulation_manager.py`), the base federation
code and HAR encoders (`trainers/evaluation.py`, `fed_avg_trainer.py`,
`fed_rs_trainer.py`, `optimizer.py`, `scaffold_trainer.py`,
`server_trainer.py`, `model/common_fusions.py`, `mm_models.py`,
`unimodal_models.py`), and the UCI-HAR/KU-HAR experiment entry points are
derived from FedMultimodal. They have been modified to add FedPCD training,
client-level missingness, the mHealth/KU-HAR pipelines, and the baselines
used in the paper. Each modified file carries a prominent notice at the top.

**New in FedPCD.** The FedPCD losses and modules (`utils/`), the MFCPL model
and trainers, the FedMEKT server, the mHealth and KU-HAR data pipelines, the
fixed client-to-modality assignments, the run scripts, and the incomplete-input
evaluation are new. They are marked as new at the top of each file. `NOTICE`
contains the complete list.

**License.** `LICENSE.md` is the upstream Apache License, Version 2.0, retained
verbatim. The upstream repository does not contain a NOTICE file; `NOTICE` is
provided by the FedPCD authors to document the derivation and modifications,
as required by Apache-2.0 §4. Original author headers in upstream files have
been retained. The datasets are not redistributed; please follow their own
terms of use.

---

## 9. Citation

If you use this code, please cite the FedPCD paper and the FedMultimodal
benchmark.

```bibtex
@inproceedings{fedpcd2026,
  title     = {FedPCD: Prototype-Cross Distillation for Multimodal Federated Learning},
  author    = {The FedPCD Authors},
  booktitle = {Under review},
  year      = {2026}
}

@inproceedings{feng2023fedmultimodal,
  title     = {FedMultimodal: A Benchmark for Multimodal Federated Learning},
  author    = {Feng, Tiantian and Bose, Digbalay and Zhang, Tuo and Hebbar, Rajat
               and Ramakrishna, Anil and Gupta, Rahul and Zhang, Mi
               and Avestimehr, Salman and Narayanan, Shrikanth},
  booktitle = {Proceedings of the 29th ACM SIGKDD Conference on Knowledge
               Discovery and Data Mining},
  pages     = {4035--4045},
  year      = {2023}
}
```

See also [`CITATION.cff`](CITATION.cff).
