# SEB-Cal: Spectral Evidence Bundling for Selective Reliability Estimation in Time-Series Classification

**Anonymous NeurIPS 2026 artifact** for the paper:

> **Beyond Output-Space Calibration: Spectral Evidence Bundling for Selective Reliability Estimation in Time-Series Classification**

SEB-Cal is a **fixed-label, post-hoc reliability estimation** method for time-series classifiers. It does **not** change the backbone prediction. Instead, it estimates whether the already-selected prediction should be trusted by combining conventional output-space calibration cues with deterministic whole-sample spectral evidence.

---

## 1. What problem does this artifact address?

Standard post-hoc calibration methods usually remap output scores. In time-series classification, however, two predictions can have identical confidence while being supported by very different temporal evidence. SEB-Cal targets three time-series-specific reliability gaps:

1. **Temporal-support mismatch:** high confidence may not be supported by coherent temporal structure.
2. **False high-confidence errors:** average calibration may still assign high reliability to wrong predictions.
3. **Limited input-linked auditability:** output-space recalibrators provide little information about which input-side evidence supports or weakens trust.

SEB-Cal addresses these gaps by estimating fixed-label reliability from both output-side confidence features and whole-sample spectral descriptors.

---

## 2. Method overview

SEB-Cal keeps the classifier fixed and learns only a shallow post-hoc reliability layer.

```text
Input time series x
        │
        ├── Frozen backbone fθ
        │       └── logits z(x), predicted label ŷ(x)
        │
        ├── Output-side calibration features h(x)
        │       ├── maximum softmax probability
        │       ├── logit margin
        │       └── predictive entropy
        │
        ├── Whole-sample spectral evidence g(x)
        │       ├── band energy concentration
        │       ├── spectral entropy
        │       ├── peak dominance
        │       └── phase stability
        │
        ├── Concatenate φ(x) = [h(x) || g(x)]
        │
        ├── Shallow reliability layer ψ(φ(x))
        │       └── scalar reliability score r̃(x) ≈ P(ŷ(x)=y | x, z(x))
        │
        └── Validation-gated policy
                ├── use SEB-Cal when ranking improves safely
                └── otherwise revert to Raw / safer scalar recalibrator
```

The deployed object is therefore not an unconstrained spectral score. It is a **validation-gated reliability policy**: SEB-Cal is selected only when held-out validation improves correctness-aware ranking without violating FalseConf@0.9 or AURC tolerances.

---

## 3. Main paper results reproduced by this artifact

The paper evaluates SEB-Cal on eight heterogeneous UCR/UEA datasets, eight backbone families, and standard output-space recalibrators. The main benchmark uses a matched evaluation subset of 191 dataset--model--seed configurations.

### 3.1 Aggregate fixed-label reliability comparison

| Method | ECE ↓ | Brier ↓ | NLL ↓ | Corr-AUROC ↑ | FalseConf@0.9 ↓ | AURC ↓ | Faith. ↑ |
|---|---:|---:|---:|---:|---:|---:|---:|
| Raw | 0.097 | 0.377 | 0.716 | 0.693 | 0.128 | 0.219 | -- |
| Temperature | 0.080 | **0.371** | 0.717 | 0.689 | 0.162 | 0.219 | -- |
| Platt | 0.078 | 0.384 | 0.745 | 0.671 | 0.228 | 0.218 | -- |
| Dirichlet | 0.153 | 0.422 | 0.983 | 0.673 | 0.153 | 0.226 | -- |
| Vector | 0.155 | 0.420 | 0.975 | 0.672 | 0.149 | 0.227 | -- |
| Isotonic | 0.144 | 0.421 | 1.977 | 0.644 | 0.292 | 0.227 | -- |
| Unconstrained SEB-Cal | 0.077 | 0.372 | 0.710 | 0.779 | 0.118 | 0.214 | 0.059 |
| Policy-selected SEB-Cal | **0.075** | **0.370** | **0.708** | **0.786** | **0.094** | **0.209** | -- |

Key interpretation: unconstrained SEB-Cal tests whether spectral evidence adds reliability information. Policy-selected SEB-Cal is the deployed reliability policy and gives the safest aggregate operating point.

### 3.2 Dataset-level regime structure

SEB-Cal is intentionally regime-dependent rather than universal. The strongest positive Corr-AUROC gains appear in:

| Dataset | Raw Corr | SEB Corr | Δ Corr |
|---|---:|---:|---:|
| FordA | 0.730 | 0.859 | +0.129 |
| ECG200 | 0.685 | 0.750 | +0.065 |
| UWaveGestureLibrary | 0.685 | 0.754 | +0.069 |
| SelfRegulationSCP1 | 0.567 | 0.671 | +0.104 |
| Wafer | 0.826 | 0.885 | +0.059 |

Unfavorable or weaker regimes are rejected by the validation-gated policy when safety constraints are not met.

### 3.3 Frequency-bundle ablation

The central mechanism ablation tests whether the spectral bundle `g(x)` adds ranking information beyond output-side features `h(x)`.

| Variant | Corr-AUROC ↑ |
|---|---:|
| Remove all `g(x)`: output-side only `h(x)` | 0.703 |
| `h(x)` + energy only | 0.778 |
| `h(x)` + entropy only | 0.733 |
| `h(x)` + peak only | 0.756 |
| `h(x)` + phase only | 0.734 |
| Full SEB-Cal bundle `h(x)+g(x)` | 0.786 |
| Full bundle without energy | 0.752 |
| Full bundle without entropy | 0.786 |
| Full bundle without peak | 0.784 |
| Full bundle without phase | 0.787 |

Energy concentration is the strongest single descriptor in the positive-regime slice, while the full bundle preserves a decomposable spectral diagnostic.

### 3.4 Robustness controls reported in the paper

The paper also reports the following robustness checks:

- comparison to non-output-space Proximity-style reliability estimators;
- output-feature-only `h(x)` control;
- time-domain summary baseline;
- STFT-SEB-Cal alternative time-frequency variant;
- input-space and feature-space masking controls;
- computational overhead analysis;
- paired bootstrap uncertainty over matched configurations;
- dependence diagnostics showing that spectral descriptors are not redundant with output features.

---

## 4. Repository structure

Expected artifact layout:

```text
.
├── README.md
├── main.tex
├── references.bib
├── requirements.txt
├── run_benchmark.sh
├── src/
│   ├── __init__.py
│   ├── datasets.py
│   ├── models.py
│   ├── training.py
│   ├── calibration.py
│   ├── evaluation.py
│   ├── run_one.py
│   └── run_all.py
├── results/
│   ├── master_metrics.csv                  # generated or supplied
│   └── runs/                               # per-run JSON outputs
├── tables/
│   ├── main_results_table.tex              # generated
│   └── ablation_table.tex                  # generated
├── figures/
│   ├── corr_auroc_by_calibrator.png        # generated
│   └── ece_vs_falseconf.png                # generated
└── logs/
    └── progress.log                        # generated
```

**Important:** preserve the `src/` directory structure. The commands below use Python module execution (`python -m src.run_one` and `python -m src.run_all`). If files are uploaded flat at the repository root, the relative imports will fail.

---

## 5. Installation

Create a fresh environment:

```bash
python -m venv .venv
source .venv/bin/activate        # Linux/macOS
# .venv\Scripts\activate         # Windows PowerShell
pip install --upgrade pip
pip install -r requirements.txt
```

The code uses public UCR/UEA time-series datasets through `aeon` and downloads them automatically when first requested.

---

## 6. Quick smoke test

Run one dataset, one backbone, and all calibration methods:

```bash
python -m src.run_one \
  --dataset ECG200 \
  --model mlp \
  --seed 7 \
  --gpu 0 \
  --calibrators none temperature platt isotonic vector dirichlet sebcal \
  --results_dir results \
  --tables_dir tables \
  --figures_dir figures \
  --logs_dir logs
```

Expected outputs:

```text
results/runs/ECG200__mlp__seed7__*.json
results/master_metrics.csv
tables/main_results_table.tex
tables/ablation_table.tex
figures/corr_auroc_by_calibrator.png
figures/ece_vs_falseconf.png
logs/progress.log
```

---

## 7. Full benchmark run

Run the full UCR/UEA × backbone × seed grid:

```bash
python -m src.run_all \
  --gpus 0 1 \
  --datasets ECG200 FordA Wafer ElectricDevices UWaveGestureLibrary BasicMotions SelfRegulationSCP1 AtrialFibrillation \
  --models mlp lstm gru tcn fcn resnet1d inceptionlite transformer \
  --calibrators none temperature platt isotonic vector dirichlet sebcal \
  --seeds 7 13 21 \
  --results_dir results \
  --tables_dir tables \
  --figures_dir figures \
  --logs_dir logs
```

Alternatively:

```bash
bash run_benchmark.sh
```

The launcher writes outputs incrementally after each completed `(dataset, model, seed, calibrator)` run. This means partial results can be inspected before the full benchmark completes.

---

## 8. How to reproduce the paper tables

After the benchmark finishes, inspect:

```bash
python - <<'PY'
import pandas as pd
m = pd.read_csv('results/master_metrics.csv')
print(m.head())
print(m.groupby('calibrator')[['ece','corr_auroc','falseconf_0.9','risk_coverage_auc']].mean())
PY
```

Generated LaTeX tables are written to:

```text
tables/main_results_table.tex
tables/ablation_table.tex
```

For exact paper reproduction, the artifact should include the configuration-level metrics used for the manuscript, ideally as:

```text
results/master_metrics.csv
paper_artifacts/table1_headline_summary.csv
paper_artifacts/table2_dataset_decomposition.csv
paper_artifacts/table3_backbone_decomposition.csv
paper_artifacts/table4_frequency_bundle_ablation.csv
paper_artifacts/validation_gate_decisions.csv
paper_artifacts/bootstrap_uncertainty.csv
```

If these files are included, reviewers can verify the reported numbers directly without rerunning the full benchmark.

---

## 9. Core implementation files

### `src/calibration.py`

Implements:

- temperature scaling;
- Platt-style logistic calibration;
- isotonic regression;
- vector scaling;
- Dirichlet calibration;
- SEB-Cal spectral bundle extraction;
- SEB-Cal shallow fixed-label reliability layer.

### `src/models.py`

Implements the eight backbone families:

- MLP;
- LSTM;
- GRU;
- TCN;
- FCN;
- 1D ResNet;
- InceptionLite;
- Transformer encoder.

### `src/evaluation.py`

Computes:

- accuracy and macro-F1 for frozen-backbone context;
- ECE;
- Brier score;
- NLL;
- Corr-AUROC;
- FalseConf@0.9;
- AURC;
- masking-based diagnostic faithfulness;
- incremental result files, tables, and figures.

---

## 10. Notes on the validation gate

The paper defines the deployed method as a validation-gated policy. In the manuscript, the gate selects SEB-Cal only when:

```text
Δ Corr-AUROC > δ_rank
Δ FalseConf@0.9 ≤ τ_fc
Δ AURC ≤ τ_aurc
```

relative to Raw and the strongest scalar recalibrator on a held-out gate-validation split.

The simplified code in this repository implements the SEB-Cal reliability model and standard metric generation. For full paper-level reproduction, the artifact should also include the exact gate-validation split records and the script used to generate the policy-selected aggregate.

---

## 11. Computational requirements

The benchmark was designed for a workstation with two GPUs. The paper reports experiments on two NVIDIA RTX A6000 GPUs.

Approximate compute profile:

- backbone training: GPU-heavy;
- post-hoc calibration: CPU-light;
- FFT feature extraction: CPU/GPU-light relative to backbone training;
- full benchmark: several hundred GPU-hours depending on hardware and early stopping.

SEB-Cal adds one real FFT per channel and deterministic spectral reductions:

```text
O(C T log T) + O(CF + B)
```

where `C` is the number of channels, `T` is sequence length, `F` is the number of retained Fourier frequencies, and `B` is the number of frequency bands.

---

## 12. Artifact checklist

A strong NeurIPS artifact should contain:

- [x] source code for datasets, backbones, calibration, SEB-Cal, and evaluation;
- [x] requirements file;
- [x] public dataset loader;
- [x] benchmark launcher;
- [x] incremental result generation;
- [ ] exact split identifiers for train/calibration/gate-validation/test;
- [ ] precomputed configuration-level metrics matching the paper;
- [ ] exact table-generation scripts for every manuscript and appendix table;
- [ ] validation-gate decision script;
- [ ] proximity, time-domain, STFT, dependence-diagnostic, bootstrap, and feature-space masking scripts if claimed as fully reproducible;
- [ ] trained logits or run-level JSON files for direct verification without rerunning all backbones.

---

## 13. Citation

This repository is anonymous for review. Citation information will be added after the review process.

