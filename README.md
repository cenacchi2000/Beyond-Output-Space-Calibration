# SEB-Cal: Spectral Evidence Bundling for Selective Reliability Estimation in Time-Series Classification

Anonymous artifact for NeurIPS 2026 submission:

**Beyond Output-Space Calibration: Spectral Evidence Bundling for Selective Reliability Estimation in Time-Series Classification**

This repository contains the implementation used to evaluate **SEB-Cal**, a validation-gated, fixed-label post-hoc reliability policy for time-series classification. SEB-Cal leaves the trained backbone and predicted label unchanged, augments output-side confidence features with deterministic whole-sample spectral descriptors, and estimates whether the selected prediction should be trusted.

Recommended anonymous code link format for the paper:

```text
https://anonymous.4open.science/r/Spectral-Evidence-Bundling-SEBCal-XXXX/
```

After uploading the repository to Anonymous 4open Science, replace `XXXX` with the generated anonymous repository identifier and cite the root URL in the paper/checklist.

---

## 1. What this artifact supports

The artifact supports the main claims of the paper:

1. **Fixed-label reliability estimation**: post-hoc methods do not change the predicted class; they estimate the reliability of the already-selected prediction.
2. **Spectral Evidence Bundling (SEB-Cal)**: output-side cues are augmented with whole-sample spectral descriptors: band energy, spectral entropy, peak dominance, and phase stability.
3. **Matched benchmark evaluation**: experiments cover eight UCR/UEA time-series classification datasets, eight backbone families, and standard output-space recalibrators.
4. **Selective-reliability metrics**: evaluation reports ECE, Brier score, NLL, Corr-AUROC, FalseConf@0.9, AURC, and diagnostic faithfulness where applicable.
5. **Incremental reproducibility**: each completed dataset--model--seed run writes a run-level result and updates aggregate CSV, table, and figure outputs.

The repository is designed to be readable and editable rather than heavily engineered.

---

## 2. Repository structure

```text
.
├── README.md
├── requirements.txt
├── run_benchmark.sh
├── main.tex
├── references.bib
├── src/
│   ├── __init__.py
│   ├── calibration.py      # scalar recalibrators + SEB-Cal spectral reliability layer
│   ├── datasets.py         # UCR/UEA loading and train/calibration/test split construction
│   ├── evaluation.py       # reliability metrics, diagnostic faithfulness, tables, figures
│   ├── models.py           # MLP, LSTM, GRU, TCN, FCN, ResNet1D, InceptionLite, Transformer
│   ├── run_all.py          # multi-job launcher over datasets, backbones, seeds
│   ├── run_one.py          # one dataset--model--seed experiment
│   ├── training.py         # backbone training and logit extraction
│   └── utils.py
├── results/                # generated run-level JSON files and master_metrics.csv
├── tables/                 # generated LaTeX tables
├── figures/                # generated figures
└── logs/                   # progress logs
```

Optional paper-artifact folders can also be included when available:

```text
paper_artifacts/
├── master_metrics.csv
├── table1_headline_summary.csv
├── dataset_decomposition.csv
├── backbone_decomposition.csv
├── validation_gate_decisions.csv
├── appendix_outputs/
└── trained_outputs/
```

Including these optional files lets reviewers verify reported tables without rerunning the full benchmark.

---

## 3. Installation

Create a clean environment, then install dependencies:

```bash
conda create -n sebcal python=3.10 -y
conda activate sebcal
pip install -r requirements.txt
```

The artifact uses public UCR/UEA datasets loaded through `aeon`. Dataset files are downloaded automatically by `aeon` when first requested.

---

## 4. Quick smoke test

Run a small experiment on one dataset, one backbone, one seed, and a subset of calibrators:

```bash
python -m src.run_one \
  --dataset ECG200 \
  --model mlp \
  --seed 7 \
  --gpu 0 \
  --calibrators none temperature platt sebcal \
  --results_dir results \
  --tables_dir tables \
  --figures_dir figures \
  --logs_dir logs
```

Expected outputs:

```text
results/runs/ECG200__mlp__seed7__none.json
results/runs/ECG200__mlp__seed7__temperature.json
results/runs/ECG200__mlp__seed7__platt.json
results/runs/ECG200__mlp__seed7__sebcal.json
results/master_metrics.csv
tables/main_results_table.tex
tables/ablation_table.tex
figures/corr_auroc_by_calibrator.png
figures/ece_vs_falseconf.png
logs/progress.log
```

---

## 5. Full benchmark command

The full benchmark used in the paper covers:

- **Datasets**: `ECG200`, `FordA`, `Wafer`, `ElectricDevices`, `UWaveGestureLibrary`, `BasicMotions`, `SelfRegulationSCP1`, `AtrialFibrillation`
- **Backbones**: `mlp`, `lstm`, `gru`, `tcn`, `fcn`, `resnet1d`, `inceptionlite`, `transformer`
- **Seeds**: `7`, `13`, `21`
- **Recalibrators**: `none`, `temperature`, `platt`, `isotonic`, `vector`, `dirichlet`, `sebcal`

Run on two GPUs:

```bash
bash run_benchmark.sh
```

or equivalently:

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

The launcher runs one job per dataset--model--seed configuration and evaluates all requested calibrators for that trained backbone.

---

## 6. Metrics produced by the code

For each run, the code reports:

- `accuracy`: frozen-backbone accuracy, reported as task context only;
- `macro_f1`: frozen-backbone macro-F1, reported as task context only;
- `ece`: expected calibration error;
- `brier`: multiclass Brier score;
- `nll`: negative log-likelihood;
- `corr_auroc`: AUROC for ranking correct predictions above incorrect predictions;
- `falseconf_0.9`: fraction of errors assigned confidence/reliability at least 0.9;
- `risk_coverage_auc`: area under the risk-coverage curve;
- `faithfulness_spearman`: Spearman alignment between spectral diagnostic scores and reliability drops under masking, for SEB-Cal.

These metrics match the fixed-label selective-reliability framing of the paper: post-hoc methods are evaluated by the reliability score assigned to an unchanged backbone prediction.

---

## 7. SEB-Cal implementation details

The main implementation is in:

```text
src/calibration.py
```

SEB-Cal constructs:

1. output-side features from logits, including maximum softmax probability, logit margin, predictive entropy, and top probabilities;
2. deterministic spectral features from the input time series using a real FFT;
3. band-level energy, spectral entropy, phase concentration, and peak dominance descriptors;
4. a shallow logistic reliability layer trained on binary correctness of the frozen prediction.

The SEB-Cal prediction preserves the class ranking of the frozen backbone and replaces only the top-class reliability assigned to the already-selected label.

---

## 8. Generated tables and figures

During execution, results are updated incrementally:

```text
results/master_metrics.csv
```

is regenerated after every completed run. The following files are also regenerated:

```text
tables/main_results_table.tex
tables/ablation_table.tex
figures/corr_auroc_by_calibrator.png
figures/ece_vs_falseconf.png
```

This means reviewers can inspect intermediate outputs without waiting for the full benchmark to finish.

---

## 9. Reproducing paper tables

To reproduce the paper tables exactly, use one of the following workflows.

### A. Full recomputation

Run the full benchmark command in Section 5, then inspect:

```text
results/master_metrics.csv
tables/main_results_table.tex
tables/ablation_table.tex
figures/
```

### B. Verification from precomputed artifacts

If the repository includes `paper_artifacts/`, reviewers can verify the reported numbers directly from the supplied CSV files without retraining all backbones:

```text
paper_artifacts/master_metrics.csv
paper_artifacts/table1_headline_summary.csv
paper_artifacts/dataset_decomposition.csv
paper_artifacts/backbone_decomposition.csv
paper_artifacts/validation_gate_decisions.csv
```

The full benchmark is computationally heavier because it trains eight backbone families across eight datasets and multiple seeds.

---

## 10. Hardware used

The reported experiments were run on a workstation with two NVIDIA RTX A6000 GPUs. Backbone training used GPU execution. Post-hoc calibration, spectral feature extraction, diagnostic masking, and table generation are primarily CPU-side.

Approximate expected cost:

- full backbone grid: several hundred GPU-hours depending on local hardware and dataset download/cache state;
- post-hoc calibration and table generation: tens of CPU-hours or less;
- quick smoke test: minutes on a modern GPU.

---

## 11. Anonymous review notes

This artifact is anonymized for peer review. Please do not add author names, institutional paths, or non-anonymous URLs before submission.

Recommended paper/checklist wording:

```latex
\paragraph{Code availability.}
An anonymized implementation and reproduction instructions are available at
\url{https://anonymous.4open.science/r/Spectral-Evidence-Bundling-SEBCal-XXXX/}.
```

Recommended checklist wording:

```latex
The benchmark uses public UCR/UEA datasets. The anonymized supplementary artifact includes implementation code, fixed split construction, seeds, spectral feature extraction, calibration baselines, SEB-Cal reliability estimation, masking controls, generated tables/figures, and scripts needed to reproduce the reported results.
```

---

## 12. Troubleshooting

### Dataset download fails

`aeon` downloads UCR/UEA datasets automatically. If a dataset fails to download, rerun the command or pre-download/cache the dataset using `aeon` utilities.

### CUDA is unavailable

The code automatically falls back to CPU if CUDA is not available. Full benchmark runtime will be much longer on CPU.

### Full benchmark is too slow

Use the smoke test first, then run a subset:

```bash
python -m src.run_all \
  --gpus 0 \
  --datasets ECG200 FordA \
  --models mlp transformer \
  --calibrators none temperature sebcal \
  --seeds 7
```

### Reviewers only want to verify numbers

Include precomputed CSV files under `paper_artifacts/` and point reviewers to Section 9B.

---

## 13. License

This artifact is provided for anonymous peer review. Add the final project license after de-anonymization.
