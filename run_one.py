from __future__ import annotations
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, log_loss
from .calibration import ece_score, multiclass_brier, correctness_auroc, falseconf_at_thresh, risk_coverage_auc, spectral_bundle
from .utils import ensure_dir

def evaluate_probs(y_true: np.ndarray, probs: np.ndarray) -> dict:
    pred = probs.argmax(axis=1)
    out = {
        "accuracy": float(accuracy_score(y_true, pred)),
        "macro_f1": float(f1_score(y_true, pred, average="macro")),
        "ece": ece_score(y_true, probs),
        "brier": multiclass_brier(y_true, probs),
        "nll": float(log_loss(y_true, probs)),
        "corr_auroc": correctness_auroc(y_true, probs),
        "falseconf_0.9": falseconf_at_thresh(y_true, probs, 0.9),
        "risk_coverage_auc": risk_coverage_auc(y_true, probs.max(axis=1), pred),
    }
    if probs.shape[1] == 2:
        out["class_auroc"] = float(roc_auc_score(y_true, probs[:, 1]))
    else:
        out["class_auroc"] = float("nan")
    return out

def certificate_faithfulness(model, calibrator, logits_test, X_test, device=None, max_samples: int = 64) -> dict:
    if not hasattr(calibrator, "predict"):
        return {"faithfulness_spearman": float("nan")}
    result = calibrator.predict(logits_test[:max_samples], X_test[:max_samples])
    cert = result.certificate
    if cert is None:
        return {"faithfulness_spearman": float("nan")}
    spec_feats, bands = spectral_bundle(X_test[:max_samples], n_bands=len(result.metadata["bands"]))
    impacts, scores = [], []
    base_probs = result.probs.max(axis=1)
    for i in range(min(max_samples, len(X_test))):
        band_scores = []
        band_impacts = []
        n_group = len(bands)
        for b in range(n_group):
            X_mask = X_test[i:i+1].copy()
            fft = np.fft.rfft(X_mask, axis=-1)
            fft[..., bands[b]] = 0
            X_mask = np.fft.irfft(fft, n=X_test.shape[-1], axis=-1).astype(np.float32)
            masked = calibrator.predict(logits_test[i:i+1], X_mask)
            drop = float(base_probs[i] - masked.probs.max(axis=1)[0])
            band_impacts.append(drop)
            band_scores.append(float(np.mean(cert[i, b::n_group][:1]) if cert.shape[1] >= n_group else cert[i, b]))
        scores.extend(band_scores)
        impacts.extend(band_impacts)
    if len(scores) < 2:
        return {"faithfulness_spearman": float("nan")}
    rho, _ = spearmanr(scores, impacts)
    return {"faithfulness_spearman": float(rho)}

def append_and_save_metrics(row: dict, results_dir: str | Path) -> Path:
    results_dir = ensure_dir(results_dir)
    csv_path = results_dir / "master_metrics.csv"
    df = pd.DataFrame([row])
    if csv_path.exists():
        old = pd.read_csv(csv_path)
        df = pd.concat([old, df], ignore_index=True)
    df.to_csv(csv_path, index=False)
    return csv_path

def generate_tables(results_dir: str | Path, tables_dir: str | Path) -> None:
    results_dir = Path(results_dir)
    tables_dir = ensure_dir(tables_dir)
    csv_path = results_dir / "master_metrics.csv"
    if not csv_path.exists():
        return
    df = pd.read_csv(csv_path)
    if df.empty:
        return
    keep = ["dataset", "model", "calibrator", "accuracy", "ece", "corr_auroc", "falseconf_0.9", "risk_coverage_auc"]
    g = df[keep].groupby(["dataset", "model", "calibrator"], as_index=False).mean(numeric_only=True)
    main_lines = []
    main_lines.append("\\begin{tabular}{lllrrrr}")
    main_lines.append("\\toprule")
    main_lines.append("Dataset & Model & Calibrator & Acc & ECE & CorrAUROC & FalseConf@0.9 & AURC\\\\")
    main_lines.append("\\midrule")
    for _, r in g.sort_values(["dataset", "model", "calibrator"]).iterrows():
        main_lines.append(
            f"{r['dataset']} & {r['model']} & {r['calibrator']} & "
            f"{r['accuracy']:.3f} & {r['ece']:.3f} & {r['corr_auroc']:.3f} & "
            f"{r['falseconf_0.9']:.3f} & {r['risk_coverage_auc']:.3f}\\\\"
        )
    main_lines.append("\\bottomrule")
    main_lines.append("\\end{tabular}")
    (tables_dir / "main_results_table.tex").write_text("\n".join(main_lines), encoding="utf-8")

    # ablation-style aggregated table
    if "faithfulness_spearman" in df.columns:
        ag = df.groupby("calibrator", as_index=False)[["ece", "corr_auroc", "falseconf_0.9", "faithfulness_spearman"]].mean(numeric_only=True)
        ab_lines = []
        ab_lines.append("\\begin{tabular}{lrrrr}")
        ab_lines.append("\\toprule")
        ab_lines.append("Method & ECE & CorrAUROC & FalseConf@0.9 & Faithfulness\\\\")
        ab_lines.append("\\midrule")
        for _, r in ag.sort_values("calibrator").iterrows():
            ab_lines.append(
                f"{r['calibrator']} & {r['ece']:.3f} & {r['corr_auroc']:.3f} & "
                f"{r['falseconf_0.9']:.3f} & {r['faithfulness_spearman']:.3f}\\\\"
            )
        ab_lines.append("\\bottomrule")
        ab_lines.append("\\end{tabular}")
        (tables_dir / "ablation_table.tex").write_text("\n".join(ab_lines), encoding="utf-8")

def generate_figures(results_dir: str | Path, figures_dir: str | Path) -> None:
    results_dir = Path(results_dir)
    figures_dir = ensure_dir(figures_dir)
    csv_path = results_dir / "master_metrics.csv"
    if not csv_path.exists():
        return
    df = pd.read_csv(csv_path)
    if df.empty:
        return
    # Figure 1: CorrAUROC by calibrator
    ag = df.groupby("calibrator", as_index=False)["corr_auroc"].mean(numeric_only=True).sort_values("corr_auroc", ascending=False)
    plt.figure(figsize=(8, 4))
    plt.bar(ag["calibrator"], ag["corr_auroc"])
    plt.xticks(rotation=30, ha="right")
    plt.ylabel("Mean Correctness-AUROC")
    plt.tight_layout()
    plt.savefig(figures_dir / "corr_auroc_by_calibrator.png", dpi=200)
    plt.close()

    # Figure 2: ECE vs FalseConf
    plt.figure(figsize=(5, 4))
    ag2 = df.groupby("calibrator", as_index=False)[["ece", "falseconf_0.9"]].mean(numeric_only=True)
    plt.scatter(ag2["ece"], ag2["falseconf_0.9"])
    for _, r in ag2.iterrows():
        plt.text(r["ece"], r["falseconf_0.9"], r["calibrator"], fontsize=8)
    plt.xlabel("ECE")
    plt.ylabel("FalseConf@0.9")
    plt.tight_layout()
    plt.savefig(figures_dir / "ece_vs_falseconf.png", dpi=200)
    plt.close()
