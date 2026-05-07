from __future__ import annotations
import argparse
import os
from datetime import datetime
from pathlib import Path
import numpy as np
from sklearn.model_selection import train_test_split
import torch

from .datasets import load_dataset_bundle
from .models import build_model
from .training import train_model, predict_logits
from .calibration import fit_calibrator
from .evaluation import evaluate_probs, certificate_faithfulness, append_and_save_metrics, generate_tables, generate_figures
from .utils import ensure_dir, write_json

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--gpu", type=str, default="")
    parser.add_argument("--calibrators", nargs="+", default=["none", "temperature", "platt", "isotonic", "vector", "dirichlet", "sebcal"])
    parser.add_argument("--results_dir", default="results")
    parser.add_argument("--tables_dir", default="tables")
    parser.add_argument("--figures_dir", default="figures")
    parser.add_argument("--logs_dir", default="logs")
    args = parser.parse_args()

    if args.gpu != "":
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    device = "cuda" if torch.cuda.is_available() else "cpu"
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    bundle = load_dataset_bundle(args.dataset, seed=args.seed)
    X_train, X_val, y_train, y_val = train_test_split(
        bundle.X_train, bundle.y_train, test_size=0.2, random_state=args.seed, stratify=bundle.y_train
    )

    model = build_model(args.model, bundle.n_channels, bundle.series_length, bundle.n_classes)
    model = train_model(model, X_train, y_train, X_val, y_val, device=device)

    logits_cal = predict_logits(model, bundle.X_cal, device=device)
    logits_test = predict_logits(model, bundle.X_test, device=device)

    run_dir = ensure_dir(Path(args.results_dir) / "runs")
    log_dir = ensure_dir(args.logs_dir)

    for cal_name in args.calibrators:
        calibrator = fit_calibrator(cal_name, logits_cal, bundle.y_cal, bundle.X_cal if cal_name == "sebcal" else None)
        if cal_name == "sebcal":
            cal_result = calibrator.predict(logits_test, bundle.X_test)
            probs = cal_result.probs
            faith = certificate_faithfulness(model, calibrator, logits_test, bundle.X_test, device=device)
        else:
            probs = calibrator.predict_proba(logits_test)
            faith = {"faithfulness_spearman": float("nan")}

        metrics = evaluate_probs(bundle.y_test, probs)
        row = {
            "timestamp": datetime.utcnow().isoformat(),
            "dataset": args.dataset,
            "model": args.model,
            "seed": args.seed,
            "calibrator": cal_name,
            **metrics,
            **faith,
        }

        run_file = run_dir / f"{args.dataset}__{args.model}__seed{args.seed}__{cal_name}.json"
        write_json(run_file, row)
        append_and_save_metrics(row, args.results_dir)
        generate_tables(args.results_dir, args.tables_dir)
        generate_figures(args.results_dir, args.figures_dir)

        with open(Path(log_dir) / "progress.log", "a", encoding="utf-8") as f:
            f.write(f"{datetime.utcnow().isoformat()} finished {args.dataset} {args.model} seed={args.seed} cal={cal_name}\n")

if __name__ == "__main__":
    main()
