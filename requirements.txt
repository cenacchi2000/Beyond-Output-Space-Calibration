from __future__ import annotations
import argparse
import itertools
import os
import subprocess
import sys
from pathlib import Path

DEFAULT_DATASETS = ["ECG200", "FordA", "Wafer", "ElectricDevices", "UWaveGestureLibrary", "BasicMotions", "SelfRegulationSCP1", "AtrialFibrillation"]
DEFAULT_MODELS = ["mlp", "lstm", "gru", "tcn", "fcn", "resnet1d", "inceptionlite", "transformer"]
DEFAULT_CALIBRATORS = ["none", "temperature", "platt", "isotonic", "vector", "dirichlet", "sebcal"]
DEFAULT_SEEDS = [7, 13, 21]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpus", nargs="+", default=["0", "1"])
    parser.add_argument("--datasets", nargs="+", default=DEFAULT_DATASETS)
    parser.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    parser.add_argument("--calibrators", nargs="+", default=DEFAULT_CALIBRATORS)
    parser.add_argument("--seeds", nargs="+", type=int, default=DEFAULT_SEEDS)
    parser.add_argument("--results_dir", default="results")
    parser.add_argument("--tables_dir", default="tables")
    parser.add_argument("--figures_dir", default="figures")
    parser.add_argument("--logs_dir", default="logs")
    args = parser.parse_args()

    jobs = list(itertools.product(args.datasets, args.models, args.seeds))
    active = []

    def launch(job, gpu):
        dataset, model, seed = job
        cmd = [
            sys.executable, "-m", "src.run_one",
            "--dataset", dataset,
            "--model", model,
            "--seed", str(seed),
            "--gpu", gpu,
            "--results_dir", args.results_dir,
            "--tables_dir", args.tables_dir,
            "--figures_dir", args.figures_dir,
            "--logs_dir", args.logs_dir,
            "--calibrators", *args.calibrators,
        ]
        return subprocess.Popen(cmd)

    while jobs or active:
        while jobs and len(active) < len(args.gpus):
            gpu = args.gpus[len(active) % len(args.gpus)]
            job = jobs.pop(0)
            proc = launch(job, gpu)
            active.append((proc, gpu, job))
        new_active = []
        for proc, gpu, job in active:
            ret = proc.poll()
            if ret is None:
                new_active.append((proc, gpu, job))
            else:
                if ret != 0:
                    print(f"[WARN] job failed on gpu={gpu}: {job}", flush=True)
        active = new_active

if __name__ == "__main__":
    main()
