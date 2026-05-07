from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple
import numpy as np
from sklearn.model_selection import train_test_split
from aeon.datasets import load_classification

@dataclass
class DatasetBundle:
    X_train: np.ndarray
    y_train: np.ndarray
    X_cal: np.ndarray
    y_cal: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray
    n_channels: int
    series_length: int
    n_classes: int

DEFAULT_DATASETS = [
    "ECG200",
    "FordA",
    "Wafer",
    "ElectricDevices",
    "UWaveGestureLibrary",
    "BasicMotions",
    "SelfRegulationSCP1",
    "AtrialFibrillation",
]

def _to_nct(X: np.ndarray) -> np.ndarray:
    X = np.asarray(X, dtype=np.float32)
    if X.ndim == 2:
        X = X[:, None, :]
    if X.ndim != 3:
        raise ValueError(f"Expected 3D time-series array, got shape {X.shape}")
    return X

def load_dataset_bundle(name: str, seed: int, cal_size: float = 0.2) -> DatasetBundle:
    X_train, y_train = load_classification(name=name, split="train")
    X_test, y_test = load_classification(name=name, split="test")
    X_train = _to_nct(X_train)
    X_test = _to_nct(X_test)
    classes, y_train_idx = np.unique(y_train, return_inverse=True)
    class_map = {c: i for i, c in enumerate(classes)}
    y_test_idx = np.array([class_map[c] for c in y_test], dtype=np.int64)
    X_train_sub, X_cal, y_train_sub, y_cal = train_test_split(
        X_train, y_train_idx, test_size=cal_size, random_state=seed, stratify=y_train_idx
    )
    return DatasetBundle(
        X_train=X_train_sub,
        y_train=y_train_sub,
        X_cal=X_cal,
        y_cal=y_cal,
        X_test=X_test,
        y_test=y_test_idx,
        n_channels=X_train.shape[1],
        series_length=X_train.shape[2],
        n_classes=len(classes),
    )
