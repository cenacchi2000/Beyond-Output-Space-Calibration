from __future__ import annotations
import numpy as np
from dataclasses import dataclass
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from scipy.special import softmax, expit

EPS = 1e-8

def multiclass_brier(y_true: np.ndarray, probs: np.ndarray) -> float:
    oh = np.eye(probs.shape[1])[y_true]
    return float(np.mean(np.sum((probs - oh) ** 2, axis=1)))

def ece_score(y_true: np.ndarray, probs: np.ndarray, n_bins: int = 15) -> float:
    conf = probs.max(axis=1)
    pred = probs.argmax(axis=1)
    acc = (pred == y_true).astype(float)
    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        mask = (conf > bins[i]) & (conf <= bins[i+1] if i < n_bins - 1 else conf <= bins[i+1] + 1e-12)
        if mask.any():
            ece += abs(acc[mask].mean() - conf[mask].mean()) * mask.mean()
    return float(ece)

def risk_coverage_auc(y_true: np.ndarray, scores: np.ndarray, preds: np.ndarray) -> float:
    order = np.argsort(-scores)
    correct = (preds[order] == y_true[order]).astype(float)
    risks = []
    covers = []
    cum_correct = np.cumsum(correct)
    n = len(correct)
    for i in range(1, n + 1):
        coverage = i / n
        risk = 1.0 - (cum_correct[i-1] / i)
        covers.append(coverage)
        risks.append(risk)
    return float(np.trapz(risks, covers))

def falseconf_at_thresh(y_true: np.ndarray, probs: np.ndarray, thresh: float = 0.9) -> float:
    conf = probs.max(axis=1)
    pred = probs.argmax(axis=1)
    wrong = pred != y_true
    if wrong.sum() == 0:
        return 0.0
    return float(np.mean(conf[wrong] >= thresh))

def correctness_auroc(y_true: np.ndarray, probs: np.ndarray) -> float:
    from sklearn.metrics import roc_auc_score
    conf = probs.max(axis=1)
    pred = probs.argmax(axis=1)
    correct = (pred == y_true).astype(int)
    if len(np.unique(correct)) < 2:
        return 0.5
    return float(roc_auc_score(correct, conf))

def make_output_features(logits: np.ndarray) -> np.ndarray:
    probs = softmax(logits, axis=1)
    sorted_logits = np.sort(logits, axis=1)
    margin = sorted_logits[:, -1] - sorted_logits[:, -2] if logits.shape[1] > 1 else sorted_logits[:, -1]
    msp = probs.max(axis=1)
    ent = -np.sum(probs * np.log(np.clip(probs, EPS, 1.0)), axis=1)
    topk = np.sort(probs, axis=1)[:, ::-1][:, :min(3, probs.shape[1])]
    return np.column_stack([msp, margin, ent, topk])

def spectral_bundle(X: np.ndarray, n_bands: int = 8) -> tuple[np.ndarray, list[list[int]]]:
    # X: [N, C, T]
    N, C, T = X.shape
    fft = np.fft.rfft(X, axis=-1)
    amp = np.abs(fft)
    phase = np.angle(fft)
    F = amp.shape[-1]
    band_edges = np.linspace(0, F, n_bands + 1, dtype=int)
    bands = [list(range(band_edges[i], max(band_edges[i+1], band_edges[i] + 1))) for i in range(n_bands)]
    feats = []
    for i in range(N):
        a = amp[i]
        ph = phase[i]
        channel_energy = np.sum(a**2, axis=0)
        total = channel_energy.sum() + EPS
        band_energy = []
        band_entropy = []
        band_phase = []
        for band in bands:
            e = float(np.sum(channel_energy[band]))
            band_energy.append(np.log1p(e))
            p = channel_energy[band] / (np.sum(channel_energy[band]) + EPS)
            ent = -np.sum(p * np.log(np.clip(p, EPS, 1.0))) / np.log(max(len(band), 2))
            band_entropy.append(float(ent))
            unit = np.exp(1j * ph[:, band])
            conc = np.abs(np.mean(unit))
            band_phase.append(float(conc))
        flat = channel_energy / total
        spectral_entropy = -np.sum(flat * np.log(np.clip(flat, EPS, 1.0))) / np.log(len(flat))
        peaks = np.sort(channel_energy)[::-1]
        peak1 = float(peaks[:1].sum() / total)
        peak3 = float(peaks[:3].sum() / total)
        peak5 = float(peaks[:5].sum() / total)
        feats.append(np.array(band_energy + band_entropy + band_phase + [spectral_entropy, peak1, peak3, peak5], dtype=np.float32))
    return np.stack(feats), bands

@dataclass
class CalibrationResult:
    probs: np.ndarray
    certificate: np.ndarray | None = None
    metadata: dict | None = None

class TemperatureCalibrator:
    def fit(self, logits_cal, y_cal):
        self.T = 1.0
        from scipy.optimize import minimize
        def nll(logT):
            T = np.exp(logT[0])
            p = softmax(logits_cal / T, axis=1)
            return -np.mean(np.log(np.clip(p[np.arange(len(y_cal)), y_cal], EPS, 1.0)))
        res = minimize(nll, x0=np.array([0.0]), method="L-BFGS-B")
        self.T = float(np.exp(res.x[0]))
        return self
    def predict_proba(self, logits):
        return softmax(logits / self.T, axis=1)

class VectorScalingCalibrator:
    def fit(self, logits_cal, y_cal):
        self.scaler = StandardScaler()
        X = self.scaler.fit_transform(logits_cal)
        self.lr = LogisticRegression(max_iter=2000, multi_class="multinomial")
        self.lr.fit(X, y_cal)
        return self
    def predict_proba(self, logits):
        return self.lr.predict_proba(self.scaler.transform(logits))

class DirichletCalibrator:
    def fit(self, logits_cal, y_cal):
        probs = softmax(logits_cal, axis=1)
        X = np.log(np.clip(probs, EPS, 1.0))
        self.scaler = StandardScaler()
        Xs = self.scaler.fit_transform(X)
        self.lr = LogisticRegression(max_iter=2000, multi_class="multinomial")
        self.lr.fit(Xs, y_cal)
        return self
    def predict_proba(self, logits):
        probs = softmax(logits, axis=1)
        X = np.log(np.clip(probs, EPS, 1.0))
        return self.lr.predict_proba(self.scaler.transform(X))

class PlattLikeCalibrator:
    def fit(self, logits_cal, y_cal):
        probs = softmax(logits_cal, axis=1)
        msp = probs.max(axis=1)
        pred = probs.argmax(axis=1)
        target = (pred == y_cal).astype(int)
        self.lr = LogisticRegression(max_iter=1000)
        self.lr.fit(msp[:, None], target)
        self.n_classes = probs.shape[1]
        return self
    def predict_proba(self, logits):
        probs = softmax(logits, axis=1)
        msp = probs.max(axis=1)
        rel = self.lr.predict_proba(msp[:, None])[:, 1]
        out = probs.copy()
        pred = probs.argmax(axis=1)
        # rescale top-class confidence while preserving class ranking
        for i in range(len(out)):
            k = pred[i]
            old = out[i, k]
            rem = max(1.0 - old, EPS)
            out[i, :] *= (1.0 - rel[i]) / rem
            out[i, k] = rel[i]
        return out / out.sum(axis=1, keepdims=True)

class IsotonicCalibrator:
    def fit(self, logits_cal, y_cal):
        probs = softmax(logits_cal, axis=1)
        self.n_classes = probs.shape[1]
        self.models = []
        for c in range(self.n_classes):
            ir = IsotonicRegression(out_of_bounds="clip")
            ir.fit(probs[:, c], (y_cal == c).astype(float))
            self.models.append(ir)
        return self
    def predict_proba(self, logits):
        probs = softmax(logits, axis=1)
        out = np.column_stack([m.transform(probs[:, i]) for i, m in enumerate(self.models)])
        out = np.clip(out, EPS, 1.0)
        return out / out.sum(axis=1, keepdims=True)

class SebCalibrator:
    def __init__(self, n_bands: int = 8):
        self.n_bands = n_bands
    def fit(self, logits_cal, y_cal, X_cal):
        self.output_scaler = StandardScaler()
        self.spec_scaler = StandardScaler()
        out_feats = make_output_features(logits_cal)
        spec_feats, bands = spectral_bundle(X_cal, self.n_bands)
        self.bands = bands
        X = np.concatenate([
            self.output_scaler.fit_transform(out_feats),
            self.spec_scaler.fit_transform(spec_feats)
        ], axis=1)
        probs = softmax(logits_cal, axis=1)
        pred = probs.argmax(axis=1)
        target = (pred == y_cal).astype(int)
        self.lr = LogisticRegression(max_iter=2000)
        self.lr.fit(X, target)
        self.n_classes = probs.shape[1]
        self.spec_dim = spec_feats.shape[1]
        return self
    def predict(self, logits, X_raw):
        base_probs = softmax(logits, axis=1)
        pred = base_probs.argmax(axis=1)
        out_feats = self.output_scaler.transform(make_output_features(logits))
        spec_feats, _ = spectral_bundle(X_raw, self.n_bands)
        spec_std = self.spec_scaler.transform(spec_feats)
        X = np.concatenate([out_feats, spec_std], axis=1)
        rel = self.lr.predict_proba(X)[:, 1]
        out = base_probs.copy()
        for i in range(len(out)):
            k = pred[i]
            old = out[i, k]
            rem = max(1.0 - old, EPS)
            out[i, :] *= (1.0 - rel[i]) / rem
            out[i, k] = rel[i]
        out = out / out.sum(axis=1, keepdims=True)
        coef = self.lr.coef_[0]
        cert = spec_std * coef[-self.spec_dim:]
        return CalibrationResult(probs=out, certificate=cert, metadata={"bands": self.bands})

def fit_calibrator(name: str, logits_cal: np.ndarray, y_cal: np.ndarray, X_cal: np.ndarray | None = None):
    name = name.lower()
    if name == "none":
        class Identity:
            def predict_proba(self, logits): return softmax(logits, axis=1)
        return Identity()
    if name == "temperature":
        return TemperatureCalibrator().fit(logits_cal, y_cal)
    if name == "platt":
        return PlattLikeCalibrator().fit(logits_cal, y_cal)
    if name == "isotonic":
        return IsotonicCalibrator().fit(logits_cal, y_cal)
    if name == "vector":
        return VectorScalingCalibrator().fit(logits_cal, y_cal)
    if name == "dirichlet":
        return DirichletCalibrator().fit(logits_cal, y_cal)
    if name == "sebcal":
        if X_cal is None:
            raise ValueError("sebcal requires raw calibration sequences")
        return SebCalibrator().fit(logits_cal, y_cal, X_cal)
    raise ValueError(f"Unknown calibrator {name}")
