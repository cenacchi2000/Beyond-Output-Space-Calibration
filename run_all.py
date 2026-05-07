from __future__ import annotations
import copy
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

class TSData(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)
    def __len__(self): return len(self.X)
    def __getitem__(self, idx): return self.X[idx], self.y[idx]

def train_model(model, X_train, y_train, X_val, y_val, device, epochs=30, batch_size=64, lr=1e-3):
    train_loader = DataLoader(TSData(X_train, y_train), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(TSData(X_val, y_val), batch_size=batch_size, shuffle=False)
    model = model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    crit = nn.CrossEntropyLoss()
    best_state = None
    best_val = float("inf")
    patience = 6
    bad = 0

    for _ in range(epochs):
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            loss = crit(model(xb), yb)
            loss.backward()
            opt.step()
        model.eval()
        val_loss = 0.0
        n = 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                logits = model(xb)
                loss = crit(logits, yb)
                val_loss += float(loss) * len(xb)
                n += len(xb)
        val_loss /= max(n, 1)
        if val_loss < best_val:
            best_val = val_loss
            best_state = copy.deepcopy(model.state_dict())
            bad = 0
        else:
            bad += 1
            if bad >= patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model

def predict_logits(model, X, device, batch_size=128):
    loader = DataLoader(torch.tensor(X, dtype=torch.float32), batch_size=batch_size, shuffle=False)
    model.eval()
    outs = []
    with torch.no_grad():
        for xb in loader:
            xb = xb.to(device)
            outs.append(model(xb).cpu())
    return torch.cat(outs, dim=0).numpy()
