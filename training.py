from __future__ import annotations
import math
from typing import Dict
import torch
from torch import nn

class MLP1D(nn.Module):
    def __init__(self, c: int, t: int, k: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(c*t, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, k),
        )
    def forward(self, x): return self.net(x)

class FCN1D(nn.Module):
    def __init__(self, c: int, t: int, k: int):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(c, 128, 8, padding=4), nn.BatchNorm1d(128), nn.ReLU(),
            nn.Conv1d(128, 256, 5, padding=2), nn.BatchNorm1d(256), nn.ReLU(),
            nn.Conv1d(256, 128, 3, padding=1), nn.BatchNorm1d(128), nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.head = nn.Linear(128, k)
    def forward(self, x):
        h = self.features(x).squeeze(-1)
        return self.head(h)

class ResidualBlock(nn.Module):
    def __init__(self, cin, cout, k):
        super().__init__()
        pad = k // 2
        self.block = nn.Sequential(
            nn.Conv1d(cin, cout, k, padding=pad), nn.BatchNorm1d(cout), nn.ReLU(),
            nn.Conv1d(cout, cout, k, padding=pad), nn.BatchNorm1d(cout), nn.ReLU(),
            nn.Conv1d(cout, cout, k, padding=pad), nn.BatchNorm1d(cout),
        )
        self.skip = nn.Conv1d(cin, cout, 1) if cin != cout else nn.Identity()
        self.act = nn.ReLU()
    def forward(self, x):
        return self.act(self.block(x) + self.skip(x))

class ResNet1D(nn.Module):
    def __init__(self, c, t, k):
        super().__init__()
        self.net = nn.Sequential(
            ResidualBlock(c, 64, 7),
            ResidualBlock(64, 128, 5),
            ResidualBlock(128, 128, 3),
            nn.AdaptiveAvgPool1d(1),
        )
        self.head = nn.Linear(128, k)
    def forward(self, x):
        h = self.net(x).squeeze(-1)
        return self.head(h)

class TCN1D(nn.Module):
    def __init__(self, c, t, k):
        super().__init__()
        layers = []
        dims = [c, 64, 64, 128]
        dilations = [1, 2, 4]
        for i, d in enumerate(dilations):
            cin, cout = dims[i], dims[i+1]
            layers.extend([
                nn.Conv1d(cin, cout, 3, padding=d, dilation=d),
                nn.ReLU(),
                nn.BatchNorm1d(cout),
            ])
        self.features = nn.Sequential(*layers, nn.AdaptiveAvgPool1d(1))
        self.head = nn.Linear(128, k)
    def forward(self, x):
        h = self.features(x).squeeze(-1)
        return self.head(h)

class RNNBase(nn.Module):
    def __init__(self, c, t, k, kind="lstm", hidden=128):
        super().__init__()
        rnn_cls = nn.LSTM if kind == "lstm" else nn.GRU
        self.rnn = rnn_cls(input_size=c, hidden_size=hidden, batch_first=True, num_layers=2, dropout=0.2)
        self.head = nn.Linear(hidden, k)
        self.kind = kind
    def forward(self, x):
        x = x.transpose(1, 2)
        out, h = self.rnn(x)
        if self.kind == "lstm":
            h = h[0][-1]
        else:
            h = h[-1]
        return self.head(h)

class InceptionBlock(nn.Module):
    def __init__(self, cin, cout=32):
        super().__init__()
        self.b1 = nn.Conv1d(cin, cout, 9, padding=4)
        self.b2 = nn.Conv1d(cin, cout, 19, padding=9)
        self.b3 = nn.Conv1d(cin, cout, 39, padding=19)
        self.pool = nn.Sequential(nn.MaxPool1d(3, stride=1, padding=1), nn.Conv1d(cin, cout, 1))
        self.bn = nn.BatchNorm1d(cout*4)
        self.act = nn.ReLU()
    def forward(self, x):
        h = torch.cat([self.b1(x), self.b2(x), self.b3(x), self.pool(x)], dim=1)
        return self.act(self.bn(h))

class InceptionLite(nn.Module):
    def __init__(self, c, t, k):
        super().__init__()
        self.net = nn.Sequential(
            InceptionBlock(c, 32),
            InceptionBlock(128, 32),
            InceptionBlock(128, 32),
            nn.AdaptiveAvgPool1d(1),
        )
        self.head = nn.Linear(128, k)
    def forward(self, x):
        h = self.net(x).squeeze(-1)
        return self.head(h)

class Transformer1D(nn.Module):
    def __init__(self, c, t, k, d_model=128, nhead=4):
        super().__init__()
        self.proj = nn.Linear(c, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=256, dropout=0.1, batch_first=True
        )
        self.enc = nn.TransformerEncoder(encoder_layer, num_layers=3)
        self.cls = nn.Parameter(torch.zeros(1, 1, d_model))
        self.head = nn.Linear(d_model, k)
    def forward(self, x):
        x = x.transpose(1, 2)
        x = self.proj(x)
        cls = self.cls.expand(x.shape[0], -1, -1)
        x = torch.cat([cls, x], dim=1)
        h = self.enc(x)[:, 0]
        return self.head(h)

def build_model(name: str, c: int, t: int, k: int) -> nn.Module:
    name = name.lower()
    registry: Dict[str, nn.Module] = {
        "mlp": MLP1D(c, t, k),
        "lstm": RNNBase(c, t, k, kind="lstm"),
        "gru": RNNBase(c, t, k, kind="gru"),
        "tcn": TCN1D(c, t, k),
        "fcn": FCN1D(c, t, k),
        "resnet1d": ResNet1D(c, t, k),
        "inceptionlite": InceptionLite(c, t, k),
        "transformer": Transformer1D(c, t, k),
    }
    if name not in registry:
        raise ValueError(f"Unknown model: {name}")
    return registry[name]
