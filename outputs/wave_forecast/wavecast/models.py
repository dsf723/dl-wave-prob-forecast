"""Matched nonnegative Bernstein quantile heads; direct multi-step outputs."""
import math
import torch
from torch import nn
from torch.nn import functional as F


class ResidualMLP(nn.Module):
    def __init__(self, n_in, n_out, hidden, dropout):
        super().__init__()
        self.body = nn.Sequential(nn.Linear(n_in, hidden), nn.GELU(),
                                  nn.Dropout(dropout), nn.Linear(hidden, n_out))
        self.skip = nn.Linear(n_in, n_out) if n_in != n_out else nn.Identity()
        self.norm = nn.LayerNorm(n_out)

    def forward(self, x):
        return self.norm(self.body(x) + self.skip(x))


class BernsteinHead(nn.Module):
    def __init__(self, hidden, degree=12, n_quantiles=99):
        super().__init__()
        self.linear = nn.Linear(hidden, degree + 1)
        # Midpoint quadrature covers the full (0,1) interval, including tails.
        tau = (torch.arange(n_quantiles, dtype=torch.float32) + .5) / n_quantiles
        k = torch.arange(degree + 1, dtype=torch.float32)
        coeff = torch.tensor([math.comb(degree, j) for j in range(degree + 1)])
        basis = coeff * tau[:, None] ** k * (1 - tau[:, None]) ** (degree - k)
        self.register_buffer("tau", tau)
        self.register_buffer("basis", basis)

    def forward(self, x):
        # Positive cumulative coefficients enforce nonnegative, ordered quantiles.
        alpha = F.softplus(self.linear(x)).cumsum(-1) / self.basis.shape[1]
        return alpha @ self.basis.T


class TiDE(nn.Module):
    """TiDE-style dense encoder/decoder, temporal covariates, linear skip.

    An adaptation with a probability head, not an exact original-paper replica.
    """
    def __init__(self, n_features, history, horizon, cfg):
        super().__init__()
        d = cfg["hidden"]
        self.horizon = horizon
        self.project = ResidualMLP(n_features, 8, d, cfg["dropout"])
        blocks = [ResidualMLP((history + horizon) * 8, d, d * 2, cfg["dropout"])]
        blocks += [ResidualMLP(d, d, d * 2, cfg["dropout"]) for _ in range(cfg["layers"] - 1)]
        self.encoder = nn.Sequential(*blocks)
        self.decoder = ResidualMLP(d, horizon * 16, d * 2, cfg["dropout"])
        self.temporal = ResidualMLP(24, d, d, cfg["dropout"])
        self.linear_skip = nn.Linear(history, horizon)
        self.head = BernsteinHead(d, cfg["degree"])

    def forward(self, past, future):
        p, f = self.project(past), self.project(future)
        z = self.encoder(torch.cat([p.flatten(1), f.flatten(1)], dim=-1))
        z = self.decoder(z).reshape(-1, self.horizon, 16)
        z = self.temporal(torch.cat([z, f], dim=-1))
        # Feature zero is normalized historical target; add skip before positive head.
        z = z + self.linear_skip(past[:, :, 0])[:, :, None]
        return self.head(z)


class Transformer(nn.Module):
    def __init__(self, n_features, history, horizon, cfg):
        super().__init__()
        d = cfg["hidden"]
        self.project = nn.Linear(n_features, d)
        self.position = nn.Parameter(torch.randn(history + horizon, d) * .02)
        enc = nn.TransformerEncoderLayer(d, cfg["heads"], 4*d,
                                        cfg["dropout"], batch_first=True, activation="gelu")
        dec = nn.TransformerDecoderLayer(d, cfg["heads"], 4*d,
                                        cfg["dropout"], batch_first=True, activation="gelu")
        self.encoder = nn.TransformerEncoder(enc, cfg["layers"])
        self.decoder = nn.TransformerDecoder(dec, cfg["layers"])
        self.head = BernsteinHead(d, cfg["degree"])
        self.history = history

    def forward(self, past, future):
        p = self.project(past) + self.position[:self.history]
        f = self.project(future) + self.position[self.history:]
        # Entire future forecast trajectory is already known at decision time.
        return self.head(self.decoder(f, self.encoder(p)))


class GRUNet(nn.Module):
    """GRU encoder-decoder baseline (cf. Jin et al. 2025, Ocean Eng. 338: GRU best among RNN/CNN/LSTM/GRU).

    The encoder reads the past window; its final state starts a decoder GRU that reads the known future
    forecast trajectory; the shared Bernstein head turns each future step into ordered quantiles.
    """
    def __init__(self, n_features, history, horizon, cfg):
        super().__init__()
        d = cfg["hidden"]
        self.project = nn.Linear(n_features, d)
        drop = cfg["dropout"] if cfg["layers"] > 1 else 0.0
        self.encoder = nn.GRU(d, d, cfg["layers"], batch_first=True, dropout=drop)
        self.decoder = nn.GRU(d, d, cfg["layers"], batch_first=True, dropout=drop)
        self.norm = nn.LayerNorm(d)
        self.head = BernsteinHead(d, cfg["degree"])

    def forward(self, past, future):
        _, h = self.encoder(self.project(past))
        z, _ = self.decoder(self.project(future), h)
        return self.head(self.norm(z))


class CNNBiGRU(nn.Module):
    """CNN + bidirectional GRU + multi-head attention baseline (cf. Xia et al. 2026, Ocean Eng. 368, CNN-BiGRU-MA).

    Past and future steps are joined in time; a 1-D convolution extracts local patterns, a BiGRU models the
    sequence in both directions, self-attention re-weights steps; the future part feeds the Bernstein head.
    """
    def __init__(self, n_features, history, horizon, cfg):
        super().__init__()
        d = cfg["hidden"]
        self.history = history
        self.conv = nn.Sequential(nn.Conv1d(n_features, d, kernel_size=3, padding=1), nn.GELU(),
                                  nn.Dropout(cfg["dropout"]))
        drop = cfg["dropout"] if cfg["layers"] > 1 else 0.0
        self.bigru = nn.GRU(d, d // 2, cfg["layers"], batch_first=True, bidirectional=True, dropout=drop)
        self.attention = nn.MultiheadAttention(d, cfg["heads"], dropout=cfg["dropout"], batch_first=True)
        self.norm = nn.LayerNorm(d)
        self.head = BernsteinHead(d, cfg["degree"])

    def forward(self, past, future):
        x = torch.cat([past, future], dim=1)                     # (batch, history + horizon, features)
        x = self.conv(x.transpose(1, 2)).transpose(1, 2)
        x, _ = self.bigru(x)
        a, _ = self.attention(x, x, x, need_weights=False)
        x = self.norm(x + a)
        return self.head(x[:, self.history:])


def crps_loss(q, y, tau):
    e = y[..., None] - q
    return 2 * torch.maximum(tau * e, (tau - 1) * e).mean()


def crps_loss_masked(q, y, tau):
    """CRPS averaged over finite targets only (sites / times without an observed value are skipped)."""
    ok = torch.isfinite(y)
    if not ok.any():
        return q.sum() * 0.0
    e = torch.where(ok, y, torch.zeros_like(y))[..., None] - q
    per = 2 * torch.maximum(tau * e, (tau - 1) * e).mean(-1)
    return per[ok].mean()


TE_SCALE = 5.0          # the period head predicts Te / 5 s, so its outputs are of the same order as Hs in metres


def add_te_head(model, cfg):
    """Second Bernstein head for the energy period Te on the same hidden state as the Hs head (multi-task).

    A forward hook keeps the input of the Hs head; te_quantiles(model) maps it to Te quantiles (seconds)."""
    d = model.head.linear.in_features
    model.te_head = BernsteinHead(d, cfg["degree"])
    def keep(module, inputs, output):
        model._te_hidden = inputs[0]
    model.head.register_forward_hook(keep)
    return model


def te_quantiles(model):
    return model.te_head(model._te_hidden) * TE_SCALE


def twcrps_loss(q, y, tau, threshold):
    """Threshold-weighted CRPS with weight 1{z >= threshold} (Gneiting & Ranjan 2011; Allen et al. 2023).

    Equals the CRPS of forecast and observation both censored below the threshold; censoring is monotone,
    so the censored quantiles stay valid. A proper score: adding it to the CRPS emphasises large waves
    without rewarding hedging toward extremes (unlike weighting cases by the observed outcome).
    """
    return crps_loss(torch.clamp(q, min=threshold), torch.clamp(y, min=threshold), tau)
