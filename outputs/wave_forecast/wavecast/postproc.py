"""Statistical benchmark and diagnostics on the shared 99-quantile grid.

EMOS (Gneiting et al. 2005) with truncated-normal (Thorarinsdottir & Gneiting 2010) and log-normal
(Baran & Lerch 2015) predictive distributions, fitted per site and lead day by minimum CRPS;
CRPS reliability / resolution / uncertainty decomposition (Hersbach 2000); threshold-conditional PIT
(Allen et al. 2023); quantile-wise one-sided conformalized quantile regression (Romano et al. 2019).
"""
import numpy as np
from scipy.optimize import minimize
from scipy.stats import norm
from .metrics import TAU

SQPI = 1 / np.sqrt(np.pi)


def crps_tn(mu, sig, y):
    """CRPS of the normal distribution left-truncated at zero."""
    r = mu / sig
    z = (y - mu) / sig
    p = norm.cdf(r)
    return sig / p**2 * (z * p * (2 * norm.cdf(z) + p - 2) + 2 * norm.pdf(z) * p - SQPI * norm.cdf(np.sqrt(2) * r))


def crps_ln(m, s, y):
    w = (np.log(np.maximum(y, 1e-3)) - m) / s
    return y * (2 * norm.cdf(w) - 1) - 2 * np.exp(m + s**2 / 2) * (norm.cdf(w - s) + norm.cdf(s / np.sqrt(2)) - 1)


def _params(kind, th, mean, spread):
    a, b, c, d = th
    loc = a + b * mean
    sd = np.sqrt(c**2 + d**2 * spread**2) + 1e-3
    if kind == "tn":
        return loc, sd
    # log-normal: EMOS fits the predictive mean and variance; convert to log-space parameters
    loc = np.maximum(loc, .05)
    s2 = np.log1p(sd**2 / loc**2)
    return np.log(loc) - s2 / 2, np.sqrt(s2)


def _crps(kind, th, mean, spread, y):
    p1, p2 = _params(kind, th, mean, spread)
    return (crps_tn if kind == "tn" else crps_ln)(p1, p2, y)


def emos_fit(mean, spread, y, groups, kind):
    """Minimum-CRPS coefficients (a, b, c, d) per group; one fit per site × lead day."""
    out = {}
    for g in np.unique(groups):
        k = groups == g
        res = minimize(lambda th: _crps(kind, th, mean[k], spread[k], y[k]).mean(), x0=[0., 1., .3, 1.],
                       method="L-BFGS-B")
        out[g] = res.x
    return out


def emos_quantiles(coef, mean, spread, groups, kind):
    q = np.empty((len(mean), len(TAU)), np.float32)
    for g in np.unique(groups):
        k = groups == g
        th = coef.get(g)
        if th is None:
            raise ValueError(f"No EMOS coefficients for group {g}")
        p1, p2 = _params(kind, th, mean[k], spread[k])
        if kind == "tn":
            lo = norm.cdf(-p1 / p2)
            q[k] = p1[:, None] + p2[:, None] * norm.ppf(lo[:, None] + TAU * (1 - lo[:, None]))
        else:
            q[k] = np.exp(p1[:, None] + p2[:, None] * norm.ppf(TAU))
    return q


def ensemble_inputs(samples, names):
    """Ensemble mean (the raw target column) and spread for every forecast row."""
    mean = np.concatenate([s["raw_mean"] for s in samples]).astype(float)
    spread = np.concatenate([s["future"][:, names.index("hs_spread")] for s in samples]).astype(float)
    return mean, np.nan_to_num(spread, nan=np.nanmean(spread))


def emos_group_keys(samples):
    return np.array([f"{m['site_id']}|{m['lead_day']}" for s in samples for m in s["metadata"]])


def emos_benchmark(train, target, names, kinds=("tn", "ln")):
    """Fit EMOS on the training split and return {kind: quantiles on `target`}."""
    mt, st = ensemble_inputs(train, names)
    yt = np.concatenate([s["y"] for s in train]).astype(float)
    gt, gv = emos_group_keys(train), emos_group_keys(target)
    mv, sv = ensemble_inputs(target, names)
    out = {}
    for kind in kinds:
        coef = emos_fit(mt, st, yt, gt, kind)
        # a target group absent from training (e.g. a site without training years) uses the same lead day
        # pooled over sites: average coefficients of that lead day
        for g in np.unique(gv):
            if g not in coef:
                day = g.split("|")[1]
                pool = [v for k, v in coef.items() if k.split("|")[1] == day]
                coef[g] = np.mean(pool, axis=0)
        out[kind] = emos_quantiles(coef, mv, sv, gv, kind)
    return out


def hersbach(q, y):
    """Hersbach (2000) decomposition of the CRPS of the 99-quantile forecasts, read as equally weighted
    members (the score in metrics.score equals the CRPS of that empirical distribution).
    Returns crps = reliability + potential, potential = uncertainty - resolution."""
    x = np.sort(np.asarray(q, float), axis=1)
    y = np.asarray(y, float)
    n = x.shape[1]
    p = np.arange(n + 1) / n
    lo, hi = x[:, :-1], x[:, 1:]
    width = hi - lo
    alpha = np.zeros((len(y), n + 1)); beta = np.zeros((len(y), n + 1))
    alpha[:, 1:n] = np.clip(y[:, None] - lo, 0, width)
    beta[:, 1:n] = np.clip(hi - y[:, None], 0, width)
    beta[:, 0] = np.maximum(x[:, 0] - y, 0)
    alpha[:, n] = np.maximum(y - x[:, -1], 0)
    a, b = alpha.mean(0), beta.mean(0)
    g = a + b
    o = np.divide(b, g, out=np.zeros_like(g), where=g > 0)
    o0, on = (y < x[:, 0]).mean(), 1 - (y > x[:, -1]).mean()
    o[0], o[n] = o0, on
    g[0] = b[0] / o0 if o0 > 0 else 0.
    g[n] = a[n] / (1 - on) if on < 1 else 0.
    reli = float((g * (o - p) ** 2).sum())
    pot = float((g * o * (1 - o)).sum())
    ys = np.sort(y); k = np.arange(1, len(ys)) / len(ys)
    unc = float((k * (1 - k) * np.diff(ys)).sum())
    crps = float((alpha * p**2 + beta * (1 - p) ** 2).sum(1).mean())
    return dict(crps=crps, reliability=reli, resolution=unc - pot, uncertainty=unc, potential=pot)


def cdf_at(q, v, rng=None):
    """Forecast CDF at v (per row) by linear interpolation of the quantiles; beyond the outermost quantiles
    the remaining 0.5 % tail mass is spread uniformly (randomised) when rng is given, else set to the edge."""
    q = np.asarray(q, float); v = np.broadcast_to(np.asarray(v, float), (len(q),))
    f = np.array([np.interp(t, row, TAU) for row, t in zip(q, v)])
    if rng is not None:
        u = rng.random(len(q))
        f = np.where(v < q[:, 0], TAU[0] * u, f)
        f = np.where(v > q[:, -1], TAU[-1] + (1 - TAU[-1]) * u, f)
    else:
        f = np.where(v < q[:, 0], 0., np.where(v > q[:, -1], 1., f))
    return f


def pit(q, y, rng):
    return cdf_at(q, y, rng)


def conditional_pit(q, y, threshold, rng):
    """PIT of the forecast distribution conditioned on exceeding the threshold, for cases whose outcome exceeds it
    (Allen et al. 2023): (F(y) - F(t)) / (1 - F(t)). A forecast with no mass above t gets PIT 1."""
    k = np.asarray(y) > threshold
    f_y, f_t = cdf_at(q[k], y[k], rng), cdf_at(q[k], threshold)
    tail = 1 - f_t
    return np.where(tail > 1e-6, (f_y - f_t) / np.maximum(tail, 1e-6), 1.), k


def conformal_shifts(q, y, groups):
    """Quantile-wise one-sided CQR (Romano et al. 2019, asymmetric form): for each group and level tau the
    shift is the ceil((n+1) tau)-th smallest residual y - q_tau, so that q_tau + shift covers y with frequency tau."""
    q = np.asarray(q, float); y = np.asarray(y, float)
    out = {}
    for g in np.unique(groups):
        k = groups == g
        r = y[k, None] - q[k]
        n = k.sum()
        lev = np.minimum(np.ceil((n + 1) * TAU) / n, 1.)
        out[g] = np.array([np.quantile(r[:, j], lev[j], method="higher") for j in range(len(TAU))])
    return out


def linear_pool(a, b, w):
    """Quantiles of the linear pool (1 - w) F_a + w F_b (CDF mixture, evaluated on a dense grid per row);
    Wessel et al. 2025 pool a body-oriented and a tail-oriented forecast this way."""
    out = np.empty_like(np.asarray(a, np.float32))
    for i, (x, z) in enumerate(zip(a, b)):
        grid = np.unique(np.concatenate([x, z, [0.]]))
        cdf = (1 - w) * np.interp(grid, x, TAU, left=0, right=1) + w * np.interp(grid, z, TAU, left=0, right=1)
        out[i] = np.interp(TAU, cdf, grid)
    return out


def apply_shifts(q, groups, shifts):
    """Add the group's shifts and restore monotone quantiles by sorting (rearrangement); groups without
    shifts and negative values are left / clipped."""
    q = np.array(q, float)
    for g in np.unique(groups):
        if g in shifts:
            k = groups == g
            q[k] = q[k] + shifts[g]
    return np.maximum(np.sort(q, axis=1), 0.).astype(np.float32)
