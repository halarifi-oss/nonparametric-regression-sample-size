"""
Monte Carlo study: effect of error distribution on the sample size needed
by kernel regression (Nadaraya-Watson) and a robust kernel (weighted-median) smoother.
"""
import numpy as np
from scipy import stats

# ---------------------------------------------------------------- true function
def f(x):
    return np.sin(x) + 0.5 * np.sin(3 * x)

# ---------------------------------------------------------------- error laws
# Every error law is (i) centred so that E[eps]=0 when the mean exists (skewed laws),
# symmetric laws are centred at 0 by construction, and
# (ii) rescaled so that its inter-quartile range equals that of N(0,1) (=1.349),
# so that the "bulk" of the noise has the same scale for all laws and only
# the tails/skewness differ.
IQR_N = stats.norm.ppf(.75) - stats.norm.ppf(.25)

def _spec(name):
    if name == "Normal":      d = stats.norm();            mean = 0.0
    elif name == "Exponential": d = stats.expon(scale=1);  mean = 1.0
    elif name == "Gamma":     d = stats.gamma(a=2, scale=1); mean = 2.0
    elif name.startswith("t"):
        df = float(name[2:-1]); d = stats.t(df);           mean = 0.0
    elif name == "Cauchy":    d = stats.cauchy();          mean = 0.0
    else: raise ValueError(name)
    iqr = d.ppf(.75) - d.ppf(.25)
    s = IQR_N / iqr
    return d, mean, s

DISTS = ["Normal", "Exponential", "Gamma", "t(3)", "Cauchy"]
# fixed RNG-stream id per law (so results are reproducible regardless of list order)
STREAM = {"Normal": 0, "Exponential": 1, "Gamma": 2, "t(3)": 5, "Cauchy": 7}
SYMMETRIC = {"Normal", "t(3)", "Cauchy"}

def draw_error(name, n, rng):
    d, mean, s = _spec(name)
    return (d.rvs(size=n, random_state=rng) - mean) * s

def error_facts(name):
    d, mean, s = _spec(name)
    if name in ("t(2)", "Cauchy"): var = np.inf
    else: var = float(d.var()) * s * s
    return dict(scale=s, variance=var)

# ---------------------------------------------------------------- smoothers
def hgrid(n):
    # 14 log-spaced bandwidths; lower bound = 1.5 average spacings (3/n) to avoid interpolation
    return np.geomspace(max(0.02, 3.0 / n), 0.9, 14)

def _loo_subset(n, rng, m=400):
    return np.arange(n) if n <= m else rng.choice(n, m, replace=False)

def nw_predict(xg, X, Y, h):
    W = np.exp(-0.5 * ((xg[:, None] - X[None, :]) / h) ** 2)
    s = W.sum(1)
    out = np.where(s > 1e-300, (W @ Y) / np.maximum(s, 1e-300), Y.mean())
    return out

def nw_cv_bandwidth(X, Y, rng):
    n = len(X); idx = _loo_subset(n, rng)
    D2 = (X[idx][:, None] - X[None, :]) ** 2
    best = (np.inf, None)
    ybar = Y.mean()
    for h in hgrid(n):
        W = np.exp(-D2 / (2 * h * h))
        W[np.arange(len(idx)), idx] = 0.0             # leave one out
        s = W.sum(1)
        pred = np.where(s > 1e-300, (W @ Y) / np.maximum(s, 1e-300), ybar)
        e = np.mean((Y[idx] - pred) ** 2)
        if e < best[0]: best = (e, h)
    return best[1]

def wmedian_predict(xg, X, Y, h, order=None):
    """Gaussian-kernel weighted median of Y (local-constant L1 smoother)."""
    if order is None: order = np.argsort(Y)
    Ys = Y[order]; Xs = X[order]
    W = np.exp(-0.5 * ((xg[:, None] - Xs[None, :]) / h) ** 2)
    cs = np.cumsum(W, axis=1)
    tot = cs[:, -1]
    j = (cs >= 0.5 * tot[:, None]).argmax(1)
    out = Ys[j]
    return np.where(tot > 1e-300, out, np.median(Y))

def wmedian_cv_bandwidth(X, Y, rng):
    n = len(X); idx = _loo_subset(n, rng)
    order = np.argsort(Y); Ys = Y[order]; Xs = X[order]
    pos = np.empty(n, int); pos[order] = np.arange(n)   # position of obs i in sorted order
    D2 = (X[idx][:, None] - Xs[None, :]) ** 2
    best = (np.inf, None)
    for h in hgrid(n):
        W = np.exp(-D2 / (2 * h * h))
        W[np.arange(len(idx)), pos[idx]] = 0.0          # leave one out
        cs = np.cumsum(W, axis=1); tot = cs[:, -1]
        j = (cs >= 0.5 * tot[:, None]).argmax(1)
        pred = np.where(tot > 1e-300, Ys[j], np.median(Y))
        e = np.mean(np.abs(Y[idx] - pred))
        if e < best[0]: best = (e, h)
    return best[1]

# ---------------------------------------------------------------- one replication
XG = np.linspace(0, 2, 201)
INT = (XG >= 0.1) & (XG <= 1.9)      # interior (boundary-free) evaluation region
FG = f(XG)

def one_rep(dist, n, rng, robust):
    X = rng.uniform(0, 2, n)
    Y = f(X) + draw_error(dist, n, rng)
    h1 = nw_cv_bandwidth(X, Y, rng)
    fh1 = nw_predict(XG, X, Y, h1)
    res = dict(h_nw=h1, fh_nw=fh1,
               mse_nw=np.mean((fh1[INT] - FG[INT]) ** 2),
               mse_nw_full=np.mean((fh1 - FG) ** 2))
    if robust:
        h2 = wmedian_cv_bandwidth(X, Y, rng)
        fh2 = wmedian_predict(XG, X, Y, h2)
        res.update(h_rb=h2, fh_rb=fh2,
                   mse_rb=np.mean((fh2[INT] - FG[INT]) ** 2),
                   mse_rb_full=np.mean((fh2 - FG) ** 2))
    return res
