import numpy as np
from sklearn.linear_model import LogisticRegression


def _conf_quantile(residuals, alpha):
    n = len(residuals)
    if n == 0:
        return np.inf
    level = min(1.0, np.ceil((n + 1) * (1 - alpha)) / n)
    return float(np.quantile(residuals, level, method="higher"))


def split_conformal(calib_pred, calib_y, test_pred, alpha=0.1):
    res = np.abs(calib_y - calib_pred)
    q = _conf_quantile(res, alpha)
    lo, hi = test_pred - q, test_pred + q
    width = np.full_like(test_pred, 2 * q, dtype=float)
    return lo, hi, width


def mondrian_conformal(calib_pred, calib_y, calib_dist, test_pred, test_dist, alpha=0.1, n_bins=5):
    """Conformal intervals stratified by distance-to-train (adapts to shift)."""
    res = np.abs(calib_y - calib_pred)
    edges = np.quantile(calib_dist, np.linspace(0, 1, n_bins + 1))
    edges[0], edges[-1] = -np.inf, np.inf
    lo = np.empty_like(test_pred, dtype=float)
    hi = np.empty_like(test_pred, dtype=float)
    width = np.empty_like(test_pred, dtype=float)
    for b in range(n_bins):
        c_mask = (calib_dist >= edges[b]) & (calib_dist < edges[b + 1])
        t_mask = (test_dist >= edges[b]) & (test_dist < edges[b + 1])
        if t_mask.sum() == 0:
            continue
        q = _conf_quantile(res[c_mask], alpha) if c_mask.sum() >= 5 else _conf_quantile(res, alpha)
        lo[t_mask] = test_pred[t_mask] - q
        hi[t_mask] = test_pred[t_mask] + q
        width[t_mask] = 2 * q
    return lo, hi, width


def weighted_conformal(calib_pred, calib_y, calib_fp, test_pred, test_fp, alpha=0.1):
    """Covariate-shift conformal with domain-classifier importance weights."""
    X = np.vstack([calib_fp, test_fp])
    y = np.concatenate([np.zeros(len(calib_fp)), np.ones(len(test_fp))])
    clf = LogisticRegression(max_iter=300, C=1.0)
    clf.fit(X, y)
    p = np.clip(clf.predict_proba(calib_fp)[:, 1], 1e-4, 1 - 1e-4)
    w = p / (1 - p)                         # ~ p_test / p_calib
    res = np.abs(calib_y - calib_pred)

    order = np.argsort(res)
    res_s, w_s = res[order], w[order]
    cw = np.cumsum(w_s)
    total = cw[-1] + w.mean()               # include test-point weight mass
    target = (1 - alpha) * total
    k = np.searchsorted(cw, target)
    q = float(res_s[min(k, len(res_s) - 1)])
    lo, hi = test_pred - q, test_pred + q
    width = np.full_like(test_pred, 2 * q, dtype=float)
    return lo, hi, width


def coverage(lo, hi, y):
    return float(np.mean((y >= lo) & (y <= hi)))


# ---- classification ----

def lac_sets(calib_prob, calib_y, test_prob, alpha=0.1):
    """Least-ambiguous-set conformal for binary tasks. Returns per-test set size and whether true label in set."""
    cp = np.stack([1 - calib_prob, calib_prob], axis=1)
    scores = 1 - cp[np.arange(len(calib_y)), calib_y.astype(int)]   # nonconformity
    q = _conf_quantile(scores, alpha)
    tp = np.stack([1 - test_prob, test_prob], axis=1)
    in_set = (1 - tp) <= q                  # (n,2) bool
    size = in_set.sum(axis=1)
    return in_set, size, q


def ece(prob, y, n_bins=10):
    edges = np.linspace(0, 1, n_bins + 1)
    e = 0.0
    for b in range(n_bins):
        m = (prob >= edges[b]) & (prob < edges[b + 1])
        if m.sum() == 0:
            continue
        conf = prob[m].mean()
        acc = y[m].mean()
        e += (m.sum() / len(prob)) * abs(conf - acc)
    return float(e)
