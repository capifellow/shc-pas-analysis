"""Descriptive statistics and model diagnostics.

* clopper_pearson - exact binomial confidence interval for a proportion
* proportion      - formatted "k/n (%; 95% CI)" summary
* vif             - variance inflation factors for a design matrix
* variance_share  - proportion of the variance of one variable explained by
                    another (used to quantify how far the alcohol-to-cyst
                    volume ratio is determined by baseline cyst volume)
"""
import numpy as np
from scipy import stats


def clopper_pearson(k, n, alpha=0.05):
    """Exact (Clopper-Pearson) 95% CI for a binomial proportion."""
    k, n = int(k), int(n)
    if n == 0:
        return np.nan, np.nan, np.nan
    lo = 0.0 if k == 0 else stats.beta.ppf(alpha / 2, k, n - k + 1)
    hi = 1.0 if k == n else stats.beta.ppf(1 - alpha / 2, k + 1, n - k)
    return k / n, float(lo), float(hi)


def proportion(k, n, alpha=0.05):
    """Formatted proportion with its exact confidence interval."""
    if n == 0:
        return 'n/a'
    p, lo, hi = clopper_pearson(k, n, alpha)
    return f'{k}/{n} ({100*p:.1f}%; 95% CI {100*lo:.1f}-{100*hi:.1f})'


def vif(X):
    """Variance inflation factors for the columns of a design matrix.

    Constant columns and columns that are collinear to numerical precision
    return inf. No intercept is required in X; one is added internally.
    """
    X = np.asarray(X, float)
    n, p = X.shape
    out = np.empty(p)
    for j in range(p):
        y = X[:, j]
        Z = np.column_stack([np.ones(n), np.delete(X, j, axis=1)])
        beta, *_ = np.linalg.lstsq(Z, y, rcond=None)
        resid = y - Z @ beta
        sst = float(np.sum((y - y.mean()) ** 2))
        r2 = 1.0 - float(np.sum(resid ** 2)) / sst if sst > 0 else 1.0
        out[j] = np.inf if r2 >= 1 - 1e-12 else 1.0 / (1.0 - r2)
    return out


def variance_share(y, x):
    """R^2 of a simple regression of y on x (both on the analysed scale)."""
    y, x = np.asarray(y, float), np.asarray(x, float)
    r = np.corrcoef(x, y)[0, 1]
    return float(r ** 2)
