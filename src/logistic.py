"""Logistic regression and restricted cubic splines.

Implements maximum-likelihood and Firth penalised logistic regression,
cluster-robust (sandwich) variance at the patient level, and Harrell's
restricted cubic spline basis. Only NumPy and SciPy are used.

Logistic regression is used for the successful regression endpoint, which
could not be analysed on the time scale because the date on which the volume
reduction rate first reached 90% was not recorded.
"""
import numpy as np
from scipy import stats


def _loglik(X, y, b):
    eta = X @ b
    return np.sum(y * eta - np.logaddexp(0, eta))


def _pen_loglik(X, y, b):
    """Firth penalised log-likelihood: loglik + 0.5 * log|I(beta)|."""
    p = 1.0 / (1.0 + np.exp(-(X @ b)))
    W = p * (1 - p)
    I = X.T @ (X * W[:, None])
    _, ld = np.linalg.slogdet(I)
    return _loglik(X, y, b) + 0.5 * ld


class LogitFit:
    """Container for a fitted logistic model."""


def logit_fit(X, y, firth=False, maxit=60, tol=1e-8):
    """Logistic regression by Newton-Raphson with step halving.

    Parameters
    ----------
    X : (n, p) design matrix; include an intercept column explicitly
    y : binary outcome
    firth : if True, use Firth penalised likelihood (reduces small-sample bias
        and is stable under separation)
    """
    X = np.asarray(X, float)
    y = np.asarray(y, float)
    n, k = X.shape
    b = np.zeros(k)
    obj = _pen_loglik if firth else _loglik
    cur = obj(X, y, b)

    for _ in range(maxit):
        p = 1.0 / (1.0 + np.exp(-(X @ b)))
        W = np.clip(p * (1 - p), 1e-10, None)
        I = X.T @ (X * W[:, None])
        Iinv = np.linalg.pinv(I)
        if firth:
            R = X * np.sqrt(W)[:, None]
            h = np.einsum('ij,jk,ik->i', R, Iinv, R)
            U = X.T @ (y - p + h * (0.5 - p))
        else:
            U = X.T @ (y - p)
        step = Iinv @ U
        ok = False
        for _s in range(20):
            nb = b + step
            nv = obj(X, y, nb)
            if np.isfinite(nv) and nv >= cur - 1e-12:
                ok = True
                break
            step = step / 2.0
        if not ok:
            break
        move = np.max(np.abs(nb - b))
        b, cur = nb, nv
        if move < tol:
            break

    p = 1.0 / (1.0 + np.exp(-(X @ b)))
    W = np.clip(p * (1 - p), 1e-10, None)
    I = X.T @ (X * W[:, None])

    f = LogitFit()
    f.beta, f.X, f.y, f.p = b, X, y, p
    f.cov = np.linalg.pinv(I)
    f.loglik = _loglik(X, y, b)
    f.firth = firth
    f.n, f.k = n, k
    return f


def cluster_cov(f, groups):
    """Cluster-robust covariance with a small-sample correction."""
    groups = np.asarray(groups)
    u = f.X * (f.y - f.p)[:, None]
    M = np.zeros((f.k, f.k))
    G = 0
    for g in np.unique(groups):
        s = u[groups == g].sum(axis=0)
        M += np.outer(s, s)
        G += 1
    c = (G / (G - 1.0)) * ((f.n - 1.0) / (f.n - f.k))
    return c * (f.cov @ M @ f.cov)


def wald(f, cov):
    """Odds ratios with 95% CIs and Wald P values."""
    se = np.sqrt(np.diag(cov))
    z = f.beta / se
    lo, hi = f.beta - 1.96 * se, f.beta + 1.96 * se
    return dict(beta=f.beta, se=se, p=2 * stats.norm.sf(np.abs(z)),
                OR=np.exp(f.beta), OR_lo=np.exp(lo), OR_hi=np.exp(hi))


def lr_test(f_full, f_red):
    """Likelihood-ratio test between nested models."""
    chi2 = 2 * (f_full.loglik - f_red.loglik)
    df = f_full.k - f_red.k
    return chi2, df, stats.chi2.sf(chi2, df)


# ------------------------------------------------------ restricted cubic spline
def rcs(x, knots):
    """Harrell restricted cubic spline basis; returns len(knots) - 1 columns."""
    x = np.asarray(x, float)
    t = np.asarray(knots, float)
    k = len(t)
    den = (t[-1] - t[0]) ** 2
    cols = [x]
    for j in range(k - 2):
        cp = lambda u: np.clip(u, 0, None) ** 3
        term = (cp(x - t[j])
                - cp(x - t[k - 2]) * (t[k - 1] - t[j]) / (t[k - 1] - t[k - 2])
                + cp(x - t[k - 1]) * (t[k - 2] - t[j]) / (t[k - 1] - t[k - 2]))
        cols.append(term / den)
    return np.column_stack(cols)


def knot_quantiles(x, nk=4):
    """Default knot placement (Harrell): quantiles of the observed distribution."""
    q = {3: [.10, .50, .90],
         4: [.05, .35, .65, .95],
         5: [.05, .275, .50, .725, .95]}[nk]
    return np.quantile(x, q)
