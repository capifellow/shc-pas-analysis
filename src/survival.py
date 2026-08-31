"""Survival analysis routines used in the primary analysis of complete regression.

Implements Kaplan-Meier estimation, the log-rank test, Cox proportional hazards
regression with Breslow handling of ties, Lin-Wei cluster-robust (sandwich)
variance at the patient level, the Grambsch-Therneau test of the proportional
hazards assumption, and Harrell's c statistic for right-censored data.

No third-party survival package is required; only NumPy and SciPy are used.
Every routine was verified against simulated data with known parameters
(see scripts/validate_implementations.py).
"""
import numpy as np
from scipy import stats


# --------------------------------------------------------------- Kaplan-Meier
def km(t, e):
    """Kaplan-Meier estimate.

    Parameters
    ----------
    t : array of follow-up times
    e : array of event indicators (1 = event, 0 = censored)

    Returns
    -------
    ndarray with one row per distinct event time and columns
    [time, n at risk, n events, S(t), lower 95% CI of S, upper 95% CI of S].
    Confidence limits use the Greenwood variance estimate.
    """
    t = np.asarray(t, float)
    e = np.asarray(e, int)
    times = np.unique(t[e == 1])
    S, var, out = 1.0, 0.0, []
    for tt in times:
        n = np.sum(t >= tt - 1e-12)
        d = np.sum((np.abs(t - tt) < 1e-12) & (e == 1))
        S *= (1 - d / n)
        var += d / (n * max(n - d, 1e-9))
        se = S * np.sqrt(var)
        out.append((tt, n, d, S, max(0.0, S - 1.96 * se), min(1.0, S + 1.96 * se)))
    return np.array(out) if out else np.zeros((0, 6))


def km_at(tab, t):
    """Cumulative incidence (1 - S) at time `t`, with 95% CI."""
    if len(tab) == 0:
        return 0.0, 0.0, 0.0
    m = tab[:, 0] <= t
    if not m.any():
        return 0.0, 0.0, 0.0
    r = tab[m][-1]
    return 1 - r[3], 1 - r[5], 1 - r[4]


def logrank(t, e, g):
    """Two-group log-rank test. Returns (chi-square, P)."""
    t = np.asarray(t, float)
    e = np.asarray(e, int)
    g = np.asarray(g)
    lev = np.unique(g)
    if len(lev) != 2:
        raise ValueError('logrank requires exactly two groups')
    O_E, V = 0.0, 0.0
    for tt in np.unique(t[e == 1]):
        n = np.sum(t >= tt - 1e-12)
        d = np.sum((np.abs(t - tt) < 1e-12) & (e == 1))
        n1 = np.sum((t >= tt - 1e-12) & (g == lev[1]))
        d1 = np.sum((np.abs(t - tt) < 1e-12) & (e == 1) & (g == lev[1]))
        O_E += d1 - d * n1 / n
        if n > 1:
            V += d * (n1 / n) * (1 - n1 / n) * (n - d) / (n - 1)
    chi2 = O_E ** 2 / V if V > 0 else 0.0
    return chi2, stats.chi2.sf(chi2, 1)


# ------------------------------------------------------------------- Cox model
class CoxFit:
    """Container for a fitted Cox model."""


def cox_fit(X, t, e, maxit=100, tol=1e-9):
    """Cox proportional hazards regression, Breslow handling of ties.

    Parameters
    ----------
    X : (n, p) design matrix, without an intercept
    t : follow-up times
    e : event indicators (1 = event, 0 = censored)

    Returns
    -------
    CoxFit with attributes beta, cov (model-based), loglik, res (score
    residuals used for the robust variance), n, k, nevent.
    """
    X = np.asarray(X, float)
    t = np.asarray(t, float)
    e = np.asarray(e, int)
    n, k = X.shape
    order = np.argsort(-t)
    Xs, ts, es = X[order], t[order], e[order]

    b = np.zeros(k)
    for _ in range(maxit):
        eta = Xs @ b
        wts = np.exp(eta - eta.max())
        S0 = np.cumsum(wts)
        S1 = np.cumsum(wts[:, None] * Xs, axis=0)
        S2 = np.cumsum(wts[:, None, None] * (Xs[:, :, None] * Xs[:, None, :]), axis=0)
        hi = np.searchsorted(-ts, -ts, side='right') - 1
        s0, s1, s2 = np.clip(S0[hi], 1e-300, None), S1[hi], S2[hi]
        xbar = s1 / s0[:, None]
        U = np.sum((Xs - xbar)[es == 1], axis=0)
        I = np.zeros((k, k))
        for i in np.where(es == 1)[0]:
            I += s2[i] / s0[i] - np.outer(xbar[i], xbar[i])
        step = np.linalg.pinv(I) @ U
        b = b + step
        if np.max(np.abs(step)) < tol:
            break

    eta = Xs @ b
    wts = np.exp(eta - eta.max())
    S0 = np.cumsum(wts)
    S1 = np.cumsum(wts[:, None] * Xs, axis=0)
    hi = np.searchsorted(-ts, -ts, side='right') - 1
    s0, s1 = np.clip(S0[hi], 1e-300, None), S1[hi]
    xbar = s1 / s0[:, None]
    ll = np.sum(eta[es == 1] - (np.log(s0[es == 1]) + eta.max()))

    # score residuals (Breslow), used for the cluster-robust sandwich
    res = np.zeros((n, k))
    ev = np.where(es == 1)[0]
    for i in range(n):
        r = (Xs[i] - xbar[i]) * es[i]
        m = ev[ts[ev] <= ts[i] + 1e-12]
        if len(m):
            r = r - np.sum((Xs[i] - xbar[m]) * (wts[i] / s0[m])[:, None], axis=0)
        res[i] = r

    f = CoxFit()
    f.X, f.t, f.e = X, t, e
    f.beta = b
    f.cov = np.linalg.pinv(I)
    f.loglik = ll
    f.res = res[np.argsort(order)]
    f.n, f.k, f.nevent = n, k, int(e.sum())
    return f


def cox_robust(f, groups):
    """Lin-Wei cluster-robust covariance, clustering on `groups` (patient ID)."""
    groups = np.asarray(groups)
    D = f.res @ f.cov
    M = np.zeros((f.k, f.k))
    G = 0
    for g in np.unique(groups):
        s = D[groups == g].sum(axis=0)
        M += np.outer(s, s)
        G += 1
    return M * (G / max(G - 1.0, 1.0))


def cox_table(f, cov, names):
    """Format hazard ratios with 95% CIs and Wald P values."""
    se = np.sqrt(np.diag(cov))
    z = f.beta / se
    p = 2 * stats.norm.sf(np.abs(z))
    return [dict(name=nm, HR=np.exp(b), lo=np.exp(b - 1.96 * s),
                 hi=np.exp(b + 1.96 * s), p=pv)
            for nm, b, s, pv in zip(names, f.beta, se, p)]


def breslow_cumhaz(f, t0):
    """Breslow estimate of the baseline cumulative hazard at time `t0`."""
    lp = f.X @ f.beta
    return sum(1.0 / np.sum(np.exp(lp[f.t >= f.t[i] - 1e-12]))
               for i in np.where(f.e == 1)[0] if f.t[i] <= t0)


# ------------------------------------------- proportional hazards assumption
def cox_zph(f, t, e, transform='rank'):
    """Grambsch-Therneau test based on scaled Schoenfeld residuals.

    `transform` selects g(t): 'rank', 'log', or 'identity'.
    Returns (per-covariate list of dicts, global test dict).
    """
    t = np.asarray(t, float)
    e = np.asarray(e, int)
    order = np.argsort(-t)
    Xs, ts, es = f.X[order], t[order], e[order]
    eta = Xs @ f.beta
    wts = np.exp(eta - eta.max())
    S0 = np.cumsum(wts)
    S1 = np.cumsum(wts[:, None] * Xs, axis=0)
    hi = np.searchsorted(-ts, -ts, side='right') - 1
    xbar = S1[hi] / np.clip(S0[hi], 1e-300, None)[:, None]
    ev = np.where(es == 1)[0]
    r = Xs[ev] - xbar[ev]
    et = ts[ev]
    dtot = len(ev)

    if transform == 'rank':
        g = stats.rankdata(et)
    elif transform == 'log':
        g = np.log(et)
    else:
        g = et.astype(float)
    gc = g - g.mean()

    I = np.linalg.pinv(f.cov)
    u = gc @ r
    V = (np.sum(gc ** 2) / dtot) * I
    per = []
    for j in range(f.k):
        chi2 = u[j] ** 2 / V[j, j]
        per.append(dict(rho=np.corrcoef(gc, r[:, j])[0, 1], chi2=chi2,
                        p=stats.chi2.sf(chi2, 1)))
    gchi2 = float(u @ np.linalg.pinv(V) @ u)
    return per, dict(chi2=gchi2, df=f.k, p=stats.chi2.sf(gchi2, f.k))


# -------------------------------------------------------------- performance
def c_index_surv(t, e, lp):
    """Harrell's c statistic for right-censored data."""
    t = np.asarray(t, float)
    e = np.asarray(e, int)
    lp = np.asarray(lp, float)
    conc = tie = tot = 0.0
    for i in range(len(t)):
        if e[i] != 1:
            continue
        m = t > t[i] + 1e-12
        k = int(m.sum())
        if k == 0:
            continue
        tot += k
        conc += float(np.sum(lp[i] > lp[m]))
        tie += float(np.sum(lp[i] == lp[m]))
    return (conc + 0.5 * tie) / tot if tot else np.nan


def cox_calib_slope(t, e, lp):
    """Calibration slope: coefficient from refitting Cox on the linear predictor."""
    try:
        return cox_fit(np.asarray(lp, float).reshape(-1, 1), t, e).beta[0]
    except Exception:
        return np.nan
