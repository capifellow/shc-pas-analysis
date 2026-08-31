"""Threshold derivation, bootstrap procedures, and segmented regression.

Contains the analyses used to establish and validate the 5% alcohol-to-cyst
volume ratio threshold:

* spline_dose_response - four-knot restricted cubic spline of the log ratio
  within the Cox model, with a likelihood-ratio test for departure from
  log-linearity
* cutpoint_scan        - exhaustive evaluation of every candidate cut-point
* bootstrap_cutpoint   - repetition of the entire cut-point search in
  patient-level bootstrap samples
* bootstrap_optimism   - Harrell optimism correction of the c statistic and
  calibration slope
* segmented_regression - breakpoint model for volume reduction rate on
  alcohol retention time (supplemental analysis)
"""
import numpy as np
from scipy import stats

from survival import (cox_fit, cox_robust, c_index_surv, cox_calib_slope)
from logistic import rcs, knot_quantiles


# ------------------------------------------------------------ dose-response
def spline_dose_response(log_ratio, log_volume, t, e, groups, nk=4):
    """Cox model with a restricted cubic spline of the log ratio.

    Returns a dict with the fitted model, cluster-robust covariance, knots,
    and likelihood-ratio tests for the overall association and for departure
    from log-linearity.
    """
    kn = knot_quantiles(log_ratio, nk)
    X_spline = np.column_stack([rcs(log_ratio, kn), log_volume])
    X_linear = np.column_stack([log_ratio, log_volume])
    X_null = np.asarray(log_volume, float).reshape(-1, 1)

    f_s = cox_fit(X_spline, t, e)
    f_l = cox_fit(X_linear, t, e)
    f_n = cox_fit(X_null, t, e)

    chi2_nl = 2 * (f_s.loglik - f_l.loglik)
    chi2_all = 2 * (f_s.loglik - f_n.loglik)
    return dict(fit=f_s, cov=cox_robust(f_s, groups), knots=kn,
                p_overall=stats.chi2.sf(chi2_all, f_s.k - f_n.k),
                p_nonlinearity=stats.chi2.sf(chi2_nl, f_s.k - f_l.k))


def hazard_ratio_curve(res, grid, reference, log_volume_at=0.0):
    """Adjusted hazard ratio across `grid`, relative to `reference`.

    The covariate held fixed cancels in the contrast, so `log_volume_at` does
    not affect the result; it is retained for clarity.
    """
    kn = res['knots']
    ref = np.concatenate([rcs(np.array([np.log(reference)]), kn)[0], [log_volume_at]])
    rows = np.column_stack([rcs(np.log(grid), kn),
                            np.full(len(grid), log_volume_at)])
    L = rows - ref
    est = L @ res['fit'].beta
    se = np.sqrt(np.einsum('ij,jk,ik->i', L, res['cov'], L))
    return dict(x=grid, hr=np.exp(est),
                lo=np.exp(est - 1.96 * se), hi=np.exp(est + 1.96 * se))


# --------------------------------------------------------------- cut-points
def cutpoint_scan(ratio, log_volume, t, e, grid, min_group=5):
    """Likelihood-ratio statistic for each candidate cut-point.

    At each candidate the ratio is dichotomised and entered into a Cox model
    adjusted for log baseline volume. Candidates leaving fewer than
    `min_group` observations in either group are not evaluated (NaN).
    """
    base = np.asarray(log_volume, float).reshape(-1, 1)
    f0 = cox_fit(base, t, e)
    out = np.full(len(grid), np.nan)
    for i, c in enumerate(grid):
        z = (np.asarray(ratio) >= c).astype(float)
        if z.sum() < min_group or len(z) - z.sum() < min_group:
            continue
        f1 = cox_fit(np.column_stack([base, z]), t, e)
        out[i] = 2 * (f1.loglik - f0.loglik)
    return out


def bootstrap_cutpoint(ratio, log_volume, t, e, groups, grid, B=1500, seed=20260828):
    """Repeat the entire cut-point search in B patient-level bootstrap samples.

    Resampling is at the level of `groups` (patient), so that cysts from the
    same patient are kept together. The search is repeated from scratch in
    every replicate, so the variability of the selection procedure itself is
    captured.
    """
    rng = np.random.default_rng(seed)
    ratio = np.asarray(ratio)
    log_volume = np.asarray(log_volume, float)
    t = np.asarray(t, float)
    e = np.asarray(e, int)
    groups = np.asarray(groups)
    gs = np.unique(groups)
    gidx = {g: np.where(groups == g)[0] for g in gs}

    selected = []
    while len(selected) < B:
        idx = np.concatenate([gidx[g] for g in rng.choice(gs, len(gs), replace=True)])
        try:
            ch = cutpoint_scan(ratio[idx], log_volume[idx], t[idx], e[idx], grid)
            if np.all(np.isnan(ch)):
                continue
            selected.append(grid[np.nanargmax(ch)])
        except Exception:
            continue
    return np.array(selected)


# ------------------------------------------------------- internal validation
def bootstrap_optimism(build, df, groups, t, e, B=500, seed=5):
    """Harrell optimism correction with patient-level resampling.

    `build(dataframe) -> design matrix`. The entire modelling procedure,
    including any cut-point selection, must live inside `build` so that its
    optimism is captured. Median optimism across replicates is subtracted
    from the apparent performance.
    """
    rng = np.random.default_rng(seed)
    groups = np.asarray(groups)
    gs = np.unique(groups)
    gidx = {g: np.where(groups == g)[0] for g in gs}

    X0 = build(df)
    f0 = cox_fit(X0, t, e)
    lp0 = X0 @ f0.beta
    app_c = c_index_surv(t, e, lp0)
    app_s = cox_calib_slope(t, e, lp0)

    oc, os_ = [], []
    for _ in range(B):
        idx = np.concatenate([gidx[g] for g in rng.choice(gs, len(gs), replace=True)])
        db = df.iloc[idx].reset_index(drop=True)
        tb, eb = t[idx], e[idx]
        if eb.sum() < 8:
            continue
        try:
            Xb = build(db)
            fb = cox_fit(Xb, tb, eb)
            if np.max(np.abs(fb.beta)) > 20:
                continue
            lpb = Xb @ fb.beta
            lpo = build(df) @ fb.beta
            cb, co = c_index_surv(tb, eb, lpb), c_index_surv(t, e, lpo)
            sb, so = cox_calib_slope(tb, eb, lpb), cox_calib_slope(t, e, lpo)
            if not all(np.isfinite([cb, co, sb, so])) or abs(sb) > 10 or abs(so) > 10:
                continue
            oc.append(cb - co)
            os_.append(sb - so)
        except Exception:
            continue

    opt_c, opt_s = float(np.median(oc)), float(np.median(os_))
    return dict(apparent_c=app_c, optimism=opt_c, corrected_c=app_c - opt_c,
                calibration_slope=app_s - opt_s, B_used=len(oc))


# ------------------------------------------------------ segmented regression
def segmented_regression(x, y, groups, candidates=np.arange(15, 125, 5.0), scale=20.0):
    """Segmented linear regression with a single breakpoint.

    Candidate breakpoints are compared by the Bayesian information criterion;
    standard errors are clustered on `groups`. Slopes are expressed per
    `scale` units of x. Used for volume reduction rate on retention time.
    """
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    groups = np.asarray(groups)

    def fit(bp):
        X = np.column_stack([np.ones(len(x)), x / scale,
                             np.clip(x - bp, 0, None) / scale])
        b, *_ = np.linalg.lstsq(X, y, rcond=None)
        r = y - X @ b
        n, k = X.shape
        XtXi = np.linalg.pinv(X.T @ X)
        M = np.zeros((k, k))
        for g in np.unique(groups):
            m = groups == g
            s = X[m].T @ r[m]
            M += np.outer(s, s)
        G = len(np.unique(groups))
        V = (G / (G - 1)) * ((n - 1) / (n - k)) * XtXi @ M @ XtXi
        bic = n * np.log(np.sum(r ** 2) / n) + k * np.log(n)
        return b, V, bic

    bp = min(((c, fit(c)[2]) for c in candidates), key=lambda z: z[1])[0]
    b, V, _ = fit(bp)
    se = np.sqrt(np.diag(V))
    L = np.array([0, 1, 1])
    s2, se2 = b[1] + b[2], np.sqrt(L @ V @ L)
    return dict(
        breakpoint=bp,
        slope_before=b[1], slope_before_ci=(b[1] - 1.96 * se[1], b[1] + 1.96 * se[1]),
        p_before=2 * stats.norm.sf(abs(b[1] / se[1])),
        slope_after=s2, slope_after_ci=(s2 - 1.96 * se2, s2 + 1.96 * se2),
        p_after=2 * stats.norm.sf(abs(s2 / se2)),
        p_slope_change=2 * stats.norm.sf(abs(b[2] / se[2])))
