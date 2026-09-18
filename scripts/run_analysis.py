"""Run the full analysis pipeline.

Reproduces, in order, every statistical analysis reported in the article and
its supplement:

   1. Kaplan-Meier estimates and log-rank test (complete regression)
   2. Univariable and multivariable Cox regression, patient-clustered SE,
      with variance inflation factors for the multivariable model
   3. Grambsch-Therneau test of the proportional hazards assumption, for the
      model as reported and for the reduced model
   4. Sensitivity analysis restricted to cysts with >= 24 months of follow-up
   5. Univariable and multivariable logistic regression (successful regression)
   6. Spline dose-response, cut-point scan, and bootstrap cut-point stability
   7. Bootstrap internal validation, including a model in which the cut-point
      is re-selected inside every replicate
   8. Alcohol-to-cyst volume ratio vs baseline cyst volume: collinearity,
      effect modification, and outcomes within volume strata
   9. Symptom resolution (patient level, exact confidence intervals)
  10. Segmented regression of volume reduction rate on retention time

Usage
-----
    python scripts/make_synthetic_data.py
    python scripts/run_analysis.py [path/to/data.csv]

With the synthetic dataset the pipeline runs end to end but the numbers will
not match the published results.
"""
import sys
import os
import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from survival import (km, km_at, logrank, cox_fit, cox_robust, cox_table, cox_zph)
from logistic import logit_fit, cluster_cov, wald, lr_test
from threshold import (spline_dose_response, cutpoint_scan, bootstrap_cutpoint,
                       bootstrap_optimism, segmented_regression)
from descriptive import proportion, vif, variance_share

CSV = sys.argv[1] if len(sys.argv) > 1 else 'data/synthetic_example.csv'
BOOT_CUTPOINT = int(os.environ.get('BOOT_CUTPOINT', 1500))
BOOT_VALIDATE = int(os.environ.get('BOOT_VALIDATE', 500))

d = pd.read_csv(CSV)
d['log_volume'] = np.log(d.cyst_volume_mL)
d['log_ratio'] = np.log(d.ratio_pct)
d['volume_100'] = d.cyst_volume_mL / 100
d['alcohol_10'] = d.alcohol_mL / 10
d['retention_20'] = d.retention_min / 20
g = d.patient_id.values
t = d.time_months.values
e = d.complete_regression.values

line = lambda s: print('\n' + '=' * 72 + f'\n{s}\n' + '=' * 72)
has = lambda c: c in d.columns

# ------------------------------------------------------------------- 1. KM
line('1. Kaplan-Meier estimates of complete regression')
tab = km(t, e)
for m in (12, 24, 36, 48):
    c, lo, hi = km_at(tab, m)
    print(f'  {m:3d} months: {100*c:5.1f}% (95% CI {100*lo:.1f}-{100*hi:.1f})'
          f'   n at risk {int((t >= m).sum())}')
chi2, p = logrank(t, e, d.ratio_ge5.values)
print(f'  log-rank, ratio <5% vs >=5%: chi2 = {chi2:.2f}, p = {p:.4g}')

# ------------------------------------------------------------------ 2. Cox
line('2. Cox regression for complete regression (patient-clustered SE)')
CANDIDATES = [('age_ge65', 'Age >=65 years'),
              ('male', 'Male sex'),
              ('symptomatic', 'Preprocedural symptoms present'),
              ('volume_100', 'Baseline cyst volume, per 100 mL'),
              ('complicated_cyst', 'Complicated cyst'),
              ('hemorrhagic_cyst', 'Hemorrhagic cyst'),
              ('infected_cyst', 'Infected cyst'),
              ('catheter_10F', 'Pigtail catheter diameter, 10 F'),
              ('catheter_days', 'Catheter indwelling duration, per day'),
              ('alcohol_10', 'Total alcohol volume, per 10 mL'),
              ('ratio_ge5', 'Ratio >=5%'),
              ('retention_20', 'Total retention time, per 20 min'),
              ('reposition', 'Patient repositioning'),
              ('multisession', 'Multisession PAS')]
CANDIDATES = [(v, nm) for v, nm in CANDIDATES if has(v)]
print('  Univariable')
entered = []
for v, nm in CANDIDATES:
    try:
        f = cox_fit(d[v].values.astype(float).reshape(-1, 1), t, e)
        r = cox_table(f, cox_robust(f, g), [nm])[0]
    except Exception as exc:
        print(f'    {nm:38s} not estimable ({exc})')
        continue
    flag = ' *' if r['p'] < .10 else ''
    print(f'    {nm:38s} HR {r["HR"]:6.2f} ({r["lo"]:5.2f}-{r["hi"]:7.2f})  p = {r["p"]:.3f}{flag}')
    if r['p'] < .10:
        entered.append((v, nm))

print(f'\n  Multivariable (variables with p <0.10 entered: {len(entered)})')
X = np.column_stack([d[v].values.astype(float) for v, _ in entered])
fm = cox_fit(X, t, e)
cov = cox_robust(fm, g)
for r in cox_table(fm, cov, [nm for _, nm in entered]):
    print(f'    {r["name"]:38s} HR {r["HR"]:6.2f} ({r["lo"]:5.2f}-{r["hi"]:7.2f})  p = {r["p"]:.3f}')
print(f'    events {fm.nevent}, events per variable {fm.nevent/len(entered):.1f}')
print('\n  Collinearity of the multivariable model')
for (v, nm), val in zip(entered, vif(X)):
    print(f'    {nm:38s} VIF {val:5.2f}')

# ------------------------------------------------------- 3. PH assumption
line('3. Proportional hazards assumption (Grambsch-Therneau)')
print('  Multivariable model as reported')
for tr in ('rank', 'log', 'identity'):
    per, gl = cox_zph(fm, t, e, transform=tr)
    if tr == 'rank':
        for (v, nm), r in zip(entered, per):
            print(f'    {nm:38s} rho {r["rho"]:+.3f}  chi2 {r["chi2"]:5.2f}  p = {r["p"]:.3f}')
    print(f'    global, g(t) = {tr:9s}: chi2 = {gl["chi2"]:5.2f} (df {gl["df"]}), p = {gl["p"]:.3f}')

REDUCED = [(v, nm) for v, nm in
           [('volume_100', 'Baseline cyst volume, per 100 mL'),
            ('ratio_ge5', 'Ratio >=5%'),
            ('reposition', 'Patient repositioning')] if has(v)]
print('\n  Reduced model, independently associated covariates only')
Xr = np.column_stack([d[v].values.astype(float) for v, _ in REDUCED])
fr = cox_fit(Xr, t, e)
for r in cox_table(fr, cox_robust(fr, g), [nm for _, nm in REDUCED]):
    print(f'    {r["name"]:38s} HR {r["HR"]:6.2f} ({r["lo"]:5.2f}-{r["hi"]:7.2f})  p = {r["p"]:.3f}')
for tr in ('rank', 'log', 'identity'):
    per, gl = cox_zph(fr, t, e, transform=tr)
    if tr == 'rank':
        for (v, nm), r in zip(REDUCED, per):
            print(f'    {nm:38s} rho {r["rho"]:+.3f}  chi2 {r["chi2"]:5.2f}  p = {r["p"]:.3f}')
    print(f'    global, g(t) = {tr:9s}: chi2 = {gl["chi2"]:5.2f} (df {gl["df"]}), p = {gl["p"]:.3f}')

# --------------------------------------------------- 4. Sensitivity analysis
line('4. Sensitivity analysis, cysts with >= 24 months of follow-up')
lm = d[d.follow_up_months >= 24]
a = lm[lm.ratio_ge5 == 1].complete_regression
b = lm[lm.ratio_ge5 == 0].complete_regression
_, pf = stats.fisher_exact([[int(a.sum()), len(a) - int(a.sum())],
                            [int(b.sum()), len(b) - int(b.sum())]])
print(f'    n = {len(lm)};  ratio <5%: {int(b.sum())}/{len(b)};  '
      f'ratio >=5%: {int(a.sum())}/{len(a)};  Fisher p = {pf:.4g}')

# ------------------------------------------------------------- 5. Logistic
line('5. Logistic regression for successful regression (patient-clustered SE)')
y = d.successful_regression.values
LOG_CANDIDATES = [(v, nm) for v, nm in CANDIDATES if v != 'infected_cyst']
sel = []
print('  Univariable')
for v, nm in LOG_CANDIDATES:
    try:
        f = logit_fit(np.column_stack([np.ones(len(d)), d[v].values.astype(float)]), y)
        w = wald(f, cluster_cov(f, g))
        assert np.all(np.isfinite(w['OR'])) and w['OR'][1] < 1e4
    except Exception:
        print(f'    {nm:38s} not estimable (sparse data)')
        continue
    flag = ' *' if w['p'][1] < .10 else ''
    print(f'    {nm:38s} OR {w["OR"][1]:6.2f} ({w["OR_lo"][1]:5.2f}-{w["OR_hi"][1]:7.2f})  p = {w["p"][1]:.3f}{flag}')
    if w['p'][1] < .10:
        sel.append((v, nm))
if sel:
    print('\n  Multivariable')
    Xl = np.column_stack([np.ones(len(d))] + [d[v].values.astype(float) for v, _ in sel])
    fl = logit_fit(Xl, y)
    wl = wald(fl, cluster_cov(fl, g))
    for i, (v, nm) in enumerate(sel, start=1):
        print(f'    {nm:38s} OR {wl["OR"][i]:6.2f} ({wl["OR_lo"][i]:5.2f}-{wl["OR_hi"][i]:7.2f})  p = {wl["p"][i]:.3f}')

# ----------------------------------------------------- 6. Threshold analysis
line('6. Threshold derivation for the alcohol-to-cyst volume ratio')
res = spline_dose_response(d.log_ratio.values, d.log_volume.values, t, e, g)
print(f'    spline, overall association  p = {res["p_overall"]:.4f}')
print(f'    spline, departure from log-linearity  p = {res["p_nonlinearity"]:.4f}')

grid = np.arange(2.0, 20.01, 0.25)
ch = cutpoint_scan(d.ratio_pct.values, d.log_volume.values, t, e, grid)
best = grid[np.nanargmax(ch)]
within = grid[np.where(ch >= np.nanmax(ch) - 3.84)[0]]
print(f'    cut-point scan: optimum {best:.2f}% (chi2 = {np.nanmax(ch):.2f}); '
      f'within 3.84 chi2 units: {within.min():.2f}-{within.max():.2f}%')
at5 = ch[np.argmin(np.abs(grid - 5.0))]
print(f'    likelihood-ratio chi2 at 5%: {at5:.2f} (p = {stats.chi2.sf(at5, 1):.4f})')

print(f'    bootstrap re-selection (B = {BOOT_CUTPOINT}) ...')
sel_b = bootstrap_cutpoint(d.ratio_pct.values, d.log_volume.values, t, e, g,
                           grid, B=BOOT_CUTPOINT)
print(f'      median {np.median(sel_b):.2f}%, IQR {np.percentile(sel_b,25):.2f}-'
      f'{np.percentile(sel_b,75):.2f}%, '
      f'95% interval {np.percentile(sel_b,2.5):.2f}-{np.percentile(sel_b,97.5):.2f}%, '
      f'{100*np.mean((sel_b>=4)&(sel_b<=6)):.1f}% selected 4-6%')


# ------------------------------------------------- 7. Internal validation
def reselected_cutpoint_design(df):
    """Design matrix in which the cut-point is re-selected from the data."""
    tt = df.time_months.values
    ee = df.complete_regression.values
    lv = np.log(df.cyst_volume_mL.values)
    stat = cutpoint_scan(df.ratio_pct.values, lv, tt, ee, grid)
    cut = grid[np.nanargmax(stat)]
    return np.column_stack([(df.ratio_pct.values >= cut).astype(float), lv])


line('7. Bootstrap internal validation (Harrell optimism)')
MODELS = {
    'Multivariable model as reported':
        lambda df: np.column_stack([df[v].values.astype(float) for v, _ in entered]),
    'Ratio >=5% (fixed) + baseline volume':
        lambda df: np.column_stack([df.ratio_ge5.values, np.log(df.cyst_volume_mL.values)]),
    'Cut-point re-selected in each sample': reselected_cutpoint_design,
    'Baseline cyst volume alone':
        lambda df: np.log(df.cyst_volume_mL.values).reshape(-1, 1),
}
print(f'    {"model":40s} {"c app":>7s} {"optim":>8s} {"c corr":>7s} {"slope":>7s}')
for nm, build in MODELS.items():
    r = bootstrap_optimism(build, d, g, t, e, B=BOOT_VALIDATE)
    print(f'    {nm:40s} {r["apparent_c"]:7.3f} {r["optimism"]:+8.3f} '
          f'{r["corrected_c"]:7.3f} {r["calibration_slope"]:7.2f}')

# --------------------------------- 8. Ratio vs baseline cyst volume
line('8. Alcohol-to-cyst volume ratio vs baseline cyst volume')
rho, p_rho = stats.spearmanr(d.ratio_pct.values, d.cyst_volume_mL.values)
print(f'    Spearman rho = {rho:+.2f} (p = {p_rho:.4g})')
print(f'    variance of log ratio explained by log volume: '
      f'{100*variance_share(d.log_ratio.values, d.log_volume.values):.0f}%')

f_main = cox_fit(np.column_stack([d.ratio_ge5.values, d.log_volume.values]), t, e)
f_int = cox_fit(np.column_stack([d.ratio_ge5.values, d.log_volume.values,
                                 d.ratio_ge5.values * d.log_volume.values]), t, e)
chi2_i, df_i, p_i = lr_test(f_int, f_main)
print(f'    ratio x log volume interaction: chi2 = {chi2_i:.2f} (df {df_i}), p = {p_i:.3f}')

print('\n    Complete regression within strata of baseline cyst volume')
STRATA = [('<500 mL', 0, 500), ('500-999 mL', 500, 1000), ('>=1,000 mL', 1000, np.inf)]
print(f'    {"stratum":12s} {"n":>4s}   {"KM at 36 mo, <5%":>18s} {"KM at 36 mo, >=5%":>19s}')
for nm, lo, hi in STRATA:
    s = d[(d.cyst_volume_mL >= lo) & (d.cyst_volume_mL < hi)]
    cells = []
    for grp in (0, 1):
        ss = s[s.ratio_ge5 == grp]
        if len(ss) == 0:
            cells.append('n/a')
            continue
        c, _, _ = km_at(km(ss.time_months.values, ss.complete_regression.values), 36)
        cells.append(f'{100*c:5.1f}% (n = {len(ss):2d})')
    print(f'    {nm:12s} {len(s):4d}   {cells[0]:>18s} {cells[1]:>19s}')

print('\n    Sensitivity analysis within strata (>= 24 months of follow-up)')
for nm, lo, hi in STRATA:
    s = lm[(lm.cyst_volume_mL >= lo) & (lm.cyst_volume_mL < hi)]
    aa = s[s.ratio_ge5 == 1].complete_regression
    bb = s[s.ratio_ge5 == 0].complete_regression
    if len(aa) and len(bb):
        _, pp = stats.fisher_exact([[int(aa.sum()), len(aa) - int(aa.sum())],
                                    [int(bb.sum()), len(bb) - int(bb.sum())]])
        pp = f'{pp:.4g}'
    else:
        pp = 'n/a'
    print(f'    {nm:12s} n = {len(s):3d}   <5%: {int(bb.sum())}/{len(bb):2d}   '
          f'>=5%: {int(aa.sum())}/{len(aa):2d}   Fisher p = {pp}')

# ------------------------------------------------- 9. Symptom resolution
line('9. Symptom resolution (patient level)')
if has('symptomatic') and has('symptom_resolved'):
    pat = d.drop_duplicates('patient_id').copy()
    worst = d.groupby('patient_id').successful_regression.min()
    allcr = d.groupby('patient_id').complete_regression.min()
    pat['all_successful'] = worst.reindex(pat.patient_id).values
    pat['all_complete'] = allcr.reindex(pat.patient_id).values
    sym = pat[pat.symptomatic == 1]
    print(f'    indication: symptomatic {int(pat.symptomatic.sum())}/{len(pat)}, '
          f'asymptomatic enlargement {int((pat.symptomatic == 0).sum())}/{len(pat)}')
    print(f'    all symptomatic patients        {proportion(int((sym.symptom_resolved == 1).sum()), len(sym))}')
    if has('symptom_type'):
        for ty in ('discomfort', 'pain', 'fever'):
            s = sym[sym.symptom_type == ty]
            if len(s):
                print(f'      {ty:28s}  {proportion(int((s.symptom_resolved == 1).sum()), len(s))}')
    groups = [('complete regression', sym.all_complete == 1),
              ('extensive regression', (sym.all_successful == 1) & (sym.all_complete == 0)),
              ('partial regression', sym.all_successful == 0)]
    for nm, mask in groups:
        s = sym[mask]
        if len(s):
            print(f'      {nm:28s}  {proportion(int((s.symptom_resolved == 1).sum()), len(s))}')
else:
    print('    columns symptomatic / symptom_resolved not present; skipped')

# --------------------------------------------------- 10. Segmented regression
line('10. Segmented regression: volume reduction rate on retention time')
sr = segmented_regression(d.retention_min.values, d.vrr_pct.values, g)
print(f'    breakpoint {sr["breakpoint"]:.0f} minutes')
print(f'    slope before: {sr["slope_before"]:+.2f} points per 20 min '
      f'(95% CI {sr["slope_before_ci"][0]:.1f}-{sr["slope_before_ci"][1]:.1f}), '
      f'p = {sr["p_before"]:.3f}')
print(f'    slope after : {sr["slope_after"]:+.2f} points per 20 min '
      f'(95% CI {sr["slope_after_ci"][0]:.2f}-{sr["slope_after_ci"][1]:.2f}), '
      f'p = {sr["p_after"]:.3f}')
print(f'    change in slope p = {sr["p_slope_change"]:.3f}')

print('\nDone.')
