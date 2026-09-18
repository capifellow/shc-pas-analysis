# Statistical analysis code

Analysis code for the study **"Alcohol-to-cyst volume ratio of at least 5% predicts long-term complete regression of simple hepatic cysts after sclerotherapy"** (under review at *Journal of Hepatology*).

This repository contains the Python implementations of every statistical method reported in the article and its supplement. The estimators were implemented from first principles using only NumPy and SciPy, and each was verified against simulated data with known parameters before use.

## What is here

```
src/
  survival.py     Kaplan-Meier, log-rank test, Cox proportional hazards regression
                  (Breslow ties), Lin-Wei cluster-robust variance, Grambsch-Therneau
                  test of proportional hazards, Harrell c statistic
  logistic.py     maximum-likelihood and Firth penalised logistic regression,
                  cluster-robust variance, likelihood-ratio test, restricted cubic
                  spline basis
  threshold.py    spline dose-response, exhaustive cut-point scan, bootstrap
                  cut-point re-selection, Harrell bootstrap optimism correction,
                  segmented linear regression
  descriptive.py  exact (Clopper-Pearson) confidence intervals for proportions,
                  variance inflation factors, variance decomposition
scripts/
  make_synthetic_data.py       generate a synthetic dataset with the study structure
  run_analysis.py              run the full analysis pipeline
  validate_implementations.py  verify the estimators against known simulated truth
data/
  synthetic_example.csv        synthetic dataset (generated; not patient data)
```

## Data availability

Patient-level data cannot be shared publicly. `scripts/make_synthetic_data.py` generates a synthetic dataset with the same variables, coding, and patient-level clustering structure as the analysed cohort, so that the pipeline can be executed end to end. **Numerical results obtained from the synthetic data will not reproduce those reported in the article.** The real data are available from the corresponding author on reasonable request.

### Data dictionary

One row per treated cyst. Indication and symptom variables are patient-level and are constant within a patient; all other variables are cyst-level.

| Column | Description |
| --- | --- |
| `patient_id` | clustering unit; some patients contributed more than one cyst |
| `age_ge65`, `male` | patient age >=65 years; male sex (0/1) |
| `symptomatic` | indication was a symptom (abdominal discomfort, pain, or fever) rather than asymptomatic progressive enlargement (0/1) |
| `symptom_type` | `discomfort`, `pain`, `fever`, or `none` |
| `cyst_volume_mL` | baseline cyst volume |
| `alcohol_mL`, `ratio_pct`, `ratio_ge5` | cumulative alcohol volume; alcohol-to-cyst volume ratio (vol%); ratio >=5% (0/1) |
| `retention_min`, `catheter_days`, `catheter_10F` | total alcohol retention time; catheter indwelling duration; final catheter diameter 10 F (0/1) |
| `reposition`, `multisession` | patient repositioning during retention; more than one session (0/1) |
| `complicated_cyst`, `hemorrhagic_cyst`, `infected_cyst` | baseline cyst complication status (0/1) |
| `follow_up_months` | total imaging follow-up |
| `time_months`, `complete_regression` | time to the first CT examination showing radiologic disappearance, or the last examination if censored; event indicator |
| `vrr_pct`, `successful_regression` | volume reduction rate at final follow-up; volume reduction rate >=90% (0/1) |
| `symptom_resolved` | symptoms resolved at the first outpatient visit after treatment (1 = resolved, 0 = persistent, -1 = not applicable in asymptomatic patients) |

## Requirements

Python 3.10 or later.

```
numpy >= 1.24
scipy >= 1.10
pandas >= 2.0
```

Install with `pip install -r requirements.txt`.

## Usage

```
pip install -r requirements.txt

# verify the implementations against simulated data with known truth
python scripts/validate_implementations.py

# generate the synthetic dataset and run the full pipeline
python scripts/make_synthetic_data.py
python scripts/run_analysis.py
```

The bootstrap procedures dominate the runtime. For a quick check, reduce the number of resamples:

```
BOOT_CUTPOINT=200 BOOT_VALIDATE=100 python scripts/run_analysis.py
```

The published analyses used `BOOT_CUTPOINT=1500` and `BOOT_VALIDATE=500` (the defaults).

To analyse your own data, pass a CSV with the columns listed in the data dictionary above:

```
python scripts/run_analysis.py path/to/your_data.csv
```

`run_analysis.py` prints, in order: Kaplan-Meier estimates and the log-rank test; univariable and multivariable Cox regression with variance inflation factors; the proportional hazards assessment for the reported and the reduced model; the sensitivity analysis restricted to cysts with at least 24 months of follow-up; logistic regression for successful regression; the threshold analyses; bootstrap internal validation; the relationship between the ratio and baseline cyst volume; symptom resolution; and the segmented regression of volume reduction rate on retention time.

## Methods implemented

**Complete regression** was analysed as a time-to-event outcome, with the event time defined as the date of the first CT examination showing radiologic disappearance and cysts without complete regression censored at their last examination.

- Kaplan-Meier estimation with Greenwood variance; two-group log-rank test
- Cox proportional hazards regression, Breslow handling of ties
- Lin-Wei cluster-robust (sandwich) standard errors at the patient level, because some patients contributed more than one cyst
- Grambsch-Therneau test of the proportional hazards assumption based on scaled Schoenfeld residuals, under rank, logarithmic, and identity transformations of event time, for the model as reported and for a reduced model containing only the independently associated covariates

**Successful regression** (volume reduction rate of at least 90%) could not be analysed on the time scale, because the date on which the volume reduction rate first reached 90% was not recorded. It was analysed as a binary outcome at final follow-up by logistic regression with patient-level cluster-robust standard errors. Firth penalised likelihood is available for sparse-data settings.

**Threshold derivation** for the alcohol-to-cyst volume ratio did not presuppose a cut-point:

- the log-transformed ratio was modelled with a four-knot restricted cubic spline within the Cox model, adjusted for log baseline cyst volume, with a likelihood-ratio test for departure from log-linearity
- every candidate cut-point between 2% and 20% was evaluated in 0.25% increments by likelihood-ratio statistic
- the entire search was repeated in 1500 patient-level bootstrap samples, so that the variability of the selection procedure itself was captured

**Internal validation** used the Harrell bootstrap optimism correction with patient-level resampling. The entire modelling procedure, including cut-point selection where applicable, is repeated inside each replicate; median optimism is subtracted from the apparent performance. Four models are reported, including one in which the cut-point is re-selected inside every replicate, so that the optimism attributable to data-driven threshold selection is isolated from the performance of a fixed 5% threshold.

**Ratio and baseline cyst volume.** Because the single-session alcohol volume was capped, the ratio is partly determined by cyst size. This relationship is quantified by the Spearman correlation and by the share of the variance of the log ratio explained by log cyst volume; collinearity by variance inflation factors; effect modification by a likelihood-ratio test of a ratio-by-log-volume interaction; and outcomes are additionally compared within strata of baseline cyst volume, both by Kaplan-Meier estimates and, restricted to cysts with at least 24 months of follow-up, by the Fisher exact test.

**Symptom resolution** was summarised at the patient level, overall, by presenting symptom, and by the volumetric outcome of the treated cysts, with exact (Clopper-Pearson) confidence intervals.

**Segmented linear regression** with a single breakpoint, selected by the Bayesian information criterion with cluster-robust standard errors, was used for the supplemental analysis of volume reduction rate on total alcohol retention time.

## Verification

Because the estimators were implemented from first principles rather than taken from an established package, `scripts/validate_implementations.py` checks each against data simulated from a known model:

| Check | Result |
| --- | --- |
| Kaplan-Meier vs the analytic survival function | maximum deviation < 0.02 |
| Cox coefficient recovery (true 0.80, -0.50) | maximum error < 0.05 |
| Cluster-robust variance without clustering | matches the model-based SE |
| Cluster-robust variance under perfect clustering | recovers the true SE where the model-based SE is falsely small |
| Grambsch-Therneau test when the assumption holds | no false alarm |
| Grambsch-Therneau test when the assumption is violated | violation detected in the offending covariate |
| Logistic coefficient recovery | maximum error < 0.10 |
| Firth penalisation under complete separation | finite estimate where maximum likelihood diverges |

## Changelog

**v2 (this version).** Updated for the *Journal of Hepatology* submission: patient-level indication and symptom variables added, with symptom resolution summarised using exact confidence intervals; variance inflation factors, the ratio-by-volume interaction test, and the volume-stratified comparisons added; the proportional hazards assessment extended to the reduced model; a cut-point re-selection model added to the internal validation; candidate variable lists aligned with the tables in the article.

**v1.** Version accompanying the earlier submission of the same study.

## License

MIT. See `LICENSE`.
