# Statistical analysis code

Analysis code for the study **"Percutaneous Alcohol Sclerotherapy for Simple Hepatic Cysts: An Alcohol-to-Cyst Volume Ratio of 5% or Greater Predicts Long-term Complete Regression"** (submitted to *Radiology*).

This repository contains the Python implementations of every statistical method reported in the article and its supplement. The estimators were implemented from first principles using only NumPy and SciPy, and each was verified against simulated data with known parameters before use.

## What is here

```
src/
  survival.py     Kaplan-Meier, log-rank test, Cox proportional hazards regression
                  (Breslow ties), Lin-Wei cluster-robust variance, Grambsch-Therneau
                  test of proportional hazards, Harrell c statistic
  logistic.py     maximum-likelihood and Firth penalised logistic regression,
                  cluster-robust variance, restricted cubic spline basis
  threshold.py    spline dose-response, exhaustive cut-point scan, bootstrap
                  cut-point re-selection, Harrell bootstrap optimism correction,
                  segmented linear regression
scripts/
  make_synthetic_data.py       generate a synthetic dataset with the study structure
  run_analysis.py              run the full analysis pipeline
  validate_implementations.py  verify the estimators against known simulated truth
data/
  synthetic_example.csv        synthetic dataset (generated; not patient data)
```

## Data availability

Patient-level data cannot be shared publicly. `scripts/make_synthetic_data.py` generates a synthetic dataset with the same variables, coding, and patient-level clustering structure as the analysed cohort, so that the pipeline can be executed end to end. **Numerical results obtained from the synthetic data will not reproduce those reported in the article.** The real data are available from the corresponding author on reasonable request.

## Requirements

Python 3.10 or later.

```
numpy >= 1.24
scipy >= 1.10
pandas >= 2.0
```

Install with `pip install -r requirements.txt`.

## Usage

```bash
pip install -r requirements.txt

# verify the implementations against simulated data with known truth
python scripts/validate_implementations.py

# generate the synthetic dataset and run the full pipeline
python scripts/make_synthetic_data.py
python scripts/run_analysis.py
```

The bootstrap procedures dominate the runtime. For a quick check, reduce the number of resamples:

```bash
BOOT_CUTPOINT=200 BOOT_VALIDATE=100 python scripts/run_analysis.py
```

The published analyses used `BOOT_CUTPOINT=1500` and `BOOT_VALIDATE=500` (the defaults).

To analyse your own data, pass a CSV with the columns listed in `data/synthetic_example.csv`:

```bash
python scripts/run_analysis.py path/to/your_data.csv
```

## Methods implemented

**Complete regression** was analysed as a time-to-event outcome, with the event time defined as the date of the first CT examination showing radiologic disappearance and cysts without complete regression censored at their last examination.

- Kaplan-Meier estimation with Greenwood variance; two-group log-rank test
- Cox proportional hazards regression, Breslow handling of ties
- Lin-Wei cluster-robust (sandwich) standard errors at the patient level, because some patients contributed more than one cyst
- Grambsch-Therneau test of the proportional hazards assumption based on scaled Schoenfeld residuals, under rank, logarithmic, and identity transformations of event time

**Successful regression** (volume reduction rate of at least 90%) could not be analysed on the time scale, because the date on which the volume reduction rate first reached 90% was not recorded. It was analysed as a binary outcome at final follow-up by logistic regression with patient-level cluster-robust standard errors. Firth penalised likelihood is available for sparse-data settings.

**Threshold derivation** for the alcohol-to-cyst volume ratio did not presuppose a cut-point:

- the log-transformed ratio was modelled with a four-knot restricted cubic spline within the Cox model, adjusted for log baseline cyst volume, with a likelihood-ratio test for departure from log-linearity
- every candidate cut-point between 2% and 20% was evaluated in 0.25% increments by likelihood-ratio statistic
- the entire search was repeated in 1500 patient-level bootstrap samples, so that the variability of the selection procedure itself was captured

**Internal validation** used the Harrell bootstrap optimism correction with patient-level resampling. The entire modelling procedure, including cut-point selection where applicable, is repeated inside each replicate; median optimism is subtracted from the apparent performance.

**Segmented linear regression** with a single breakpoint, selected by the Bayesian information criterion with cluster-robust standard errors, was used for the supplemental analysis of volume reduction rate on total alcohol retention time.

## Verification

Because the estimators were implemented from first principles rather than taken from an established package, `scripts/validate_implementations.py` checks each against data simulated from a known model:

| Check | Result |
| --- | --- |
| Kaplan-Meier vs the analytic survival function | maximum deviation < 0.02 |
| Cox coefficient recovery (true 0.80, −0.50) | maximum error < 0.05 |
| Cluster-robust variance without clustering | matches the model-based SE |
| Cluster-robust variance under perfect clustering | recovers the true SE where the model-based SE is falsely small |
| Grambsch-Therneau test when the assumption holds | no false alarm |
| Grambsch-Therneau test when the assumption is violated | violation detected in the offending covariate |
| Logistic coefficient recovery | maximum error < 0.10 |
| Firth penalisation under complete separation | finite estimate where maximum likelihood diverges |

## License

MIT. See `LICENSE`.
