"""Verify the custom implementations against simulated data with known truth.

Because the estimators in src/ are implemented from first principles rather
than taken from an established package, each is checked against data
simulated from a known model. Running this script should reproduce the
checks reported below.

Usage
-----
    python scripts/validate_implementations.py
"""
import sys
import os
import warnings
import numpy as np

warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from survival import km, cox_fit, cox_robust, cox_zph
from logistic import logit_fit

ok = lambda cond: 'PASS' if cond else 'FAIL'
print('=' * 72)
print('Validation of the custom implementations')
print('=' * 72)

# ------------------------------------------------- 1. Kaplan-Meier estimator
rng = np.random.default_rng(1)
T = rng.exponential(1.0, 6000)
C = rng.exponential(2.0, 6000)
t, e = np.minimum(T, C), (T <= C).astype(int)
tab = km(t, e)
print('\n1. Kaplan-Meier against the analytic survival function S(t) = exp(-t)')
worst = 0.0
for q in (0.5, 1.0, 2.0):
    s = tab[tab[:, 0] <= q][-1, 3]
    worst = max(worst, abs(s - np.exp(-q)))
    print(f'   S({q:.1f}) estimated {s:.4f}, analytic {np.exp(-q):.4f}')
print(f'   maximum absolute deviation {worst:.4f}   [{ok(worst < 0.02)}]')

# --------------------------------------------- 2. Cox coefficient recovery
rng = np.random.default_rng(2)
n = 4000
X = rng.normal(size=(n, 2))
beta_true = np.array([0.80, -0.50])
T = rng.exponential(np.exp(-(X @ beta_true)))
C = rng.exponential(1.2, n)
t, e = np.minimum(T, C), (T <= C).astype(int)
f = cox_fit(X, t, e)
err = np.max(np.abs(f.beta - beta_true))
print('\n2. Cox regression coefficient recovery')
print(f'   true  {beta_true}')
print(f'   fitted {np.round(f.beta, 3)}   ({f.nevent} events)')
print(f'   maximum absolute error {err:.3f}   [{ok(err < 0.05)}]')

# --------------------------------- 3. cluster-robust variance behaviour
rng = np.random.default_rng(3)
n = 1500
X = rng.normal(size=(n, 2))
T = rng.exponential(np.exp(-(0.6 * X[:, 0])))
C = rng.exponential(1.5, n)
t, e = np.minimum(T, C), (T <= C).astype(int)
f = cox_fit(X, t, e)
se_model = np.sqrt(np.diag(f.cov))
se_robust = np.sqrt(np.diag(cox_robust(f, np.arange(n))))
idx = np.repeat(np.arange(n), 2)
f2 = cox_fit(X[idx], t[idx], e[idx])
se_model_dup = np.sqrt(np.diag(f2.cov))
se_robust_dup = np.sqrt(np.diag(cox_robust(f2, idx)))
print('\n3. Cluster-robust variance')
print(f'   singleton clusters: model SE {np.round(se_model,4)}, robust SE {np.round(se_robust,4)}')
print(f'   each subject duplicated: model SE {np.round(se_model_dup,4)} (falsely small), '
      f'robust SE {np.round(se_robust_dup,4)}')
c1 = np.allclose(se_model, se_robust, rtol=0.05)
c2 = np.allclose(se_robust_dup, se_model, rtol=0.10)
print(f'   robust equals model SE without clustering [{ok(c1)}]; '
      f'robust recovers the true SE under perfect clustering [{ok(c2)}]')

# ------------------------------------------ 4. proportional hazards test
rng = np.random.default_rng(11)
n = 1500
X = rng.normal(size=(n, 2))
T = rng.exponential(np.exp(-(0.7 * X[:, 0])))
C = rng.exponential(1.5, n)
t, e = np.minimum(T, C), (T <= C).astype(int)
per_ok, gl_ok = cox_zph(cox_fit(X, t, e), t, e)
u = rng.uniform(size=n)
T2 = np.where(u < 0.5, rng.exponential(np.exp(-1.5 * X[:, 0])), rng.exponential(np.ones(n)))
C2 = rng.exponential(1.5, n)
t2, e2 = np.minimum(T2, C2), (T2 <= C2).astype(int)
per_bad, gl_bad = cox_zph(cox_fit(X, t2, e2), t2, e2)
print('\n4. Grambsch-Therneau test of proportional hazards')
print(f'   assumption holds:    covariate P values {[round(x["p"],3) for x in per_ok]}')
print(f'   assumption violated: covariate P values {[round(x["p"],4) for x in per_bad]}')
print(f'   no false alarm when PH holds [{ok(per_ok[0]["p"] > .05)}]; '
      f'violation detected in the offending covariate [{ok(per_bad[0]["p"] < .05)}]')

# ------------------------------------ 5. logistic regression, ML and Firth
rng = np.random.default_rng(4)
n = 6000
X = np.column_stack([np.ones(n), rng.normal(size=(n, 2))])
beta_true = np.array([-0.5, 1.0, -0.7])
p = 1 / (1 + np.exp(-(X @ beta_true)))
y = rng.binomial(1, p)
fml = logit_fit(X, y)
err = np.max(np.abs(fml.beta - beta_true))
print('\n5. Logistic regression coefficient recovery')
print(f'   true  {beta_true}')
print(f'   fitted {np.round(fml.beta, 3)}')
print(f'   maximum absolute error {err:.3f}   [{ok(err < 0.10)}]')

Xs = np.column_stack([np.ones(20), np.r_[np.zeros(10), np.ones(10)]])
ys = np.r_[np.zeros(10), np.ones(10)]          # complete separation
f_ml = logit_fit(Xs, ys)
f_fi = logit_fit(Xs, ys, firth=True)
print('\n6. Firth penalisation under complete separation')
print(f'   maximum likelihood slope {f_ml.beta[1]:10.2f} (diverges)')
print(f'   Firth penalised slope    {f_fi.beta[1]:10.2f} (finite)')
print(f'   Firth gives a finite estimate [{ok(abs(f_fi.beta[1]) < 20)}]')

print('\nAll checks complete.')
