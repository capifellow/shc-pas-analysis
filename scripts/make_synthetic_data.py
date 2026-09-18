"""Generate a synthetic dataset with the structure of the study data.

Patient-level data cannot be shared publicly. This script creates a synthetic
dataset with the same variables, coding, and clustering structure as the
analysed cohort so that the analysis pipeline can be executed end to end.

The synthetic data are simulated from a model in which an alcohol-to-cyst
volume ratio of at least 5% increases the hazard of complete regression;
numerical results will NOT reproduce those reported in the article.

Variables are described in the data dictionary in README.md. Indication and
symptom variables are patient-level and are therefore constant within a
patient; all other variables are cyst-level.
"""
import numpy as np
import pandas as pd

OUT = 'data/synthetic_example.csv'


def make(n_patients=92, seed=42):
    rng = np.random.default_rng(seed)

    # 82 patients contribute one cyst, nine contribute two, one contributes three
    counts = np.array([1] * 82 + [2] * 9 + [3])
    rng.shuffle(counts)
    counts = counts[:n_patients]
    pid = np.repeat(np.arange(1, n_patients + 1), counts)
    n = len(pid)

    # ---------------------------------------------------- patient-level block
    # indication: symptomatic (abdominal discomfort, pain, or fever) vs
    # asymptomatic progressive enlargement
    p_sympt = rng.binomial(1, 0.66, n_patients)
    p_type = np.where(p_sympt == 1,
                      rng.choice(['discomfort', 'pain', 'fever'], n_patients,
                                 p=[.62, .35, .03]),
                      'none')
    p_age = rng.binomial(1, 0.46, n_patients)
    p_male = rng.binomial(1, 0.18, n_patients)

    take = lambda v: np.repeat(v, counts)
    symptomatic, symptom_type = take(p_sympt), take(p_type)
    age_ge65, male = take(p_age), take(p_male)

    # ------------------------------------------------------- cyst-level block
    cyst_volume = np.exp(rng.normal(np.log(700), 0.9, n)).clip(50, 4500)
    alcohol = np.minimum(cyst_volume * rng.uniform(0.015, 0.22, n), 100)
    ratio = alcohol / cyst_volume * 100
    ge5 = (ratio >= 5).astype(int)

    reposition = rng.binomial(1, 0.82, n)
    multisession = rng.binomial(1, 0.45, n)
    hemorrhagic = rng.binomial(1, 0.11, n)
    infected = rng.binomial(1, 0.12, n) * (1 - hemorrhagic)
    complicated = ((hemorrhagic + infected) > 0).astype(int)
    catheter_10F = rng.binomial(1, 0.18, n)
    retention = rng.choice([10, 20, 30, 60, 120, 240], n,
                           p=[.19, .19, .21, .21, .16, .04]).astype(float)
    catheter_days = rng.poisson(8, n).clip(1, 53).astype(float)

    # time to complete regression
    lp = (1.9 * ge5 + 1.1 * reposition - 0.55 * np.log(cyst_volume / 700))
    T = rng.exponential(950.0 / np.exp(lp))
    C = 13 + rng.exponential(32, n).clip(0, 145)
    time = np.minimum(T, C)
    event = (T <= C).astype(int)

    follow_up = np.where(event == 1, np.maximum(time, rng.uniform(13, 158, n)), time)
    vrr = 100 - 100 * np.exp(-0.012 * follow_up - 1.05 * ge5
                             - 0.45 * reposition - rng.exponential(0.95, n))
    vrr = np.where(event == 1, 100.0, np.clip(vrr, 10, 99.9))
    successful = (vrr >= 90).astype(int)

    df = pd.DataFrame(dict(
        patient_id=pid,
        age_ge65=age_ge65,
        male=male,
        symptomatic=symptomatic,
        symptom_type=symptom_type,
        cyst_volume_mL=cyst_volume.round(0),
        alcohol_mL=alcohol.round(0),
        ratio_pct=ratio.round(2),
        ratio_ge5=ge5,
        retention_min=retention,
        catheter_days=catheter_days,
        catheter_10F=catheter_10F,
        reposition=reposition,
        multisession=multisession,
        complicated_cyst=complicated,
        hemorrhagic_cyst=hemorrhagic,
        infected_cyst=infected,
        follow_up_months=follow_up.round(1),
        time_months=time.round(1),
        complete_regression=event,
        vrr_pct=vrr.round(1),
        successful_regression=successful))

    # symptom resolution at the first outpatient visit after treatment;
    # patient-level, recorded only for symptomatic patients. Resolution is
    # slightly less likely when a treated cyst showed partial regression.
    worst = df.groupby('patient_id').successful_regression.min()
    p_res = np.where(worst.values == 1, 0.99, 0.93)
    resolved = rng.binomial(1, p_res)
    resolved = np.where(p_sympt == 1, resolved, -1)   # -1 = not applicable
    df['symptom_resolved'] = take(resolved)
    return df


if __name__ == '__main__':
    df = make()
    df.to_csv(OUT, index=False)
    p = df.drop_duplicates('patient_id')
    print(f'wrote {OUT}: {len(df)} cysts in {df.patient_id.nunique()} patients, '
          f'{int(df.complete_regression.sum())} complete regressions, '
          f'{int(df.successful_regression.sum())} successful regressions, '
          f'{int(p.symptomatic.sum())} symptomatic patients '
          f'({int((p.symptom_resolved == 1).sum())} with symptom resolution)')
