"""
A1 (reviewers 11–13): standard face-verification metrics from the EXISTING embedding distances (no retraining).

Distances behind the paper's face-matching table (machine 28, kannika/.../Tripletloss_unlearn/csv/result-cal):
  baseline = calculated_distances_effnet_base_8type.csv ; unlearned = combined_calculated_distances.csv
  type 5 = same-person pairs (8,339) , type 6 = different-person pairs (75,808)
Verifier: "same person" if distance < t.
  ROC-AUC = P(D_diff > D_same) (= CLES in the paper), EER, TAR @ FAR = 1% and 0.1%.
95% CI: identity-level cluster bootstrap (resample identities of the first image in each pair), B = 2000.

usage (machine 28): python experiments/analysis/a1_face_metrics.py --out /media/HDD/Model_unlearn_2026/analysis
"""
import argparse
import os
import re

import numpy as np
import pandas as pd

D = '/home/kannika/Machine-Unlearning-USAI-CCA-2025/Tripletloss_unlearn/csv/result-cal'
FILES = {'baseline': f'{D}/calculated_distances_effnet_base_8type.csv', 'unlearned': f'{D}/combined_calculated_distances.csv'}
SAME, DIFF, B, SEED = 5, 6, 2000, 2026


def identity(path):
    m = re.search(r'/(?:Add_star_re|Non_LR)/([^/]+)/', path)
    return m.group(1) if m else path.split('/')[-3]


def auc(ds, dd):  # P(D_diff > D_same) + 0.5 P(tie), via ranks
    from scipy.stats import rankdata
    r = rankdata(np.concatenate([ds, dd]))
    return (r[len(ds):].sum() - len(dd) * (len(dd) + 1) / 2) / (len(ds) * len(dd))


def eer(ds, dd):
    t = np.unique(np.concatenate([ds, dd]))
    t = t[np.linspace(0, len(t) - 1, min(len(t), 4000)).astype(int)]
    far = np.searchsorted(np.sort(dd), t, side='left') / len(dd)        # diff pairs accepted (d < t)
    frr = 1 - np.searchsorted(np.sort(ds), t, side='left') / len(ds)    # same pairs rejected (d >= t)
    i = np.argmin(np.abs(far - frr))
    return (far[i] + frr[i]) / 2


def tar_at_far(ds, dd, far):
    t = np.quantile(dd, far)  # threshold accepting `far` of different-person pairs
    return (ds < t).mean()


def metrics(ds, dd):
    return dict(roc_auc=auc(ds, dd), eer=eer(ds, dd), tar_far1=tar_at_far(ds, dd, 0.01), tar_far01=tar_at_far(ds, dd, 0.001))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default='analysis')
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    rng = np.random.default_rng(SEED)

    data = {}
    for name, f in FILES.items():
        d = pd.read_csv(f)
        d = d[d['type'].isin([SAME, DIFF])].copy()
        d['id1'] = d['path_img1'].map(identity)
        data[name] = d
    ids = np.unique(np.concatenate([data[n]['id1'].unique() for n in data]))
    rows, boots = [], {n: [] for n in data}
    for name, d in data.items():
        ds, dd = d.loc[d.type == SAME, 'distance'].values, d.loc[d.type == DIFF, 'distance'].values
        rows.append(dict(model=name, n_same=len(ds), n_diff=len(dd), n_identities=d['id1'].nunique(), **metrics(ds, dd)))
    groups = {n: {i: g for i, g in d.groupby('id1')} for n, d in data.items()}
    for _ in range(B):
        samp = rng.choice(ids, len(ids), replace=True)
        for n in data:
            g = pd.concat([groups[n][i] for i in samp if i in groups[n]])
            boots[n].append(metrics(g.loc[g.type == SAME, 'distance'].values, g.loc[g.type == DIFF, 'distance'].values))
    s = pd.DataFrame(rows)
    for k in ('roc_auc', 'eer', 'tar_far1', 'tar_far01'):
        for n in data:
            a = np.array([b[k] for b in boots[n]])
            s.loc[s.model == n, f'{k}_ci'] = f'{np.percentile(a, 2.5):.4f}-{np.percentile(a, 97.5):.4f}'
        diff = np.array([bu[k] - bb[k] for bb, bu in zip(boots['baseline'], boots['unlearned'])])
        s.loc[s.model == 'unlearned', f'delta_{k}'] = s.loc[s.model == 'unlearned', k].values[0] - s.loc[s.model == 'baseline', k].values[0]
        s.loc[s.model == 'unlearned', f'delta_{k}_ci'] = f'{np.percentile(diff, 2.5):+.4f} to {np.percentile(diff, 97.5):+.4f}'
    s.to_csv(f'{args.out}/A1_face_verification.csv', index=False)
    pd.set_option('display.width', 250)
    print(f'identity-level bootstrap B={B} over {len(ids)} identities\n')
    for _, r in s.iterrows():
        print(f"== {r.model}: {r.n_same} same / {r.n_diff} diff pairs, {r.n_identities} identities")
        for k, lab in (('roc_auc', 'ROC-AUC'), ('eer', 'EER'), ('tar_far1', 'TAR@FAR=1%'), ('tar_far01', 'TAR@FAR=0.1%')):
            extra = f"   Δ {r['delta_' + k]:+.4f} [{r['delta_' + k + '_ci']}]" if r.model == 'unlearned' else ''
            print(f"  {lab:13s} {r[k]:.4f} [{r[k + '_ci']}]{extra}")


if __name__ == '__main__':
    main()
