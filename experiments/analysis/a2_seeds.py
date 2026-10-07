"""
A2 v2 (reviewers 6, 8): original vs unlearned trained with the SAME pipeline (train.py), 3 seeds each.
Uses the per-image predictions written by evaluate.py ($PRED_DIR/{model}_seed{k}_{lab|field}.csv).

Per seed pair k (original seed k vs unlearned seed k):
  accuracy, macro-F1 (Table 3/4 convention), balanced accuracy, Normal-vs-Abnormal sensitivity / specificity,
  case-level bootstrap 95% CI of Δ = unlearned − original, exact McNemar on per-image correctness.
Pooled over seeds:
  mean Δ over the 3 pairs, 95% CI from a case-level bootstrap that resamples the SAME patients for all pairs.
Per class: mean ± SD over seeds of sensitivity and specificity for each model.

usage (machine 29): python experiments/analysis/a2_seeds.py --pred_dir /media/tohn/HDD2/Model_unlearn_2026/predictions \
                       --out /media/tohn/HDD2/Model_unlearn_2026/analysis
"""
import argparse
import os

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score, classification_report

try:
    from scipy.stats import binomtest
    def binom_p(k, n): return binomtest(k, n, 0.5).pvalue
except ImportError:  # scipy < 1.7
    from scipy.stats import binom_test
    def binom_p(k, n): return binom_test(k, n, 0.5)

TESTSETS = {  # same filters as evaluate.py -> same row order as the prediction files
    'lab':   ('/media/tohn/HDD/VISION_dataset/Testdf_fold1_2_v1.csv', 'Path Crop',
              lambda d: d[(d['Path Crop'] != 'None') & (d['Path Crop'] != 'Nan')]),
    'field': ('/home/yupaporn/CSV_file/UICCA_DiagRadioExp_AzureDb51Case813Image.csv', 'Path Crop_I7',
              lambda d: d[d['Sub_Class_15AB'] != 'None']),
}
SEEDS, B, SEED = (1, 2, 3), 2000, 2026
METRICS = ('acc', 'macro_f1', 'bal_acc', 'abn_sens', 'abn_spec')


def metrics(y, p):
    rep = classification_report(y, p, output_dict=True, zero_division=0)
    ay, ap = (y != 'Normal'), (p != 'Normal')
    return dict(acc=accuracy_score(y, p), macro_f1=rep['macro avg']['f1-score'], bal_acc=balanced_accuracy_score(y, p),
                abn_sens=(ay & ap).sum() / max(ay.sum(), 1), abn_spec=(~ay & ~ap).sum() / max((~ay).sum(), 1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pred_dir', required=True)
    ap.add_argument('--out', default='analysis')
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    rng = np.random.default_rng(SEED)
    pair_rows, pooled_rows, cls_rows = [], [], []

    for ts, (csv, pcol, keep) in TESTSETS.items():
        ref = keep(pd.read_csv(csv)).reset_index(drop=True)
        cases = ref['Case'].values
        uniq = np.unique(cases)
        idx_by_case = {c: np.where(cases == c)[0] for c in uniq}
        y, P = None, {}
        for m in ('original', 'unlearned'):
            for k in SEEDS:
                d = pd.read_csv(f'{args.pred_dir}/{m}_seed{k}_{ts}.csv')
                assert len(d) == len(ref) and (d['img_path'].values == ref[pcol].values).all(), f'{m} seed {k} {ts}: row order mismatch'
                y = d['label'].values if y is None else y
                P[(m, k)] = d['pred'].values

        # per seed pair
        for k in SEEDS:
            po, pu = P[('original', k)], P[('unlearned', k)]
            mo, mu = metrics(y, po), metrics(y, pu)
            boot = []
            for _ in range(B):
                idx = np.concatenate([idx_by_case[c] for c in rng.choice(uniq, len(uniq))])
                bo, bu = metrics(y[idx], po[idx]), metrics(y[idx], pu[idx])
                boot.append({m: bu[m] - bo[m] for m in METRICS})
            co, cu = po == y, pu == y
            b_, c_ = int((co & ~cu).sum()), int((~co & cu).sum())
            row = dict(testset=ts, seed=k, n_images=len(y), n_cases=len(uniq), mcnemar_orig_only=b_, mcnemar_unl_only=c_,
                       mcnemar_p=binom_p(min(b_, c_), b_ + c_) if b_ + c_ else 1.0)
            for m in METRICS:
                a = np.array([bb[m] for bb in boot])
                row.update({f'orig_{m}': mo[m], f'unl_{m}': mu[m], f'delta_{m}': mu[m] - mo[m],
                            f'delta_{m}_lo': np.percentile(a, 2.5), f'delta_{m}_hi': np.percentile(a, 97.5)})
            pair_rows.append(row)

        # pooled over seeds: same resampled patients for all pairs, average Δ
        boot = []
        for _ in range(B):
            idx = np.concatenate([idx_by_case[c] for c in rng.choice(uniq, len(uniq))])
            ds = [{m: metrics(y[idx], P[('unlearned', k)][idx])[m] - metrics(y[idx], P[('original', k)][idx])[m]
                   for m in METRICS} for k in SEEDS]
            boot.append({m: np.mean([d[m] for d in ds]) for m in METRICS})
        for m in METRICS:
            o = [metrics(y, P[('original', k)])[m] for k in SEEDS]
            u = [metrics(y, P[('unlearned', k)])[m] for k in SEEDS]
            a = np.array([bb[m] for bb in boot])
            pooled_rows.append(dict(testset=ts, metric=m, orig_mean=np.mean(o), orig_sd=np.std(o, ddof=1),
                                    unl_mean=np.mean(u), unl_sd=np.std(u, ddof=1), mean_delta=np.mean(u) - np.mean(o),
                                    ci_lo=np.percentile(a, 2.5), ci_hi=np.percentile(a, 97.5),
                                    seeds_positive=int(sum(ui > oi for oi, ui in zip(o, u)))))

        # per class
        for cls in sorted(set(y)):
            for m in ('original', 'unlearned'):
                sens, spec = [], []
                for k in SEEDS:
                    p = P[(m, k)]
                    sens.append(((y == cls) & (p == cls)).sum() / (y == cls).sum())
                    spec.append(((y != cls) & (p != cls)).sum() / (y != cls).sum())
                cls_rows.append(dict(testset=ts, cls=cls, support=int((y == cls).sum()), model=m,
                                     sens_mean=np.mean(sens), sens_sd=np.std(sens, ddof=1),
                                     spec_mean=np.mean(spec), spec_sd=np.std(spec, ddof=1)))

    pr, po, pc = pd.DataFrame(pair_rows), pd.DataFrame(pooled_rows), pd.DataFrame(cls_rows)
    pr.to_csv(f'{args.out}/A2v2_per_seed_pairs.csv', index=False)
    po.to_csv(f'{args.out}/A2v2_pooled.csv', index=False)
    pc.to_csv(f'{args.out}/A2v2_per_class.csv', index=False)

    print(f'case-level bootstrap B={B}\n')
    for ts in TESTSETS:
        print(f'===== {ts} =====')
        for _, r in pr[pr.testset == ts].iterrows():
            print(f"  seed {r.seed}: acc {r.orig_acc:.3f}->{r.unl_acc:.3f} Δ{r.delta_acc:+.3f} [{r.delta_acc_lo:+.3f},{r.delta_acc_hi:+.3f}] | "
                  f"F1 {r.orig_macro_f1:.3f}->{r.unl_macro_f1:.3f} Δ{r.delta_macro_f1:+.3f} [{r.delta_macro_f1_lo:+.3f},{r.delta_macro_f1_hi:+.3f}] | "
                  f"balAcc Δ{r.delta_bal_acc:+.3f} [{r.delta_bal_acc_lo:+.3f},{r.delta_bal_acc_hi:+.3f}] | "
                  f"McNemar {r.mcnemar_orig_only}/{r.mcnemar_unl_only} p={r.mcnemar_p:.3g}")
        print('  pooled over 3 seeds:')
        for _, r in po[po.testset == ts].iterrows():
            print(f"    {r.metric:9s} orig {r.orig_mean:.3f}±{r.orig_sd:.3f}  unl {r.unl_mean:.3f}±{r.unl_sd:.3f}  "
                  f"Δ {r.mean_delta:+.3f} [{r.ci_lo:+.3f}, {r.ci_hi:+.3f}]  seeds unl>orig: {r.seeds_positive}/3")
        print()
    print('saved A2v2_per_seed_pairs.csv, A2v2_pooled.csv, A2v2_per_class.csv')


if __name__ == '__main__':
    main()
