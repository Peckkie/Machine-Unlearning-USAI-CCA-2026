"""
A2 (reviewers 6, 8): statistics from the EXISTING predictions behind Table 3/4 — no retraining.

For each pair (original vs unlearned) on each test set:
  * accuracy, macro precision / recall / F1 (sklearn 'macro avg' over classes in y_true ∪ y_pred = Table 3/4),
    balanced accuracy (mean recall over classes present in y_true), Normal-vs-Abnormal sensitivity / specificity
  * case-level bootstrap 95% CI (resample patients = 'Case', B = 2000) for each model and for Δ = unlearned − original
  * exact McNemar test on per-image correctness (paired)
  * per-class sensitivity / specificity with Clopper–Pearson 95% CI

usage (machine 29):  /home/kannika/miniconda3/envs/AI/bin/python experiments/analysis/a2_stats.py --out /media/tohn/HDD2/Model_unlearn_2026/analysis
"""
import argparse
import os

import numpy as np
import pandas as pd
from scipy.stats import beta
try:
    from scipy.stats import binomtest
    def binom_p(k, n): return binomtest(k, n, 0.5).pvalue
except ImportError:  # scipy < 1.7
    from scipy.stats import binom_test
    def binom_p(k, n): return binom_test(k, n, 0.5)
from sklearn.metrics import accuracy_score, balanced_accuracy_score, classification_report

D = '/home/kannika/code'
PAIRS = [  # (backbone, testset, original csv, unlearned csv, path col, label col)
    ('EfficientNet-B5', 'lab', f'{D}/Result_EffNetB5_paper_test_on_newtestset.csv',
     f'{D}/Result_EffNetB5_unlearn_R2_UnfreezeB4-B7_USAI_unalanced.csv', 'Path Crop', 'Sub_class_New'),
    ('EfficientNet-B5', 'field', f'{D}/Result_EffNetB5_paper_test_on_FieldTestset.csv',
     f'{D}/Result_EffNetB5_Fine-Tune-Machine-Unlearning-unfreezeB4-B7_R2_FieldTestset.csv', 'Path Crop_I7', 'Sub_Class_15AB'),
    ('ResNet-152V2', 'lab', f'{D}/ResNet152v2_MLorigin_USAI_unfreeze_conv3_block-conv5_block-R2_unbalanced.csv',
     f'{D}/ResNet152v2_Unlearn_USAI_unfreeze_conv3_block-conv5_block-R2_unbalanced.csv', 'Path Crop', 'Sub_class_New'),
    ('ResNet-152V2', 'field', f'{D}/Result_ResNet152v2_MLorigin_USAI_unfreeze_conv3_block-conv5_block-R2_unbalanced_FieldTestset.csv',
     f'{D}/Result_ResNet152v2_MLunlearn_USAI_unfreeze_conv3_block-conv5_block-R2_unbalanced_FieldTestset.csv', 'Path Crop_I7', 'Sub_Class_15AB'),
    ('ViT-L/32', 'field', f'{D}/Result_VITL32_Original_USAI15ab-R1_unbalanced_FieldTestset.csv',
     f'{D}/Result_VITL32_Unlearn_USAI15ab-R1_unbalanced_FieldTestset.csv', 'Path Crop_I7', 'Sub_Class_15AB'),
]
B, SEED = 2000, 2026


def metrics(y, p):
    rep = classification_report(y, p, output_dict=True, zero_division=0)
    ab_y, ab_p = (y != 'Normal'), (p != 'Normal')
    return dict(acc=accuracy_score(y, p), macro_p=rep['macro avg']['precision'], macro_r=rep['macro avg']['recall'],
                macro_f1=rep['macro avg']['f1-score'], bal_acc=balanced_accuracy_score(y, p),
                abn_sens=(ab_y & ab_p).sum() / max(ab_y.sum(), 1), abn_spec=(~ab_y & ~ab_p).sum() / max((~ab_y).sum(), 1))


def cp(x, n):
    if n == 0:
        return np.nan, np.nan
    return (beta.ppf(0.025, x, n - x + 1) if x > 0 else 0.0), (beta.ppf(0.975, x + 1, n - x) if x < n else 1.0)


def load_pair(orig_csv, unl_csv, path_col, label_col):
    o, u = pd.read_csv(orig_csv), pd.read_csv(unl_csv)
    keep = ['Case', path_col, label_col, 'category']
    o, u = o[keep].reset_index(drop=True), u[keep].reset_index(drop=True)
    if len(o) == len(u) and (o[path_col].values == u[path_col].values).all():
        m = o.copy(); m['pred_o'] = o['category']; m['pred_u'] = u['category']
    else:  # align by image path
        m = o.merge(u[[path_col, 'category']], on=path_col, suffixes=('_o', '_u'), validate='one_to_one')
        m = m.rename(columns={'category_o': 'pred_o', 'category_u': 'pred_u'})
    assert (m[label_col].notna()).all()
    return m.rename(columns={label_col: 'y'})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default='analysis')
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    rng = np.random.default_rng(SEED)
    summary, perclass = [], []

    for bb, ts, oc, uc, pc, lc in PAIRS:
        m = load_pair(oc, uc, pc, lc)
        y, po, pu = m['y'].values, m['pred_o'].values, m['pred_u'].values
        mo, mu = metrics(y, po), metrics(y, pu)

        # case-level bootstrap (resample patients)
        cases = m['Case'].values
        uniq = np.unique(cases)
        idx_by_case = {c: np.where(cases == c)[0] for c in uniq}
        boots = {k: [] for k in ('acc', 'macro_f1', 'bal_acc')}
        for _ in range(B):
            idx = np.concatenate([idx_by_case[c] for c in rng.choice(uniq, size=len(uniq), replace=True)])
            bo, bu = metrics(y[idx], po[idx]), metrics(y[idx], pu[idx])
            for k in boots:
                boots[k].append((bo[k], bu[k], bu[k] - bo[k]))

        # exact McNemar on correctness
        co, cu = (po == y), (pu == y)
        b_, c_ = int((co & ~cu).sum()), int((~co & cu).sum())
        p_mcn = binom_p(min(b_, c_), b_ + c_) if b_ + c_ > 0 else 1.0

        row = dict(backbone=bb, testset=ts, n_images=len(m), n_cases=len(uniq),
                   mcnemar_orig_only_correct=b_, mcnemar_unl_only_correct=c_, mcnemar_p=p_mcn)
        for name, mm in (('orig', mo), ('unl', mu)):
            for k, v in mm.items():
                row[f'{name}_{k}'] = v
        for k, arr in boots.items():
            a = np.array(arr)
            row[f'orig_{k}_ci'] = f'{np.percentile(a[:, 0], 2.5):.4f}-{np.percentile(a[:, 0], 97.5):.4f}'
            row[f'unl_{k}_ci'] = f'{np.percentile(a[:, 1], 2.5):.4f}-{np.percentile(a[:, 1], 97.5):.4f}'
            row[f'delta_{k}'] = mu[k] - mo[k]
            row[f'delta_{k}_ci'] = f'{np.percentile(a[:, 2], 2.5):+.4f} to {np.percentile(a[:, 2], 97.5):+.4f}'
        summary.append(row)

        for cls in sorted(set(y)):
            for name, p in (('original', po), ('unlearned', pu)):
                tp = int(((y == cls) & (p == cls)).sum()); fn = int(((y == cls) & (p != cls)).sum())
                tn = int(((y != cls) & (p != cls)).sum()); fp = int(((y != cls) & (p == cls)).sum())
                sl, sh = cp(tp, tp + fn); pl, ph = cp(tn, tn + fp)
                perclass.append(dict(backbone=bb, testset=ts, model=name, cls=cls, support=tp + fn,
                                     sensitivity=tp / (tp + fn) if tp + fn else np.nan, sens_ci=f'{sl:.3f}-{sh:.3f}',
                                     specificity=tn / (tn + fp) if tn + fp else np.nan, spec_ci=f'{pl:.3f}-{ph:.3f}'))

    s = pd.DataFrame(summary); s.to_csv(f'{args.out}/A2_summary.csv', index=False)
    pd.DataFrame(perclass).to_csv(f'{args.out}/A2_per_class.csv', index=False)

    print(f'case-level bootstrap B={B}, seed={SEED}\n')
    for _, r in s.iterrows():
        print(f"== {r.backbone} | {r.testset} | {r.n_images} images / {r.n_cases} cases ==")
        for k, lab in (('acc', 'accuracy'), ('macro_f1', 'macro F1'), ('bal_acc', 'balanced acc')):
            print(f"  {lab:13s} orig {r['orig_' + k]:.4f} [{r['orig_' + k + '_ci']}]  unl {r['unl_' + k]:.4f} [{r['unl_' + k + '_ci']}]"
                  f"  Δ {r['delta_' + k]:+.4f} [{r['delta_' + k + '_ci']}]")
        print(f"  macro P/R    orig {r.orig_macro_p:.3f}/{r.orig_macro_r:.3f}  unl {r.unl_macro_p:.3f}/{r.unl_macro_r:.3f}"
              f"  | abnormal sens/spec orig {r.orig_abn_sens:.3f}/{r.orig_abn_spec:.3f}  unl {r.unl_abn_sens:.3f}/{r.unl_abn_spec:.3f}")
        print(f"  McNemar: only-orig-correct {r.mcnemar_orig_only_correct}, only-unl-correct {r.mcnemar_unl_only_correct}, exact p = {r.mcnemar_p:.3g}\n")
    print(f'saved: {args.out}/A2_summary.csv, {args.out}/A2_per_class.csv')


if __name__ == '__main__':
    main()
