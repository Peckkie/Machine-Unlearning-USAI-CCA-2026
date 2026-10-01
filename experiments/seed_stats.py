"""
X2 repeated runs: mean ± SD over seeds and paired test across seeds.

Input CSV (one row per model x seed x test set), see results_template.csv
    model,seed,testset,accuracy,macro_f1
    original,1,lab,0.81,0.78
    unlearned,1,lab,0.84,0.80
    ...
model   : original, unlearned, C1, C2
testset : lab, field

Example
    python3 seed_stats.py --csv results_seeds.csv --ref unlearned
"""
import argparse

import pandas as pd
from scipy import stats


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--csv', required=True)
    p.add_argument('--ref', default='unlearned', help='model compared against every other model')
    p.add_argument('--out', default='seed_stats_summary.csv')
    args = p.parse_args()

    df = pd.read_csv(args.csv)
    metrics = [c for c in df.columns if c not in ('model', 'seed', 'testset')]

    rows = []
    for testset, d_t in df.groupby('testset'):
        for metric in metrics:
            wide = d_t.pivot(index='seed', columns='model', values=metric)
            for model in wide.columns:
                row = {'testset': testset, 'metric': metric, 'model': model,
                       'n_seeds': int(wide[model].notna().sum()),
                       'mean': wide[model].mean(), 'sd': wide[model].std(ddof=1)}
                if model != args.ref:
                    pair = wide[[args.ref, model]].dropna()  # pair by seed
                    diff = pair[args.ref] - pair[model]
                    row['mean_diff_vs_ref'] = diff.mean()
                    if len(pair) >= 2:
                        row['paired_t'], row['p_value'] = stats.ttest_rel(pair[args.ref], pair[model])
                rows.append(row)

    out = pd.DataFrame(rows)
    out.to_csv(args.out, index=False)
    pd.set_option('display.width', 200)
    for (testset, metric), d in out.groupby(['testset', 'metric']):
        print(f'\n=== {testset} test | {metric} | ref = {args.ref} ===')
        for _, r in d.iterrows():
            line = f"  {r['model']:<10} {r['mean']:.4f} ± {r['sd']:.4f}  (n={r['n_seeds']})"
            if 'p_value' in r and pd.notna(r.get('p_value')):
                line += f"   Δ(ref-model) = {r['mean_diff_vs_ref']:+.4f}  paired t = {r['paired_t']:.2f}  p = {r['p_value']:.4f}"
            print(line)
    print(f'\nSaved: {args.out}')
    print('Note: n = 3 seeds -> paired t-test has df = 2; also report per-seed differences (all same sign or not).')


if __name__ == '__main__':
    main()
