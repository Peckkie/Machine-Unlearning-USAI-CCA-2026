"""
A3 (reviewers 1, 14, 15): flip-consistency score — representation-level evidence, forward passes only.

For every test image x:  s(x) = cos( f(x), f(flip_h(x)) ),  f = global-average-pooled penultimate features.
Higher s = more flip-invariant. Also reports prediction agreement P[argmax(x) == argmax(flip(x))] for full models.
Pairs compared (paired by image): Wilcoxon signed-rank + case-level bootstrap 95% CI of the mean difference.

Same preprocessing as training / evaluate.py: load_img(target_size) -> /255.
usage (machine 29, CPU):
  CUDA_VISIBLE_DEVICES=-1 /home/kannika/miniconda3/envs/AI/bin/python experiments/analysis/a3_flip_consistency.py \
      --out /media/tohn/HDD2/Model_unlearn_2026/analysis
"""
import argparse
import os

import numpy as np
import pandas as pd

H = '/media/tohn/HDD2'
MODELS = {  # name: (kind, path)
    'EffNetB5 backbone ImageNet':          ('efn_imagenet', '/home/kannika/.keras/models/efficientnet-b5_weights_tf_dim_ordering_tf_kernels_autoaugment.h5'),
    'EffNetB5 backbone unlearned (B4-B7)': ('unlearn_backbone', f'{H}/mini-ImageNet/EffNetB5Model_unlearn/R2/unfreezeB4-B7/on_epoch_end/modelEffNetB5_Unlearning_unfreezeB4-B7_R2_epoch50.h5'),
    'EffNetB5 original (Table 3)':         ('full', '/media/tohn/SSD/ModelTrainByImages/R2_1/models/B5R2_block5_15AB_1FC_3.h5'),
    'EffNetB5 unlearned (Table 3)':        ('full', f'{H}/Model_unlearn/EffNetB5Model/MLunlearn_USAI/R2_unbalanced/unfreezeBlock5a_se_excite/exp_unfreezeB4-B7/models/modelEffNetB5_MLunlearn_USAI_unfreezeBlock5a_se_excite_exp_unfreezeB4-B7-R2_unbalanced.h5'),
    'ResNet152V2 original (Table 3)':      ('full', f'{H}/Model_unlearn/ResNet152v2Model/MLorigin_USAI/R2_unbalanced/unfreeze_conv3_block-conv5_block/models/modelResNet152v2_MLorigin_USAI_unfreeze_conv3_block-conv5_block-R2_unbalanced.h5'),
    'ResNet152V2 unlearned (Table 3)':     ('full', f'{H}/Model_unlearn/ResNet152v2Model/MLunlearn_USAI/R2_unbalanced/unfreeze_conv3_block-conv5_block/models/modelResNet152v2_MLunlearn_USAI_unfreeze_conv3_block-conv5_block-R2_unbalanced.h5'),
}
PAIRS = [('EffNetB5 backbone ImageNet', 'EffNetB5 backbone unlearned (B4-B7)'),
         ('EffNetB5 original (Table 3)', 'EffNetB5 unlearned (Table 3)'),
         ('ResNet152V2 original (Table 3)', 'ResNet152V2 unlearned (Table 3)')]
TESTSETS = {'lab':   ('/media/tohn/HDD/VISION_dataset/Testdf_fold1_2_v1.csv', 'Path Crop',
                      lambda d: d[(d['Path Crop'] != 'None') & (d['Path Crop'] != 'Nan')]),
            'field': ('/home/yupaporn/CSV_file/UICCA_DiagRadioExp_AzureDb51Case813Image.csv', 'Path Crop_I7',
                      lambda d: d[d['Sub_Class_15AB'] != 'None'])}


def build(kind, path):
    """return (feature_model, full_model_or_None, image_size)"""
    import efficientnet.tfkeras as efn
    from tensorflow.keras import layers, Model, Input
    from tensorflow.keras.models import load_model
    if kind == 'efn_imagenet':
        m = efn.EfficientNetB5(weights=None, include_top=True)
        m.load_weights(path)
        gap = [l for l in m.layers if isinstance(l, layers.GlobalAveragePooling2D)][-1]
        return Model(m.input, gap.output), None, 456
    if kind == 'unlearn_backbone':  # the nested backbone exactly as downstream R1 uses it (loadmodelUnlearn)
        bb = load_model(path, compile=False).get_layer('efficientnet-b5')
        x = Input((456, 456, 3))
        return Model(x, layers.GlobalAveragePooling2D()(bb(x))), None, 456
    m = load_model(path, compile=False)
    gap = [l for l in m.layers if isinstance(l, layers.GlobalAveragePooling2D)][-1]
    return Model(m.input, [gap.output, m.output]), m, m.input_shape[1]


def run_model(name, kind, path, paths, bs):
    from tensorflow.keras.preprocessing import image
    feat, full, size = build(kind, path)
    sims, agree = [], []
    for i in range(0, len(paths), bs):
        x = np.stack([image.img_to_array(image.load_img(p, target_size=(size, size))) for p in paths[i:i + bs]]) / 255.
        xf = x[:, :, ::-1, :]                                     # horizontal flip (exact pixel permutation)
        o, of = feat.predict(x, verbose=0), feat.predict(xf, verbose=0)
        if full is not None:
            (f, pr), (ff, prf) = o, of
            agree.extend(pr.argmax(1) == prf.argmax(1))
        else:
            f, ff = o, of
        sims.extend(np.sum(f * ff, 1) / (np.linalg.norm(f, axis=1) * np.linalg.norm(ff, axis=1) + 1e-12))
        print(f'\r  {name}: {min(i + bs, len(paths))}/{len(paths)}', end='', flush=True)
    print()
    return np.array(sims), (np.array(agree) if agree else None)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default='analysis')
    ap.add_argument('--testset', default='lab', choices=list(TESTSETS))
    ap.add_argument('--bs', type=int, default=16)
    ap.add_argument('--limit', type=int, default=0, help='smoke test: first N images only')
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    csv, pcol, keep = TESTSETS[args.testset]
    df = keep(pd.read_csv(csv)).reset_index(drop=True)
    if args.limit:
        df = df.head(args.limit)
    paths = df[pcol].tolist()
    print(f'[A3] {args.testset} test: {len(paths)} images')

    per = pd.DataFrame({'img_path': paths, 'Case': df['Case'].values})
    agree = {}
    for name, (kind, path) in MODELS.items():
        s, a = run_model(name, kind, path, paths, args.bs)
        per[f'cos | {name}'] = s
        if a is not None:
            per[f'agree | {name}'] = a
            agree[name] = a.mean()
        per.to_csv(f'{args.out}/A3_flip_consistency_{args.testset}_per_image.csv', index=False)  # save as we go

    from scipy.stats import wilcoxon
    rng = np.random.default_rng(2026)
    cases = per['Case'].values; uniq = np.unique(cases)
    idx_by_case = {c: np.where(cases == c)[0] for c in uniq}
    rows = []
    for name in MODELS:
        s = per[f'cos | {name}'].values
        rows.append(dict(model=name, mean_cos=s.mean(), sd_cos=s.std(ddof=1), median_cos=np.median(s),
                         pred_agreement=agree.get(name, np.nan)))
    summ = pd.DataFrame(rows)
    comp = []
    for a, b in PAIRS:
        sa, sb = per[f'cos | {a}'].values, per[f'cos | {b}'].values
        d = sb - sa
        boot = [d[np.concatenate([idx_by_case[c] for c in rng.choice(uniq, len(uniq))])].mean() for _ in range(2000)]
        comp.append(dict(reference=a, compared=b, mean_diff=d.mean(), ci=f'{np.percentile(boot, 2.5):+.4f} to {np.percentile(boot, 97.5):+.4f}',
                         frac_lower=(d < 0).mean(), wilcoxon_p=wilcoxon(sa, sb).pvalue))
    comp = pd.DataFrame(comp)
    summ.to_csv(f'{args.out}/A3_flip_consistency_{args.testset}_summary.csv', index=False)
    comp.to_csv(f'{args.out}/A3_flip_consistency_{args.testset}_pairs.csv', index=False)
    pd.set_option('display.width', 220); pd.set_option('display.max_colwidth', 60)
    print(summ.to_string(index=False)); print(); print(comp.to_string(index=False))


if __name__ == '__main__':
    main()
