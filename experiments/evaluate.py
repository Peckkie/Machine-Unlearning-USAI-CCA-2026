"""
Evaluate a USAI 15AB EfficientNet-B5 model on the Lab test set (1312) and/or Field test set (807).
Same procedure as the notebooks behind Table 3/4 of the paper:
  keras image.load_img(target_size=456) -> /255 -> argmax ; labels = fixed 15-class order ;
  macro precision/recall/F1 = sklearn classification_report 'macro avg' (labels = classes in y_true ∪ y_pred)

Output
  --results : one row per (model, seed, testset) appended -> input of seed_stats.py
  --pred_dir: per-image predictions + probabilities (for McNemar / bootstrap CI, A2)

Example
  python3 experiments/evaluate.py --model_h5 /path/model.h5 --model unlearned --seed 1 \
      --lab_csv /media/tohn/HDD/VISION_dataset/Testdf_fold1_2_v1.csv \
      --field_csv /home/yupaporn/CSV_file/UICCA_DiagRadioExp_AzureDb51Case813Image.csv \
      --results results_seeds.csv --pred_dir predictions --gpu 0
"""
import argparse
import os

import numpy as np
import pandas as pd

CLASSES = ['AB01', 'AB02', 'AB03', 'AB04', 'AB05', 'AB06', 'AB07', 'AB081', 'AB082',
           'AB083', 'AB09', 'AB10', 'AB11', 'AB12', 'Normal']

# test set definitions (path column, label column, row filter) as in the paper notebooks
TESTSETS = {
    'lab':   dict(path_col='Path Crop',    label_col='Sub_class_New',
                  keep=lambda d: d[(d['Path Crop'] != 'None') & (d['Path Crop'] != 'Nan')]),
    'field': dict(path_col='Path Crop_I7', label_col='Sub_Class_15AB',
                  keep=lambda d: d[d['Sub_Class_15AB'] != 'None']),
}


def load_model_any(args):
    import efficientnet.tfkeras  # noqa: F401  registers swish / FixedDropout
    from tensorflow.keras.models import load_model, model_from_json
    if args.model_h5:
        return load_model(args.model_h5, compile=False)
    with open(args.model_json) as f:
        model = model_from_json(f.read())
    model.load_weights(args.model_weights)
    return model


def predict(model, paths, size, batch_size):
    from tensorflow.keras.preprocessing import image
    probs = []
    for i in range(0, len(paths), batch_size):
        x = np.stack([image.img_to_array(image.load_img(p, target_size=(size, size))) for p in paths[i:i + batch_size]])
        probs.append(model.predict(x / 255., verbose=0))
        print(f'\r  {min(i + batch_size, len(paths))}/{len(paths)}', end='')
    print()
    return np.concatenate(probs)


def metrics(y_true, y_pred):
    from sklearn.metrics import accuracy_score, balanced_accuracy_score, classification_report
    rep = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    return dict(accuracy=accuracy_score(y_true, y_pred),
                macro_precision=rep['macro avg']['precision'],
                macro_recall=rep['macro avg']['recall'],
                macro_f1=rep['macro avg']['f1-score'],
                balanced_accuracy=balanced_accuracy_score(y_true, y_pred))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model_h5', default=None)
    p.add_argument('--model_json', default=None)
    p.add_argument('--model_weights', default=None)
    p.add_argument('--model', required=True, help='original | unlearned | C1 | C2')
    p.add_argument('--seed', required=True)
    p.add_argument('--lab_csv', default=None)
    p.add_argument('--field_csv', default=None)
    p.add_argument('--results', default='results_seeds.csv')
    p.add_argument('--pred_dir', default='predictions')
    p.add_argument('--batch_size', type=int, default=16)
    p.add_argument('--gpu', default='0')
    args = p.parse_args()
    if not args.model_h5 and not (args.model_json and args.model_weights):
        p.error('give --model_h5, or --model_json + --model_weights')

    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    model = load_model_any(args)
    size = model.input_shape[1]
    os.makedirs(args.pred_dir, exist_ok=True)

    for testset, csv in [('lab', args.lab_csv), ('field', args.field_csv)]:
        if not csv:
            continue
        t = TESTSETS[testset]
        df = t['keep'](pd.read_csv(csv)).reset_index(drop=True)
        print(f'[{args.model} seed {args.seed}] {testset} test: {len(df)} images')
        probs = predict(model, df[t['path_col']].tolist(), size, args.batch_size)
        y_true = df[t['label_col']].values
        y_pred = np.array(CLASSES)[probs.argmax(1)]

        out = pd.DataFrame({'img_path': df[t['path_col']], 'label': y_true, 'pred': y_pred, 'prob': probs.max(1)})
        out = pd.concat([out, pd.DataFrame(probs, columns=[f'p_{c}' for c in CLASSES])], axis=1)
        out.to_csv(f'{args.pred_dir}/{args.model}_seed{args.seed}_{testset}.csv', index=False)

        row = dict(model=args.model, seed=args.seed, testset=testset, **metrics(y_true, y_pred))
        print('  ' + '  '.join(f'{k}={v:.4f}' for k, v in row.items() if isinstance(v, float)))
        res = pd.DataFrame([row])
        if os.path.exists(args.results):
            old = pd.read_csv(args.results, dtype={'seed': str})
            old = old[~((old['model'] == args.model) & (old['seed'].astype(str) == str(args.seed)) & (old['testset'] == testset))]
            res = pd.concat([old, res], ignore_index=True)
        res.to_csv(args.results, index=False)
    print(f'Saved: {args.results} , {args.pred_dir}/')


if __name__ == '__main__':
    main()
