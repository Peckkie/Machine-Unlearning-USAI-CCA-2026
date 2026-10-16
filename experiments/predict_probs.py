"""
Softmax probabilities of a Keras model on a list of images (keras load_img -> /255, size = model input).
Needs only TensorFlow + numpy, so it runs in envs without pandas / scikit-learn (e.g. a vit_keras env).
Use with evaluate.py:  --dump_paths DIR  ->  predict_probs.py  ->  evaluate.py --probs_dir DIR

  python predict_probs.py --model_h5 model.h5 --paths DIR/lab_paths.txt --out DIR/{model}_seed1_lab.npy --gpu 0
"""
import argparse
import os

import numpy as np


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model_h5', required=True)
    p.add_argument('--paths', required=True, help='text file, one image path per line')
    p.add_argument('--out', required=True)
    p.add_argument('--batch_size', type=int, default=16)
    p.add_argument('--gpu', default='0')
    args = p.parse_args()
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    for m in ('efficientnet.tfkeras', 'vit_keras.vit'):
        try:
            __import__(m)
        except ImportError:
            pass
    from tensorflow.keras.models import load_model
    from tensorflow.keras.preprocessing import image
    model = load_model(args.model_h5, compile=False)
    size = model.input_shape[1]
    paths = [l.rstrip('\n') for l in open(args.paths) if l.strip()]
    probs = []
    for i in range(0, len(paths), args.batch_size):
        x = np.stack([image.img_to_array(image.load_img(q, target_size=(size, size))) for q in paths[i:i + args.batch_size]])
        probs.append(model.predict(x / 255., verbose=0))
        print(f'\r  {min(i + args.batch_size, len(paths))}/{len(paths)}', end='')
    print()
    np.save(args.out, np.concatenate(probs))
    print('saved', args.out)


if __name__ == '__main__':
    main()
