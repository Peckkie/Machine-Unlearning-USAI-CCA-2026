"""
X1 / C1 compute-matched control: how many extra downstream epochs equal the compute of the unlearning stage.

Two ways to match (report both in the paper, use one to set --epochs):
  [A] sample passes : extra_epochs = sum(N_mini_train * E_unlearn_stage) / N_usai_train
  [B] GPU time      : extra_epochs = sum(unlearn stage seconds) / downstream seconds-per-epoch  (read from TensorBoard logs)

Example
  # [A] only
  python3 compute_budget.py --unlearn_epochs 200 50 --n_usai_train 12000
  # [A] + [B]  (Mylogs_tensor folders of unlearn R1, unlearn R2, and the downstream R2 run)
  python3 compute_budget.py --unlearn_epochs 200 50 --n_usai_train 12000 \
      --tb_unlearn /path/R1/transfer/Mylogs_tensor /path/R2/unfreezeB5a_se_excite/Mylogs_tensor \
      --tb_downstream /path/MLorigin_USAI/.../unfreezeBlock5a_se_excite/Mylogs_tensor
"""
import argparse
import glob
import math
import os


def tb_wall_time(logdir):
    """Return (seconds, epochs) of a Keras TensorBoard log folder (train/ sub-folder, epoch_loss scalar)."""
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    files = sorted(glob.glob(os.path.join(logdir, '**', 'train', 'events.out.tfevents*'), recursive=True))
    if not files:
        raise FileNotFoundError(f'no train event file under {logdir}')
    seconds, epochs = 0.0, 0
    for f in files:  # several run_* folders when the run was resumed
        ea = EventAccumulator(f, size_guidance={'scalars': 0})
        ea.Reload()
        ev = ea.Scalars('epoch_loss')
        if len(ev) > 1:
            # first event is written at the end of epoch 1, so add one average epoch
            seconds += (ev[-1].wall_time - ev[0].wall_time) * len(ev) / (len(ev) - 1)
        epochs += len(ev)
    return seconds, epochs


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--n_mini_train', type=int, default=42000, help='mini-ImageNet unlearn train pairs (CSV subset=train)')
    p.add_argument('--unlearn_epochs', type=int, nargs='+', required=True, help='epochs used of each unlearn stage, e.g. 200 50 (R1, R2 checkpoint epoch)')
    p.add_argument('--n_usai_train', type=int, required=True, help='USAI train images of the fold (train-Kfold.py trainframe)')
    p.add_argument('--downstream_epochs', type=int, default=200, help='epochs of the normal downstream R2 run')
    p.add_argument('--tb_unlearn', nargs='*', default=[], help='Mylogs_tensor folders of each unlearn stage')
    p.add_argument('--tb_downstream', default=None, help='Mylogs_tensor folder of a normal downstream R2 run')
    args = p.parse_args()

    passes = args.n_mini_train * sum(args.unlearn_epochs)
    extra_a = math.ceil(passes / args.n_usai_train)
    print('=' * 80)
    print(f'[A] sample passes in unlearn stages : {passes:,} pairs (1 pair = 1 EffNet forward/backward)')
    print(f'    extra downstream epochs          : {extra_a}')
    print(f'    C1 --epochs (R2)                 : {args.downstream_epochs} + {extra_a} = {args.downstream_epochs + extra_a}')

    if args.tb_unlearn and args.tb_downstream:
        if len(args.tb_unlearn) != len(args.unlearn_epochs):
            p.error('--tb_unlearn ต้องมีจำนวนเท่ากับ --unlearn_epochs (1 log ต่อ 1 stage)')
        un_sec = 0.0
        for d, used in zip(args.tb_unlearn, args.unlearn_epochs):
            s, e = tb_wall_time(d)
            stage_sec = s / e * used  # time/epoch from the log x epochs actually used (e.g. checkpoint epoch 50 of 115)
            print(f'    unlearn  {e:4d} epochs logged, {used:4d} used  {stage_sec / 3600:7.2f} h  <- {d}')
            un_sec += stage_sec
        ds_sec, ds_ep = tb_wall_time(args.tb_downstream)
        sec_per_epoch = ds_sec / ds_ep
        extra_b = math.ceil(un_sec / sec_per_epoch)
        print(f'[B] unlearn GPU time                 : {un_sec / 3600:.2f} h')
        print(f'    downstream time / epoch          : {sec_per_epoch:.1f} s  ({ds_ep} epochs logged)')
        print(f'    extra downstream epochs          : {extra_b}')
        print(f'    C1 --epochs (R2)                 : {args.downstream_epochs} + {extra_b} = {args.downstream_epochs + extra_b}')
    print('=' * 80)


if __name__ == '__main__':
    main()
