# Matbench MP Gap Experiments

This file records clean internal-validation experiments for `matbench_mp_gap`.
The held-out Matbench test fold is kept blind during optimization.

## Split And Leakage Policy

- Task: `matbench_mp_gap`
- Split: official Matbench fold 0
- Internal validation: 10% split from the official train+validation fold
- Test handling: `include_test_targets=False`, `--eval-test` not used
- Selection/calibration data: train/internal-val only
- Resource policy: single 24 GB GPU, CPU workers capped to 4-8

## Current Best Clean Internal-Val Result

Run directory:

`checkpoints/20260609_213826_matbench_mp_gap_comp_cewald_cnn_dropcoords_resumed`

Warm-restart command:

```bash
env PYTHONUNBUFFERED=1 pixi run python train.py \
  --dataset matbench_mp_gap --model-size large \
  --add-composition --add-comp-ewald --add-comp-nn --drop-coords \
  --matbench-split official --matbench-fold 0 --matbench-val-ratio 0.1 \
  --max-epochs 15 --patience 15 --max-cpu-workers 4 \
  --checkpoint-interval 5 --log-every 250 --eval-train-every 0 \
  --structural-bias-lr-mult 20 --loss-compression-scale 5.0 \
  --resume checkpoints/20260609_211539_matbench_mp_gap_comp_cewald_cnn_dropcoords_resumed/matbench_mp_gap_train_best_val.pth \
  --warm-restart --lr 2e-6
```

This loaded only the E74 tail-resume best model weights and restarted the
optimizer/scheduler at a controlled low learning rate.

Training result:

- Best raw internal-val MAE: `0.1909816447128995`
- Best raw epoch: `9`
- Final epoch raw internal-val MAE: `0.19169204973260823`
- Best checkpoint: `matbench_mp_gap_train_best_val.pth`

Train-only calibration result:

- `prediction_scale`: `0.9788793103448277`
- `prediction_bias`: `0.0020258320262655618`
- `prediction_min_value`: `0.0`
- `prediction_zero_threshold`: `0.08749280689075462`
- Train raw MAE: `0.06243138569799237`
- Train calibrated MAE: `0.05116716518026227`
- Internal-val raw MAE: `0.1909633870402973`
- Internal-val calibrated MAE: `0.18526953161707985`

## 80-Epoch Tail Resume Reference

Run directory:

`checkpoints/20260609_211539_matbench_mp_gap_comp_cewald_cnn_dropcoords_resumed`

Resume command:

```bash
env PYTHONUNBUFFERED=1 pixi run python train.py \
  --dataset matbench_mp_gap --model-size large \
  --add-composition --add-comp-ewald --add-comp-nn --drop-coords \
  --matbench-split official --matbench-fold 0 --matbench-val-ratio 0.1 \
  --max-epochs 80 --patience 80 --max-cpu-workers 4 \
  --checkpoint-interval 10 --log-every 250 --eval-train-every 0 \
  --structural-bias-lr-mult 20 --loss-compression-scale 5.0 \
  --resume checkpoints/20260609_203927_matbench_mp_gap_comp_cewald_cnn_dropcoords/matbench_mp_gap_train_best_val.pth
```

This resumed from the 50-epoch best-val checkpoint with optimizer/scheduler/RNG
state. Because `max_epochs` was changed from 50 to 80, the restored scheduler
continued with the new total step count; this behaved like a controlled tail
resume, not a pure fixed-low-LR finetune.

Training result:

- Best raw internal-val MAE: `0.19149403465853215`
- Best raw epoch: `74`
- Final epoch raw internal-val MAE: `0.19228556685072942`
- Best checkpoint: `matbench_mp_gap_train_best_val.pth`

Train-only calibration result:

- `prediction_scale`: `0.9768965517241379`
- `prediction_bias`: `0.002490254961724939`
- `prediction_min_value`: `0.0`
- `prediction_zero_threshold`: `0.07`
- Train raw MAE: `0.06538834212617449`
- Train calibrated MAE: `0.053470605442049735`
- Internal-val raw MAE: `0.1915013018724585`
- Internal-val calibrated MAE: `0.18574099966490124`

## 50-Epoch Schedule Reference

Run directory:

`checkpoints/20260609_203927_matbench_mp_gap_comp_cewald_cnn_dropcoords`

Training command:

```bash
env PYTHONUNBUFFERED=1 pixi run python train.py \
  --dataset matbench_mp_gap --model-size large \
  --add-composition --add-comp-ewald --add-comp-nn --drop-coords \
  --matbench-split official --matbench-fold 0 --matbench-val-ratio 0.1 \
  --max-epochs 50 --patience 50 --max-cpu-workers 4 \
  --checkpoint-interval 10 --log-every 250 --eval-train-every 0 \
  --structural-bias-lr-mult 20 --loss-compression-scale 5.0
```

Training result:

- Best raw internal-val MAE: `0.1958390816335572`
- Best raw epoch: `49`
- Final epoch raw internal-val MAE: `0.19595054178300966`
- Best checkpoint: `matbench_mp_gap_train_best_val.pth`

Train-only calibration command:

```bash
env PYTHONUNBUFFERED=1 pixi run python scripts/calibrate_matbench_gap.py \
  --checkpoint checkpoints/20260609_203927_matbench_mp_gap_comp_cewald_cnn_dropcoords/matbench_mp_gap_train_best_val.pth \
  --dataset matbench_mp_gap --model-size large \
  --add-composition --add-comp-ewald --add-comp-nn --drop-coords \
  --matbench-fold 0 --matbench-val-ratio 0.1 \
  --batch-size 256 --max-cpu-workers 4 --loss-compression-scale 5.0 \
  --output checkpoints/20260609_203927_matbench_mp_gap_comp_cewald_cnn_dropcoords/best_val_calibration.json
```

Calibration result:

- `prediction_scale`: `0.9768965517241379`
- `prediction_bias`: `-0.0017141314669801243`
- `prediction_min_value`: `0.0`
- `prediction_zero_threshold`: `0.114`
- Train raw MAE: `0.08068698849458913`
- Train calibrated MAE: `0.06842921710414535`
- Internal-val raw MAE: `0.1958378539378899`
- Internal-val calibrated MAE: `0.1889698844736923`

## Element/NN/Ewald Statistics Feature Result

Run directory:

`checkpoints/20260609_222021_matbench_mp_gap_comp_elprops_cewald_nn_cnn_dropcoords_cewalds_cnns`

Training command:

```bash
env PYTHONUNBUFFERED=1 MALLOC_ARENA_MAX=2 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 pixi run python train.py \
  --dataset matbench_mp_gap --model-size large \
  --add-composition --add-comp-ewald --add-comp-nn \
  --add-comp-ewald-stats --add-comp-nn-stats --add-nn-stats --add-element-props \
  --drop-coords \
  --matbench-split official --matbench-fold 0 --matbench-val-ratio 0.1 \
  --max-epochs 50 --patience 50 --max-cpu-workers 4 \
  --checkpoint-interval 10 --log-every 250 --eval-train-every 0 \
  --structural-bias-lr-mult 20 --loss-compression-scale 5.0
```

This feature set adds per-element electronegativity/ionization/electron-affinity
properties, global nearest-neighbor summary tokens, and per-composition
Ewald/nearest-neighbor distribution statistics. It keeps the same official fold0
train/internal-val protocol and leaves the test fold blind.

Training result:

- Best raw internal-val MAE: `0.19380969149196547`
- Best raw epoch: `46`
- Final epoch raw internal-val MAE: `0.19417301546992194`
- Best checkpoint: `matbench_mp_gap_train_best_val.pth`

Train-only calibration result:

- `prediction_scale`: `0.9788793103448277`
- `prediction_bias`: `-0.0027797758539110937`
- `prediction_min_value`: `0.0`
- `prediction_zero_threshold`: `0.09211876527839818`
- Train raw MAE: `0.06721239903507012`
- Train calibrated MAE: `0.05566641541358365`
- Internal-val raw MAE: `0.19380021669323552`
- Internal-val calibrated MAE: `0.18829754023237796`

This is a small positive result over the 50-epoch base-feature schedule, but it
does not beat the current low-LR warm-restart best (`0.18526953161707985`
calibrated internal-val MAE). It may still be a useful starting point for a
controlled low-LR warm restart or for a lighter feature ablation.

## Baseline Reference

Run directory:

`checkpoints/20260609_162945_matbench_mp_gap_comp_cewald_cnn_dropcoords_resumed`

Result:

- 30-epoch raw internal-val MAE: `0.21436249205258304`
- Train-only calibrated internal-val MAE: `0.2065424962885058` to `0.2065564961817524`

The 50-epoch schedule improves over this baseline by about `0.0176` MAE after
the same train-only calibration procedure. The 80-epoch tail resume improves
over the 30-epoch baseline by about `0.0208` MAE after train-only calibration.
The controlled low-LR warm restart improves over the 30-epoch baseline by about
`0.0213` MAE after train-only calibration.

## Density Feature Negative Result

Run directory:

`checkpoints/20260609_202143_matbench_mp_gap_comp_cewald_dens_cnn_dropcoords`

Command differences from the current best:

- Added `--add-density`
- Trained for 20 epochs

Result:

- Best raw internal-val MAE: `0.24584001029610478`
- Train-only calibrated internal-val MAE: `0.2370622604588617`

Density/volume features were not competitive with the current base feature
recipe under this schedule.

## Notes

- These are not final official Matbench test results.
- The test fold was not evaluated during any experiment recorded here.
- The strongest clean improvement so far comes from a longer cosine schedule,
  the 80-epoch tail resume, a controlled low-LR warm restart, and train-only
  scale/bias/nonnegative/zero-threshold calibration.
