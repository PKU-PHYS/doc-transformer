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

## Baseline Reference

Run directory:

`checkpoints/20260609_162945_matbench_mp_gap_comp_cewald_cnn_dropcoords_resumed`

Result:

- 30-epoch raw internal-val MAE: `0.21436249205258304`
- Train-only calibrated internal-val MAE: `0.2065424962885058` to `0.2065564961817524`

The 50-epoch schedule improves over this baseline by about `0.0189` MAE after
the same train-only calibration procedure.

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
- The strongest clean improvement so far comes from a longer 50-epoch cosine
  schedule plus train-only scale/bias/nonnegative/zero-threshold calibration.
