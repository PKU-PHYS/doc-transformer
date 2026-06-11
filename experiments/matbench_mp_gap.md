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

`checkpoints/20260611_075420_matbench_mp_gap_comp_cewald_cnn_dropcoords`

Checkpoint:

`matbench_mp_gap_train_best_val.pth`

This is a single checkpoint from an 80-epoch cosine-schedule run enabling
`numeric_path_film` under the default dropout setting. It applies train-only
scalar calibration plus a conservative binned residual correction fitted only
on the official train subset. No held-out test targets are loaded or used.

Result:

- Raw internal-val MAE: `0.18469890111309578`
- Raw best epoch: `70`
- Final epoch raw internal-val MAE: `0.1854295078517309`
- Scalar calibrated internal-val MAE: `0.1786675109253744`
- Binned-residual calibrated internal-val MAE: `0.17866467743857514`
- Binned-residual calibrated train MAE: `0.04138452037057726`
- `prediction_scale`: `0.9749137931034483`
- `prediction_bias`: `0.0040830272564600254`
- `prediction_min_value`: `0.0`
- `prediction_zero_threshold`: `0.068`
- Binned-residual bins: `6`
- Binned-residual shrinkage: `5000.0`
- Binned-residual zero threshold: `0.06078463622176789`
- Checkpoint source: single best-val checkpoint, not a checkpoint average

This improves the previous clean train-only-calibrated internal-validation best
from `0.18055832` to `0.17866468`, a no-leakage single-fold gain of about
`0.00189` MAE. The raw single-checkpoint best remains the three-way
`dropout=0.05 + numeric_path_film + bias_discrete_depths` run at `0.18349456`;
the `numeric_path_film`-only run is currently the best calibrated model, not
the best raw model.

## Previous Checkpoint-Averaged Result

Run directory:

`checkpoints/20260609_213826_matbench_mp_gap_comp_cewald_cnn_dropcoords_resumed`

Averaged checkpoint:

`avg_e10_e15.pth`

This averages the two saved low-LR warm-restart checkpoints from epoch 10 and
epoch 15, then applies train-only scalar calibration plus a conservative binned
residual correction fitted only on the official train subset.

Result:

- Raw internal-val MAE: `0.1915938070899512`
- Scalar calibrated internal-val MAE: `0.18508102969846527`
- Binned-residual calibrated internal-val MAE: `0.18496897995763836`
- Binned-residual calibrated train MAE: `0.05090135484499366`
- `prediction_scale`: `0.9768965517241379`
- `prediction_bias`: `-0.0003890366042996275`
- `prediction_min_value`: `0.0`
- `prediction_zero_threshold`: `0.08361550587698303`
- Binned-residual bins: `6`
- Binned-residual shrinkage: `5000.0`
- Binned-residual zero threshold: `0.066`
- Checkpoint source: weight average of `matbench_mp_gap_train_e10.pth` and
  `matbench_mp_gap_train_e15.pth`

## Previous Best Single-Checkpoint Result

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

## Statistics Feature Low-LR Warm Restart

Run directory:

`checkpoints/20260609_234526_matbench_mp_gap_comp_elprops_cewald_nn_cnn_dropcoords_cewalds_cnns_resumed`

Warm-restart command:

```bash
env PYTHONUNBUFFERED=1 MALLOC_ARENA_MAX=2 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 pixi run python train.py \
  --dataset matbench_mp_gap --model-size large \
  --add-composition --add-element-props --add-comp-ewald --add-nn-stats \
  --add-comp-nn --add-comp-ewald-stats --add-comp-nn-stats --drop-coords \
  --matbench-split official --matbench-fold 0 --matbench-val-ratio 0.1 \
  --max-epochs 15 --patience 15 --max-cpu-workers 4 \
  --checkpoint-interval 5 --log-every 250 --eval-train-every 0 \
  --structural-bias-lr-mult 20 --loss-compression-scale 5.0 \
  --resume checkpoints/20260609_222021_matbench_mp_gap_comp_elprops_cewald_nn_cnn_dropcoords_cewalds_cnns/matbench_mp_gap_train_best_val.pth \
  --warm-restart --lr 2e-6
```

Training result:

- Best raw internal-val MAE: `0.19300220214263744`
- Best raw epoch: `8`
- Final epoch raw internal-val MAE: `0.19340285103839483`
- Best checkpoint: `matbench_mp_gap_train_best_val.pth`

Train-only calibration result:

- `prediction_scale`: `0.9768965517241379`
- `prediction_bias`: `0.00047254222091929664`
- `prediction_min_value`: `0.0`
- `prediction_zero_threshold`: `0.094`
- Train raw MAE: `0.06498312749288561`
- Train calibrated MAE: `0.053346660631316344`
- Internal-val raw MAE: `0.19301664289693302`
- Internal-val calibrated MAE: `0.18749063962038412`

The warm restart improves the statistics-feature run, but still does not beat
the current base-feature low-LR warm-restart best.

Postprocessing check:

- Averaged checkpoint: `avg_e10_e15.pth`
- Scalar calibrated internal-val MAE: `0.18731926731852908`
- Binned-residual calibrated internal-val MAE: `0.18726354259815234`

Conclusion: same-trajectory checkpoint averaging plus the later residual
calibration still leaves the statistics-feature run well behind the base-feature
candidate. The added feature tokens appear to hurt this fold after the stronger
schedule/calibration stack is applied.

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

## Shared JSON Structural Biases

Commit:

`642fb4e feat: add shared json structural biases`

Run directory:

`checkpoints/20260610_002848_matbench_mp_gap_comp_cewald_cnn_dropcoords`

Command:

```bash
env PYTHONUNBUFFERED=1 MALLOC_ARENA_MAX=2 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 pixi run python train.py \
  --dataset matbench_mp_gap --model-size large \
  --add-composition --add-comp-ewald --add-comp-nn --drop-coords \
  --matbench-split official --matbench-fold 0 --matbench-val-ratio 0.1 \
  --max-epochs 50 --patience 50 --max-cpu-workers 4 \
  --checkpoint-interval 10 --log-every 250 --eval-train-every 0 \
  --structural-bias-lr-mult 20 --loss-compression-scale 5.0 \
  --bias-same-parent --bias-shared-group-depth
```

This adds two general JSON structural attention-bias signals:

- `same_parent`: sibling leaves under the same JSON object/list instance.
- `shared_group_depth`: number of shared array-instance ancestors.

Result:

- Best raw internal-val MAE: `0.19670155086126112`
- Best raw epoch: `43`
- Final epoch raw internal-val MAE: `0.1973`
- Best checkpoint: `matbench_mp_gap_train_best_val.pth`

Train-only calibration result:

- `prediction_scale`: `0.9808620689655173`
- `prediction_bias`: `-0.0033167513069728843`
- `prediction_min_value`: `0.0`
- `prediction_zero_threshold`: `0.11`
- Train raw MAE: `0.07995345388770221`
- Train calibrated MAE: `0.06997571399168996`
- Internal-val raw MAE: `0.196717696954291`
- Internal-val calibrated MAE: `0.19023055898414332`

Conclusion: this general bias pair is close to the 50-epoch base-feature run,
but does not beat the current clean best (`0.18526953161707985` calibrated
internal-val MAE). The next useful ablation is to enable only one of the two new
signals at a time, starting with `same_parent`, because the paired signal may
add redundant or conflicting attention priors.

## Same-Parent Structural Bias Ablation

Run directory:

`checkpoints/20260610_010626_matbench_mp_gap_comp_cewald_cnn_dropcoords`

Command:

```bash
env PYTHONUNBUFFERED=1 MALLOC_ARENA_MAX=2 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 pixi run python train.py \
  --dataset matbench_mp_gap --model-size large \
  --add-composition --add-comp-ewald --add-comp-nn --drop-coords \
  --matbench-split official --matbench-fold 0 --matbench-val-ratio 0.1 \
  --max-epochs 50 --patience 50 --max-cpu-workers 4 \
  --checkpoint-interval 10 --log-every 250 --eval-train-every 0 \
  --structural-bias-lr-mult 20 --loss-compression-scale 5.0 \
  --bias-same-parent
```

Result:

- Best raw internal-val MAE: `0.19516862255564735`
- Best raw epoch: `50`
- Best checkpoint: `matbench_mp_gap_train_best_val.pth`

Train-only calibration result:

- `prediction_scale`: `0.9749137931034483`
- `prediction_bias`: `3.7703006769177214e-05`
- `prediction_min_value`: `0.0`
- `prediction_zero_threshold`: `0.116`
- Train raw MAE: `0.07989941724063926`
- Train calibrated MAE: `0.0669026149064464`
- Internal-val raw MAE: `0.19518531278645382`
- Internal-val calibrated MAE: `0.18822475042954695`

Conclusion: `same_parent` alone is a small positive result over the base
50-epoch schedule (`0.18897` calibrated internal-val MAE), and is cleaner than
the paired shared-group-depth run. It still does not beat the current best, so
the next test is a controlled low-LR warm restart from this checkpoint.

## Same-Parent Low-LR Warm Restart

Run directory:

`checkpoints/20260610_014156_matbench_mp_gap_comp_cewald_cnn_dropcoords_resumed`

Warm-restart command:

```bash
env PYTHONUNBUFFERED=1 MALLOC_ARENA_MAX=2 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 pixi run python train.py \
  --dataset matbench_mp_gap --model-size large \
  --add-composition --add-comp-ewald --add-comp-nn --drop-coords \
  --matbench-split official --matbench-fold 0 --matbench-val-ratio 0.1 \
  --max-epochs 15 --patience 15 --max-cpu-workers 4 \
  --checkpoint-interval 5 --log-every 250 --eval-train-every 0 \
  --structural-bias-lr-mult 20 --loss-compression-scale 5.0 \
  --bias-same-parent \
  --resume checkpoints/20260610_010626_matbench_mp_gap_comp_cewald_cnn_dropcoords/matbench_mp_gap_train_best_val.pth \
  --warm-restart --lr 2e-6
```

Result:

- Best raw internal-val MAE: `0.19431427347915828`
- Best warm-restart epoch: `11`
- Final epoch raw internal-val MAE: `0.19446154055347664`
- Best checkpoint: `matbench_mp_gap_train_best_val.pth`

Train-only calibration result:

- `prediction_scale`: `0.9768965517241379`
- `prediction_bias`: `3.06204243980605e-05`
- `prediction_min_value`: `0.0`
- `prediction_zero_threshold`: `0.108`
- Train raw MAE: `0.07687364391452096`
- Train calibrated MAE: `0.06463537889658519`
- Internal-val raw MAE: `0.1942986494543743`
- Internal-val calibrated MAE: `0.1878585800739121`

Conclusion: the low-LR warm restart improves the same-parent ablation over its
50-epoch checkpoint, but it still does not beat the current best base-feature
low-LR warm restart (`0.18526953161707985` calibrated internal-val MAE).

## Shared-Group-Depth Structural Bias Ablation

Run directory:

`checkpoints/20260610_015714_matbench_mp_gap_comp_cewald_cnn_dropcoords`

Command:

```bash
env PYTHONUNBUFFERED=1 MALLOC_ARENA_MAX=2 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 pixi run python train.py \
  --dataset matbench_mp_gap --model-size large \
  --add-composition --add-comp-ewald --add-comp-nn --drop-coords \
  --matbench-split official --matbench-fold 0 --matbench-val-ratio 0.1 \
  --max-epochs 50 --patience 50 --max-cpu-workers 4 \
  --checkpoint-interval 10 --log-every 250 --eval-train-every 0 \
  --structural-bias-lr-mult 20 --loss-compression-scale 5.0 \
  --bias-shared-group-depth
```

Result:

- Best raw internal-val MAE: `0.19571158876236153`
- Best raw epoch: `47`
- Final epoch raw internal-val MAE: `0.1959`
- Best checkpoint: `matbench_mp_gap_train_best_val.pth`

Train-only calibration result:

- `prediction_scale`: `0.9768965517241379`
- `prediction_bias`: `-0.0014718137870960194`
- `prediction_min_value`: `0.0`
- `prediction_zero_threshold`: `0.116`
- Train raw MAE: `0.07928337416675008`
- Train calibrated MAE: `0.06691746980746403`
- Internal-val raw MAE: `0.19571158876236153`
- Internal-val calibrated MAE: `0.18878109177362704`

Conclusion: `shared_group_depth` alone is roughly comparable to the 50-epoch
base-feature run, but it is weaker than the `same_parent` ablation and does not
beat the current best. The paired shared-bias result is therefore not hiding a
useful `shared_group_depth` effect; the next structural-bias work should keep
`same_parent` as the only promising added signal.

## L1 Numeric Loss Ablation

Commit:

`240fc1e feat: add numeric loss option`

Run directory:

`checkpoints/20260610_023615_matbench_mp_gap_comp_cewald_cnn_dropcoords`

Command:

```bash
env PYTHONUNBUFFERED=1 MALLOC_ARENA_MAX=2 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 pixi run python train.py \
  --dataset matbench_mp_gap --model-size large \
  --add-composition --add-comp-ewald --add-comp-nn --drop-coords \
  --matbench-split official --matbench-fold 0 --matbench-val-ratio 0.1 \
  --max-epochs 50 --patience 50 --max-cpu-workers 4 \
  --checkpoint-interval 10 --log-every 250 --eval-train-every 0 \
  --structural-bias-lr-mult 20 --loss-compression-scale 5.0 \
  --numeric-loss l1
```

Result:

- Best raw internal-val MAE: `0.20050058203986626`
- Best raw epoch: `50`
- Best checkpoint: `matbench_mp_gap_train_best_val.pth`

Train-only calibration result:

- `prediction_scale`: `0.9749137931034483`
- `prediction_bias`: `0.0030287794584151484`
- `prediction_min_value`: `0.0`
- `prediction_zero_threshold`: `0.010468318050149198`
- Train raw MAE: `0.10989800494202821`
- Train calibrated MAE: `0.10153049916868487`
- Internal-val raw MAE: `0.20050122286720512`
- Internal-val calibrated MAE: `0.1956320670321587`
- Internal-val calibrated zero fraction: `0.46306985510660853`

Conclusion: replacing the compressed-space Huber loss with pure L1 is a clear
negative result. It makes near-zero predictions very aggressive, but weakens
the nonzero band-gap regression enough that calibration cannot recover the
baseline. The next numeric-loss test should keep Huber smoothing and tune its
delta/transition point rather than moving all the way to L1.

## Huber Delta 0.5 Ablation

Commit:

`5be520f feat: make numeric huber delta configurable`

Run directory:

`checkpoints/20260610_033844_matbench_mp_gap_comp_cewald_cnn_dropcoords`

Command:

```bash
env PYTHONUNBUFFERED=1 MALLOC_ARENA_MAX=2 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 pixi run python train.py \
  --dataset matbench_mp_gap --model-size large \
  --add-composition --add-comp-ewald --add-comp-nn --drop-coords \
  --matbench-split official --matbench-fold 0 --matbench-val-ratio 0.1 \
  --max-epochs 50 --patience 50 --max-cpu-workers 4 \
  --checkpoint-interval 10 --log-every 250 --eval-train-every 0 \
  --structural-bias-lr-mult 20 --loss-compression-scale 5.0 \
  --numeric-huber-delta 0.5
```

Result:

- Best raw internal-val MAE: `0.19819920464534632`
- Best raw epoch: `48`
- Best checkpoint: `matbench_mp_gap_train_best_val.pth`

Train-only calibration result:

- `prediction_scale`: `0.9788793103448277`
- `prediction_bias`: `0.00011889314768707444`
- `prediction_min_value`: `0.0`
- `prediction_zero_threshold`: `0.11196256685849482`
- Train raw MAE: `0.08586654288756558`
- Train calibrated MAE: `0.07603812450730713`
- Internal-val raw MAE: `0.1982043898309007`
- Internal-val calibrated MAE: `0.19258874847583277`
- Internal-val calibrated zero fraction: `0.4683708328425021`

Conclusion: decreasing the compressed-space Huber transition point is also
negative. Together with the L1 ablation, this suggests that making the numeric
loss more absolute-error-like over a wider region hurts nonzero band-gap
regression more than it helps zero-gap cases. The next numeric-loss ablation
should test a larger Huber delta, keeping more errors in the smooth quadratic
region.

## Huber Delta 2.0 Ablation

Run directory:

`checkpoints/20260610_041643_matbench_mp_gap_comp_cewald_cnn_dropcoords`

Command:

```bash
env PYTHONUNBUFFERED=1 MALLOC_ARENA_MAX=2 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 pixi run python train.py \
  --dataset matbench_mp_gap --model-size large \
  --add-composition --add-comp-ewald --add-comp-nn --drop-coords \
  --matbench-split official --matbench-fold 0 --matbench-val-ratio 0.1 \
  --max-epochs 50 --patience 50 --max-cpu-workers 4 \
  --checkpoint-interval 10 --log-every 250 --eval-train-every 0 \
  --structural-bias-lr-mult 20 --loss-compression-scale 5.0 \
  --numeric-huber-delta 2.0
```

Result:

- Best raw internal-val MAE: `0.19895589746380493`
- Best raw epoch: `46`
- Best checkpoint: `matbench_mp_gap_train_best_val.pth`

Train-only calibration result:

- `prediction_scale`: `0.9788793103448277`
- `prediction_bias`: `0.0010011857817077946`
- `prediction_min_value`: `0.0`
- `prediction_zero_threshold`: `0.10427563471061276`
- Train raw MAE: `0.08620126506470294`
- Train calibrated MAE: `0.07561406811105438`
- Internal-val raw MAE: `0.19897113366743738`
- Internal-val calibrated MAE: `0.19269166288450437`
- Internal-val calibrated zero fraction: `0.4558840852868418`

Conclusion: increasing the Huber transition point is also negative. The
0.5/1.0/2.0 sweep suggests that the current compressed-space Huber shape is not
the limiting factor for this fold; schedule and output-bias calibration remain
more promising than further Huber-delta tuning.

## Isotonic Calibration Ablation

Run directory:

`checkpoints/20260609_213826_matbench_mp_gap_comp_cewald_cnn_dropcoords_resumed`

Command:

```bash
env PYTHONUNBUFFERED=1 MALLOC_ARENA_MAX=2 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 pixi run python scripts/calibrate_matbench_gap.py \
  --checkpoint checkpoints/20260609_213826_matbench_mp_gap_comp_cewald_cnn_dropcoords_resumed/matbench_mp_gap_train_best_val.pth \
  --dataset matbench_mp_gap --model-size large \
  --add-composition --add-comp-ewald --add-comp-nn --drop-coords \
  --matbench-fold 0 --matbench-val-ratio 0.1 \
  --batch-size 256 --max-cpu-workers 4 --loss-compression-scale 5.0 \
  --also-fit-isotonic --isotonic-input scale_bias \
  --output checkpoints/20260609_213826_matbench_mp_gap_comp_cewald_cnn_dropcoords_resumed/best_val_isotonic_scale_bias_calibration.json
```

Result:

- Scalar calibrated internal-val MAE: `0.1852694723865815`
- Isotonic calibrated internal-val MAE: `0.18821647991570978`
- Scalar calibrated train MAE: `0.05116647935437652`
- Isotonic calibrated train MAE: `0.05278508173376005`

Conclusion: a more flexible train-only monotonic mapping does not improve this
checkpoint, even before worrying about possible overfit. The simple global
scale/bias/nonnegative/zero-threshold calibration is still the strongest
no-leakage output-bias correction tested so far.

## Low-LR Checkpoint Averaging Ablation

Commit:

`9cd3961 feat: add checkpoint averaging utility`

Run directory:

`checkpoints/20260609_213826_matbench_mp_gap_comp_cewald_cnn_dropcoords_resumed`

Checkpoint generation commands:

```bash
env PYTHONUNBUFFERED=1 MALLOC_ARENA_MAX=2 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 pixi run python scripts/average_checkpoints.py \
  --output checkpoints/20260609_213826_matbench_mp_gap_comp_cewald_cnn_dropcoords_resumed/avg_best_e10.pth \
  checkpoints/20260609_213826_matbench_mp_gap_comp_cewald_cnn_dropcoords_resumed/matbench_mp_gap_train_best_val.pth \
  checkpoints/20260609_213826_matbench_mp_gap_comp_cewald_cnn_dropcoords_resumed/matbench_mp_gap_train_e10.pth

env PYTHONUNBUFFERED=1 MALLOC_ARENA_MAX=2 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 pixi run python scripts/average_checkpoints.py \
  --output checkpoints/20260609_213826_matbench_mp_gap_comp_cewald_cnn_dropcoords_resumed/avg_e10_e15.pth \
  checkpoints/20260609_213826_matbench_mp_gap_comp_cewald_cnn_dropcoords_resumed/matbench_mp_gap_train_e10.pth \
  checkpoints/20260609_213826_matbench_mp_gap_comp_cewald_cnn_dropcoords_resumed/matbench_mp_gap_train_e15.pth

env PYTHONUNBUFFERED=1 MALLOC_ARENA_MAX=2 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 pixi run python scripts/average_checkpoints.py \
  --output checkpoints/20260609_213826_matbench_mp_gap_comp_cewald_cnn_dropcoords_resumed/avg_best_e10_e15.pth \
  checkpoints/20260609_213826_matbench_mp_gap_comp_cewald_cnn_dropcoords_resumed/matbench_mp_gap_train_best_val.pth \
  checkpoints/20260609_213826_matbench_mp_gap_comp_cewald_cnn_dropcoords_resumed/matbench_mp_gap_train_e10.pth \
  checkpoints/20260609_213826_matbench_mp_gap_comp_cewald_cnn_dropcoords_resumed/matbench_mp_gap_train_e15.pth
```

Result:

| Averaged checkpoint | Raw internal-val MAE | Calibrated internal-val MAE |
| --- | ---: | ---: |
| `avg_best_e10.pth` | `0.19121448945627767` | `0.18521468148499592` |
| `avg_e10_e15.pth` | `0.1915938070899512` | `0.18508102969846527` |
| `avg_best_e10_e15.pth` | `0.19133006325526028` | `0.1851493717075686` |

Early-checkpoint averaging check with the later `6 bins, shrinkage 5000`
residual calibration:

| Averaged checkpoint | Scalar val MAE | Residual val MAE |
| --- | ---: | ---: |
| `avg_e5_e10.pth` | `0.18533222822378245` | `0.1852857875033103` |
| `avg_e5_e10_e15.pth` | `0.18520787081540352` | `0.1851468613929959` |

Prediction-level ensemble check with E10 and E15 checkpoints:

- Scalar calibrated internal-val MAE: `0.18508245778908303`
- Binned-residual calibrated internal-val MAE: `0.18500007567257207`

Conclusion: same-trajectory checkpoint averaging gives a small but clean
improvement over the previous best single checkpoint. The best candidate is the
later-tail average `avg_e10_e15.pth`, suggesting that the low-LR trajectory
contains a slightly better flat-region solution than the raw best-val epoch
alone. Mixing in the earlier E5 checkpoint degrades both scalar and residual
calibrated results. Prediction-level ensembling of E10/E15 is close, but it is
slightly weaker than weight averaging after residual calibration.

## Second Low-LR Warm Restart From Averaged Checkpoint

Run directory:

`checkpoints/20260610_130805_matbench_mp_gap_comp_cewald_cnn_dropcoords_resumed`

Warm-restart command:

```bash
env PYTHONUNBUFFERED=1 MALLOC_ARENA_MAX=2 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 pixi run python train.py \
  --dataset matbench_mp_gap --model-size large \
  --add-composition --add-comp-ewald --add-comp-nn --drop-coords \
  --matbench-split official --matbench-fold 0 --matbench-val-ratio 0.1 \
  --max-epochs 12 --patience 12 --max-cpu-workers 4 \
  --checkpoint-interval 4 --log-every 250 --eval-train-every 0 \
  --structural-bias-lr-mult 20 --loss-compression-scale 5.0 \
  --resume checkpoints/20260609_213826_matbench_mp_gap_comp_cewald_cnn_dropcoords_resumed/avg_e10_e15.pth \
  --warm-restart --lr 1e-6
```

Training result:

- Best raw internal-val MAE: `0.1913256779724466`
- Best raw epoch: `7`
- Final epoch raw internal-val MAE: `0.19169478168120715`
- GPU peak during training: about `2.0G`

Train-only calibration result:

- Best-val checkpoint calibrated internal-val MAE: `0.18530835251332392`
- Best-val checkpoint calibrated train MAE: `0.051135470277544795`
- Average of previous best and new best calibrated internal-val MAE:
  `0.18519139201333076`
- Average of previous best and new best calibrated train MAE:
  `0.05101651318911314`

Conclusion: the second lower-LR warm restart did not improve on the
same-trajectory checkpoint average (`0.18508102969846527`). It slightly improved
raw MAE relative to the averaged checkpoint but moved the train-only calibration
in the wrong direction, so this path is not the next best use of compute.

## Very-Low-LR Warm Restart From Averaged Checkpoint

Run directory:

`checkpoints/20260610_135904_matbench_mp_gap_comp_cewald_cnn_dropcoords_resumed`

Warm-restart command:

```bash
env PYTHONUNBUFFERED=1 MALLOC_ARENA_MAX=2 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 pixi run python train.py \
  --dataset matbench_mp_gap --model-size large \
  --add-composition --add-comp-ewald --add-comp-nn --drop-coords \
  --matbench-split official --matbench-fold 0 --matbench-val-ratio 0.1 \
  --max-epochs 8 --patience 8 --max-cpu-workers 4 \
  --checkpoint-interval 4 --log-every 250 --eval-train-every 0 \
  --structural-bias-lr-mult 20 --loss-compression-scale 5.0 \
  --resume checkpoints/20260609_213826_matbench_mp_gap_comp_cewald_cnn_dropcoords_resumed/avg_e10_e15.pth \
  --warm-restart --lr 5e-7
```

Training result:

- Best raw internal-val MAE: `0.19164062293901407`
- Best raw epoch: `5`
- Final epoch raw internal-val MAE: `0.1917677242029105`
- GPU peak during training: about `2.0G`

Train-only residual calibration result:

| Candidate | Scalar val MAE | Residual val MAE | Residual train MAE |
| --- | ---: | ---: | ---: |
| Best-val checkpoint | `0.18518028033741638` | `0.18513878437336823` | `0.05105686816086606` |
| Average of previous best and new best | `0.18511917696447328` | `0.1850627706124678` | `0.05093547505329013` |

Conclusion: reducing the second warm-restart LR from `1e-6` to `5e-7` still did
not improve on the earlier same-trajectory average plus residual calibration
(`0.18496897995763836`). The best result from this run comes from averaging
back toward the previous best, which confirms that the extra finetuning mostly
moves away from the best calibrated region.

## Higher-LR Warm Restart From E74 Checkpoint

Run directory:

`checkpoints/20260610_142023_matbench_mp_gap_comp_cewald_cnn_dropcoords_resumed`

Warm-restart command:

```bash
env PYTHONUNBUFFERED=1 MALLOC_ARENA_MAX=2 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 pixi run python train.py \
  --dataset matbench_mp_gap --model-size large \
  --add-composition --add-comp-ewald --add-comp-nn --drop-coords \
  --matbench-split official --matbench-fold 0 --matbench-val-ratio 0.1 \
  --max-epochs 12 --patience 12 --max-cpu-workers 4 \
  --checkpoint-interval 4 --log-every 250 --eval-train-every 0 \
  --structural-bias-lr-mult 20 --loss-compression-scale 5.0 \
  --resume checkpoints/20260609_211539_matbench_mp_gap_comp_cewald_cnn_dropcoords_resumed/matbench_mp_gap_train_best_val.pth \
  --warm-restart --lr 3e-6
```

This run was stopped after epoch 4 because the raw internal-val curve was
clearly worse than the established `2e-6` warm restart:

- Raw internal-val MAE by epoch: `0.19291248483379492`,
  `0.19290110317856451`, `0.19267057437232488`, `0.19259954716252897`
- Best raw internal-val MAE before stopping: `0.19259954716252897`
- Best raw epoch before stopping: `4`
- GPU peak during training: about `2.2G`

Train-only residual calibration result:

- Scalar calibrated internal-val MAE: `0.18573838961080438`
- Binned-residual calibrated internal-val MAE: `0.18565765939532783`
- Binned-residual calibrated train MAE: `0.05216159807361745`

Conclusion: `3e-6` is too aggressive for this warm-restart point. It moves the
model away from the useful low-LR basin and does not merit a full 12-epoch run.

## Binned Residual Calibration Ablation

Commit:

`20af48b feat: report binned residual calibration`

Run directory:

`checkpoints/20260609_213826_matbench_mp_gap_comp_cewald_cnn_dropcoords_resumed`

Checkpoint:

`avg_e10_e15.pth`

Calibration command pattern:

```bash
env PYTHONUNBUFFERED=1 MALLOC_ARENA_MAX=2 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 pixi run python scripts/calibrate_matbench_gap.py \
  --checkpoint checkpoints/20260609_213826_matbench_mp_gap_comp_cewald_cnn_dropcoords_resumed/avg_e10_e15.pth \
  --dataset matbench_mp_gap --model-size large \
  --add-composition --add-comp-ewald --add-comp-nn --drop-coords \
  --matbench-fold 0 --matbench-val-ratio 0.1 \
  --batch-size 256 --max-cpu-workers 4 --loss-compression-scale 5.0 \
  --also-fit-binned-residual --binned-residual-bins <bins> \
  --binned-residual-shrinkage <shrinkage>
```

Result:

| Calibration | Scalar val MAE | Residual val MAE | Residual train MAE |
| --- | ---: | ---: | ---: |
| `8 bins, shrinkage 5000` | `0.1850801124643769` | `0.18498806346515725` | `0.05086737283664464` |
| `6 bins, shrinkage 5000` | `0.18508102969846527` | `0.18496897995763836` | `0.05090135484499366` |
| `8 bins, shrinkage 10000` | `0.1850851433462705` | `0.18500740347798172` | `0.050881150634927604` |

Stability check with `6 bins, shrinkage 5000`:

| Averaged checkpoint | Scalar val MAE | Residual val MAE | Residual train MAE |
| --- | ---: | ---: | ---: |
| `avg_best_e10.pth` | `0.1852221302268657` | `0.18512544229429684` | `0.051024934730922344` |
| `avg_e10_e15.pth` | `0.18508102969846527` | `0.18496897995763836` | `0.05090135484499366` |
| `avg_best_e10_e15.pth` | `0.18514711496903086` | `0.18503170834575763` | `0.05091816786536162` |

Train-only CV selection check:

- Grid: `bins in {4,6,8}`, `shrinkage in {5000,10000}`
- Folds: `5`
- Selected by train-only CV: `8 bins, shrinkage 5000`
- Mean train-CV MAE: `0.05089767019235632`
- Internal-val MAE after refitting selected mapping on all train:
  `0.18498948106981508`

Fixed-edge residual check:

- Edges: `0,0.01,0.05,0.15,0.5,1,2,4,8`
- Scalar calibrated internal-val MAE: `0.1850822293877643`
- Fixed-edge residual internal-val MAE: `0.18502181266380344`
- Fixed-edge residual train MAE: `0.05057749182759186`

Conclusion: a small, shrinkage-regularized residual correction improves the
averaged checkpoint without using held-out test labels. The best single-fold
candidate uses 6 quantile bins and shrinkage 5000; the nearby settings also
improve over scalar calibration, but by a smaller amount. Applying the same
residual calibration to neighboring averaged checkpoints also improves them,
while preserving `avg_e10_e15.pth` as the best candidate. The train-only CV
selection chooses the neighboring `8 bins, shrinkage 5000` setting and lands at
`0.18498948106981508`, which is slightly weaker than the single-fold best but
supports the residual-calibration effect without using val to choose the
hyperparameters. Fixed physical-looking edges lower train MAE more aggressively
but generalize worse than quantile bins on internal validation.

## Weight Decay 0.02 Ablation

Commit:

`54247bd feat: add weight decay override`

Run directory:

`checkpoints/20260610_143043_matbench_mp_gap_comp_cewald_cnn_dropcoords`

Training command:

```bash
env PYTHONUNBUFFERED=1 MALLOC_ARENA_MAX=2 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 pixi run python train.py \
  --dataset matbench_mp_gap --model-size large \
  --add-composition --add-comp-ewald --add-comp-nn --drop-coords \
  --matbench-split official --matbench-fold 0 --matbench-val-ratio 0.1 \
  --max-epochs 50 --patience 50 --max-cpu-workers 4 \
  --checkpoint-interval 10 --log-every 250 --eval-train-every 0 \
  --structural-bias-lr-mult 20 --loss-compression-scale 5.0 \
  --weight-decay 0.02
```

Result:

- Best raw internal-val MAE: `0.2019501742731977` at epoch 48
- Train-only scalar calibrated internal-val MAE: `0.19444512656307547`
- Train-only binned-residual calibrated internal-val MAE:
  `0.19441525051407796`

Conclusion: increasing AdamW weight decay from `0.01` to `0.02` is a clear
negative result on the single official fold0 internal validation split. It
lags the base 50-epoch run both before calibration (`0.20195` vs `0.19584`)
and after the same train-only calibration (`0.19442` vs the current
`0.18497` best candidate). This branch should not receive a warm restart; the
next training-regularization test should instead vary dropout while keeping
the base weight decay.

## Dropout 0.05 Ablation

Commit:

`3470afc feat: add dropout override`

Run directory:

`checkpoints/20260610_150732_matbench_mp_gap_comp_cewald_cnn_dropcoords`

Training command:

```bash
env PYTHONUNBUFFERED=1 MALLOC_ARENA_MAX=2 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 pixi run python train.py \
  --dataset matbench_mp_gap --model-size large \
  --add-composition --add-comp-ewald --add-comp-nn --drop-coords \
  --matbench-split official --matbench-fold 0 --matbench-val-ratio 0.1 \
  --max-epochs 50 --patience 50 --max-cpu-workers 4 \
  --checkpoint-interval 10 --log-every 250 --eval-train-every 0 \
  --structural-bias-lr-mult 20 --loss-compression-scale 5.0 \
  --dropout 0.05
```

Result:

- Best raw internal-val MAE: `0.19299828914686676` at epoch 46
- Train-only scalar calibrated internal-val MAE: `0.18872184477529746`
- Train-only binned-residual calibrated internal-val MAE:
  `0.18872986891595864`

Conclusion: reducing transformer dropout from `0.10` to `0.05` is a genuine
structural training improvement on this single official fold0 internal
validation split. It improves the 50-epoch raw best from `0.19584` to
`0.19300`, and slightly improves the matched train-only scalar calibration
from `0.18897` to `0.18872`. It does not beat the longer-schedule/warm-restart
candidate (`0.18497`), and the binned-residual correction does not help this
checkpoint. The next low-cost follow-up should warm-restart from the epoch 46
best checkpoint with the same `dropout=0.05` setting.

## Dropout 0.05 Low-LR Warm Restart

Run directory:

`checkpoints/20260610_154412_matbench_mp_gap_comp_cewald_cnn_dropcoords_resumed`

Source checkpoint:

`checkpoints/20260610_150732_matbench_mp_gap_comp_cewald_cnn_dropcoords/matbench_mp_gap_train_best_val.pth`

Training command:

```bash
env PYTHONUNBUFFERED=1 MALLOC_ARENA_MAX=2 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 pixi run python train.py \
  --resume checkpoints/20260610_150732_matbench_mp_gap_comp_cewald_cnn_dropcoords/matbench_mp_gap_train_best_val.pth \
  --warm-restart \
  --dataset matbench_mp_gap --model-size large \
  --add-composition --add-comp-ewald --add-comp-nn --drop-coords \
  --matbench-split official --matbench-fold 0 --matbench-val-ratio 0.1 \
  --max-epochs 20 --patience 20 --max-cpu-workers 4 \
  --checkpoint-interval 5 --log-every 250 --eval-train-every 0 \
  --structural-bias-lr-mult 20 --loss-compression-scale 5.0 \
  --dropout 0.05 --lr 2e-6
```

Result:

- Best raw internal-val MAE: `0.19217697372522413` at restart epoch 10
- Train-only scalar calibrated internal-val MAE: `0.1880308904109926`
- Train-only binned-residual calibrated internal-val MAE:
  `0.18799017998228945`

Conclusion: the low-LR warm restart preserves the dropout improvement and
continues the same structural training direction, improving the 50-epoch
dropout run from `0.19300` raw / `0.18872` scalar to `0.19218` raw /
`0.18799` with train-only binned-residual calibration. It still does not beat
the current best `0.18497` candidate from the longer baseline schedule, but it
confirms that reduced dropout is useful and worth combining with a longer
training schedule.

## Dropout 0.05 80-Epoch Long Schedule

Run directory:

`checkpoints/20260610_204411_matbench_mp_gap_comp_cewald_cnn_dropcoords`

Training command:

```bash
env PYTHONUNBUFFERED=1 MALLOC_ARENA_MAX=2 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 pixi run python train.py \
  --dataset matbench_mp_gap --model-size large \
  --add-composition --add-comp-ewald --add-comp-nn --drop-coords \
  --matbench-split official --matbench-fold 0 --matbench-val-ratio 0.1 \
  --max-epochs 80 --patience 80 --max-cpu-workers 4 \
  --checkpoint-interval 10 --log-every 250 --eval-train-every 0 \
  --structural-bias-lr-mult 20 --loss-compression-scale 5.0 \
  --dropout 0.05
```

Training result:

- Best raw internal-val MAE: `0.18445097342720354`
- Best raw epoch: `74`
- Final epoch raw internal-val MAE: `0.18463109290718835`
- Best checkpoint: `matbench_mp_gap_train_best_val.pth`

Train-only calibration result:

- `prediction_scale`: `0.9868103448275862`
- `prediction_bias`: `0.0002859855594980563`
- `prediction_min_value`: `0.0`
- `prediction_zero_threshold`: `0.056`
- Train raw MAE: `0.03619367770028811`
- Train calibrated MAE: `0.028306797247887622`
- Internal-val raw MAE: `0.18447844981848474`
- Internal-val calibrated MAE: `0.18119688913386592`
- Binned-residual calibrated train MAE: `0.028336464163975245`
- Binned-residual calibrated internal-val MAE: `0.18117659119360632`

Conclusion: combining the lower dropout (`0.05`) with the full 80-epoch cosine
schedule is the first clear structural improvement past the previous
checkpoint-averaged `0.18497` result. The raw single-checkpoint validation MAE
already reaches `0.18445`, and train-only scalar calibration brings the matched
single-fold internal validation result to `0.18120`. The binned residual
correction is negligible here (`0.18120` → `0.18118`), so the main gain is from
training dynamics rather than post-processing.

## Dropout 0.05 80-Epoch Low-LR Warm Restart

Run directory:

`checkpoints/20260610_213910_matbench_mp_gap_comp_cewald_cnn_dropcoords_resumed`

Source checkpoint:

`checkpoints/20260610_204411_matbench_mp_gap_comp_cewald_cnn_dropcoords/matbench_mp_gap_train_best_val.pth`

Training command:

```bash
env PYTHONUNBUFFERED=1 MALLOC_ARENA_MAX=2 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 pixi run python train.py \
  --resume checkpoints/20260610_204411_matbench_mp_gap_comp_cewald_cnn_dropcoords/matbench_mp_gap_train_best_val.pth \
  --warm-restart \
  --dataset matbench_mp_gap --model-size large \
  --add-composition --add-comp-ewald --add-comp-nn --drop-coords \
  --matbench-split official --matbench-fold 0 --matbench-val-ratio 0.1 \
  --max-epochs 20 --patience 20 --max-cpu-workers 4 \
  --checkpoint-interval 5 --log-every 250 --eval-train-every 0 \
  --structural-bias-lr-mult 20 --loss-compression-scale 5.0 \
  --dropout 0.05 --lr 2e-6
```

Training result:

- Best raw internal-val MAE: `0.18415902546685092`
- Best raw epoch: `6`
- Final epoch raw internal-val MAE: `0.18426521545898678`
- Best checkpoint: `matbench_mp_gap_train_best_val.pth`

Train-only calibration result:

- `prediction_scale`: `0.988793103448276`
- `prediction_bias`: `-0.0011113769174071737`
- `prediction_min_value`: `0.0`
- `prediction_zero_threshold`: `0.052000000000000005`
- Train raw MAE: `0.034357663771868004`
- Train calibrated MAE: `0.027655111099689`
- Internal-val raw MAE: `0.1841691656442252`
- Internal-val calibrated MAE: `0.18122382824986974`
- Binned-residual calibrated train MAE: `0.027655089714201695`
- Binned-residual calibrated internal-val MAE: `0.1812252191894463`

Conclusion: the low-LR warm restart slightly improves the raw validation MAE
from `0.18445` to `0.18416`, but it does not improve the calibrated score
(`0.18122` vs the current `0.18118`). This is a useful sanity check that the
long-schedule checkpoint is already near its calibrated optimum; it should not
replace the current best.

## Same-Path-Template Structural Bias

Run directory:

`checkpoints/20260610_220232_matbench_mp_gap_comp_cewald_cnn_dropcoords`

Training command:

```bash
env PYTHONUNBUFFERED=1 MALLOC_ARENA_MAX=2 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 pixi run python train.py \
  --dataset matbench_mp_gap --model-size large \
  --add-composition --add-comp-ewald --add-comp-nn --drop-coords \
  --matbench-split official --matbench-fold 0 --matbench-val-ratio 0.1 \
  --max-epochs 50 --patience 50 --max-cpu-workers 4 \
  --checkpoint-interval 10 --log-every 250 --eval-train-every 0 \
  --structural-bias-lr-mult 20 --loss-compression-scale 5.0 \
  --dropout 0.05 --bias-same-path-template
```

Training result:

- Best raw internal-val MAE: `0.19534491940074195`
- Best raw epoch: `49`
- Final epoch raw internal-val MAE: `0.19534491940074195`
- Best checkpoint: `matbench_mp_gap_train_best_val.pth`

Train-only calibration result:

- `prediction_scale`: `0.988793103448276`
- `prediction_bias`: `-0.0015172184533279005`
- `prediction_min_value`: `0.0`
- `prediction_zero_threshold`: `0.098`
- Train raw MAE: `0.05339481046868737`
- Train calibrated MAE: `0.045237716025358476`
- Internal-val raw MAE: `0.19533241128744447`
- Internal-val calibrated MAE: `0.1905814128853145`
- Binned-residual calibrated train MAE: `0.0452635286399252`
- Binned-residual calibrated internal-val MAE: `0.19061356858935802`

Conclusion: `same_path_template` is a clean JSON-structural signal and is
slightly stronger than the earlier `same_parent` structural-bias ablation, but
it does not beat the matched 50-epoch dropout baseline (`0.192998` raw /
`0.188722` calibrated) and is far behind the current 80-epoch dropout best
(`0.181177` calibrated). The next bias experiment should keep this separated
from other new signals and test `value_type_pair` alone, because it addresses a
different generic relation: the directed attention role between numeric,
string, boolean, and mask tokens.

## Value-Type-Pair Structural Bias

Run directory:

`checkpoints/20260610_224239_matbench_mp_gap_comp_cewald_cnn_dropcoords`

Training command:

```bash
env PYTHONUNBUFFERED=1 MALLOC_ARENA_MAX=2 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 pixi run python train.py \
  --dataset matbench_mp_gap --model-size large \
  --add-composition --add-comp-ewald --add-comp-nn --drop-coords \
  --matbench-split official --matbench-fold 0 --matbench-val-ratio 0.1 \
  --max-epochs 50 --patience 50 --max-cpu-workers 4 \
  --checkpoint-interval 10 --log-every 250 --eval-train-every 0 \
  --structural-bias-lr-mult 20 --loss-compression-scale 5.0 \
  --dropout 0.05 --bias-value-type-pair
```

Training result:

- Best raw internal-val MAE: `0.19714943298914378`
- Best raw epoch: `48`
- Final epoch raw internal-val MAE: `0.19714943298914378`
- Best checkpoint: `matbench_mp_gap_train_best_val.pth`

Train-only calibration result:

- `prediction_scale`: `0.988793103448276`
- `prediction_bias`: `-0.001480775336927638`
- `prediction_min_value`: `0.0`
- `prediction_zero_threshold`: `0.098`
- Train raw MAE: `0.05530583274488042`
- Train calibrated MAE: `0.0475221594922396`
- Internal-val raw MAE: `0.19715597465907725`
- Internal-val calibrated MAE: `0.1925050967768701`
- Binned-residual calibrated train MAE: `0.0475399576442407`
- Binned-residual calibrated internal-val MAE: `0.19257943952733467`

Conclusion: `value_type_pair` is a clean, directed JSON-token relation, but by
itself it is weaker than `same_path_template` and clearly worse than the
matched 50-epoch dropout baseline. The current structural-bias additions do not
explain the remaining gap to the 80-epoch best; the next promising generic
direction is input-side numeric representation, especially making numeric
encoding aware of the field/path context rather than adding more target-derived
or output-only corrections.

## Numeric Path FiLM Probe

Run directory:

`checkpoints/20260610_232345_matbench_mp_gap_comp_cewald_cnn_dropcoords`

Training command:

```bash
env PYTHONUNBUFFERED=1 MALLOC_ARENA_MAX=2 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 pixi run python train.py \
  --dataset matbench_mp_gap --model-size large \
  --add-composition --add-comp-ewald --add-comp-nn --drop-coords \
  --matbench-split official --matbench-fold 0 --matbench-val-ratio 0.1 \
  --max-epochs 20 --patience 20 --max-cpu-workers 4 \
  --checkpoint-interval 10 --log-every 250 --eval-train-every 0 \
  --structural-bias-lr-mult 20 --loss-compression-scale 5.0 \
  --dropout 0.05 --numeric-path-film
```

Training result:

- Best raw internal-val MAE: `0.22548361537685974`
- Best raw epoch: `20`
- Best checkpoint: `matbench_mp_gap_train_best_val.pth`
- Short-run curve comparison:
  - Matched dropout baseline E10/E15/E20: `0.29045779890786744` /
    `0.2529337770222255` / `0.23568223023731574`
  - Numeric path FiLM E10/E15/E20: `0.27216284520585693` /
    `0.23928665507798033` / `0.22548361537685974`

Train-only calibration result:

- `prediction_scale`: `0.9947413793103449`
- `prediction_bias`: `-0.003990163869658035`
- `prediction_min_value`: `0.0`
- `prediction_zero_threshold`: `0.23600000000000002`
- Train raw MAE: `0.14209556640475754`
- Train calibrated MAE: `0.13341964200018178`
- Internal-val raw MAE: `0.22549287994336392`
- Internal-val calibrated MAE: `0.21875686652256626`
- Binned-residual calibrated train MAE: `0.1333308648807274`
- Binned-residual calibrated internal-val MAE: `0.21860361977455064`

Conclusion: `numeric_path_film` is a clean input-side numeric representation
change, not a target-derived prior or output-only correction. It is initialized
to preserve the old numeric encoder at step 0, then learns field-conditioned
scale/shift terms from the JSON path embedding for number tokens. The 20-epoch
probe is ahead of the matched dropout baseline at the same epoch, but it is not
yet evidence of a final improvement over the current 80-epoch best
(`0.181177` binned calibrated). This direction is worth a longer controlled run
or a smaller beta-only variant; it should stay separate from the rejected
target-prior direction.

## Numeric Path Beta Probe

Run directory:

`checkpoints/20260611_002050_matbench_mp_gap_comp_cewald_cnn_dropcoords`

Training command:

```bash
env PYTHONUNBUFFERED=1 MALLOC_ARENA_MAX=2 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 pixi run python train.py \
  --dataset matbench_mp_gap --model-size large \
  --add-composition --add-comp-ewald --add-comp-nn --drop-coords \
  --matbench-split official --matbench-fold 0 --matbench-val-ratio 0.1 \
  --max-epochs 20 --patience 20 --max-cpu-workers 4 \
  --checkpoint-interval 10 --log-every 250 --eval-train-every 0 \
  --structural-bias-lr-mult 20 --loss-compression-scale 5.0 \
  --dropout 0.05 --numeric-path-beta
```

Training result:

- Best raw internal-val MAE: `0.22815094612611625`
- Best raw epoch: `20`
- Best checkpoint: `matbench_mp_gap_train_best_val.pth`
- Short-run curve comparison:
  - Matched dropout baseline E10/E15/E20: `0.29045779890786744` /
    `0.2529337770222255` / `0.23568223023731574`
  - Numeric path FiLM E10/E15/E20: `0.27216284520585693` /
    `0.23928665507798033` / `0.22548361537685974`
  - Numeric path beta E10/E15/E20: `0.2771256336458818` /
    `0.23767161282216867` / `0.22815094612611625`

Train-only calibration result:

- `prediction_scale`: `0.9947413793103449`
- `prediction_bias`: `-0.006147264121742598`
- `prediction_min_value`: `0.0`
- `prediction_zero_threshold`: `0.23600000000000002`
- Train raw MAE: `0.14584798282419445`
- Train calibrated MAE: `0.13660233288846577`
- Internal-val raw MAE: `0.2281535316455807`
- Internal-val calibrated MAE: `0.22108429056684045`
- Binned-residual calibrated train MAE: `0.13646759015376245`
- Binned-residual calibrated internal-val MAE: `0.22094347563668637`

Conclusion: the beta-only path-conditioned numeric encoder is a positive
short-run signal versus the matched dropout baseline, though slightly weaker
than the full FiLM variant at 20 epochs. The result supports the field-aware
numeric-encoding direction while suggesting that multiplicative path scaling is
currently useful. Separately, structural-bias ablations should not be discarded
only because they fail to beat the strongest dropout run in isolation; small
positive or mechanism-distinct signals such as `same_parent` should be tested as
combinations on the current strong baseline.

## Same-Parent Bias on Dropout Baseline

Run directory:

`checkpoints/20260611_003857_matbench_mp_gap_comp_cewald_cnn_dropcoords`

Training command:

```bash
env PYTHONUNBUFFERED=1 MALLOC_ARENA_MAX=2 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 pixi run python train.py \
  --dataset matbench_mp_gap --model-size large \
  --add-composition --add-comp-ewald --add-comp-nn --drop-coords \
  --matbench-split official --matbench-fold 0 --matbench-val-ratio 0.1 \
  --max-epochs 20 --patience 20 --max-cpu-workers 4 \
  --checkpoint-interval 10 --log-every 250 --eval-train-every 0 \
  --structural-bias-lr-mult 20 --loss-compression-scale 5.0 \
  --dropout 0.05 --bias-same-parent
```

Training result:

- Best raw internal-val MAE: `0.22355039390925138`
- Best raw epoch: `20`
- Best checkpoint: `matbench_mp_gap_train_best_val.pth`
- Short-run curve comparison:
  - Matched dropout baseline E10/E15/E20: `0.29045779890786744` /
    `0.2529337770222255` / `0.23568223023731574`
  - Numeric path FiLM E10/E15/E20: `0.27216284520585693` /
    `0.23928665507798033` / `0.22548361537685974`
  - Numeric path beta E10/E15/E20: `0.2771256336458818` /
    `0.23767161282216867` / `0.22815094612611625`
  - Same-parent bias E10/E15/E20: `0.2672322339495805` /
    `0.23332767412909886` / `0.22355039390925138`

Train-only calibration result:

- `prediction_scale`: `0.9947413793103449`
- `prediction_bias`: `-0.0060340707234492336`
- `prediction_min_value`: `0.0`
- `prediction_zero_threshold`: `0.242`
- Train raw MAE: `0.13682169227984667`
- Train calibrated MAE: `0.12724007723483796`
- Internal-val raw MAE: `0.22355356949276195`
- Internal-val calibrated MAE: `0.21613417406257987`
- Binned-residual calibrated train MAE: `0.1270392292145749`
- Binned-residual calibrated internal-val MAE: `0.21581298228973428`

Conclusion: `same_parent` is a real positive structural-bias signal when tested
as a combination on the current strong dropout baseline. The earlier isolated
result did not beat the 80-epoch best, but that was too strict a filter for
combination candidates. At 20 epochs, this run beats the matched dropout
baseline and both path-conditioned numeric-encoding probes. The next controlled
combination should test whether `same_parent` composes with `numeric_path_film`.

## Same-Parent Plus Numeric Path FiLM

Run directory:

`checkpoints/20260611_005630_matbench_mp_gap_comp_cewald_cnn_dropcoords`

Training command:

```bash
env PYTHONUNBUFFERED=1 MALLOC_ARENA_MAX=2 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 pixi run python train.py \
  --dataset matbench_mp_gap --model-size large \
  --add-composition --add-comp-ewald --add-comp-nn --drop-coords \
  --matbench-split official --matbench-fold 0 --matbench-val-ratio 0.1 \
  --max-epochs 20 --patience 20 --max-cpu-workers 4 \
  --checkpoint-interval 10 --log-every 250 --eval-train-every 0 \
  --structural-bias-lr-mult 20 --loss-compression-scale 5.0 \
  --dropout 0.05 --bias-same-parent --numeric-path-film
```

Training result:

- Best raw internal-val MAE: `0.22441326740978518`
- Best raw epoch: `19`
- Final epoch raw internal-val MAE: `0.22468994033614004`
- Best checkpoint: `matbench_mp_gap_train_best_val.pth`
- Short-run curve comparison:
  - Same-parent bias E10/E15/E20: `0.2672322339495805` /
    `0.23332767412909886` / `0.22355039390925138`
  - Numeric path FiLM E10/E15/E20: `0.27216284520585693` /
    `0.23928665507798033` / `0.22548361537685974`
  - Same-parent + FiLM E10/E15/E20: `0.28283596255920135` /
    `0.23217157084155363` / `0.22468994033614004`

Train-only calibration result:

- `prediction_scale`: `0.9967241379310345`
- `prediction_bias`: `-0.004168601747248964`
- `prediction_min_value`: `0.0`
- `prediction_zero_threshold`: `0.246`
- Train raw MAE: `0.14274681628578056`
- Train calibrated MAE: `0.13411015176710628`
- Internal-val raw MAE: `0.22440170794780204`
- Internal-val calibrated MAE: `0.21810612102866203`
- Binned-residual calibrated train MAE: `0.13386405785215164`
- Binned-residual calibrated internal-val MAE: `0.21787665855156063`

Conclusion: this combination is stronger than the FiLM-only probe but weaker
than `same_parent` alone on the same 20-epoch schedule. The two mechanisms do
not show clean additive gains here; the best next use of budget is to extend the
`same_parent` dropout-baseline run rather than prioritizing a longer
`same_parent + numeric_path_film` run.

## Same-Parent Bias 80-Epoch Dropout Schedule

Run directory:

`checkpoints/20260611_012809_matbench_mp_gap_comp_cewald_cnn_dropcoords`

Training command:

```bash
env PYTHONUNBUFFERED=1 MALLOC_ARENA_MAX=2 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 pixi run python train.py \
  --dataset matbench_mp_gap --model-size large \
  --add-composition --add-comp-ewald --add-comp-nn --drop-coords \
  --matbench-split official --matbench-fold 0 --matbench-val-ratio 0.1 \
  --max-epochs 80 --patience 80 --max-cpu-workers 4 \
  --checkpoint-interval 10 --log-every 250 --eval-train-every 0 \
  --structural-bias-lr-mult 20 --loss-compression-scale 5.0 \
  --dropout 0.05 --bias-same-parent
```

Training result:

- Best raw internal-val MAE: `0.18810501787043746`
- Best raw epoch: `79`
- Final epoch raw internal-val MAE: `0.18810822931385138`
- Best checkpoint: `matbench_mp_gap_train_best_val.pth`
- Late-epoch raw internal-val MAE:
  - E60/E65/E70/E75/E80: `0.19026114966998056` /
    `0.19068618441266086` / `0.18852155044899715` /
    `0.1883318535039353` / `0.18810822931385138`

Train-only calibration result:

- `prediction_scale`: `0.9868103448275862`
- `prediction_bias`: `0.0011149373750106014`
- `prediction_min_value`: `0.0`
- `prediction_zero_threshold`: `0.054`
- Train raw MAE: `0.034445618728721594`
- Train calibrated MAE: `0.02639768811925706`
- Internal-val raw MAE: `0.18811670790730553`
- Internal-val calibrated MAE: `0.18508393961892597`
- Binned-residual calibrated train MAE: `0.026357638035022897`
- Binned-residual calibrated internal-val MAE: `0.18503255766405102`

Conclusion: `same_parent` remains a positive structural-bias signal on the full
80-epoch dropout schedule, but it does not beat the current `dropout=0.05`
80-epoch best (`0.18445097342720354` raw / `0.18117659119360632` binned
calibrated). The value of this result is directional: a categorical sibling
relation still improves the strong baseline over shorter matched schedules and
holds up through a long run, which makes discrete structural encodings worth a
controlled probe. The next experiment should test whether integer tree-depth
relations (`first_diff`, `tree_dist`, and optionally `shared_group_depth`) work
better as learnable categorical bias embeddings than as continuous sinusoidal
features.

## Discrete Depth Bias on Dropout Baseline

Run directory:

`checkpoints/20260611_022632_matbench_mp_gap_comp_cewald_cnn_dropcoords`

Training command:

```bash
env PYTHONUNBUFFERED=1 MALLOC_ARENA_MAX=2 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 pixi run python train.py \
  --dataset matbench_mp_gap --model-size large \
  --add-composition --add-comp-ewald --add-comp-nn --drop-coords \
  --matbench-split official --matbench-fold 0 --matbench-val-ratio 0.1 \
  --max-epochs 20 --patience 20 --max-cpu-workers 4 \
  --checkpoint-interval 10 --log-every 250 --eval-train-every 0 \
  --structural-bias-lr-mult 20 --loss-compression-scale 5.0 \
  --dropout 0.05 --bias-discrete-depths
```

Training result:

- Best raw internal-val MAE: `0.22810707737377245`
- Best raw epoch: `20`
- Best checkpoint: `matbench_mp_gap_train_best_val.pth`
- Short-run curve comparison:
  - Matched dropout baseline E10/E15/E20: `0.29045779890786744` /
    `0.2529337770222255` / `0.23568223023731574`
  - Numeric path beta E10/E15/E20: `0.2771256336458818` /
    `0.23767161282216867` / `0.22815094612611625`
  - Numeric path FiLM E10/E15/E20: `0.27216284520585693` /
    `0.23928665507798033` / `0.22548361537685974`
  - Same-parent bias E10/E15/E20: `0.2672322339495805` /
    `0.23332767412909886` / `0.22355039390925138`
  - Discrete depth bias E10/E15/E20: `0.2715337269990462` /
    `0.24170798617673578` / `0.22810707737377245`

Train-only calibration result:

- `prediction_scale`: `0.9927586206896553`
- `prediction_bias`: `-0.006689328326811565`
- `prediction_min_value`: `0.0`
- `prediction_zero_threshold`: `0.23800000000000002`
- Train raw MAE: `0.14354800477083407`
- Train calibrated MAE: `0.1338551270204161`
- Internal-val raw MAE: `0.22810451551416833`
- Internal-val calibrated MAE: `0.22062377767914837`
- Binned-residual calibrated train MAE: `0.13364277704475555`
- Binned-residual calibrated internal-val MAE: `0.22038613350896188`

Conclusion: switching `first_diff` and `tree_dist` from continuous sinusoidal
features to learnable categorical depth embeddings is a clean positive result
over the matched dropout baseline. It is slightly stronger than the
`numeric_path_beta` probe at 20 epochs, but weaker than `numeric_path_film` and
`same_parent`. Because `same_parent` is largely a special case of
`tree_dist == 2`, the next structural-bias work should avoid stacking the two
directly and instead separate which generic depth signal benefits most from
discretization.

## Dropout Plus Discrete Depth Plus Numeric Path FiLM

Run directory:

`checkpoints/20260611_030957_matbench_mp_gap_comp_cewald_cnn_dropcoords`

Training command:

```bash
env PYTHONUNBUFFERED=1 MALLOC_ARENA_MAX=2 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 pixi run python train.py \
  --dataset matbench_mp_gap --model-size large \
  --add-composition --add-comp-ewald --add-comp-nn --drop-coords \
  --matbench-split official --matbench-fold 0 --matbench-val-ratio 0.1 \
  --max-epochs 80 --patience 80 --max-cpu-workers 4 \
  --checkpoint-interval 10 --log-every 0 --sample-every 0 --eval-train-every 0 \
  --structural-bias-lr-mult 20 --loss-compression-scale 5.0 \
  --dropout 0.05 --numeric-path-film --bias-discrete-depths
```

Training result:

- Best raw internal-val MAE: `0.18349455533800266`
- Best raw epoch: `73`
- Final epoch raw internal-val MAE: `0.18352685070425476`
- Best checkpoint: `matbench_mp_gap_train_best_val.pth`
- Late-epoch raw internal-val MAE:
  - E60/E65/E70/E75/E80: `0.18621687214007937` /
    `0.18659174881981977` / `0.18461913495467108` /
    `0.18388267049839596` / `0.18352685070425476`

Train-only calibration result:

- `prediction_scale`: `0.988793103448276`
- `prediction_bias`: `0.0006901980361676423`
- `prediction_min_value`: `0.0`
- `prediction_zero_threshold`: `0.06`
- Train raw MAE: `0.03516331944635005`
- Train calibrated MAE: `0.02808856545825752`
- Internal-val raw MAE: `0.18349573230856453`
- Internal-val calibrated MAE: `0.18056009869746822`
- Binned-residual calibrated train MAE: `0.027997064197959623`
- Binned-residual calibrated internal-val MAE: `0.1805583177254605`

Conclusion: the three-way combination is the new clean fold0 internal-validation
best so far. It improves the previous `dropout=0.05` 80-epoch best from
`0.18445097342720354` raw / `0.18117659119360632` binned calibrated to
`0.18349455533800266` raw / `0.1805583177254605` binned calibrated. This
supports the user's hypothesis that short-run rankings should not eliminate
structural/numeric-bias factors; the combination was not best at 20 epochs but
became best by the full 80-epoch schedule. The remaining nonempty subsets of
`dropout=0.05`, `numeric_path_film`, and `bias_discrete_depths` should still be
run to completion before drawing a final interaction conclusion.

## Dropout Plus Discrete Depth 80-Epoch Schedule

Run directory:

`checkpoints/20260611_040752_matbench_mp_gap_comp_cewald_cnn_dropcoords`

Training command:

```bash
env PYTHONUNBUFFERED=1 MALLOC_ARENA_MAX=2 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 pixi run python train.py \
  --dataset matbench_mp_gap --model-size large \
  --add-composition --add-comp-ewald --add-comp-nn --drop-coords \
  --matbench-split official --matbench-fold 0 --matbench-val-ratio 0.1 \
  --max-epochs 80 --patience 80 --max-cpu-workers 4 \
  --checkpoint-interval 10 --log-every 0 --sample-every 0 --eval-train-every 0 \
  --structural-bias-lr-mult 20 --loss-compression-scale 5.0 \
  --dropout 0.05 --bias-discrete-depths
```

Training result:

- Best raw internal-val MAE: `0.18587612239330312`
- Best raw epoch: `73`
- Final epoch raw internal-val MAE: `0.18587973439291053`
- Best checkpoint: `matbench_mp_gap_train_best_val.pth`
- Late-epoch raw internal-val MAE:
  - E60/E65/E70/E75/E80: `0.18838964485315687` /
    `0.18694659481214064` / `0.1860130556824993` /
    `0.18638523591928548` / `0.18587973439291053`

Train-only calibration result:

- `prediction_scale`: `0.9868103448275862`
- `prediction_bias`: `0.0008076076186935843`
- `prediction_min_value`: `0.0`
- `prediction_zero_threshold`: `0.062`
- Train raw MAE: `0.037194299125385855`
- Train calibrated MAE: `0.02879570962308884`
- Internal-val raw MAE: `0.18587566384432672`
- Internal-val calibrated MAE: `0.18233078996050622`
- Binned-residual calibrated train MAE: `0.028729555340199414`
- Binned-residual calibrated internal-val MAE: `0.18226782116896223`

Conclusion: `dropout=0.05 + bias_discrete_depths` is a positive long-run result
relative to many earlier structural-bias probes, but it does not beat either
the previous dropout-only 80-epoch calibrated best or the new three-way
`dropout + discrete + FiLM` result. This suggests the discrete-depth signal is
useful but composes most effectively with `numeric_path_film` rather than
standing alone as the dominant improvement.

## Dropout Plus Numeric Path FiLM 80-Epoch Schedule

Run directory:

`checkpoints/20260611_050428_matbench_mp_gap_comp_cewald_cnn_dropcoords`

Training command:

```bash
env PYTHONUNBUFFERED=1 MALLOC_ARENA_MAX=2 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 pixi run python train.py \
  --dataset matbench_mp_gap --model-size large \
  --add-composition --add-comp-ewald --add-comp-nn --drop-coords \
  --matbench-split official --matbench-fold 0 --matbench-val-ratio 0.1 \
  --max-epochs 80 --patience 80 --max-cpu-workers 4 \
  --checkpoint-interval 10 --log-every 0 --sample-every 0 --eval-train-every 0 \
  --structural-bias-lr-mult 20 --loss-compression-scale 5.0 \
  --dropout 0.05 --numeric-path-film
```

Training result:

- Best raw internal-val MAE: `0.1849508279662171`
- Best raw epoch: `75`
- Final epoch raw internal-val MAE: `0.18501803992899016`
- Best checkpoint: `matbench_mp_gap_train_best_val.pth`
- Late-epoch raw internal-val MAE:
  - E60/E65/E70/E75/E80: `0.1891040019851296` /
    `0.18536893658598574` / `0.185075655899327` /
    `0.1849508279662171` / `0.18501803992899016`

Train-only calibration result:

- `prediction_scale`: `0.9868103448275862`
- `prediction_bias`: `0.00046773190577996187`
- `prediction_min_value`: `0.0`
- `prediction_zero_threshold`: `0.05`
- Train raw MAE: `0.03331615803422045`
- Train calibrated MAE: `0.026027095241415416`
- Internal-val raw MAE: `0.18493986210893149`
- Internal-val calibrated MAE: `0.1822814564514201`
- Binned-residual calibrated train MAE: `0.025982146492600156`
- Binned-residual calibrated internal-val MAE: `0.18218630031045166`

Conclusion: `dropout=0.05 + numeric_path_film` is a positive long-run result,
roughly matching the raw dropout-only schedule but landing behind it after
train-only calibration. It is stronger than `dropout=0.05 +
bias_discrete_depths`, while the three-way `dropout + discrete + FiLM`
combination remains best. This keeps FiLM as a useful generic numeric-path
bias, but suggests it needs the discrete depth encoding to produce the best
fold0 interaction.

## Discrete Depth Plus Numeric Path FiLM 80-Epoch Schedule

Run directory:

`checkpoints/20260611_060200_matbench_mp_gap_comp_cewald_cnn_dropcoords`

Training command:

```bash
env PYTHONUNBUFFERED=1 MALLOC_ARENA_MAX=2 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 pixi run python train.py \
  --dataset matbench_mp_gap --model-size large \
  --add-composition --add-comp-ewald --add-comp-nn --drop-coords \
  --matbench-split official --matbench-fold 0 --matbench-val-ratio 0.1 \
  --max-epochs 80 --patience 80 --max-cpu-workers 4 \
  --checkpoint-interval 10 --log-every 0 --sample-every 0 --eval-train-every 0 \
  --structural-bias-lr-mult 20 --loss-compression-scale 5.0 \
  --numeric-path-film --bias-discrete-depths
```

Training result:

- Best raw internal-val MAE: `0.18776998618917828`
- Best raw epoch: `75`
- Final epoch raw internal-val MAE: `0.18830131182724483`
- Best checkpoint: `matbench_mp_gap_train_best_val.pth`
- Late-epoch raw internal-val MAE:
  - E60/E65/E70/E75/E80: `0.18947460201829303` /
    `0.18985934007631142` / `0.18899139986592828` /
    `0.18776998618917828` / `0.18830131182724483`

Train-only calibration result:

- `prediction_scale`: `0.9749137931034483`
- `prediction_bias`: `0.0022739766123865187`
- `prediction_min_value`: `0.0`
- `prediction_zero_threshold`: `0.06973106137476874`
- Train raw MAE: `0.057141752507096945`
- Train calibrated MAE: `0.04282554916595107`
- Internal-val raw MAE: `0.1877844599754231`
- Internal-val calibrated MAE: `0.18130163223504703`
- Binned-residual calibrated train MAE: `0.042692398710919066`
- Binned-residual calibrated internal-val MAE: `0.1811977855022765`

Conclusion: `bias_discrete_depths + numeric_path_film` is a useful generic
bias-only combination under the default dropout setting. Its raw MAE is weaker
than the `dropout=0.05` variants, but train-only calibration recovers much of
the gap and nearly matches the dropout-only calibrated baseline. The three-way
result remains the best evidence: dropout, discrete depth, and numeric-path
FiLM appear to be complementary when all three are present.

## Discrete Depth 80-Epoch Schedule

Run directory:

`checkpoints/20260611_065838_matbench_mp_gap_comp_cewald_cnn_dropcoords`

Training command:

```bash
env PYTHONUNBUFFERED=1 MALLOC_ARENA_MAX=2 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 pixi run python train.py \
  --dataset matbench_mp_gap --model-size large \
  --add-composition --add-comp-ewald --add-comp-nn --drop-coords \
  --matbench-split official --matbench-fold 0 --matbench-val-ratio 0.1 \
  --max-epochs 80 --patience 80 --max-cpu-workers 4 \
  --checkpoint-interval 10 --log-every 0 --sample-every 0 --eval-train-every 0 \
  --structural-bias-lr-mult 20 --loss-compression-scale 5.0 \
  --bias-discrete-depths
```

Training result:

- Best raw internal-val MAE: `0.19237995491523208`
- Best raw epoch: `64`
- Final epoch raw internal-val MAE: `0.19256771545088402`
- Best checkpoint: `matbench_mp_gap_train_best_val.pth`
- Late-epoch raw internal-val MAE:
  - E60/E65/E70/E75/E80: `0.19371400053859256` /
    `0.1932591599112969` / `0.1933578684317808` /
    `0.1934131735184451` / `0.19256771545088402`

Train-only calibration result:

- `prediction_scale`: `0.9788793103448277`
- `prediction_bias`: `-0.002348988940798158`
- `prediction_min_value`: `0.0`
- `prediction_zero_threshold`: `0.064`
- Train raw MAE: `0.05828074941524517`
- Train calibrated MAE: `0.04699884233085141`
- Internal-val raw MAE: `0.19237494918298126`
- Internal-val calibrated MAE: `0.18795474080870514`
- Binned-residual calibrated train MAE: `0.04679732113733563`
- Binned-residual calibrated internal-val MAE: `0.18782283592840035`

Conclusion: discrete depth encoding alone is not competitive with the
dropout-only or FiLM-containing schedules at 80 epochs. The feature is still a
clean structural-bias component, but its value appears to come from interaction
with numeric-path FiLM and lower dropout rather than from replacing the
continuous depth features by itself.

## Numeric Path FiLM 80-Epoch Schedule

Run directory:

`checkpoints/20260611_075420_matbench_mp_gap_comp_cewald_cnn_dropcoords`

Training command:

```bash
env PYTHONUNBUFFERED=1 MALLOC_ARENA_MAX=2 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 pixi run python train.py \
  --dataset matbench_mp_gap --model-size large \
  --add-composition --add-comp-ewald --add-comp-nn --drop-coords \
  --matbench-split official --matbench-fold 0 --matbench-val-ratio 0.1 \
  --max-epochs 80 --patience 80 --max-cpu-workers 4 \
  --checkpoint-interval 10 --log-every 0 --sample-every 0 --eval-train-every 0 \
  --structural-bias-lr-mult 20 --loss-compression-scale 5.0 \
  --numeric-path-film
```

Training result:

- Best raw internal-val MAE: `0.18469890111309578`
- Best raw epoch: `70`
- Final epoch raw internal-val MAE: `0.1854295078517309`
- Best checkpoint: `matbench_mp_gap_train_best_val.pth`
- Late-epoch raw internal-val MAE:
  - E60/E65/E70/E75/E80: `0.1876296930863932` /
    `0.18760970986573355` / `0.18469890111309578` /
    `0.18501574255333011` / `0.1854295078517309`

Train-only calibration result:

- `prediction_scale`: `0.9749137931034483`
- `prediction_bias`: `0.0040830272564600254`
- `prediction_min_value`: `0.0`
- `prediction_zero_threshold`: `0.068`
- Train raw MAE: `0.05542935738508976`
- Train calibrated MAE: `0.04132544349283468`
- Internal-val raw MAE: `0.18469758320306529`
- Internal-val calibrated MAE: `0.1786675109253744`
- Binned-residual calibrated train MAE: `0.04138452037057726`
- Binned-residual calibrated internal-val MAE: `0.17866467743857514`

Conclusion: `numeric_path_film` alone is the best train-only-calibrated result
so far, even though the three-way `dropout=0.05 + discrete + FiLM` run still
has the best raw MAE. This makes FiLM the strongest generic bias found in the
seven-run matrix. It also shows that lower raw MAE and better train-fitted
calibration are not perfectly aligned, so follow-up work should evaluate FiLM
variants that improve raw MAE without destroying this favorable calibration
behavior.

## Numeric Path FiLM Plus Same-Path-Template 80-Epoch Schedule

Run directory:

`checkpoints/20260611_085351_matbench_mp_gap_comp_cewald_cnn_dropcoords`

Training command:

```bash
env PYTHONUNBUFFERED=1 MALLOC_ARENA_MAX=2 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 pixi run python train.py \
  --dataset matbench_mp_gap --model-size large \
  --add-composition --add-comp-ewald --add-comp-nn --drop-coords \
  --matbench-split official --matbench-fold 0 --matbench-val-ratio 0.1 \
  --max-epochs 80 --patience 80 --max-cpu-workers 4 \
  --checkpoint-interval 10 --log-every 0 --sample-every 0 --eval-train-every 0 \
  --structural-bias-lr-mult 20 --loss-compression-scale 5.0 \
  --numeric-path-film --bias-same-path-template
```

Training result:

- Best raw internal-val MAE: `0.19043274807943122`
- Best raw epoch: `75`
- Final epoch raw internal-val MAE: `0.19096141426642235`
- Best checkpoint: `matbench_mp_gap_train_best_val.pth`
- Late-epoch raw internal-val MAE:
  - E60/E65/E70/E75/E80: `0.19648339801560036` /
    `0.19230500079506221` / `0.19091610911265797` /
    `0.19043274807943122` / `0.19096141426642235`

Train-only calibration result:

- `prediction_scale`: `0.9729310344827586`
- `prediction_bias`: `0.00031862725953346697`
- `prediction_min_value`: `0.0`
- `prediction_zero_threshold`: `0.058`
- Train raw MAE: `0.05869130651316069`
- Train calibrated MAE: `0.041816384083505166`
- Internal-val raw MAE: `0.19044858991308417`
- Internal-val calibrated MAE: `0.18411754153387358`
- Binned-residual calibrated train MAE: `0.04169296227988686`
- Binned-residual calibrated internal-val MAE: `0.18407671758863553`

Conclusion: adding `same_path_template` to `numeric_path_film` is a clear
negative interaction on this fold. It is much worse than FiLM alone in both raw
and train-only-calibrated MAE, so the path-conditioned numeric embedding is
not helped by an additional same-template attention bias here. Future work
should focus on FiLM parameterization and numeric encoding rather than stacking
this structural relation.

## Numeric Path Beta 80-Epoch Schedule

Run directory:

`checkpoints/20260611_095040_matbench_mp_gap_comp_cewald_cnn_dropcoords`

Training command:

```bash
env PYTHONUNBUFFERED=1 MALLOC_ARENA_MAX=2 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 pixi run python train.py \
  --dataset matbench_mp_gap --model-size large \
  --add-composition --add-comp-ewald --add-comp-nn --drop-coords \
  --matbench-split official --matbench-fold 0 --matbench-val-ratio 0.1 \
  --max-epochs 80 --patience 80 --max-cpu-workers 4 \
  --checkpoint-interval 10 --log-every 0 --sample-every 0 --eval-train-every 0 \
  --structural-bias-lr-mult 20 --loss-compression-scale 5.0 \
  --numeric-path-beta
```

Training result:

- Best raw internal-val MAE: `0.1854656840265659`
- Best raw epoch: `72`
- Final epoch raw internal-val MAE: `0.1864444200833746`
- Best checkpoint: `matbench_mp_gap_train_best_val.pth`
- Late-epoch raw internal-val MAE:
  - E60/E65/E70/E75/E80: `0.1926359522576631` /
    `0.18844940193173026` / `0.18822398324447118` /
    `0.18683868030427445` / `0.1864444200833746`

Train-only calibration result:

- `prediction_scale`: `0.9788793103448277`
- `prediction_bias`: `0.002472669333380101`
- `prediction_min_value`: `0.0`
- `prediction_zero_threshold`: `0.0632273206938508`
- Train raw MAE: `0.05453122441355281`
- Train calibrated MAE: `0.04374708309146335`
- Internal-val raw MAE: `0.18547552619715424`
- Internal-val calibrated MAE: `0.18049102536700246`
- Binned-residual calibrated train MAE: `0.04372943309704913`
- Binned-residual calibrated internal-val MAE: `0.18039327389929605`

Conclusion: beta-only path conditioning captures much of the benefit of
`numeric_path_film`, but it does not match full FiLM. The raw result is close
to the better long-run schedules, while train-only calibration lands near the
three-way `dropout + discrete + FiLM` result and clearly behind FiLM-only. This
strongly suggests that path-conditioned additive offsets are useful, but the
FiLM multiplicative gamma branch provides the extra calibration/gap needed for
the current best.

## Numeric Path Gamma 80-Epoch Schedule

Run directory:

`checkpoints/20260611_105200_matbench_mp_gap_comp_cewald_cnn_dropcoords`

Training command:

```bash
env PYTHONUNBUFFERED=1 MALLOC_ARENA_MAX=2 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 pixi run python train.py \
  --dataset matbench_mp_gap --model-size large \
  --add-composition --add-comp-ewald --add-comp-nn --drop-coords \
  --matbench-split official --matbench-fold 0 --matbench-val-ratio 0.1 \
  --max-epochs 80 --patience 80 --max-cpu-workers 4 \
  --checkpoint-interval 10 --log-every 0 --sample-every 0 --eval-train-every 0 \
  --structural-bias-lr-mult 20 --loss-compression-scale 5.0 \
  --numeric-path-gamma
```

Training result:

- Best raw internal-val MAE: `0.1861711605434854`
- Best raw epoch: `73`
- Final epoch raw internal-val MAE: `0.1864775159734493`
- Best checkpoint: `matbench_mp_gap_train_best_val.pth`
- Late-epoch raw internal-val MAE:
  - E60/E65/E70/E75/E80: `0.19145814810225448` /
    `0.18723653024045323` / `0.1878447578741025` /
    `0.18661861754267467` / `0.1864775159734493`

Train-only calibration result:

- `prediction_scale`: `0.9788793103448277`
- `prediction_bias`: `0.0013427254133697214`
- `prediction_min_value`: `0.0`
- `prediction_zero_threshold`: `0.060594581244192355`
- Train raw MAE: `0.05542669501621422`
- Train calibrated MAE: `0.04398530146551161`
- Internal-val raw MAE: `0.18616940212120175`
- Internal-val calibrated MAE: `0.18120019323619593`
- Binned-residual calibrated train MAE: `0.04394524239027421`
- Binned-residual calibrated internal-val MAE: `0.18113993278640972`

Conclusion: gamma-only path conditioning is useful but weaker than beta-only
and much weaker than full FiLM after train-only calibration. The result
supports an interaction view: beta and gamma each help, but FiLM's calibrated
gain comes from learning both a path-conditioned additive offset and a
path-conditioned multiplicative rescaling together.

## Notes

- These are not final official Matbench test results.
- The test fold was not evaluated during any experiment recorded here.
- The strongest clean improvement so far comes from lowering dropout to `0.05`
  and training the same single checkpoint for an 80-epoch cosine schedule,
  followed by train-only scale/bias/nonnegative/zero-threshold calibration.
