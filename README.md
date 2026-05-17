# Document Transformer

A **Masked Prediction Transformer** that natively handles deeply nested JSON documents (MongoDB-style) for structured property prediction.

## Requirements

```bash
pixi install
```

## Training

### Matbench (Crystal Property Prediction)

```bash
pixi run python train.py --dataset matbench_mp_gap
pixi run python train.py --dataset matbench_dielectric --add-angles --no-coords
```

Available options: `--add-angles`, `--add-bonds`, `--add-composition`, `--no-coords`, `--add-ewald`.

### Tabular Regression

```bash
pixi run python train.py --dataset california_housing
```

### Resume Training

```bash
pixi run python train.py --dataset matbench_mp_gap --resume checkpoints/xxx.pth
pixi run python train.py --dataset matbench_mp_gap --resume checkpoints/xxx.pth --warm-restart
```

Checkpoints are saved to `checkpoints/`. TensorBoard logs to `runs/`.

## Inference

```bash
pixi run python inference.py
```
