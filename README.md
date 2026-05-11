# Document Transformer Testing & Prototype

This directory contains the prototype implementation of the **Document Transformer**, a Masked Prediction model designed to natively handle deeply nested MongoDB JSON documents without manually engineered schemas or SQL masks.

## Requirements

Ensure `pytorch` and `sentence-transformers` are installed in your environment:
```bash
pixi install
```

## Running the Validation Pipeline

The model must pass a strict 3-stage sanity check to prove gradient flow and architectural soundness. 
Run these commands from the project root (`AIMaterials/`):

1. **Stage 1: Zero-Loss Overfit**
   ```bash
   pixi run python tmp/testing/tests/test_stage1_overfit.py
   ```
2. **Stage 2: Logic Copy & Addressing**
   ```bash
   pixi run python tmp/testing/tests/test_stage2_copy.py
   ```
3. **Stage 3: Padding Masking Block**
   ```bash
   pixi run python tmp/testing/tests/test_stage3_padding.py
   ```

## Training on Synthetic Data

To launch an end-to-end training loop over synthetic hierarchical data:
```bash
pixi run python tmp/testing/train.py
```
Checkpoints will be saved to `tmp/testing/checkpoints/`.

## Inference Demo

To test the trained checkpoint on a single manual dictionary logic completion:
```bash
pixi run python tmp/testing/inference.py
```
