"""Average model weights from compatible training checkpoints.

The output keeps only the averaged model state plus lightweight provenance. It
is intended for cheap SWA/weight-soup ablations without touching held-out data.
"""

import argparse
import json
import pathlib
import sys
import time

import torch

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from train import get_rng_states


def _load_model_state(path):
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if "model" not in checkpoint:
        raise KeyError(f"Checkpoint has no 'model' state: {path}")
    return checkpoint["model"], checkpoint


def average_model_states(paths):
    if not paths:
        raise ValueError("At least one checkpoint is required")

    paths = [pathlib.Path(p) for p in paths]
    avg_state = None
    reference_keys = None
    reference_dtypes = None
    reference_shapes = None
    provenance = []

    for index, path in enumerate(paths):
        state, checkpoint = _load_model_state(path)
        keys = set(state.keys())
        dtypes = {key: tensor.dtype for key, tensor in state.items()}
        shapes = {key: tuple(tensor.shape) for key, tensor in state.items()}

        if index == 0:
            reference_keys = keys
            reference_dtypes = dtypes
            reference_shapes = shapes
            avg_state = {
                key: tensor.detach().clone().float()
                if tensor.is_floating_point()
                else tensor.detach().clone()
                for key, tensor in state.items()
            }
        else:
            if keys != reference_keys:
                missing = sorted(reference_keys - keys)
                extra = sorted(keys - reference_keys)
                raise ValueError(
                    f"Checkpoint keys differ for {path}: missing={missing[:5]}, extra={extra[:5]}"
                )
            for key, tensor in state.items():
                if tuple(tensor.shape) != reference_shapes[key]:
                    raise ValueError(
                        f"Shape mismatch for {key} in {path}: "
                        f"{tuple(tensor.shape)} != {reference_shapes[key]}"
                    )
                if tensor.dtype != reference_dtypes[key]:
                    raise ValueError(
                        f"Dtype mismatch for {key} in {path}: "
                        f"{tensor.dtype} != {reference_dtypes[key]}"
                    )
                if tensor.is_floating_point():
                    avg_state[key].add_(tensor.detach().float())
                elif not torch.equal(avg_state[key], tensor):
                    raise ValueError(f"Non-floating state differs for {key} in {path}")

        provenance.append(
            {
                "path": str(path),
                "epoch": checkpoint.get("epoch"),
                "best_loss": checkpoint.get("best_loss"),
                "stage": checkpoint.get("stage"),
            }
        )

    count = float(len(paths))
    for key, tensor in list(avg_state.items()):
        if tensor.is_floating_point():
            averaged = tensor.div_(count)
            avg_state[key] = averaged.to(reference_dtypes[key])

    return avg_state, provenance


def main():
    parser = argparse.ArgumentParser(
        description="Average compatible checkpoint model weights",
        allow_abbrev=False,
    )
    parser.add_argument("--output", required=True, help="Output .pth path")
    parser.add_argument("checkpoints", nargs="+", help="Input checkpoint .pth paths")
    args = parser.parse_args()

    output = pathlib.Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    avg_state, provenance = average_model_states(args.checkpoints)
    averaged_checkpoint = {
        "model": avg_state,
        "stage": "averaged",
        "averaged_from": provenance,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "rng_states": get_rng_states(),
    }
    torch.save(averaged_checkpoint, output)

    provenance_path = output.with_suffix(output.suffix + ".json")
    with provenance_path.open("w") as f:
        json.dump({"output": str(output), "averaged_from": provenance}, f, indent=2)

    print(f"Saved averaged checkpoint: {output}")
    print(f"Saved provenance: {provenance_path}")


if __name__ == "__main__":
    main()
