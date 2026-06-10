import tempfile
import unittest
from pathlib import Path

import torch

from scripts.average_checkpoints import average_model_states


class AverageCheckpointsTest(unittest.TestCase):
    def test_averages_only_floating_tensors(self):
        with tempfile.TemporaryDirectory() as tmp:
            path1 = Path(tmp) / "a.pth"
            path2 = Path(tmp) / "b.pth"
            torch.save(
                {
                    "model": {
                        "weight": torch.tensor([1.0, 3.0], dtype=torch.float32),
                        "counter": torch.tensor([7], dtype=torch.int64),
                    },
                    "epoch": 1,
                },
                path1,
            )
            torch.save(
                {
                    "model": {
                        "weight": torch.tensor([3.0, 5.0], dtype=torch.float32),
                        "counter": torch.tensor([7], dtype=torch.int64),
                    },
                    "epoch": 2,
                },
                path2,
            )

            state, provenance = average_model_states([path1, path2])

            torch.testing.assert_close(state["weight"], torch.tensor([2.0, 4.0]))
            torch.testing.assert_close(state["counter"], torch.tensor([7]))
            self.assertEqual([entry["epoch"] for entry in provenance], [1, 2])

    def test_rejects_changed_non_floating_tensors(self):
        with tempfile.TemporaryDirectory() as tmp:
            path1 = Path(tmp) / "a.pth"
            path2 = Path(tmp) / "b.pth"
            torch.save({"model": {"counter": torch.tensor([7])}}, path1)
            torch.save({"model": {"counter": torch.tensor([8])}}, path2)

            with self.assertRaisesRegex(ValueError, "Non-floating state differs"):
                average_model_states([path1, path2])


if __name__ == "__main__":
    unittest.main()
