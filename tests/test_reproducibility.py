import json
import os
import random
import subprocess
import sys
import unittest
from collections import Counter
from pathlib import Path

from data.matbench.loader import _stratified_sample


REPO_ROOT = Path(__file__).resolve().parents[1]


class StableHashTests(unittest.TestCase):
    def _hashes_with_seed(self, seed: int):
        env = os.environ.copy()
        env["PYTHONHASHSEED"] = str(seed)
        code = (
            "import json; "
            "from model.json_parser import JSONParser; "
            "print(json.dumps([JSONParser._key_hash(k) "
            "for k in ['crystal', 'target', 'sites']]))"
        )
        result = subprocess.run(
            [sys.executable, "-c", code],
            cwd=REPO_ROOT,
            env=env,
            check=True,
            capture_output=True,
            text=True,
        )
        return json.loads(result.stdout)

    def test_key_hash_is_stable_across_python_hash_seeds(self):
        hashes_a = self._hashes_with_seed(1)
        hashes_b = self._hashes_with_seed(2)

        self.assertEqual(hashes_a, hashes_b)
        for key_hash in hashes_a:
            self.assertGreaterEqual(key_hash, 1_000_000)


class StratifiedSampleTests(unittest.TestCase):
    def test_unique_elements_over_budget_are_sampled_not_hard_truncated(self):
        sites = [
            {"element": element, "idx": idx}
            for idx, element in enumerate(["Li", "Na", "K", "Rb", "Cs"])
        ]

        sampled = _stratified_sample(sites, 3)
        sampled_elements = [site["element"] for site in sampled]

        self.assertEqual(len(sampled), 3)
        self.assertEqual(len(set(sampled_elements)), 3)
        self.assertNotEqual(sampled_elements, ["Li", "Na", "K"])
        self.assertTrue({"Rb", "Cs"} & set(sampled_elements))

    def test_sampling_is_deterministic_and_does_not_consume_global_random(self):
        sites = (
            [{"element": "Li", "idx": idx} for idx in range(6)]
            + [{"element": "Na", "idx": 6 + idx} for idx in range(3)]
            + [{"element": "K", "idx": 9 + idx} for idx in range(1)]
        )

        random.seed(1234)
        state_before = random.getstate()
        first = _stratified_sample(sites, 5)
        state_after = random.getstate()

        random.seed(9999)
        second = _stratified_sample(sites, 5)

        self.assertEqual(state_before, state_after)
        self.assertEqual(first, second)
        self.assertEqual(len(first), 5)
        self.assertEqual(Counter(site["element"] for site in first), Counter({
            "Li": 3,
            "Na": 1,
            "K": 1,
        }))


if __name__ == "__main__":
    unittest.main()
