"""python3 -m unittest discover -s okc_bridge/tests"""

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
os.environ["AKO_OKC_STATE"] = tempfile.mkdtemp()

import broker  # noqa: E402


def limits(checks_used, runs_used, checks_left=None, runs_left=None):
    return {"checks": {"used": checks_used, "left": 20 - checks_used if checks_left is None else checks_left},
            "scored": {"used": runs_used, "left": 10 - runs_used if runs_left is None else runs_left}}


class Schedule(unittest.TestCase):
    cfg = {"schedule": {"checks_per_run": 2, "reserve_checks": 0, "reserve_runs": 0}}

    def test_full_day_is_check_check_run(self):
        c = r = 0
        seq = ""
        while (k := broker.choose_kind(limits(c, r), self.cfg)):
            seq += {"check": "C", "run": "R"}[k]
            c, r = c + (k == "check"), r + (k == "run")
        self.assertEqual(seq, "CCR" * 10)

    def test_follows_live_counts_after_manual_spending(self):
        self.assertEqual(broker.choose_kind(limits(5, 0), self.cfg), "run")
        self.assertEqual(broker.choose_kind(limits(0, 3), self.cfg), "check")

    def test_falls_back_when_one_kind_is_spent(self):
        self.assertEqual(broker.choose_kind(limits(20, 2, checks_left=0), self.cfg), "run")
        self.assertEqual(broker.choose_kind(limits(4, 10, runs_left=0), self.cfg), "check")
        self.assertIsNone(broker.choose_kind(limits(20, 10, 0, 0), self.cfg))

    def test_reserves(self):
        cfg = {"schedule": {"checks_per_run": 2, "reserve_checks": 2, "reserve_runs": 1}}
        self.assertIsNone(broker.choose_kind(limits(18, 9), cfg))


class Summary(unittest.TestCase):
    def test_real_check_result(self):
        raw = json.loads((HERE / "fixtures" / "check_result.json").read_text())
        s = broker.summarize(raw)
        self.assertTrue(s["passed"])
        self.assertAlmostEqual(s["speedup"], 0.90997, places=4)
        self.assertEqual(s["tests"], {"passed": 9, "total": 9})
        self.assertEqual(len(s["shapes"]), 30)
        self.assertEqual(set(s["loads"]), {"decode 1", "decode 16", "decode 32", "verify 1", "verify 16", "verify 32"})


if __name__ == "__main__":
    unittest.main()
