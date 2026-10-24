"""python3 -m unittest discover -s okc_bridge/tests"""

import hashlib
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
os.environ.setdefault("AKO_OKC_STATE", tempfile.mkdtemp())

import broker  # noqa: E402
import dashboard  # noqa: E402

P, Q, R = "muse_bf16_proj", "muse_attention_decode", "muse_moe_w4a16"
BEFORE = dashboard.LOCK_FIX_TS - 3600
AFTER = dashboard.LOCK_FIX_TS + 3600


class State:
    """A throwaway $AKO_OKC_STATE with helpers that write rows the way broker.py and local.py do."""

    def __init__(self):
        self.root = Path(tempfile.mkdtemp())
        self.n = 0

    def _code(self, problem, where, body, header):
        code = header + body
        sha = hashlib.sha256(code.encode()).hexdigest()
        d = self.root / problem / where
        d.mkdir(parents=True, exist_ok=True)
        (d / f"{sha}.py").write_text(code)
        return sha

    def _append(self, problem, name, row):
        d = self.root / problem
        d.mkdir(parents=True, exist_ok=True)
        with open(d / name, "a") as f:
            f.write(json.dumps(row) + "\n")

    def official(self, problem, body, speedup, *, kind="check", ts=None, status="done", child="ako4x-run-x-r2",
                 label=None, header=None, job=None, flags=()):
        self.n += 1
        sha = self._code(problem, "code", body, header or f"# okc: problem={problem}\n")
        job = job or f"job{self.n}"
        self._append(problem, "evals.jsonl", {"event": "started", "ts": ts or self.n, "job": job, "kind": kind,
                                              "sha256": sha, "child": child, "label": label or f"iter-{self.n}"})
        self._append(problem, "evals.jsonl", {
            "event": "finished", "ts": ts or self.n, "problem": problem, "child": child,
            "label": label or f"iter-{self.n}", "sha256": sha, "kind": kind, "job": job, "status": status,
            "passed": True, "speedup": speedup, "ci": [speedup - 0.01, speedup + 0.01], "flags": list(flags)})
        return job

    def local(self, problem, body, speedup, *, ts=None, child="ako4x-run-x-r3", label=None, header=None,
              flags=(), passed=True):
        self.n += 1
        sha = self._code(problem, "local", body, header or f"# okc: problem={problem}\n")
        self._append(problem, "local.jsonl", {
            "event": "finished", "ts": ts or self.n, "problem": problem, "child": child,
            "label": label or f"local-{self.n}", "sha256": sha, "kind": "local", "job": f"local-{sha[:12]}",
            "status": "done" if passed else "failed", "passed": passed, "speedup": speedup if passed else None,
            "ci": None, "flags": list(flags), "source": "local"})

    def calibration(self, cal):
        d = self.root / "sidecar"
        d.mkdir(parents=True, exist_ok=True)
        (d / "calibration.json").write_text(json.dumps(cal))


class Dashboard(unittest.TestCase):
    def setUp(self):
        self.s = State()
        self.prev = os.environ.get("AKO_OKC_STATE")
        os.environ["AKO_OKC_STATE"] = str(self.s.root)
        dashboard._hash_cache.clear()

    def tearDown(self):
        if self.prev is None:
            os.environ.pop("AKO_OKC_STATE", None)
        else:
            os.environ["AKO_OKC_STATE"] = self.prev

    def prob(self, data, problem):
        return next(d for d in data["problems"] if d["problem"] == problem)

    def test_uses_the_state_root(self):
        self.assertEqual(broker.state_root(), self.s.root)
        self.s.official(P, "a\n", 1.1)
        self.assertEqual(self.prob(dashboard.build(), P)["counts"]["total"], 1)

    def test_duplicates_failures_calibration_and_test_rows_are_not_runs(self):
        s = self.s
        s.official(P, "a\n", 1.0, ts=10)
        s.official(P, "a\n", 1.0, ts=11, status="duplicate")
        s.official(P, "b\n", 1.1, ts=12, kind="run")
        s.local(P, "a\n", 1.02, ts=13, child="calibration", label="calibrate:job1")
        s.local(P, "a\n", 1.02, ts=14, child="e2e-test", label="e2e")
        s.local(P, "c\n", 1.3, ts=15, child="ako4x-run-x-r4", label="calibrate:old")
        s.local(P, "d\n", 1.2, ts=16)
        s.local(P, "e\n", None, ts=17, passed=False)
        d = self.prob(dashboard.build(), P)
        self.assertEqual(d["counts"], {"check": 1, "run": 1, "local": 1, "total": 3})
        self.assertEqual([p["kind"] for p in d["points"]], ["check", "run", "local"])
        self.assertEqual(len(d["calibration_rows"]), 3)
        # ...but the calibration rows still pair: file "a" was judged 1.0 and timed 1.02 locally
        self.assertEqual(d["ratio"]["source"], "matched pairs")
        self.assertAlmostEqual(d["ratio"]["ratio"], 1.0 / 1.02)

    def test_pairing_ignores_the_okc_header_line(self):
        s = self.s
        s.official(Q, "body\n", 0.96, header=f"# okc: problem={Q}\n")
        s.local(Q, "body\n", 1.0, header=f"# okc: problem={Q} child=ako4x-run-q-r5\n")
        s.local(Q, "other body\n", 2.0)
        pr = dashboard.pairs(Q, s.root)
        self.assertEqual(len(pr), 1)
        self.assertAlmostEqual(pr[0]["ratio"], 0.96)
        self.assertNotEqual(dashboard.strip_header("x = 1\n"), "")  # no header: the code as is

    def test_projection_and_ratio_fallbacks(self):
        s = self.s
        s.calibration({P: {"n": 4, "ratio_median": 0.99}})
        s.official(P, "p\n", 1.0)
        s.local(P, "p\n", 2.0)  # a pair that calibration.json overrides
        s.official(Q, "q\n", 0.96)
        s.local(Q, "q\n", 1.0)
        s.official(R, "r\n", 1.5)
        s.local(R, "new\n", 2.0)  # R has no pair: falls back to the median of P and Q
        data = dashboard.build()
        p, q, r = (self.prob(data, x)["ratio"] for x in (P, Q, R))
        self.assertEqual((p["source"], p["ratio"], p["n"], p["uncalibrated"]), ("sidecar calibration", 0.99, 4, False))
        self.assertEqual((q["source"], q["n"], q["uncalibrated"]), ("matched pairs", 1, False))
        self.assertTrue(r["uncalibrated"])
        self.assertAlmostEqual(r["ratio"], (0.99 + 0.96) / 2)
        pt = next(x for x in self.prob(data, R)["points"] if x["kind"] == "local")
        self.assertAlmostEqual(pt["y"], 2.0 * r["ratio"])
        self.assertEqual(pt["local_speedup"], 2.0)
        self.assertIn("projection uncalibrated", pt["hover"])
        self.assertIn("sidecar score: 2.0000x", pt["hover"])
        # with no ratio anywhere, projections are 1:1 and uncalibrated
        self.assertEqual(dashboard.ratios(State().root)[P]["ratio"], 1.0)

    def test_runs_are_numbered_in_time_order_across_kinds(self):
        s = self.s
        s.local(P, "l1\n", 1.1, ts=300)
        s.official(P, "o1\n", 1.0, ts=100)
        s.official(P, "o2\n", 1.2, ts=400, kind="run", child="promote", label="promote:job2")
        s.official(P, "o3\n", 0.9, ts=200, kind="run", child="ako4x-run-x-r7")
        d = self.prob(dashboard.build(), P)
        self.assertEqual([(p["n"], p["ts"]) for p in d["points"]], [(1, 100), (2, 200), (3, 300), (4, 400)])
        self.assertEqual([p["round"] for p in d["points"]], ["r2", "r7", "r3", "promote"])
        self.assertEqual(d["board"]["y"], 1.2)
        self.assertEqual(d["latest_run"]["y"], 1.2)

    def test_withdrawn_runs_are_marked_and_leave_the_board(self):
        s = self.s
        s.official(P, "a\n", 1.1, kind="run", job="J1", ts=1)
        s.official(P, "b\n", 1.3, kind="run", job="J2", ts=2)
        s._append(P, "evals.jsonl", {"event": "withdrawn", "ts": 3, "problem": P, "job": "J2", "reason": "bug"})
        d = self.prob(dashboard.build(), P)
        self.assertEqual(d["board"]["job"], "J1")
        j2 = next(p for p in d["points"] if p["job"] == "J2")
        self.assertTrue(j2["withdrawn"])
        self.assertIn("WITHDRAWN", j2["hover"])

    def test_pre_lock_fix_only_for_early_self_timing_files(self):
        s = self.s
        s.local(P, "import time\n", 1.0, ts=BEFORE)
        s.local(P, "x = 1\n", 1.0, ts=BEFORE + 1, flags=["line 9: torch.cuda.Event: CUDA event timing in a submission"])
        s.local(P, "y = 1\n", 1.0, ts=BEFORE + 2)
        s.local(P, "import time\n# later\n", 1.0, ts=AFTER)
        d = self.prob(dashboard.build(), P)
        self.assertEqual([p["pre_lock_fix"] for p in d["points"]], [True, True, False, False])
        self.assertIn("pre lock fix", d["points"][0]["hover"])

    def test_partial_last_line_is_skipped_and_page_is_written(self):
        s = self.s
        s.official(P, "a\n", 1.1, label="m<=32 </script>")
        with open(s.root / P / "evals.jsonl", "a") as f:
            f.write('{"event": "finished", "sta')
        out = Path(tempfile.mkdtemp()) / "dash"
        data = dashboard.generate(out, refresh_s=30)
        self.assertEqual(self.prob(data, P)["counts"]["total"], 1)
        page = (out / "index.html").read_text()
        self.assertIn("plotly.js/2.27.0/plotly.min.js", page)
        self.assertIn("const REFRESH_S = 30;", page)
        self.assertNotIn("</script>\"", page)  # labels can't close the inline script
        self.assertIn("m&lt;=32", self.prob(json.loads((out / "data.json").read_text()), P)["points"][0]["hover"])


class Board(unittest.TestCase):
    def test_rank_counts_only_other_teams_above_us(self):
        snap = {"ts": 0, "leader": 2.0, "leader_team": "lead", "others": [2.0, 1.8, 1.5, 1.2] + [1.0] * 8}
        r = dashboard.rank_on(snap, 1.5)
        self.assertEqual((r["rank"], r["of"], r["top10"]), (3, 13, 1.0))
        self.assertEqual(dashboard.rank_on(snap, 2.1)["rank"], 1)
        self.assertIsNone(dashboard.rank_on(None, 1.0))

    def test_snapshots_give_the_leader_at_each_run_time(self):
        root = Path(tempfile.mkdtemp())
        (root / "leaderboard").mkdir()
        rows = [{"ts": 100, "leader": 1.5, "leader_team": "a", "others": [1.5]},
                {"ts": 200, "leader": 1.9, "leader_team": "b", "others": [1.9, 1.5]}]
        (root / "leaderboard" / "muse_bf16_proj.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
        snaps = dashboard.board_snapshots("muse_bf16_proj", root, fetch=False)
        self.assertEqual([s["leader"] for s in snaps], [1.5, 1.9])
        then = [s for s in snaps if s["ts"] <= 150]
        self.assertEqual(then[-1]["leader"], 1.5)


if __name__ == "__main__":
    unittest.main()
