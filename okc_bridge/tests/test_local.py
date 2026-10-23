"""python3 -m unittest discover -s okc_bridge/tests"""

import json
import os
import sys
import tempfile
import types
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
os.environ.setdefault("AKO_OKC_STATE", tempfile.mkdtemp())

import local  # noqa: E402

CODE = "# okc: problem=muse_bf16_proj\ndef kernel_fn(data, config):\n    return data\n"


class LocalEval(unittest.TestCase):
    def setUp(self):
        self.raw = json.loads((HERE / "fixtures" / "check_result.json").read_text())
        self.calls = []
        local.run_sidecar = lambda problem, file, cfg: self.calls.append(problem) or self.raw
        self.f = Path(tempfile.mkdtemp()) / "submission.py"

    def args(self, code):
        self.f.write_text(code)
        return types.SimpleNamespace(problem="muse_bf16_proj", file=str(self.f), child="c1", label="local-1")

    def test_renders_like_the_judge_and_caches_by_file(self):
        row = local.cmd_eval(self.args(CODE + "# a\n"), {})
        self.assertEqual((row["event"], row["kind"], row["source"]), ("finished", "local", "local"))
        self.assertTrue(row["job"].startswith("local-"))
        text = local.render(row)
        self.assertIn("LOCAL sidecar H200", text)
        self.assertIn("NOT the judge", text)
        self.assertIn("uncalibrated", text)
        again = local.cmd_eval(self.args(CODE + "# a\n"), {})
        self.assertTrue(again.get("cached"))
        self.assertEqual(self.calls, ["muse_bf16_proj"])

    def test_lint_failure_never_reaches_the_pod(self):
        row = local.cmd_eval(self.args("# okc: problem=muse_bf16_proj\nx = 1\n"), {})
        self.assertEqual(row["status"], "LINT_REJECTED")
        self.assertEqual(self.calls, [])

    def test_unavailable_passes_through(self):
        local.run_sidecar = lambda *a: local.unavailable("pod off")
        row = local.cmd_eval(self.args(CODE + "# b\n"), {})
        self.assertEqual(row["status"], "LOCAL_UNAVAILABLE")
        self.assertIn("LOCAL LOCAL_UNAVAILABLE", local.render(row))


if __name__ == "__main__":
    unittest.main()
