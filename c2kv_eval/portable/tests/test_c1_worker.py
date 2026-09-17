from __future__ import annotations

import argparse
from pathlib import Path
import tempfile
import unittest

from c2kv_eval.portable import c1_worker


class C1WorkerTest(unittest.TestCase):
    def _args(self, root: Path, benchmark: str) -> argparse.Namespace:
        return argparse.Namespace(
            benchmark=benchmark,
            task_id="0" if benchmark == "tau2" else "get_wifi",
            agent_base_url="http://127.0.0.1:38810",
            user_base_url="http://127.0.0.1:38800",
            benchmark_dir=root,
            bench_python="/path/to/benchmark/python",
            out=root / "new-output",
            run_name="c1_mock",
            model="c1_legacy_prefill",
            task_set="airline",
            tau2_max_steps=None,
            task_timeout=3600,
            ts_agent="GPT_4_o_2024_05_13",
            ts_user="GPT_4_o_2024_05_13",
        )

    def test_agent_and_user_endpoints_are_distinct(self):
        with tempfile.TemporaryDirectory() as directory:
            args = self._args(Path(directory), "tau2")
            context = c1_worker.build_context(args)
            self.assertEqual(context.base_url, "http://127.0.0.1:38810")
            self.assertEqual(context.user_base_url, "http://127.0.0.1:38800")
            self.assertEqual(context.arm, "c1_legacy_prefill")

    def test_equal_agent_and_user_endpoint_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            args = self._args(Path(directory), "toolsandbox")
            args.user_base_url = args.agent_base_url + "/v1"
            with self.assertRaisesRegex(ValueError, "must differ"):
                c1_worker.build_context(args)

    def test_exactly_one_identity_is_forwarded(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            tau2 = c1_worker.build_context(self._args(root, "tau2"))
            toolsandbox = c1_worker.build_context(self._args(root, "toolsandbox"))
            self.assertEqual(tau2.options["tau2_task_ids"], "0")
            self.assertEqual(toolsandbox.options["ts_scenarios"], "get_wifi")

    def test_existing_output_is_rejected_before_adapter_import(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            args = self._args(root, "tau2")
            args.out.mkdir()
            with self.assertRaises(FileExistsError):
                c1_worker.run(args)


if __name__ == "__main__":
    unittest.main()
