# SPDX-License-Identifier: Apache-2.0
"""Dependency-free runner: python3 tests/ut/attention/test_turboquant_group_plan.py."""

import pathlib
import shutil
import subprocess
import tempfile
import unittest


class TestTurboQuantGroupPlan(unittest.TestCase):
    def test_native_planner(self):
        compiler = shutil.which("clang++") or shutil.which("g++")
        if compiler is None:
            self.skipTest("A C++ compiler is required for the shared device/CPU planner")
        root = pathlib.Path(__file__).resolve().parents[3]
        source = root / "tests/ut/attention/turboquant/test_group_plan.cpp"
        with tempfile.TemporaryDirectory(prefix="tq-group-plan-") as directory:
            binary = pathlib.Path(directory) / "group-plan-test"
            subprocess.run(
                [
                    compiler,
                    "-std=c++17",
                    "-O1",
                    "-g",
                    "-fsanitize=address,undefined",
                    "-fno-omit-frame-pointer",
                    "-Wall",
                    "-Wextra",
                    "-Werror",
                    "-I",
                    str(root),
                    str(source),
                    "-o",
                    str(binary),
                ],
                check=True,
            )
            subprocess.run([str(binary)], check=True)


if __name__ == "__main__":
    unittest.main()
