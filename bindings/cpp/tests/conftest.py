"""Generate canonical native archives and exact reference actions."""
import json
import os
import subprocess
from pathlib import Path

import pytest


@pytest.fixture(scope="session")
def native_fixtures(tmp_path_factory):
    root = tmp_path_factory.mktemp("native-fixtures")
    executable = os.environ.get("REXGRAPH_CPP_TEST_FIXTURE")
    if executable is None:
        build = root / "build"
        subprocess.run([
            "cmake", "-S", str(Path(__file__).resolve().parents[1]), "-B", str(build),
            "-DREXGRAPH_BINDING_TEST_FIXTURE=ON",
            f"-DPython3_EXECUTABLE={os.sys.executable}",
        ], check=True, capture_output=True, text=True)
        subprocess.run(["cmake", "--build", str(build), "--target", "binding_fixture", "-j1"],
                       check=True, capture_output=True, text=True)
        executable = build / "binding_fixture"
    result = subprocess.run([str(executable), str(root)], check=True,
                            capture_output=True, text=True)
    oracle = {(row["source"], row["kind"], row["grade"]): row
              for row in map(json.loads, result.stdout.splitlines())}
    return root, oracle


@pytest.fixture
def native_source(native_fixtures):
    from rexgraph_cpp import NativeSource
    return NativeSource((native_fixtures[0] / "diagonal.rfcxs").read_bytes())
