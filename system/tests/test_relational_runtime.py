"""System serves native program families through the common RCQL contracts."""
import importlib.util
import os
from pathlib import Path

import pytest


def test_native_family_inspection_and_packet_download_use_one_selected_record():
    pytest.importorskip("rexgraph_cpp._temporal_program")
    pytest.importorskip("httpx")
    from fastapi.testclient import TestClient
    from rcql import Result
    from rexgraph.io.relational_record import relational_record
    from rexgraph.relational_runtime import RelationalRuntime
    from system.server.app import app
    from system.state import sources
    path = Path(os.environ.get("REXGRAPH_RELATIONAL_TEST_HELPER", str(
        Path(__file__).resolve().parents[2] / "bindings/cpp/tests/test_temporal_program.py")))
    spec = importlib.util.spec_from_file_location("system_relational_declarations", path)
    helper = importlib.util.module_from_spec(spec);spec.loader.exec_module(helper)
    _, runtime, *_ = helper.fit_fixture(RelationalRuntime.create(89001))
    sources.register("native-runtime", relational_record(runtime))
    try:
        with TestClient(app) as client:
            response = client.post("/api/query", json={"query":
                'FROM REX("native-runtime") RETURN RELATIONAL_INFO(RELATIONAL_PACKET(),observation="fit")',
                "result_format": "native"})
            assert response.status_code == 200, response.text
            result = Result.from_bytes(response.content)
            assert result.values[0] == runtime.fit()
            response = client.post("/api/query", json={"query":
                'FROM REX("native-runtime") RETURN RELATIONAL_PACKET()', "result_format": "native"})
            assert response.status_code == 200, response.text
            assert Result.from_bytes(response.content).values == (runtime.to_bytes(),)
    finally:
        sources.remove("native-runtime")
