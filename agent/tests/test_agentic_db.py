"""RCDB revisions and RCQL execution through Agent and Hive."""
from fractions import Fraction
import pytest
from agent.agentic_db import AgenticDB
from agent import rcdb
from agent.hive import Hive
from rcql import Result
from rexgraph import RexGraph
from rexgraph.object_identity import object_digest


@pytest.fixture
def database(tmp_path):
    with AgenticDB("local://" + str(tmp_path / "records"), writable=True) as db:
        yield db


def graph():
    return RexGraph.from_cells([4, [[0, 1, 2], [2, 3]]])


def test_complete_native_values_and_guarded_revisions(database):
    value = graph()
    first = database.publish("data", value, expected_version=0)
    assert first.version == 1
    assert object_digest(database.extract("data", version=1)) == object_digest(value)
    second = database.publish("data", RexGraph.from_cells([2, [[0, 1]]]), expected_version=1)
    assert second.version == 2 and database.extract("data", version=1).nV == 4
    with pytest.raises(rcdb.VersionConflictError):
        database.publish("data", value, expected_version=1)


def test_typed_queries_filter_actual_record_metadata(database):
    database.publish("data", graph(), expected_version=0)
    result = database.query('FROM $db MATCH r IN RCDB_SELECT() WHERE r.signature.nE >= 2 RETURN r.id')
    assert result.values == ((("data",),),)
    assert database.records(filters={"min_nE": 2})[0]["id"] == "data"
    assert database.records(filters={"min_nE": 3}) == []


def test_read_only_path_rejects_mutation(database):
    with AgenticDB(store=database.store) as reader:
        for text in ["DROP TABLE data", "DELETE FROM data", "UPDATE data SET x=1"]:
            with pytest.raises((ValueError, TypeError, SyntaxError)):
                reader.query(text)
            with pytest.raises(PermissionError):
                reader.modify(text)
        with pytest.raises(PermissionError):
            reader.publish("data", graph(), expected_version=0)
    assert database.records() == []


@pytest.mark.parametrize("uri", ["sqlite:///unused", "postgresql://host/db", "mysql://host/db", "snowflake://host/db"])
def test_sql_connections_are_rejected_before_transport(uri, monkeypatch):
    import socket
    monkeypatch.setattr(socket, "create_connection", lambda *a, **k: pytest.fail("unexpected network access"))
    with pytest.raises(ValueError, match="SQL"):
        AgenticDB(uri)
    with pytest.raises(ValueError):
        rcdb.open_store(uri)


def test_hive_workers_run_queries_and_return_complete_results(database):
    database.publish("data", graph(), expected_version=0)
    h = Hive("records-test")
    names = database.attach_to_hive(h)
    assert "db.query" in names and "db.modify" in names
    result = h.invoke("db.query", {"query": 'FROM RCDB_VERSION($db,"data",1) RETURN COUNT(CELLS(1))'})
    assert isinstance(result, Result) and result.values == (2,)
    assert Result.from_bytes(result.to_bytes()).to_bytes() == result.to_bytes()


def test_custom_file_import_export_preserves_complete_source(database, tmp_path):
    value = graph()
    database.publish("data", value, expected_version=0)
    for fmt, suffix in [("rcbd", ".rcbd"), ("safetensors", ".safetensors"), ("hdf5", ".h5")]:
        path = tmp_path / ("source" + suffix)
        database.export_file("data", path, format=fmt)
        database.import_file(path, fmt, expected_version=0, format=fmt)
        assert object_digest(database.extract(fmt)) == object_digest(value)


def test_exact_result_fields_survive_rcdb_and_cold_read(database, tmp_path):
    expected = Result(({"alternatives": ((Fraction(1,3), Fraction(2,3)), (Fraction(2,3), Fraction(1,3))),
                        "unknown": None},))
    database.publish("family", expected, expected_version=0)
    with AgenticDB("local://" + str(tmp_path / "records")) as reader:
        assert reader.extract("family").to_bytes() == expected.to_bytes()
