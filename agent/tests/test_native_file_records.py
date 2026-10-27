"""Custom file transport retains whole sources through Agent, Hive and RCDB."""
import io
import struct
import zipfile
from contextlib import closing

import numpy as np
import pytest
from agent.agentic_db import AgenticDB
from agent.native_files import load_native_file
from agent.hive import Hive
from rexgraph import RexGraph
from rexgraph.io import load, save
from rexgraph.object_identity import object_digest


def rcbf_bytes():
    from rexgraph.io.rcbf import MAGIC
    def strings(*values):
        return b"".join(value.encode() + b"\0" for value in values)
    def array(values, dtype):
        return np.asarray(values, dtype=dtype).tobytes()
    return b"".join([
        struct.pack("<8s7IQ", MAGIC, 5, 0, 4, 3, 0, 1, 0, 0),
        strings("a", "b", "c", "d"), array([2, 1, 2], "<i4"), array([0, 1, 3], "<i4"),
        array([1, 2, 3], "<f8"), array([1, 1, 1], "<f8"), array([1, 1, 1], "u1"), array([0, 0, 0], "u1"),
        strings("relation-0", "self-loop", "relation-2"), strings("ordered", "ordered", "ordered"),
        np.zeros((3, 4), dtype="<f8").tobytes(), array([3, 2, 2], "<u2"),
        array([2, 0, 1, 1, 1, 2, 3], "<i4"), strings("activity"), array([0, 0, 3], "<f8"),
        struct.pack("<I", 0)])


def zipped_bundle(path, *, nested=False):
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, "w", zipfile.ZIP_DEFLATED) as archive:
        for member in sorted(path.rglob("*")):
            if member.is_file():
                name = member.relative_to(path).as_posix()
                archive.write(member, ("source.rcbd/" if nested else "") + name)
    return stream.getvalue()


def test_rcbf_repeated_participants_signal_zero_and_fields_roundtrip(tmp_path):
    source = tmp_path / "source.rcbf"
    source.write_bytes(rcbf_bytes())
    expected = load(source)
    assert expected.boundary_idx.tolist() == [2, 0, 1, 1, 1, 2, 3]
    with AgenticDB("local://" + str(tmp_path / "records"), writable=True) as database:
        database.import_file(source, "source", expected_version=0, format="rcbf")
        hive = Hive("file-records")
        database.attach_to_hive(hive)
        result = hive.invoke("db.query", {"query":
            'FROM RCDB_VERSION($db,"source",1) RETURN COUNT(CELLS(1))'})
        assert result.values == (3,)
        for format, suffix in [("rcbf", ".rcbf"), ("rcbd", ".rcbd"), ("safetensors", ".safetensors"), ("hdf5", ".h5")]:
            target = tmp_path / ("copy" + suffix)
            database.export_file("source", target, format=format)
            assert object_digest(load(target, format=format)) == object_digest(expected)
    with AgenticDB("local://" + str(tmp_path / "records")) as reopened:
        assert object_digest(reopened.extract("source")) == object_digest(expected)


@pytest.mark.parametrize("nested", [False, True])
def test_rcbd_directory_zip_transport_is_owning_and_lossless(tmp_path, nested):
    source = tmp_path / "source.rcbd"
    expected = RexGraph.from_cells([4, [[0, 1, 2], [2, 3]]])
    save(source, expected)
    packed = tmp_path / "upload.zip"
    packed.write_bytes(zipped_bundle(source, nested=nested))
    value = load_native_file(packed, format="rcbd")
    assert object_digest(value) == object_digest(expected)
    # The temporary extraction is gone; every array and subsequent serialization still works.
    save(tmp_path / "copied.safetensors", value)
    assert object_digest(load(tmp_path / "copied.safetensors")) == object_digest(expected)


@pytest.mark.parametrize("name", ["../escape", "/escape", "source.rcbd/../../escape", "a\\escape"])
def test_rcbd_transport_refuses_members_outside_bundle(tmp_path, name):
    path = tmp_path / "bad.zip"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr(name, b"data")
    with pytest.raises(ValueError, match="bundle"):
        load_native_file(path, format="rcbd")


def test_rcbd_transport_refuses_symlinks_and_duplicates(tmp_path):
    path = tmp_path / "bad.zip"
    entry = zipfile.ZipInfo("linked")
    entry.external_attr = 0o120777 << 16
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr(entry, "outside")
    with pytest.raises(ValueError, match="regular"):
        load_native_file(path, format="rcbd")


def test_workspace_backup_restores_native_history_and_results(tmp_path, monkeypatch):
    from agent.server import persistence
    from rcdb import LocalStore
    from fractions import Fraction
    monkeypatch.setenv("REXGRAPH_CONFIG_DIR", str(tmp_path / "config"))
    persistence.save_query("workspace", "operator", "RCQL", "exact", [{"value": Fraction(1, 3)}])
    persistence.save_activity("workspace", [("operator", "query", "source", 1)])
    persistence.save_activity("workspace", [("operator", "query", "source", 1), ("operator", "read", "result", 2)])
    source = persistence._ws_dir("workspace")
    (source / "state.db").write_bytes(b"preserved legacy data")
    target = tmp_path / "backup"
    persistence.export_workspace("workspace", str(target), fmt="rcdb")
    assert (source / "state.db").read_bytes() == b"preserved legacy data"
    assert not (target / "state.db").exists()
    with closing(LocalStore(str(target / "state.rcdb"))) as store:
        assert len(store.history("activity")) == 2
        result = store.read_record("activity", version=1).value
        assert result.values == ([("operator", "query", "source", 1)],)
        records = store.query(tags_all=["agent-query"])
        assert len(records) == 1
        value = store.read_record(records[0].id).value.values[0]
        assert value["results"] == [{"value": Fraction(1, 3)}]
