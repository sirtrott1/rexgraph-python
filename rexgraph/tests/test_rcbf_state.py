"""Complete RCBF state, legacy migration and bounded stream admission."""
from fractions import Fraction
import hashlib
import struct

import numpy as np
import pytest

from rexgraph import Absent, Relations, RexGraph, TemporalRex, VertexTable
from rexgraph.io import FileCatalog, load, save
from rexgraph.io.rcbf import MAGIC, RCBFFormatError, load_rcbf, save_rcbf
from rexgraph.object_identity import object_digest
from rexgraph.state import to_state
from rexgraph.value_codec import pack_value


def graph():
    relations = Relations.from_supports(
        [[0, 1, 2], [1, 2], [2, 2]],
        vertices=VertexTable(("a", "b", "c", "isolated")),
        weights=[Fraction(2, 3), Absent, 0], signs=[-1, 1, 1],
        relation_ids=["branch", "pair", "loop"],
        attributes={1: {0: {"payload": (2**512, Fraction(1, 17), b"\x00\xff", None)}}},
    )
    return RexGraph.from_relations(relations, c_channel="count")


def wire(state, entries=None, *, flags=0):
    header = pack_value(state.header)
    entries = entries if entries is not None else sorted(state.tensors.items())
    parts = [MAGIC, struct.pack("<IIQQ", 6, flags, len(header), len(entries)), header]
    for name, value in entries:
        encoded = name.encode("utf-8")
        value = pack_value(value)
        parts.extend([struct.pack("<QQ", len(encoded), len(value)), encoded, value])
    body = b"".join(parts)
    return body + hashlib.sha256(body).digest()


def test_complete_state_is_deterministic_and_matches_the_declared_wire(tmp_path):
    original = graph()
    path = tmp_path / "graph.rcbf"
    assert save_rcbf(path, original) == path
    assert path.read_bytes() == wire(to_state(original))
    restored = load(path)
    assert object_digest(restored) == object_digest(original)
    assert restored.relation_keys == original.relation_keys
    assert restored.relation_supports() == original.relation_supports()
    assert restored.get_metadata(1, 0, "payload") == (2**512, Fraction(1, 17), b"\x00\xff", None)
    entry = FileCatalog([tmp_path]).info("root0/graph.rcbf")
    assert entry.kind == "rcbf"


def test_temporal_rcbf_preserves_history_instead_of_selecting_a_snapshot(tmp_path):
    original = TemporalRex([])
    original.append_snapshot(graph(), at=Fraction(1, 3))
    original.append_snapshot(RexGraph.from_graph([0], [1]), at=Fraction(7, 3))
    path = tmp_path / "history.rcbf"
    save(path, original)
    restored = load(path)
    assert isinstance(restored, TemporalRex)
    assert object_digest(restored) == object_digest(original)
    assert list(restored._times) == list(original._times)
    for index in range(2):
        assert object_digest(restored.at(index)) == object_digest(original.at(index))


def test_stream_byte_limits_precede_decode_and_failed_publication_preserves_old_file(tmp_path, monkeypatch):
    path = tmp_path / "graph.rcbf"
    save(path, graph())
    before = path.read_bytes()
    import rexgraph.value_codec as codec
    def refuse(data):
        pytest.fail("decoded a stream before its total byte admission")
    monkeypatch.setattr(codec, "unpack_value", refuse)
    with pytest.raises(RCBFFormatError, match="max_bytes"):
        load_rcbf(path, max_bytes=len(before)-1)
    with pytest.raises(ValueError, match="max_bytes"):
        save_rcbf(path, graph(), max_bytes=64)
    assert path.read_bytes() == before
    assert sorted(p.name for p in tmp_path.iterdir()) == ["graph.rcbf"]


@pytest.mark.parametrize("case", ["duplicate", "unordered", "not-array", "flags", "header", "oversize-frame"])
def test_independently_framed_invalid_states_are_rejected(tmp_path, case):
    state = to_state(graph())
    entries = sorted(state.tensors.items())
    if case == "duplicate":
        raw = wire(state, entries + [entries[-1]])
    elif case == "unordered":
        raw = wire(state, list(reversed(entries)))
    elif case == "not-array":
        raw = wire(state, [(entries[0][0], 42)] + entries[1:])
    elif case == "flags":
        raw = wire(state, flags=1)
    elif case == "header":
        state.header["digest"] = "0"*64
        raw = wire(state)
    else:
        header = pack_value(state.header)
        body = MAGIC + struct.pack("<IIQQ", 6, 0, len(header), 1) + header
        body += struct.pack("<QQ", 1, 2**63) + b"a"
        raw = body + hashlib.sha256(body).digest()
    path = tmp_path / "invalid.rcbf"
    path.write_bytes(raw)
    with pytest.raises(RCBFFormatError):
        load_rcbf(path)


def test_truncated_corrupt_and_trailing_streams_reject(tmp_path):
    raw = wire(to_state(graph()))
    cases = [raw[:cut] for cut in [0, 7, 8, 11, 12, 31, 32, len(raw)//2, len(raw)-32, len(raw)-1]]
    altered = bytearray(raw)
    altered[-1] ^= 1
    cases.append(bytes(altered))
    body = raw[:-32] + b"unclaimed"
    cases.append(body + hashlib.sha256(body).digest())
    path = tmp_path / "invalid.rcbf"
    for case in cases:
        path.write_bytes(case)
        with pytest.raises(RCBFFormatError):
            load_rcbf(path)


def test_rcbf_does_not_synthesize_numeric_projections_or_accept_arbitrary_objects(tmp_path, monkeypatch):
    original = graph()
    def refuse(*args, **kwargs):
        pytest.fail("serialization computed a graph projection")
    monkeypatch.setattr(RexGraph, "to_json", refuse)
    save_rcbf(tmp_path / "graph.rcbf", original)
    with pytest.raises(TypeError, match="RexGraph or TemporalRex"):
        save_rcbf(tmp_path / "object.rcbf", np.zeros((2, 2)))
