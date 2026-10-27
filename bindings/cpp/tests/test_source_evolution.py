"""Whole native history forks retain old fields and reject guessed migrations."""
import gc
import struct

import pytest


def source():
    from rexgraph_cpp import NativeSource
    return NativeSource.from_declaration({"profile": "declared-role-source-v1",
        "namespace": 9041, "semantic_scale": 2, "max_grade": 2,
        "role_aspects": ("boundary",),
        "types": ({"name": "original relation", "min_arity": 0, "max_arity": 8},),
        "witnesses": (10, 11), "relations": ({"grade": 1, "key": 7, "type": 0,
            "occurrences": (
                {"key": 100, "lower": 10, "exists": True, "negative": True, "magnitude": 1, "aspects": (0,)},
                {"key": 101, "lower": 11, "exists": True, "negative": False, "magnitude": 1, "aspects": (0,)},
                {"key": 102, "lower": 10, "exists": False, "negative": True, "magnitude": 7, "aspects": ()},
                {"key": 103, "lower": 11, "exists": True, "negative": False, "magnitude": 0, "aspects": ()},
            )},),
        "carriers": ({"key": 20, "kind": "source_grade", "grade": 0, "coordinates": 0, "size": 2},),
        "fields": ({"key": 21, "carrier": 20, "entries": ((0, 3), (1, 0)), "support": (0, 1)},),
        "metrics": ({"key": 22, "carrier": 20, "diagonal": (2, 3)},)})


def specification(*, fill=False, assignments=()):
    return {"profile": "source-signature-additions-v1",
        "aspects": ({"name": "available", "role": "visibility", "builder_default": True,
                     "fill": fill, "assignments": assignments},),
        "types": ({"name": "future relation", "min_arity": 0, "max_arity": 9},)}


def last_frame(archive):
    """Read a frame from the documented native series container."""
    assert archive[:8] == b"RFCXS\x01\x00\x00"
    offset = 40
    for _ in range(2):
        offset += 8 + struct.unpack_from("<Q", archive, offset)[0]
    count = struct.unpack_from("<Q", archive, offset)[0]
    assert count > 0
    offset += 8
    for _ in range(count):
        size = struct.unpack_from("<Q", archive, offset)[0]
        frame = archive[offset + 8:offset + 8 + size]
        assert len(frame) == size
        offset += 8 + size
    return frame


def test_append_replays_the_actual_frame_and_rejects_wrong_parent():
    from rexgraph_cpp._source_evolution import NativeSourceEvolution
    parent = NativeSourceEvolution(source().archive)
    expected = parent.add_declarations(specification())
    frame = last_frame(expected.archive)
    child = parent.append(frame)
    assert child.archive == expected.archive
    assert child.source().at(0).archive == parent.source().archive
    assert child.source().field(21).values == parent.source().field(21).values
    with pytest.raises((ValueError, RuntimeError)):
        child.append(frame)
    assert parent.identity == (9041, 0, 0, 0)
    assert child.archive == expected.archive


def test_append_and_declaration_outputs_keep_the_receiving_limits():
    from rexgraph_cpp._source_evolution import NativeSourceEvolution
    parent = NativeSourceEvolution(source().archive)
    child = parent.add_declarations(specification())
    byte_limits = (len(parent.archive), *parent.limits[1:])
    bounded = NativeSourceEvolution(parent.archive, limits=byte_limits)
    for operation in (lambda: bounded.add_declarations(specification()),
                      lambda: bounded.append(last_frame(child.archive))):
        with pytest.raises((ValueError, RuntimeError)):
            operation()
        assert bounded.archive == parent.archive
    one_segment = NativeSourceEvolution(parent.archive,
        limits=(*parent.limits[:4], 1, *parent.limits[5:]))
    admitted = one_segment.append(last_frame(child.archive))
    grandchild = child.add_declarations({"profile": "source-signature-additions-v1",
        "aspects": (), "types": ({"name": "another", "min_arity": 0, "max_arity": 1},)})
    with pytest.raises((ValueError, RuntimeError)):
        admitted.append(last_frame(grandchild.archive))
    assert admitted.archive == child.archive


def test_owning_fork_replay_fields_and_parent_lifetime():
    from rexgraph_cpp._source_evolution import NativeSourceEvolution
    original = source()
    raw = original.archive
    parent = NativeSourceEvolution(raw)
    child = parent.add_declarations(specification())
    assert parent.archive == raw and parent.identity == (9041, 0, 0, 0)
    assert child.identity == (9041, 0, 1, 1)
    assert child.source().at(0).archive == original.at(0).archive
    assert child.source().carriers == original.carriers
    assert child.source().tensors == original.tensors
    assert child.source().field(21).values == original.field(21).values
    assert child.source().field(21).support == original.field(21).support == (0, 1)
    restored = NativeSourceEvolution(child.archive)
    del original, parent
    gc.collect()
    assert child.source().field(21).support == (0, 1)
    assert restored.source().field(21).values == (3, 0)
    with pytest.raises(RuntimeError, match="immutable"):
        restored.__init__(raw)


def test_explicit_assignments_cover_repeated_inactive_and_supported_zero_flags():
    from rexgraph_cpp._source_evolution import NativeSourceEvolution
    owner = NativeSourceEvolution(source().archive)
    assignments = tuple((1, key, key == 102) for key in (100, 101, 102, 103))
    child = owner.add_declarations(specification(fill=None, assignments=assignments))
    assert child.identity[2] == 1
    for bad in (assignments[:-1], assignments + (assignments[0],), assignments + ((1, 999, False),)):
        with pytest.raises((ValueError, RuntimeError)):
            owner.add_declarations(specification(fill=None, assignments=bad))
    assert owner.identity[2] == 0


@pytest.mark.parametrize("mutation", [
    lambda d: d.update(profile="unknown"),
    lambda d: d.update(extra=True),
    lambda d: d["aspects"][0].update(role="existence"),
    lambda d: d["aspects"][0].update(fill=None),
    lambda d: d["aspects"][0].update(fill=0),
    lambda d: d["aspects"][0].update(name="boundary"),
    lambda d: d["aspects"][0].update(name="bad\x00name"),
    lambda d: d["aspects"][0].update(assignments=((True, 100, False),)),
    lambda d: d["types"][0].update(min_arity=10),
    lambda d: d.update(aspects=(), types=()),
])
def test_invalid_addition_does_not_publish_a_child(mutation):
    from rexgraph_cpp._source_evolution import NativeSourceEvolution
    owner = NativeSourceEvolution(source().archive)
    spec = specification()
    mutation(spec)
    with pytest.raises((ValueError, TypeError, RuntimeError)):
        owner.add_declarations(spec)
    assert owner.identity == (9041, 0, 0, 0)


def test_archive_frame_and_resource_admission():
    from rexgraph_cpp._source_evolution import NativeSourceEvolution
    raw = source().archive
    owner = NativeSourceEvolution(raw)
    for archive in (raw + b"trailing", raw[:-1], b"", raw[:32] + b"bad" + raw[35:]):
        with pytest.raises((ValueError, RuntimeError)):
            NativeSourceEvolution(archive)
    with pytest.raises(ValueError):
        NativeSourceEvolution(raw, limits=(1, 1, 1, 1, 1, 1, 1))
    with pytest.raises(ValueError):
        owner.append(b"not a native frame")
    assert owner.archive == raw
