"""C++ field results reopen with exact declarations and fresh, shared owners."""
from fractions import Fraction as Q
import hashlib

import pytest

from rcql import Result
from rexgraph.source_binding import CppSourceReference
from rexgraph.tensor_field import TensorField
from rexgraph.tensor_moment import CoordinatePairing, MomentSpan
from rexgraph.type_accession import CoordinateSpace
from rexgraph.value_codec import pack_value, unpack_value


def fixture():
    cpp = pytest.importorskip("rexgraph_cpp")
    owner = cpp.NativeSource.from_declaration({
        "profile": "declared-exact-source-v1", "namespace": 991, "semantic_scale": 0,
        "max_grade": 1, "types": ({"name": "pair", "min_arity": 2, "max_arity": 2},),
        "witnesses": (), "relations": ({"grade": 1, "key": 100, "type": 0,
            "occurrences": ({"key": 1, "lower": 10, "exists": True, "negative": False, "magnitude": 1},
                            {"key": 2, "lower": 11, "exists": True, "negative": True, "magnitude": 1})},),
        "carriers": (), "fields": (), "metrics": ()})
    ref = CppSourceReference(owner, "native", 3)
    space = CoordinateSpace("original-edge", ("100",))
    powers = CoordinateSpace("polynomial powers", ("0", "1", "2"))
    field = TensorField(space, [[Q(1, 7), Q(0), Q(3)]], (powers,), ref, 1, "chain")
    form = CoordinatePairing(space, space, ((0, 0, Q(2)),))
    return owner, ref, field, MomentSpan(field, field, form).contracted_field()


def test_native_fields_moments_and_references_share_one_reopened_owner():
    owner, ref, field, moment = fixture()
    raw = Result((owner, ref, field, moment)).to_bytes()
    result = Result.from_bytes(raw)
    restored, reference, f, m = result.values
    assert restored is not owner and restored.archive == owner.archive
    assert reference.source is f.source.source is restored
    assert m.dependencies and all(r.source is restored for r in m.dependencies)
    assert f.coefficient_digest == field.coefficient_digest
    assert m.coefficient_digest == moment.coefficient_digest
    assert f.values.tolist() == [[Q(1, 7), Q(0), Q(3)]]
    assert m.values[0].tolist() == [[Q(2, 49), 0, Q(6, 7)], [0, 0, 0], [Q(6, 7), 0, 18]]
    assert result.to_bytes() == raw


@pytest.mark.parametrize("change", ("archive", "coordinates", "namespace", "tag", "reference"))
def test_new_outer_checksum_does_not_authorize_a_changed_native_basis(change):
    owner, ref, field, _ = fixture()
    raw = Result((owner, ref, field)).to_bytes()
    record = unpack_value(raw[37:])
    basis = record["graphs"][0]
    if change == "archive":
        basis["archive"] = basis["archive"][:-1]+bytes([basis["archive"][-1] ^ 1])
    elif change == "coordinates":
        basis["identity"]["coordinates"] = (("10", "11"), ("101",))
    elif change == "namespace":
        basis["identity"]["namespace"] += 1
    elif change == "tag":
        nodes = list(record["values"][1]); nodes[0] = ("graph", 0)
        record["values"] = ("tuple", tuple(nodes))
    else:
        component = record["values"][1][2]
        component[3][0][0]["record_id"] = "foreign"
    body = pack_value(record)
    with pytest.raises(ValueError):
        Result.from_bytes(raw[:5]+hashlib.sha256(body).digest()+body)


def test_native_packet_cell_admission_uses_the_decoded_source():
    from rcql.result_codec import _record_cell_counts
    owner, ref, _, _ = fixture()
    assert _record_cell_counts(Result((owner, ref)).to_bytes()) == {"nV": 2, "nE": 1, "nF": 0}


def test_detached_native_reference_stays_detached():
    _, ref, _, _ = fixture()
    restored = Result.from_bytes(Result((ref.detached(),)).to_bytes()).values[0]
    assert restored.source is None and restored.coefficient_digest == ref.coefficient_digest
