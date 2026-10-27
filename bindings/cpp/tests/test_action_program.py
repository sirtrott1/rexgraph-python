"""Native action order, signed fields, current evidence and cold ownership."""
import base64
from concurrent.futures import ThreadPoolExecutor
from fractions import Fraction as Q
import json
import os
from pathlib import Path

import pytest

native = pytest.importorskip("rexgraph_cpp._action_program")


def assets():
    data = json.loads(Path(__file__).with_name("action_program_fixture.json").read_text())
    root = os.environ.get("REXGRAPH_ACTION_FIXTURE_DIR")
    if root is not None:
        return {name: (Path(root) / name).read_bytes() for name in data}
    return {name: base64.b64decode(value) for name, value in data.items()}


def recipe(kind="field"):
    out = dict(version=1, namespace=790, sequence=7, identity=kind,
        meaning="exact ordered action on original source coordinates", output="field", result=3,
        metric=0, channels=(), reference=None,
        nodes=[dict(id=1, opcode="tensor", selector=1200, operands=(), scale=1),
               dict(id=2, opcode="tensor", selector=1202, operands=(), scale=1),
               dict(id=3, opcode="then", selector=0, operands=(1, 2), scale=1)])
    if kind == "reverse":
        out["nodes"][0]["selector"], out["nodes"][1]["selector"] = 1202, 1200
    elif kind in {"signed", "cancellation"}:
        out.update(output="moments", result=0, metric=1100, reference=3,
            channels=({"label": "first", "action": 1}, {"label": "second", "action": 2}))
        out["nodes"][1]["selector"] = 1201
        out["nodes"][2]["opcode"] = "sum"
        if kind == "cancellation":
            out["nodes"][0].update(opcode="identity", selector=10)
            out["nodes"][1].update(opcode="scale", selector=0, operands=(1,), scale=-1)
    return out


def policy(r, known=10):
    def dependency(i):
        return (781, i, 1, 0)
    return dict(request=(789, 1, 1), observation=(781, 1, 1), frontier=(100, known, 1), purpose="training",
        selected=tuple(dependency(i) for i in (1, 2, 3, 4)), fields=(dependency(1),), law=None,
        coordinates=dependency(2), selection=dependency(3), rules=(dependency(4),),
        programs=({"identity": r["identity"], "language": "rexgraph.native-action.v1",
                   "source": native.action_description(r), "rule": dependency(4)},))


def fixture(kind="field", *, graph="evidence.bin", known=10, core=False):
    data, r = assets(), recipe(kind)
    if core:
        from rexgraph.action_program import ActionEvidence, ActionProgram
        program = ActionProgram.create(data["source.rfcx"], data["endpoint.rfct"], r, 1000, ((790, 7),))
        evidence = ActionEvidence.create(data[graph], policy(r, known))
    else:
        program = native.NativeActionProgram.prepare(data["source.rfcx"], data["endpoint.rfct"], r, 1000, ((790, 7),))
        evidence = native.NativeActionEvidence.from_envelope(data[graph], policy(r, known))
    return program, evidence


def run(program, evidence, availability=((790, 7),), *, core=False, **options):
    return (program.execute(evidence, availability, **options) if core
            else program.execute(evidence, availability, 1000, **options))


@pytest.mark.parametrize("kind,values", [("field", ((1, Q(1)),)), ("reverse", ((0, Q(-1)),))])
def test_native_noncommuting_actions_retain_full_source_carrier(kind, values):
    program, evidence = fixture(kind)
    result = run(program, evidence).inspect()
    field = result["fields"][0]
    assert field["entries"] == values and field["shape"] == (2,) and field["support_count"] == 2
    assert field["carrier"] == (790, 7, 10) and len(field["dependencies"]) == 3
    assert (781, 5, 1, 0, 1) in result["admitted_dependencies"]


@pytest.mark.parametrize("kind,totals", [("signed", (5, -2, 3, 0)), ("cancellation", (6, -6, 0, 0))])
def test_native_signed_moments_keep_components_and_supported_zero(kind, totals):
    program, evidence = fixture(kind)
    result = run(program, evidence, pairs=((0, 1),)).inspect()
    assert result["totals"] == totals and result["matches_reference"]
    assert len(result["components"]) == 2 and len(result["fields"]) == 6
    assert result["interactions"][0]["doubled"] == totals[1]
    assert all(f["support_count"] == 2 for f in result["fields"])
    assert result["fields"][2]["entries"] and result["fields"][3]["entries"]
    if kind == "cancellation":
        assert result["fields"][1]["entries"] == () and result["fields"][-1]["entries"] == ()
    assert result["steps"] > 0 and result["retained_sparse_entries"] > 0


@pytest.mark.parametrize("availability", [(), ((790, 6),), ((791, 7),), ((790, 7), (790, 8))])
def test_current_publication_cutoff_required_on_every_native_execution(availability):
    program, evidence = fixture()
    with pytest.raises((ValueError, RuntimeError)):
        run(program, evidence, availability)
    assert run(program, evidence).inspect()["fields"][0]["entries"] == ((1, Q(1)),)


def test_future_enabling_dependency_and_evaluation_partition_block_before_work():
    program, evidence = fixture(graph="late.bin")
    with pytest.raises(RuntimeError, match="native execution has not started"):
        run(program, evidence)
    _, now = fixture(graph="late.bin", known=20)
    assert run(program, now).inspect()["admitted_dependencies"][-1][-1] == 1
    _, evaluation = fixture(graph="evaluation.bin")
    with pytest.raises(RuntimeError, match="native execution has not started"):
        run(program, evaluation)


def test_native_lease_budget_cancellation_inspection_and_concurrent_lifetimes():
    program, evidence = fixture("signed")
    low = native.NativeActionLease.create(limits=(1048576, 16000000, 16000000, 0, 4096))
    with pytest.raises((ValueError, RuntimeError)):
        run(program, evidence, lease=low)
    cancelled = native.NativeActionLease.create(); cancelled.cancel()
    with pytest.raises(RuntimeError):
        run(program, evidence, lease=cancelled)
    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(lambda _, program=program, evidence=evidence:
            program.execute(evidence, ((790, 7),), 1001, pairs=((0, 1),)), range(16)))
    del program, evidence
    assert all(r.inspect()["totals"] == (5, -2, 3, 0) for r in results)
    assert all(r.inspect()["fields"][0]["dependencies"][0][-1] == 1001 for r in results)
    with pytest.raises((ValueError, RuntimeError)):
        results[0].inspect(max_entries=0)


@pytest.mark.parametrize("change", ["cycle", "unused", "unknown", "float", "wrong-sequence", "wrong-selector"])
def test_invalid_native_action_declarations_reject(change):
    data, r = assets(), recipe()
    if change == "cycle": r["nodes"][2]["operands"] = (3, 1)
    elif change == "unused": r["nodes"].append(dict(id=4, opcode="identity", selector=10, operands=(), scale=1))
    elif change == "unknown": r["nodes"][0]["opcode"] = "dense-fit"
    elif change == "float": r["nodes"][0]["scale"] = 1.0
    elif change == "wrong-sequence": r["sequence"] = 8
    else: r["nodes"][0]["selector"] = 1000
    with pytest.raises((ValueError, TypeError, RuntimeError)):
        native.NativeActionProgram.prepare(data["source.rfcx"], data["endpoint.rfct"], r, 1000, ((790, 7),))


def test_exact_rule_description_is_required_in_actual_evidence_request():
    program, _ = fixture()
    bad = policy(recipe()); bad["programs"][0]["source"] += " changed"
    evidence = native.NativeActionEvidence.from_envelope(assets()["evidence.bin"], bad)
    with pytest.raises(ValueError, match="exact selected evidence rule"):
        run(program, evidence)


def test_core_immutable_cold_program_and_output_require_fresh_authority():
    from rexgraph.action_program import ActionEvidence, ActionOutput, ActionProgram
    from rexgraph.value_codec import pack_value, unpack_value
    program, evidence = fixture("cancellation", core=True)
    result = run(program, evidence, core=True, pairs=((0, 1),))
    packet = result.to_bytes()
    reopened_evidence = ActionEvidence.from_bytes(evidence.to_bytes())
    restored = ActionOutput.replay(packet, evidence=reopened_evidence, availability=((790, 7),))
    assert restored.to_bytes() == packet
    assert ActionProgram.from_bytes(program.to_bytes(), availability=((790, 7),)).to_bytes() == program.to_bytes()
    with pytest.raises(AttributeError): program._packet = b"changed"
    with pytest.raises(TypeError): program.execute(result, ((790, 7),))
    with pytest.raises(TypeError): ActionOutput.replay(packet, evidence=None, availability=((790, 7),))
    with pytest.raises(ValueError): ActionOutput.replay(packet, evidence=evidence, availability=((790, 6),))
    forged = unpack_value(packet); forged["snapshot"]["totals"] = (0, 0, 0, 0)
    with pytest.raises(ValueError, match="semantic replay"):
        ActionOutput.replay(pack_value(forged), evidence=evidence, availability=((790, 7),))
    old = unpack_value(program.to_bytes()); old["implementation"]["binding"] = "0" * 64
    with pytest.raises(ValueError, match="implementation differs"):
        ActionProgram.from_bytes(pack_value(old), availability=((790, 7),))


@pytest.mark.parametrize("suffix", ["rcbd", "json", "safetensors", "h5"])
def test_program_evidence_and_output_file_roundtrips_need_real_replay(tmp_path, suffix):
    from rexgraph.io import load, save
    from rexgraph.io.action_record import action_record, read_action_record
    program, evidence = fixture("signed", core=True)
    output = run(program, evidence, core=True, pairs=((0, 1),))
    for index, value in enumerate((program, evidence, output)):
        path = tmp_path / (str(index) + "." + suffix)
        save(path, action_record(value))
        restored = read_action_record(load(path), evidence=evidence, availability=((790, 7),))
        assert restored.to_bytes() == value.to_bytes()


def test_historical_evidence_keeps_original_bytes_and_refuses_unknown_pins():
    original=assets()['evidence.bin']
    evidence=native.NativeActionEvidence.from_envelope(original,policy(recipe()))
    assert evidence.archive()==original
    # Walk the declared model child, changing its pin while resealing both levels.
    raw=bytearray(original);position=8;size=0;shift=0
    while True:
        octet=raw[position];position+=1;size|=(octet&127)<<shift
        if octet<128:break
        shift+=7
    child=bytearray(raw[position:position+size])
    assert child[:6]==b'RXLMOD'
    child[9:73]=b'0'*64
    def seal(value):
        h=1469598103934665603
        for octet in value[:-8]:h=((h^octet)*1099511628211)&((1<<64)-1)
        value[-8:]=h.to_bytes(8,'little')
    seal(child);raw[position:position+size]=child;seal(raw)
    with pytest.raises(ValueError,match='core pin/version mismatch'):native.NativeActionEvidence.from_envelope(bytes(raw),policy(recipe()))
