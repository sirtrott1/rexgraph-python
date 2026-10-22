"""Owning native results retain source uses and complete postcondition packets."""
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy

import pytest
from test_relational_field_admission import context, prepared, publish
from test_relational_field_evidence import (
    HELD_INPUT,
    learning_declaration,
    matrix,
    read_matrix,
    reference,
    tensor,
)

from rexgraph_cpp import (
    NativeInputUseAsset,
    NativeTaskOutcome,
    task_binding_origin_payload,
    task_goal_origin_payload,
    task_outcome_build_info,
)


def contract(response, asset, owner, origins, target, *, key=900, rectangular=False, construct=True):
    count = len(target)
    direction_count = count + int(rectangular)
    directions = tuple(range(10001, 10001 + direction_count))
    accession = [[int(i == j) for j in range(count)] for i in range(direction_count)]
    reconstruction = [[int(i == j) for j in range(direction_count)] for i in range(count)]
    if rectangular:
        accession[-1][0] = accession[-1][1] = 1
        target = (*target, target[0] + target[1] + 1)
    metric = [[0] * direction_count for _ in range(direction_count)]
    offset = 0
    for stalk in learning_declaration()["stalks"]:
        if stalk["cell"][0] != 0:
            continue
        for i, j, q in stalk["metric"]["entries"]:
            metric[offset + i][offset + j] = q
        offset += len(stalk["directions"])
    assert offset == count
    task = {"profile": "rexgraph.relational-task-postcondition.v1", "key": key, "revision": 1,
        "use": asset.uses()[0]["key"], "origin": 0, "basis": key + 1, "directions": directions,
        "metric": matrix(metric), "target": tensor(target)}
    binding = {"profile": "rexgraph.relational-worker-binding.v1", "key": key + 2, "revision": 1,
        "worker": 41, "machine": 51, "operation": 61, "origin": 0,
        "interface": {"key": key + 3, "branch": key + 4, "cells": tuple((2, (0, i)) for i in range(3))},
        "view": {"profile": "rexgraph.relational-field-view.v1", "key": key + 5, "basis": task["basis"],
            "directions": directions, "accession": matrix(accession), "reconstruction": matrix(reconstruction), "physical": None}}
    task["origin"] = key + 6
    binding["origin"] = key + 7
    origins[task["origin"]] = publish(task["origin"], owner, "prospective", task_goal_origin_payload(task, asset, response))
    origins[binding["origin"]] = publish(binding["origin"], owner, "maps", task_binding_origin_payload(binding, response))
    authority = context(origins)
    outcome = NativeTaskOutcome(task, asset, binding, response, authority) if construct else None
    return outcome, task, binding, authority


def fixture(*, delta=False, rectangular=False):
    owner, program, available, row, origins, authority = prepared()
    response = available.execute_inputs((row,), authority)
    target = list(reference(HELD_INPUT, (0, 0, 0, 0, 0))[0])
    if delta:
        target[0] += 1
    asset = NativeInputUseAsset.from_archive(row["asset"])
    outcome, task, binding, authority = contract(response, asset, owner, origins, target, rectangular=rectangular)
    return owner, response, asset, origins, outcome, task, binding, authority


@pytest.mark.parametrize("delta,status", [(False, "satisfied"), (True, "refuted")])
def test_executed_whole_field_verdict_and_exact_residual(delta, status):
    owner, response, asset, origins, outcome, task, binding, authority = fixture(delta=delta)
    info = outcome.inspect(authority)
    assert info["status"] == status and info["executed_response"]
    assert info["worker"] == binding["worker"] and info["use"] == task["use"]
    proof = outcome.certificate(authority)
    expected = [-1 if delta else 0] + [0] * 6
    assert proof["residual"] == tensor(expected)
    assert len(read_matrix(proof["directions"])) == 7
    assert outcome.definition(authority)["target"] == task["target"]
    assert outcome.response(authority).archive() == response.archive()
    assert any(image == asset.source_archive() for image in outcome.dependency_archives(authority))
    assert task_outcome_build_info()["scalar_policy"] == "gmp-rational-exact"


def test_open_and_universally_satisfied_families_remain_prospective():
    from test_relational_field_evidence_admission import independent_input, learned
    from test_relational_field_evidence_admission import prepared as root
    owner, program, evidence, origins = root()
    row = independent_input(owner, program, origins, 81, HELD_INPUT)
    response = evidence.execute_inputs((row,), context(origins))
    asset = NativeInputUseAsset.from_archive(row["asset"])
    target = reference(HELD_INPUT, (0, 0, 0, 0, 0))[0]
    outcome, task, binding, authority = contract(response, asset, owner, origins, target)
    assert outcome.inspect(authority)["status"] == "open"
    assert not outcome.inspect(authority)["executed_response"]
    assert read_matrix(outcome.certificate(authority)["directions"])[0]
    from rcql import TaskOutcome
    wrapper = TaskOutcome.from_native(outcome, authority)
    assert wrapper.verdict(authority) is None
    owner, program, parent, child, origins, theta = learned()
    row = independent_input(owner, program, origins, 81, HELD_INPUT)
    response = child.execute_inputs((row,), context(origins))
    outcome, _, _, authority = contract(response, NativeInputUseAsset.from_archive(row["asset"]), owner,
        origins, reference(HELD_INPUT, theta)[0])
    assert outcome.inspect(authority)["status"] == "satisfied"
    assert not outcome.inspect(authority)["executed_response"]
    assert TaskOutcome.from_native(outcome, authority).verdict(authority) is None
    assert response.evidence().evidence().identity()["free_parameter_count"] == 2


def test_rectangular_faithful_view_retains_off_image_refutation():
    _, _, _, _, outcome, _, _, authority = fixture(rectangular=True)
    assert outcome.inspect(authority)["status"] == "refuted"
    assert outcome.certificate(authority)["residual"] == tensor([0] * 7 + [-1])
    assert len(read_matrix(outcome.definition(authority)["accession"])) == 8


@pytest.mark.parametrize("key", [1, 2, 3, 4, 5, 6, 7, 906, 907])
def test_current_origin_cutoffs_and_roles_gate_warm_and_cold_use(key):
    _, _, _, origins, outcome, _, _, authority = fixture()
    image = outcome.archive(authority)
    denied = context(origins, remove=(key,))
    with pytest.raises(ValueError):
        outcome.inspect(denied)
    with pytest.raises(ValueError):
        outcome.archive(denied)
    with pytest.raises(ValueError):
        NativeTaskOutcome.from_archive(image, denied)
    original = origins[key]
    origins[key] = publish(key, __import__('rexgraph_cpp').NativeSource(original.source_archive()), "target-value", original.payload())
    with pytest.raises(ValueError):
        NativeTaskOutcome.from_archive(image, context(origins))


def test_canonical_streaming_cold_adoption_callback_errors_and_owned_lifetimes():
    owner, response, asset, origins, outcome, task, binding, authority = fixture()
    image = outcome.archive(authority)
    parts = []
    assert outcome.stream(parts.append, authority) == len(image)
    assert b"".join(parts) == image
    with pytest.raises(OSError, match="short write"):
        outcome.stream(lambda chunk: len(chunk) - 1, authority)
    def stopped():
        raise LookupError("stop exact native work")
    with pytest.raises(LookupError, match="stop exact"):
        outcome.archive(authority, check=stopped)
    from rexgraph_cpp import NativeSource
    copied = {key: publish(key, NativeSource(value.source_archive()), value.inspect()["role"], value.payload())
              for key, value in origins.items()}
    cold_authority = context(copied)
    with pytest.raises(ValueError, match="live value origin owner"):
        outcome.admit(cold_authority)
    restored = NativeTaskOutcome.from_archive(image, cold_authority)
    assert restored.archive(cold_authority) == image
    with ThreadPoolExecutor(max_workers=3) as pool:
        assert list(pool.map(restored.archive, [cold_authority] * 3)) == [image] * 3
    with pytest.raises(TypeError, match="immutable"):
        outcome.__init__(task, asset, binding, response, authority)
    del owner, response, asset, origins, copied
    assert restored.archive(cold_authority) == image
    with pytest.raises(RuntimeError):
        NativeTaskOutcome.__new__(NativeTaskOutcome).archive(cold_authority)


def test_goal_binding_mutation_and_cold_tampering_are_rejected():
    _, response, asset, origins, outcome, task, binding, authority = fixture()
    changed = deepcopy(binding)
    changed["machine"] += 1
    with pytest.raises(ValueError):
        NativeTaskOutcome(task, asset, changed, response, authority)
    raw = bytearray(outcome.archive(authority))
    raw[-1] ^= 1
    with pytest.raises(ValueError):
        NativeTaskOutcome.from_archive(bytes(raw), authority)
    with pytest.raises(ValueError):
        NativeTaskOutcome.from_archive(outcome.archive(authority), authority, limits=(1, 1, 1, 1))
