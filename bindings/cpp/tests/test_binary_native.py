"""Packed canonical truth gates retain all lanes, owners and exact fitting ties."""
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest
from rexgraph_cpp import (
    CancellationToken,
    ExecutionCancelled,
    NativeBinaryGate,
    NativeBinaryInput,
    binary_build_info,
    build_info,
    fit_binary_gate,
)


def packed(values):
    return np.packbits(np.asarray(values, dtype=np.uint8), bitorder="little").tobytes()


@pytest.mark.parametrize("count", [0, 1, 7, 8, 9, 63, 64, 65, 127, 128, 129, 513])
@pytest.mark.parametrize("table", range(16))
def test_all_truth_tables_and_partial_states_match_independent_dense_oracle(count, table):
    a = tuple((i // 2) % 2 for i in range(count)); b = tuple(i % 2 for i in range(count))
    labels = tuple((table >> (2 * x + y)) & 1 for x, y in zip(a, b, strict=True))
    owner = NativeBinaryInput(packed(a), packed(b), count)
    assert (owner.lanes, owner.word_count, owner.first, owner.second) == (count, (count + 63) // 64, packed(a), packed(b))
    assert NativeBinaryGate(table).forward(owner) == packed(labels)
    gate, loss = fit_binary_gate(owner, packed(labels))
    expected = min(t for t in range(16) if all(((t >> (2 * x + y)) & 1) == z for x, y, z in zip(a, b, labels, strict=True)))
    assert gate.truth_table == expected and loss == 0


@pytest.mark.parametrize("count", [3, 4, 9, 65, 129, 1024])
def test_noisy_and_contradictory_observations_choose_exact_minimum_hamming_loss(count):
    a = tuple((i // 2) % 2 for i in range(count)); b = tuple(i % 2 for i in range(count))
    targets = tuple((i // 4) % 2 for i in range(count))
    candidates = tuple((sum(((t >> (2 * x + y)) & 1) != z for x, y, z in zip(a, b, targets, strict=True)), t) for t in range(16))
    loss, table = min(candidates)
    gate, actual = fit_binary_gate(NativeBinaryInput(packed(a), packed(b), count), packed(targets))
    assert (actual, gate.truth_table) == (loss, table)


@pytest.mark.parametrize("first,second,count,maximum", [(b"", b"\0", 1, 8), (b"\xff", b"\0", 1, 8),
    (b"\0", b"\xff", 1, 8), (b"\0\0", b"\0", 1, 8), (b"\0", b"\0", 9, 8),
    (b"\0", b"\0", 1, 0), (b"", b"", 0, 64000001)])
def test_plane_shape_padding_and_admission_reject(first, second, count, maximum):
    with pytest.raises((ValueError, RuntimeError)):
        NativeBinaryInput(first, second, count, max_lanes=maximum)


@pytest.mark.parametrize("value", [True, np.bool_(True), -1, 16, 1.0, 1 << 64])
def test_truth_table_requires_an_explicit_four_bit_integer(value):
    with pytest.raises((ValueError, TypeError, OverflowError)):
        NativeBinaryGate(value)


def test_handles_are_immutable_initialized_and_shared_for_concurrent_native_work():
    input = NativeBinaryInput(b"\x0c", b"\x0a", 4)
    gate = NativeBinaryGate(6)
    with pytest.raises(RuntimeError): input.__init__(b"\0", b"\0", 4)
    with pytest.raises(RuntimeError): gate.__init__(9)
    with pytest.raises(TypeError): type("Mutable", (NativeBinaryInput,), {})
    with pytest.raises(TypeError): type("MutableGate", (NativeBinaryGate,), {})
    with pytest.raises(RuntimeError): _ = NativeBinaryInput.__new__(NativeBinaryInput).lanes
    with pytest.raises(RuntimeError): _ = NativeBinaryGate.__new__(NativeBinaryGate).truth_table
    with ThreadPoolExecutor(max_workers=4) as pool:
        outputs = list(pool.map(lambda _: gate.forward(input), range(32)))
    assert outputs == [b"\x06"] * 32


def test_cancellation_is_checked_between_native_kernel_phases():
    token = CancellationToken(); token.cancel()
    input = NativeBinaryInput(b"\0", b"\0", 1)
    with pytest.raises(ExecutionCancelled): NativeBinaryInput(b"\0", b"\0", 1, cancellation=token)
    with pytest.raises(ExecutionCancelled): NativeBinaryGate(0).forward(input, cancellation=token)
    with pytest.raises(ExecutionCancelled): fit_binary_gate(input, b"\0", cancellation=token)
    with pytest.raises(TypeError): NativeBinaryGate(0).forward(input, cancellation=object())
    with pytest.raises(TypeError): fit_binary_gate(input, bytearray(b"\0"))


@pytest.mark.parametrize("targets", [b"", b"\0\0", b"\x80", b"\xff"])
def test_target_capture_requires_exact_size_and_zero_tail_padding(targets):
    input = NativeBinaryInput(b"\0", b"\0", 1)
    with pytest.raises(ValueError):
        fit_binary_gate(input, targets)


def test_packed_binding_has_its_own_receipt_and_uses_the_same_native_engine():
    receipt = binary_build_info(); original = build_info()
    assert receipt["profile"] == "binary-truth-gate-v1"
    assert receipt["engine_digest"] == original["engine_digest"]
    assert len(bytes.fromhex(original["binding_digest"])) == 32
    assert receipt["binding_digest"] != original["binding_digest"]
