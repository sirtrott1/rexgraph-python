"""Retained checkpoints check their complete content before owner reuse."""
from fractions import Fraction as Q
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest

from rexgraph import model_codec as codec
from rexgraph.model_state import ModelState, freeze_tree
from rexgraph.tensor_field import FieldSource
from rexgraph.type_accession import CoordinateSpace


@pytest.fixture(autouse=True)
def clear_checkpoints():
    with codec._checkpoint_lock:
        codec._checkpoint_cache.clear()
    yield
    with codec._checkpoint_lock:
        codec._checkpoint_cache.clear()


def model(index=0):
    return ModelState("checkpoint-fixture", "1", FieldSource(None, state_digest="a"*64),
        CoordinateSpace("rows", ("first",)), (), {"index": index},
        {"exact": np.asarray([Q(1,3), Q(2,7)], dtype=object),
         "data": np.asarray([1,2], dtype=np.int64)}, "rational")


def test_checked_receipt_copies_reuse_the_owner_without_decoding(monkeypatch):
    value=model();checkpoint=codec._model_checkpoint(value)
    retained=freeze_tree(checkpoint)
    def forbidden(_):
        raise AssertionError("unchanged checkpoint was decoded again")
    monkeypatch.setattr(codec, "unpack_model", forbidden)
    assert codec._checkpoint_model(retained) is value
    assert codec._model_checkpoint(value) is checkpoint
    assert all(not a.flags.writeable for a in checkpoint.values())


def test_public_packing_and_decoding_still_return_detached_values():
    value=model();retained=codec._model_checkpoint(value)
    first=codec.pack_model(value);second=codec.pack_model(value)
    for key in first:
        assert not np.shares_memory(first[key],second[key])
        assert not np.shares_memory(first[key],retained[key])
        assert np.array_equal(first[key],retained[key])
    decoded=codec.unpack_model(first)
    assert decoded is not value and decoded.coefficient_digest==value.coefficient_digest
    decoded.payload["data"].flags.writeable=True;decoded.payload["data"][0]=99
    value.check_state()
    with pytest.raises(ValueError):decoded.check_state()


@pytest.mark.parametrize("change", ["data", "fraction", "dtype", "shape", "spec", "seal", "missing", "extra"])
def test_supplied_checkpoint_changes_cannot_use_a_warm_owner(change):
    value=model();codec._model_checkpoint(value);checkpoint=codec.pack_model(value)
    data=next(a for a in checkpoint.values() if a.dtype==np.int64)
    if change=="data":data[0]=99
    elif change=="fraction":
        array=next(a for a in checkpoint.values() if a.dtype.hasobject)
        array[0]=Q(99)
    elif change=="dtype":
        key=next(k for k,a in checkpoint.items() if a is data)
        checkpoint[key]=data.view(np.uint64)
    elif change=="shape":
        key=next(k for k,a in checkpoint.items() if a is data)
        checkpoint[key]=data.reshape(1,2)
    elif change=="spec":checkpoint["spec"][0]^=1
    elif change=="seal":checkpoint["model_digest"][0]^=1
    elif change=="missing":checkpoint.pop("spec")
    else:checkpoint["unclaimed"]=np.asarray([0],np.uint8)
    if change=="shape":
        # The decoder accepts a storage reshape with the same declared shape.
        assert codec._checkpoint_model(checkpoint).coefficient_digest==value.coefficient_digest
    else:
        with pytest.raises(ValueError):codec._checkpoint_model(checkpoint)


def test_equal_claimed_seals_do_not_merge_different_checkpoint_content():
    first=model(1);second=model(2)
    codec._model_checkpoint(first);tensors=codec.pack_model(second)
    tensors["model_digest"][:]=np.frombuffer(first.coefficient_digest.encode(),np.uint8)
    with pytest.raises(ValueError,match="identity mismatch"):codec._checkpoint_model(tensors)


@pytest.mark.parametrize("change", ["source", "axis", "array", "fraction", "seal"])
def test_owner_content_is_checked_even_when_the_checkpoint_is_unchanged(change):
    value=model();tensors=codec._model_checkpoint(value)
    if change=="source":object.__setattr__(value.source,"record_id","foreign")
    elif change=="axis":object.__setattr__(value.space,"keys",("foreign",))
    elif change=="array":value.payload["data"].flags.writeable=True;value.payload["data"][0]=99
    elif change=="fraction":value.payload["exact"][0]._numerator=99
    else:object.__setattr__(value,"_digest","0"*64)
    with pytest.raises(ValueError):codec._checkpoint_model(tensors)
    with pytest.raises(ValueError):codec._model_checkpoint(value)


def test_modified_internal_checkpoint_is_rejected_before_receipt_reuse():
    value=model();tensors=codec._model_checkpoint(value)
    array=next(a for a in tensors.values() if a.dtype==np.int64)
    array.flags.writeable=True;array[0]=99
    with pytest.raises(ValueError,match="checkpoint changed"):codec._model_checkpoint(value)
    value.check_state()


@pytest.mark.parametrize("bound", ["_MAX_NODES", "_MAX_BYTES"])
def test_witness_admission_exhaustion_keeps_full_decoding(bound,monkeypatch):
    from rexgraph import model_integrity
    value=model();tensors=codec.pack_model(value)
    monkeypatch.setattr(model_integrity,bound,0)
    first=codec._checkpoint_model(tensors);second=codec._checkpoint_model(tensors)
    assert first is not second and first.coefficient_digest==second.coefficient_digest
    array=next(a for a in tensors.values() if a.dtype==np.int64);array[0]=99
    with pytest.raises(ValueError):codec._checkpoint_model(tensors)


def test_checkpoint_changes_during_decoding_are_rejected(monkeypatch):
    tensors=codec.pack_model(model());original=codec.unpack_model
    def changed(data):
        value=original(data);data["spec"][0]^=1
        return value
    monkeypatch.setattr(codec,"unpack_model",changed)
    with pytest.raises(ValueError,match="during decoding"):codec._checkpoint_model(tensors)


def test_evicted_checkpoints_retain_exact_cold_validation():
    value=model();tensors=codec._model_checkpoint(value)
    for i in range(1,8):codec._model_checkpoint(model(i))
    assert len(codec._checkpoint_cache)<=codec._checkpoint_limit
    restored=codec._checkpoint_model(tensors)
    assert restored is not value and restored.coefficient_digest==value.coefficient_digest
    changed=codec.pack_model(value);changed["unclaimed"]=np.asarray([1],np.uint8)
    with pytest.raises(ValueError):codec._checkpoint_model(changed)


def test_concurrent_checkpoint_checks_keep_distinct_source_content():
    values=[model(i) for i in range(8)]
    def run(i):
        value=values[i%len(values)];tensors=codec._model_checkpoint(value)
        restored=codec._checkpoint_model(freeze_tree(tensors))
        assert restored.coefficient_digest==value.coefficient_digest
        assert restored.configuration["index"]==i%len(values)
    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(run,range(32)))
