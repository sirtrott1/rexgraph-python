"""Explicit replay keeps original batches and complete independent graded families."""
from dataclasses import replace

import numpy as np
import pytest

from test_stalk_context_rules import learn
from test_graded_stalk_transport import trained
from rexgraph.cpp_stalk_transport import replay_stalk_transport_model, _prepare, _receipt_matches
from rexgraph.cpp_graded_stalk_transport import replay_graded_stalk_transport_model
from rexgraph.model_state import thaw_tree


@pytest.mark.parametrize('kind',['ordinary','graded'])
def test_changed_implementation_requires_explicit_original_observation_replay(kind):
    source,state,_=learn() if kind=='ordinary' else trained()
    c=thaw_tree(state.configuration);p=thaw_tree(state.payload)
    c['engine_receipt']['engine_headers']='a'*64;p['fit']['engine_headers']='a'*64
    if kind=='graded':p['fit']['source']['engine_headers']='a'*64
    old=replace(state,configuration=c,payload=p);digest=old.coefficient_digest
    if kind=='ordinary':
        with pytest.raises(ValueError,match='implementation'):_prepare(old,source)
        replay=replay_stalk_transport_model(old,source)
        assert _receipt_matches(state.payload['fit'],replay.payload['fit'])
    else:
        from rexgraph.cpp_graded_stalk_transport import _prepare as prepare
        with pytest.raises(ValueError,match='implementation'):prepare(old,source)
        replay=replay_graded_stalk_transport_model(old,source)
        assert _receipt_matches(state.payload['fit'],replay.payload['fit'])
    assert _receipt_matches(state.payload['observations'],replay.payload['observations'])
    assert old.coefficient_digest==digest and replay.source.matches(old.source)
    assert replay.configuration['engine_receipt']!=old.configuration['engine_receipt']


@pytest.mark.parametrize('kind',['ordinary','graded'])
@pytest.mark.parametrize('change',['coefficient','batch','input'])
def test_replay_rejects_changed_claimed_families_and_original_batches(kind,change):
    source,state,_=learn() if kind=='ordinary' else trained();p=thaw_tree(state.payload)
    if change=='coefficient':
        fit=p['fit'] if kind=='ordinary' else p['fit']['fit'];fit['particular']=(99,*fit['particular'][1:])
    elif change=='batch':p['observations'][0]['batch']='b'*64
    else:p['observations'][0]['input']=tuple(tuple(3 for _ in row) for row in p['observations'][0]['input'])
    with pytest.raises(ValueError):
        old=replace(state,payload=p)
        (replay_stalk_transport_model if kind=='ordinary' else replay_graded_stalk_transport_model)(old,source)


def test_checkpoint_tensor_receipts_preserve_dtype_shape_and_exact_bytes():
    a=np.asarray([-0.0,1.0]);b=a.copy()
    assert _receipt_matches(a,b)
    b[0]=0.0;assert not _receipt_matches(a,b)
    assert not _receipt_matches(a,a.view(np.uint64))
    assert not _receipt_matches(a,a.reshape(1,2))
