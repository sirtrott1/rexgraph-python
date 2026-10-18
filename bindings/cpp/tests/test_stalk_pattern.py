"""Compact native pattern inference and observable parameter provenance."""
from concurrent.futures import ThreadPoolExecutor
from fractions import Fraction as Q
from hashlib import sha256
import subprocess
import sys

import pytest

from rexgraph.cpp_stalk_pattern import compile_stalk_context_pattern
from rexgraph_cpp import PreparedStalkTransportPattern
from test_stalk_context_view_rules import learn, query


def compile_pattern(redundant=False):
    source, state, views = learn(redundant=redundant)
    pattern = compile_stalk_context_pattern(state, source, query(state, views), 0,
        model_namespace=904, coordinate_ids=(100,101,102))
    return source, state, pattern


@pytest.mark.parametrize('redundant', [False, True])
def test_exact_inference_owns_compact_weights_and_preserves_provenance(redundant):
    source, state, pattern = compile_pattern(redundant)
    inspected = pattern.inspect()
    assert len(inspected['weight_origin']) == 8
    assert inspected['parameter_directions']['shape'] == (8,0)
    assert all(inspected['identified'])
    assert inspected['source_sha256'] == sha256(source.archive).hexdigest()
    assert inspected['evidence_sha256'] == state.coefficient_digest
    assert inspected['observations'] == 4 and inspected['peers'] == 1
    assert len(inspected['occurrence_keys']) == 2
    assert {name:axes for _,name,axes in inspected['common_frames']} == {
        'participant':('x','y'), 'clause':('a','b','context','redundant')}
    assert len(inspected['pattern_source']) < len(source.archive)
    field = (Q(2,3), Q(-3,7), 0, Q(2,3) if redundant else 0)
    result = pattern.response(field)
    assert result['origin'] == (Q(149,105), Q(-10,7), Q(298,105), Q(-20,7))
    assert result['identified'] and result['directions']['shape'] == (4,0)
    inspected['weight_origin'] = (999,)
    assert pattern.inspect()['weight_origin'] != (999,)
    with pytest.raises(ValueError, match='outside'):
        pattern.response((0,0,1,0))


def test_cold_inference_requires_only_the_native_model_artifact(tmp_path):
    _, _, pattern = compile_pattern(True)
    image = pattern.archive()
    path = tmp_path/'pattern.rfcxs'
    path.write_bytes(image)
    script = '''
from pathlib import Path
from fractions import Fraction as Q
from rexgraph_cpp import PreparedStalkTransportPattern
import sys
p=PreparedStalkTransportPattern(Path(sys.argv[1]).read_bytes(),model_namespace=904)
r=p.response((Q(2,3),Q(-3,7),0,Q(2,3)))
assert r['origin']==(Q(149,105),Q(-10,7),Q(298,105),Q(-20,7))
assert r['identified']
assert p.archive()==Path(sys.argv[1]).read_bytes()
assert 'rexgraph.model_runtime' not in sys.modules
assert 'torch' not in sys.modules
'''
    child = subprocess.run([sys.executable,'-I','-c',script,str(path)],cwd=tmp_path,
        capture_output=True,text=True,timeout=60)
    assert child.returncode == 0, child.stderr
    loaded = PreparedStalkTransportPattern(image,model_namespace=904)
    assert loaded.inspect() == pattern.inspect()
    with ThreadPoolExecutor(max_workers=4) as pool:
        responses = tuple(pool.map(lambda _:loaded.response((Q(2,3),Q(-3,7),0,Q(2,3))),range(8)))
    assert all(r == responses[0] for r in responses)


@pytest.mark.parametrize('case', ['namespace','truncated','corrupt','dimension','entries','steps','bytes'])
def test_model_loading_enforces_canonical_namespace_and_admission(case):
    _, _, pattern = compile_pattern()
    image = pattern.archive(); ns = 904; limits = (4096,16000000,64000000,16384); maximum = len(image)
    if case == 'namespace': ns += 1
    if case == 'truncated': image = image[:-1]
    if case == 'corrupt': image = image[:-1]+bytes([image[-1]^1])
    if case == 'dimension': limits = (1,16000000,64000000,16384)
    if case == 'entries': limits = (4096,1,64000000,16384)
    if case == 'steps': limits = (4096,16000000,1,16384)
    if case == 'bytes': maximum -= 1
    with pytest.raises((ValueError, RuntimeError, OverflowError)):
        PreparedStalkTransportPattern(image,model_namespace=ns,limits=limits,max_bytes=maximum)


@pytest.mark.parametrize('case', ['unobserved','conflict','partial','coordinates','namespace'])
def test_unadmitted_patterns_do_not_become_learned_weights(case):
    source, state, views = learn(conflict=case=='conflict',partial=case=='partial')
    with pytest.raises((ValueError, RuntimeError, OverflowError)):
        compile_stalk_context_pattern(state,source,query(state,views),2 if case=='unobserved' else 0,
            model_namespace=0 if case=='namespace' else 904,
            coordinate_ids=(1,2,3) if case=='coordinates' else (100,101,102))
