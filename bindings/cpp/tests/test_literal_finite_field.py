"""Byte derived finite readers retain native recipes, owners and full fields."""
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
import gc
import importlib.util
import json
import os
from pathlib import Path
import struct

import pytest
import rexgraph_cpp as cpp

POLICY={'limits':(4096,16000000,64000000,16384),'max_bytes':64*1024*1024}


@pytest.fixture(scope='module')
def original():
    root=os.environ.get('REXGRAPH_CPP_LITERAL_READER_PREFLIGHT')
    if not root:pytest.skip('original complete-use preflight not supplied')
    root=Path(root)
    spec=importlib.util.spec_from_file_location('literal_reader_preflight',root/'preflight.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    selection=json.loads((root/'source-selection-frozen.json').read_text())
    return module,selection,root


def fixture(original,n=3):
    module,selection,root=original
    inventory=tuple(bytes.fromhex(v) for v in selection['inventory_hex'])
    obj=module.reader(selection['cases'][str(n)],210000+n,inventory)
    d=deepcopy(obj['finite_spec']);d['key']+=100;d['revision']+=1
    for symbol in d['symbols']:
        b=next(b for b in d['bindings'] if symbol['key'] in d['variables'][b['cell'][1]]['domain'])
        symbol['definition']=cpp.encode_literal_use_value(obj['asset'],b['use'],**POLICY)
    finite=cpp.NativeFiniteSource(obj['asset'],d,**POLICY)
    request={'profile':'rexgraph.relational-literal-finite-field-map.v1','key':55001,'revision':1,
             'rules':({'key':55002,'bindings':tuple(b['key'] for b in d['bindings']),
                       'symbols':tuple(v['domain'][0] for v in d['variables']),
                       'target':0,'inputs':(obj['literal'],),'unsupported':''},)}
    model=cpp.PreparedLiteralFiniteFieldMap(finite,(obj['program'],),request,**POLICY)
    return obj,d,finite,request,model


def value_oracle(spans):
    body=struct.pack('<Q',len(spans))+b''.join(struct.pack('<Q',len(s))+s for s in spans)
    checksum=1469598103934665603
    for b in body:checksum=((checksum^b)*1099511628211)&((1<<64)-1)
    return b'RFCLV\1\0\0'+struct.pack('<IIQQ',1,0,32+len(body),checksum)+body


@pytest.mark.parametrize('n',(1,3,5,8))
def test_complete_native_readings_and_fields_match_prior_declared_program(original,n):
    obj,d,finite,request,model=fixture(original,n);result=model.execute()
    assert model.source().same_owner(finite) and finite.asset().same_owner(obj['asset'])
    assert result.map().same_owner(model)
    assert model.inspect()['rules'][0]['inputs'][0].same_owner(obj['literal'])
    assert result.family().select(0,1)[0].result().field_archive(1)==obj['result'].field_archive(1)
    assert result.family().select(0,1)[0].result().field_archive(2)==obj['result'].field_archive(2)
    assert result.family().select(0,1)[0].result().field_archive(2,occurrences=True)==obj['result'].field_archive(2,occurrences=True)
    assert model.raw_map().source().same_owner(finite)
    for u in obj['asset'].uses():
        assert cpp.encode_literal_use_value(obj['asset'],u['key'],**POLICY)==value_oracle(obj['asset'].recover(u['key']))
    assert cpp.NativeLiteralFiniteFieldResult.from_archive(result.archive(),**POLICY).archive()==result.archive()
    restored=cpp.PreparedLiteralFiniteFieldMap.from_archive(model.archive(),**POLICY)
    assert restored.archive()==model.archive() and not restored.same_owner(model)
    assert restored.inspect()['rules'][0]['inputs'][0].asset().same_owner(restored.source().asset())
    assert restored.execute().archive()==result.archive()
    assert len(restored.source().asset().uses())==n


def test_old_opaque_packet_is_preserved_and_not_reinterpreted(original):
    obj,d,finite,request,model=fixture(original)
    old=obj['finite'].archive();assert cpp.NativeFiniteSource.from_archive(old,**POLICY).archive()==old
    with pytest.raises((ValueError,RuntimeError)):
        cpp.PreparedLiteralFiniteFieldMap(obj['finite'],(obj['program'],),request,**POLICY)
    assert old==obj['finite'].archive() and finite.inspect()['revision']==2


@pytest.mark.parametrize('fault',('forged-definition','underived-symbol','missing-use','nonsingleton','missing-rule-dependency','foreign-literal','foreign-program','no-input','raw-input'))
def test_missing_byte_proofs_owners_and_incomplete_consumption_refuse(original,fault):
    obj,d,finite,request,model=fixture(original);request=dict(request,rules=tuple(dict(r) for r in request['rules']));targets=(obj['program'],)
    if fault=='forged-definition':d['symbols'][0]['definition']=b'forged'
    elif fault=='underived-symbol':d['symbols']=(*d['symbols'],{'key':999999,'definition':cpp.encode_literal_use_value(obj['asset'],1,**POLICY)})
    elif fault=='missing-use':
        d['variables']=d['variables'][:-1];d['bindings']=d['bindings'][:-1];d['charts']=()
        used={v['domain'][0] for v in d['variables']};d['symbols']=tuple(v for v in d['symbols'] if v['key'] in used)
        request['rules'][0]['bindings']=request['rules'][0]['bindings'][:-1];request['rules'][0]['symbols']=request['rules'][0]['symbols'][:-1]
    elif fault=='nonsingleton':
        d['variables'][0]['domain']=(d['variables'][0]['domain'][0],d['variables'][1]['domain'][0]);d['charts']=()
    elif fault=='missing-rule-dependency':
        request['rules'][0]['bindings']=request['rules'][0]['bindings'][:-1];request['rules'][0]['symbols']=request['rules'][0]['symbols'][:-1]
    elif fault=='foreign-literal':request['rules'][0]['inputs']=(cpp.NativeLiteralInput.from_archive(obj['literal'].archive(),**POLICY),)
    elif fault=='foreign-program':targets=(cpp.PreparedAccessionFieldProgram(obj['program'].archive(),**POLICY),)
    elif fault=='no-input':request['rules'][0]['inputs']=()
    else:request['rules'][0]['inputs']=(obj['literal'].input(),)
    finite=cpp.NativeFiniteSource(obj['asset'],d,**POLICY)
    with pytest.raises((ValueError,RuntimeError,TypeError)):
        cpp.PreparedLiteralFiniteFieldMap(finite,targets,request,**POLICY)


def test_multiple_native_branches_keep_full_recipe_and_unsupported_status(original):
    obj,d,finite,request,model=fixture(original);request=dict(request,rules=tuple(dict(r) for r in request['rules']))
    request['rules']=(*request['rules'],dict(request['rules'][0],key=55003),
        dict(request['rules'][0],key=55004,target=None,inputs=(),unsupported='unresolved interpretation'))
    result=cpp.PreparedLiteralFiniteFieldMap(finite,(obj['program'],),request,**POLICY).execute()
    assert tuple(b.inspect()['status'] for b in result.family().select(0,3))==('interpreted','interpreted','unsupported')
    assert cpp.NativeLiteralFiniteFieldResult.from_archive(result.archive(),**POLICY).archive()==result.archive()


def test_ordered_repeated_overlapping_discontinuous_and_empty_spans(original):
    module,_,_=original
    row={'case_id':'reader ordered span control','raw_utf8':'aba aba!','byte_cutoff':8,'uses':[
        {'original_token_id':1,'spans':[[0,3],[4,7],[0,0]]},
        {'original_token_id':2,'spans':[[4,7],[0,3],[1,6]]},
        {'original_token_id':3,'spans':[[0,3]]}]}
    owner,asset,_,_=module.source_chart(row,230001)
    for use in (1,2,3):assert cpp.encode_literal_use_value(asset,use,**POLICY)==value_oracle(asset.recover(use))
    assert cpp.encode_literal_use_value(asset,1,**POLICY)!=cpp.encode_literal_use_value(asset,3,**POLICY)
    with pytest.raises((ValueError,RuntimeError)):cpp.encode_literal_use_value(asset,999,**POLICY)
    with pytest.raises((ValueError,RuntimeError)):cpp.encode_literal_use_value(asset,1,max_bytes=39)
    with pytest.raises((ValueError,RuntimeError)):cpp.encode_literal_use_value(asset,1,limits=(1,16000000,64000000,16384))


def test_composite_retained_after_owner_lifetime_and_concurrent_cold_replay(original):
    obj,d,finite,request,model=fixture(original);result=model.execute();packet=result.archive()
    escaped=result.map().inspect()['rules'][0]['inputs'][0];del obj,d,finite,request,model,result;gc.collect()
    assert len(escaped.asset().uses())==3
    def read(_):
        r=cpp.NativeLiteralFiniteFieldResult.from_archive(packet,**POLICY)
        return r.archive(),r.map().inspect()['rules'][0]['inputs'][0].asset().archive()==r.map().source().asset().archive()
    with ThreadPoolExecutor(max_workers=4) as pool:assert tuple(pool.map(read,range(8)))==((packet,True),)*8


def test_immutable_reentrant_and_bounded_wrappers(original):
    obj,d,finite,request,model=fixture(original)
    with pytest.raises(TypeError):model.__init__(finite,(obj['program'],),request)
    owner=cpp.PreparedLiteralFiniteFieldMap.__new__(cpp.PreparedLiteralFiniteFieldMap)
    class Targets:
        def __iter__(self):
            with pytest.raises(TypeError):owner.__init__(finite,(obj['program'],),request)
            return iter((obj['program'],))
    owner.__init__(finite,Targets(),request,**POLICY)
    assert owner.archive()==model.archive()
    for cls in (cpp.PreparedLiteralFiniteFieldMap,cpp.NativeLiteralFiniteFieldResult):
        with pytest.raises(RuntimeError):cls.__new__(cls).archive()
    with pytest.raises((ValueError,RuntimeError)):model.inspect(max_records=1)
    for obj_,cls in ((model,cpp.PreparedLiteralFiniteFieldMap),(model.execute(),cpp.NativeLiteralFiniteFieldResult)):
        packet=obj_.archive()
        for bad in (packet[:1],packet[:-1],packet+b'trailing'):
            with pytest.raises((ValueError,RuntimeError)):cls.from_archive(bad,**POLICY)
        with pytest.raises((ValueError,RuntimeError)):cls.from_archive(packet,max_bytes=32)
        changed=bytearray(packet);changed[-1]^=1
        with pytest.raises((ValueError,RuntimeError)):cls.from_archive(bytes(changed),**POLICY)
    assert cpp.literal_finite_field_build_info()['authority']=='raw-none'
