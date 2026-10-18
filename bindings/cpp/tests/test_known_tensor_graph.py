"""Known input graph ownership, declared images and exact source fields."""
from copy import deepcopy
from fractions import Fraction as Q
from concurrent.futures import ThreadPoolExecutor
import gc
import pytest
import rexgraph_cpp as cpp
from test_literal_input import declaration as literal_declaration
from test_relational_field_evidence import actual_asset, input_accession, learning_declaration, read_matrix
from test_relational_field_program import source, matrix, tensor, UPPER


def graph_declaration():
    return {'profile':'rexgraph.relational-known-tensor-graph.v1','key':8101,'revision':1,
            'blocks':({'cell':(1,0),'source_basis':4301,'lifted_basis':4501,
                       'left':(4311,),'right':(4312,),'products':(4511,)},)}


def assert_product_scalar(image, value):
    assert tuple(int.from_bytes(image[i:i+8],'little') for i in (0,8,16))==(2,1,1)
    assert image[24:]==tensor((value,))[16:]


def small(raw=b'bank', *, parent_program=None, target_program=None, generic=False):
    owner=source()
    spec=learning_declaration();spec['nodes']=spec['nodes'][:2]
    for occurrence in spec['occurrences']:
        occurrence['map']={'shape':occurrence['map']['shape'],'entries':()}
    parent_program=parent_program or cpp.PreparedAccessionFieldProgram(cpp.PreparedFieldProgram(owner.archive,spec).archive())
    asset=actual_asset(999991,111,raw)
    accession=cpp.NativeInputAccession(asset,parent_program,input_accession())
    literal=cpp.NativeLiteralInput(accession,literal_declaration())
    graph=cpp.NativeKnownTensorGraph(literal.input() if generic else literal,graph_declaration())
    stalks=tuple({**s,'metric':matrix(read_matrix(s['metric']))} for s in graph.stalks())
    if target_program is None:
        spec['revision']+=1
        spec['stalks']=tuple(s for s in spec['stalks'] if s['cell'][0]==0)+stalks
        for i,occurrence in enumerate(spec['occurrences']):
            occurrence['map']={'shape':(occurrence['map']['shape'][0],len(stalks[UPPER[i]]['directions'])),'entries':()}
        target_program=cpp.PreparedAccessionFieldProgram(cpp.PreparedFieldProgram(owner.archive,spec).archive())
    target_spec=deepcopy(input_accession());target_spec['stalks']=stalks
    target_spec['accession']=matrix(tuple(tuple(int(i==j) for j in range(5)) for i in range(5)))
    target_spec['reconstruction']=target_spec['accession']
    target=cpp.NativeInputAccession(asset,target_program,target_spec)
    admitted=graph.admit(target)
    return locals()


def test_complete_graph_lifetime_source_field_and_live_admission():
    f=small();g=f['graph'];value=f['admitted']
    assert g.literal_parent().same_owner(f['literal'])
    assert g.parent().inspect()==f['literal'].input().inspect()
    assert g.lifted_field()==tensor((1,1,1,1,0))
    assert_product_scalar(g.product((1,0)),1)
    assert value.input().inspect()['field']==g.lifted_field()
    assert value.program().execute((value.input(),)).field_archive(1)==g.lifted_field()
    assert read_matrix(g.matrix('metric'))[2][2]==6
    assert g.declaration()==graph_declaration()
    g.require_image(g.lifted_field())
    image=value.archive();del f,g;gc.collect()
    assert value.graph().literal_parent().asset().recover(111)==(b'bank',b'',b'ban')
    assert cpp.NativeKnownTensorGraphInput.from_archive(image).archive()==image


def test_generic_source_input_graph_and_original_not_represented_values():
    f=small();a=deepcopy(input_accession())
    a['accession']=matrix(((2,0,0,0),(0,2,0,0),(0,0,2,0),(0,0,0,2)))
    a['reconstruction']=matrix(((Q(1,2),0,0,0),(0,Q(1,2),0,0),(0,0,Q(1,2),0),(0,0,0,Q(1,2))))
    a['isometric']=False
    accession=cpp.NativeInputAccession(f['asset'],f['parent_program'],a)
    parent=cpp.NativeLiteralInput(accession,literal_declaration())
    assert parent.input().inspect()['field']!=parent.input().inspect()['represented']
    graph=cpp.NativeKnownTensorGraph(parent,graph_declaration())
    assert graph.lifted_field()==f['graph'].lifted_field()
    generic=cpp.NativeKnownTensorGraph(parent.input(),graph_declaration())
    assert generic.literal_parent() is None
    assert generic.lifted_field()==graph.lifted_field()


def test_complete_multistalk_graph_supported_zeros_and_complement():
    f=small();d=graph_declaration();d['blocks']+=({'cell':(1,1),'source_basis':4401,'lifted_basis':4502,'left':(4411,),'right':(4412,),'products':(4512,)},)
    graph=cpp.NativeKnownTensorGraph(f['literal'],d)
    assert graph.lifted_field()==tensor((1,1,1,1,0,0))
    assert_product_scalar(graph.product((1,1)),0)
    assert graph.stalks()[1]['directions']==(4411,4412,4512)
    with pytest.raises(ValueError):graph.product((0,0))
    with pytest.raises(ValueError):graph.require_image(tensor((1,1,0,1,0,0)))


@pytest.mark.parametrize('fault',('profile','missing','repeat-cell','basis','duplicate-left','foreign-right','missing-product','collision','grade'))
def test_declared_graph_rejects_unsupported_maps(fault):
    f=small();d=graph_declaration()
    if fault=='profile':d['profile']='other'
    elif fault=='missing':d['blocks']=()
    elif fault=='repeat-cell':d['blocks']*=2
    elif fault=='basis':d['blocks'][0]['source_basis']=999
    elif fault=='duplicate-left':d['blocks'][0]['left']=(4311,4311)
    elif fault=='foreign-right':d['blocks'][0]['right']=(999,)
    elif fault=='missing-product':d['blocks'][0]['products']=()
    elif fault=='collision':d['blocks'][0]['products']=(4311,)
    else:d['blocks'][0]['cell']=(0,0)
    with pytest.raises((ValueError,RuntimeError)):cpp.NativeKnownTensorGraph(f['literal'],d)


def test_cold_byte_policy_bounds_and_immutable_publication():
    f=small();g=f['graph'];image=g.archive();cold=cpp.NativeKnownTensorGraph.from_archive(image)
    assert cold.archive()==image and not cold.same_owner(g) and cold.same_owner(cold)
    with pytest.raises(TypeError):g.__init__(f['literal'],graph_declaration())
    with pytest.raises((ValueError,RuntimeError)):g.declaration(max_records=3)
    with pytest.raises((ValueError,RuntimeError)):g.stalks(max_records=1)
    with pytest.raises((ValueError,RuntimeError)):g.require_image(tensor((1,1,0,1,0)))
    for packet in (image[:-1],image[:-1]+bytes((image[-1]^1,)),image+b'x'):
        with pytest.raises((ValueError,RuntimeError)):cpp.NativeKnownTensorGraph.from_archive(packet)
    with pytest.raises(ValueError):cpp.NativeKnownTensorGraph.from_archive(image,max_bytes=len(image)-1)
    with pytest.raises(ValueError):cpp.NativeKnownTensorGraph.from_archive(image,limits=(4096,16000001,64000000,16384))


def test_callback_cancel_reentry_and_subclass_cold_exposure_are_safe():
    f=small();target=cpp.NativeKnownTensorGraph.__new__(cpp.NativeKnownTensorGraph)
    class Stop(Exception):pass
    error=Stop('cancel this operation')
    def cancel():
        with pytest.raises(RuntimeError):target.lifted_field()
        raise error
    with pytest.raises(Stop) as caught:target.__init__(f['literal'],graph_declaration(),check=cancel)
    assert caught.value is error
    target.__init__(f['literal'],graph_declaration())
    with pytest.raises(Stop):target.archive(check=lambda:(_ for _ in ()).throw(error))
    with pytest.raises(Stop):target.require_image(target.lifted_field(),check=lambda:(_ for _ in ()).throw(error))
    exposed=[]
    class Child(cpp.NativeKnownTensorGraph):
        def __new__(cls):
            instance=super().__new__(cls);exposed.append(instance);return instance
    def observe():
        with pytest.raises(RuntimeError):exposed[-1].admission()
        with pytest.raises(TypeError):exposed[-1].__init__(f['literal'],graph_declaration())
    restored=Child.from_archive(target.archive(),check=observe)
    assert restored.archive()==target.archive()
    with pytest.raises(TypeError):cpp.NativeKnownTensorGraph.from_archive(target.archive(),check=5)


@pytest.mark.parametrize('kind',('graph','input'))
def test_cold_subclass_cannot_rebind_an_existing_immutable_owner(kind):
    f=small();value=f['graph'] if kind=='graph' else f['admitted'];packet=value.archive()
    class Reused(type(value)):
        def __new__(cls):return value
    with pytest.raises(TypeError):Reused.from_archive(packet)
    assert value.archive()==packet


def test_concurrent_actual_owner_pins_survive_gc():
    value=small()['admitted'];image=value.archive();g=value.graph()
    def run(_):
        gc.collect()
        return value.archive(),g.archive(),value.program().execute((value.input(),)).field_archive(1)
    with ThreadPoolExecutor(max_workers=2) as pool:
        results=list(pool.map(run,range(4)))
    assert all(row==(image,g.archive(),g.lifted_field()) for row in results)
