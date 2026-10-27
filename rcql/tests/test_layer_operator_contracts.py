"""Every registered layer/function entry has an executable typed contract."""
import inspect
import pytest
from rexgraph import RexGraph
from rexgraph.type_accession import CoordinateSpace
from rexgraph.coordinate_map import CoordinateMap
from rcql import Executor,SourcePolicy,bind,query,source,call
from rcql.layer_contracts import ARGUMENTS
from rcql.operators import get_operator
from rcql.signatures import lookup
from rcql.execution_trace import capture_methods
from rcql.executor import value_exactness


def cases(name):
    from rcql import Layer,LayerInterface,LayerProduct,LayerReadout,TemporalModel,LayerOperator
    r=RexGraph.from_cells([2,[[0,1]]]);bindings={'r':bind('r',r,SourcePolicy.allow('*'))}
    local=CoordinateSpace('coefficient',('a',));common=CoordinateSpace('agreement',('a',))
    action=CoordinateMap(local,common,((0,0,1),))
    layers=(Layer.affine('left',local),Layer.affine('right',local))
    interface=LayerInterface.equality('same',(('left',action),('right',action)))
    product=LayerProduct.create(layers,(interface,));solution=product.solve((('left','a',2),))
    readout=LayerReadout.create(solution,product.system().selection('right'))
    table={
        'LAYER':('new',local), 'LAYER_INTERFACE':('interface',(('left',action),('right',action))),
        'LAYER_PRODUCT':(layers,(interface,)), 'LAYER_MODIFY':(product,()),
        'LAYER_SOLVE':(product,(('left','a',2),)), 'LAYER_PROJECT':(solution,'right'),
        'LAYER_OBSERVE':(solution,product.system().selection('right')), 'LAYER_VALUE':(readout,),
        'LAYER_INFO':(solution,), 'LAYER_RECORD':(product,), 'LAYER_READ':(product.to_record(),),
    }
    if name in table:
        return r,table[name]
    pytest.importorskip('rexgraph_temporal')
    m=TemporalModel.create(789).declare('coordinate',dict(name='t',clock='event',unit='s',origin='0'));c=m.selected
    m=m.declare('function',dict(coordinate=c,pieces=({'numerator':(1,2,3)},)));f=m.selected
    m=m.declare('function',dict(coordinate=c,pieces=({'numerator':(2,6)},)));target=m.selected
    m=m.declare('program',{'opcode':'input'});x=m.selected
    m=m.declare('program',{'opcode':'derivative','operands':(x,)});dx=m.selected
    m=m.declare('support',{'coordinate':c,'components':((0,2,True,True),)});s=m.selected
    m=m.declare('context',{'support':s});context=m.selected
    m=m.declare('set',{'coordinate':c,'members':((101,1,f,s,True),)});members=m.selected
    m=m.fit(members,context,(x,dx),target);fit=m.selected
    native_layers=(Layer.function('left',m,s,program=dx),Layer.function('right',m,s,program=dx))
    native_product=LayerProduct.create(native_layers)
    operation=LayerOperator.compile(native_product,bindings,{})
    from rexgraph.native_field import NativeFieldCalculus
    target_space=NativeFieldCalculus.from_rex(r).complex.spaces[1]
    injection=CoordinateMap(CoordinateSpace('amplitude',('q',)),target_space,((0,0,1),))
    table={
        'LAYER_FUNCTION':('function',m,s,f), 'LAYER_FAMILY':('family',m,fit),
        'LAYER_COMPILE':(native_product,bindings,{}), 'LAYER_APPLY':(operation,bindings,{1:f}),
        'FUNCTION_REPLAY':(m,m.declaration['build'],(65536,128,256,1000000,1000000,4096,16000000,64000000,4096)),
        'FUNCTION_SOURCE':(123,), 'FUNCTION_DECLARE':(m,'support',{'coordinate':c,'components':((0,1,True,True),)}),
        'FUNCTION_PROGRAM':(m,'derivative',(x,)), 'FUNCTION_APPLY':(m,dx,{1:f}),
        'FUNCTION_FIT':(m,members,context,(x,dx),target), 'FUNCTION_TRANSFER':(m,fit,{1:f}),
        'FUNCTION_JOINT':(m,(fit,)), 'FUNCTION_MODIFY':(m,dx,{'opcode':'input','operands':()}),
        'FUNCTION_VALUE':(m,2,f), 'FUNCTION_ID':(m,), 'FUNCTION_INFO':(m,),
        'FUNCTION_VERIFY':(m,), 'FUNCTION_RECORD':(m,), 'FUNCTION_READ':(m.to_record(),),
        'FUNCTION_FIELD':(m,injection,1,CoordinateSpace('powers',('0','1','2')),f),
    }
    if name in {'FUNCTION_IMPORT','FUNCTION_EXPORT'}:
        pytest.importorskip('rexgraph_cpp._temporal_program')
        table.update(FUNCTION_IMPORT=(m.select(f,'function').to_relational().to_bytes(),'function'),
                     FUNCTION_EXPORT=(m.select(f,'function'),))
    return r,table[name]


@pytest.mark.parametrize('name',tuple(ARGUMENTS))
def test_layer_operator_matches_registered_contract(name):
    r,args=cases(name)
    parameters=list(inspect.signature(get_operator(name).fn).parameters.values())[1:]
    assert lookup(name).arity==(sum(p.default is p.empty for p in parameters),len(parameters))
    with capture_methods(binding=bind('r',r,SourcePolicy.allow('*'))):
        direct=get_operator(name).fn(r,*args)
    executor=Executor(sources={'r':r})
    result=executor.execute(query(source('r'),call(name,*args)))
    assert len(result.values)==1
    assert value_exactness(direct)==result.exactness[0]
