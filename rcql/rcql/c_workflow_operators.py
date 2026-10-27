"""Existing RCQL authority applied to the closed native C workflow registry."""
from .execution_trace import record_method
from .layer_operators import _authorize,_reference
LIMITS=(32<<20,128,32,2000000,256,2048)
EXACT=(4096,16000000,64000000,1<<24)

def c_workflow_read(source,archive,limits=LIMITS,exact=EXACT):
    from .c_workflow import CWorkflow
    record_method('native-C-declaration-reconstruction')
    return CWorkflow.decode(archive,limits=limits,exact=exact,dependencies=(_reference(),))

def c_workflow_build(source,metadata,records,exports=(),limits=LIMITS,exact=EXACT):
    from .c_workflow import CWorkflow
    record_method('native-C-closed-constructors')
    return CWorkflow.create(metadata,records,exports,limits=limits,exact=exact,dependencies=(_reference(),))

def c_workflow_replay(source,workflow,expected_implementation,limits=LIMITS,exact=EXACT,sources=None):
    bindings=_authorize(workflow,sources)
    record_method('explicit-native-C-replay',original_implementation=expected_implementation)
    return workflow.replay(expected_implementation,limits=limits,exact=exact,sources=bindings)

def c_workflow_export(source,workflow,sources=None):
    _authorize(workflow,sources);workflow.native()
    return workflow.native().to_bytes()

def c_workflow_modify(source,workflow,replacements=(),exports=None,sources=None):
    _authorize(workflow,sources)
    return workflow.modify(replacements,exports=exports)

def c_value(source,workflow,index):return workflow.value(index)

def c_inspect(source,value,maximum=1000000,sources=None,scope=None):
    bindings=_authorize(value,sources);value.workflow.require_scope(scope)
    record_method('native-C-typed-inspection',constructor=value.constructor)
    return value.inspect(maximum,bindings)

def c_evaluate(source,value,operation,parameter=0,grade=0,power=0,maximum=1000000,sources=None,scope=None):
    bindings=_authorize(value,sources);value.workflow.require_scope(scope)
    record_method('native-C-typed-evaluation',constructor=value.constructor,operation=operation)
    return value.evaluate(operation,parameter,grade=grade,power=power,maximum=maximum,sources=bindings)

def c_primitives(source):
    from .c_workflow import primitive_ledger
    return {'profile':'closed-C-declarations-v1','primitives':primitive_ledger()}

def layer_c_value(source,key,value,role='participant'):
    from .layer_product import Layer
    return Layer.c_value(key,value,role=role)

def layer_c_interface(source,key,proof):
    from .c_workflow import CInterface
    return CInterface.declare(key,proof)

def install(register):
    from .c_workflow_contracts import ARGUMENTS
    for name in ARGUMENTS:register(name)(globals()[name.lower()])
