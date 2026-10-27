"""Read-authorized native demand execution; explicit archives are never memoized."""
from .layer_operators import _reference,_authorize
from .execution_trace import record_method
from .demand_gluing import DemandProgram,LIMITS

def demand_program(source,manifest,metadata,patches):
    return DemandProgram.create(manifest,metadata,patches,dependencies=(_reference(),)).data

def demand_revise(source,program,metadata,patches=None,sources=None):
    value=DemandProgram.from_bytes(program);_authorize(value,sources)
    return value.revise(metadata=metadata,patches=patches).data

def demand_run(source,program,archive,request,limits=LIMITS,sources=None):
    value=DemandProgram.from_bytes(program);bindings=_authorize(value,sources)
    for binding in bindings.values():
        binding.source.require('files');binding.source.require('file_read')
    record_method('native-demand-gluing',profile='demand-graph-identity-v1')
    return value.run(archive,request,limits=limits,sources=bindings)

def demand_replay(source,result,archive,expected_implementation,limits=LIMITS,sources=None):
    value=DemandProgram.from_bytes(result['program']);bindings=_authorize(value,sources)
    for binding in bindings.values():
        binding.source.require('files');binding.source.require('file_read')
    record_method('explicit-native-demand-replay',original_implementation=expected_implementation)
    return DemandProgram.replay(result,archive,expected_implementation,limits=limits,sources=bindings)

def install(register):
    for name in ('DEMAND_PROGRAM','DEMAND_REVISE','DEMAND_RUN','DEMAND_REPLAY'):register(name)(globals()[name.lower()])
