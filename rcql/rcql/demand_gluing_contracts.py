"""Planning retains a closed prospective program and never opens its payload store."""
from .demand_gluing_profile import LIMITS
from .types import RCType,ValueKind,Exactness,Effect
ARGUMENTS={
    'DEMAND_PROGRAM':(('manifest','metadata','patches'),()),
    'DEMAND_REVISE':(('program','metadata','patches','sources'),(None,None)),
    'DEMAND_RUN':(('program','archive','request','limits','sources'),(LIMITS,None)),
    'DEMAND_REPLAY':(('result','archive','expected_implementation','limits','sources'),(LIMITS,None)),
}
def install(register,OperatorSignature,TypePattern):
    for name,(labels,defaults) in ARGUMENTS.items():
        kinds={'DEMAND_PROGRAM':(bytes,None,None),'DEMAND_REVISE':(bytes,None,None,None),'DEMAND_RUN':(bytes,str,None,None,None),'DEMAND_REPLAY':(None,str,None,None,None)}[name]
        patterns=tuple(TypePattern(label,kind=ValueKind.BYTES,literal=bytes,optional=j>=len(labels)-len(defaults)) if kind is bytes else TypePattern(label,literal=kind,optional=j>=len(labels)-len(defaults)) for j,(label,kind) in enumerate(zip(labels,kinds,strict=True)))
        output=ValueKind.BYTES if name in ('DEMAND_PROGRAM','DEMAND_REVISE') else ValueKind.RECORD
        register(OperatorSignature(name,ValueKind.REX,patterns,RCType(output.value,kind=output,exactness=Exactness.STRUCTURAL),'rcql.demand_gluing_operators.'+name.lower(),requires=frozenset({'read','files','file_read'} if name in ('DEMAND_RUN','DEMAND_REPLAY') else {'read'}),effects=frozenset({Effect.READ,Effect.FILESYSTEM} if name in ('DEMAND_RUN','DEMAND_REPLAY') else {Effect.READ}),memoizable=False))
