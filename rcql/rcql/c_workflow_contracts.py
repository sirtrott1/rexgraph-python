"""Typed closed C declarations; planning never reconstructs native certificates."""
from .types import RCType, ValueKind, Exactness

ARGUMENTS={
    'C_WORKFLOW_READ':(('archive','limits','exact'),((32<<20,128,32,2000000,256,2048),(4096,16000000,64000000,1<<24))),
    'C_WORKFLOW_BUILD':(('metadata','records','exports','limits','exact'),((),(32<<20,128,32,2000000,256,2048),(4096,16000000,64000000,1<<24))),
    'C_WORKFLOW_REPLAY':(('workflow','expected_implementation','limits','exact','sources'),((32<<20,128,32,2000000,256,2048),(4096,16000000,64000000,1<<24),None)),
    'C_WORKFLOW_EXPORT':(('workflow','sources'),(None,)),
    'C_WORKFLOW_MODIFY':(('workflow','replacements','exports','sources'),((),None,None)),
    'C_VALUE':(('workflow','index'),()),
    'C_INSPECT':(('value','maximum','sources','scope'),(1000000,None,None)),
    'C_EVALUATE':(('value','operation','parameter','grade','power','maximum','sources','scope'),(0,0,0,1000000,None,None)),
    'C_PRIMITIVES':((),()),
    'LAYER_C_VALUE':(('key','value','role'),('participant',)),
    'LAYER_C_INTERFACE':(('key','proof'),()),
}

def install(register,OperatorSignature,TypePattern):
    v=ValueKind
    table={
        'C_WORKFLOW_READ':((bytes,None,None),v.C_WORKFLOW),
        'C_WORKFLOW_BUILD':((None,None,None,None,None),v.C_WORKFLOW),
        'C_WORKFLOW_REPLAY':((v.C_WORKFLOW,None,None,None,None),v.C_WORKFLOW),
        'C_WORKFLOW_EXPORT':((v.C_WORKFLOW,None),v.BYTES),
        'C_WORKFLOW_MODIFY':((v.C_WORKFLOW,None,None,None),v.C_WORKFLOW),
        'C_VALUE':((v.C_WORKFLOW,int),v.C_VALUE),
        'C_INSPECT':((v.C_VALUE,int,None,None),v.RECORD),
        'C_EVALUATE':((v.C_VALUE,str,None,int,int,int,None,None),v.RECORD),
        'C_PRIMITIVES':((),v.RECORD),
        'LAYER_C_VALUE':((str,v.C_VALUE,str),v.LAYER),
        'LAYER_C_INTERFACE':((str,v.C_VALUE),v.C_INTERFACE),
    }
    for name,(labels,defaults) in ARGUMENTS.items():
        kinds,output=table[name]
        patterns=tuple(TypePattern(label,kind=kind,optional=i>=len(labels)-len(defaults)) if isinstance(kind,ValueKind)
                       else TypePattern(label,literal=kind,optional=i>=len(labels)-len(defaults))
                       for i,(label,kind) in enumerate(zip(labels,kinds,strict=True)))
        result=RCType(output.value,kind=output,exactness=Exactness.STRUCTURAL,
                      layer_profile='c_workflows' if name=='LAYER_C_VALUE' else None)
        register(OperatorSignature(name,v.REX,patterns,result,'rcql.c_workflow_operators.'+name.lower(),requires=frozenset({'read'}),memoizable=True))

def literal_type(binding,value):
    from .c_workflow import CWorkflow,CValue,CInterface
    kinds={CWorkflow:ValueKind.C_WORKFLOW,CValue:ValueKind.C_VALUE,CInterface:ValueKind.C_INTERFACE}
    if type(value) in kinds:
        return RCType(type(value).__name__,kind=kinds[type(value)],exactness=Exactness.STRUCTURAL,
                      declaration_digest=value.coefficient_digest)
    return None

def refine(typed,children,context):
    context.binding.source.require('read')
    # Descriptors are prospective. No native(), decode(), solve() or evaluate().
    return typed.result,()
